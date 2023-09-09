from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import pandas as pd
import numpy as np
import random
import math
import Cytograph


def st_mean(st_exp):
    value = st_exp.mean()
    mask = st_exp >= value
    st_exp[mask] = 1
    st_exp[~mask] = 0
    return st_exp


def st_median(st_exp):
    value = st_exp.median()
    mask = st_exp >= value
    st_exp[mask] = 1
    st_exp[~mask] = 0
    return st_exp


def preprocess_st(st_data, filtering):
    if filtering == "mean":
        st_data = st_data.apply(st_mean, axis=1)
    if filtering == "median":
        st_data = st_data.apply(st_median, axis=1)
    else:
        st_data[st_data > 0] = 1
    return st_data


def get_distance(st_meta, distance_threshold):
    if 'Z' in st_meta.columns:
        st_meta = st_meta.astype({'X': 'float', 'Y': 'float', 'Z': 'float'})
        A = st_meta[["X", "Y", "Z"]].values
    if 'z' in st_meta.columns:
        st_meta = st_meta.astype({'x': 'float', 'y': 'float', 'z': 'float'})
        A = st_meta[["x", "y", "z"]].values
    if 'X' in st_meta.columns:
        st_meta = st_meta.astype({'X': 'float', 'Y': 'float'})
        A = st_meta[["X", "Y"]].values
    if 'x' in st_meta.columns:
        st_meta = st_meta.astype({'x': 'float', 'y': 'float'})
        A = st_meta[["x", "y"]].values
    else:
        raise ValueError("the coordinate information must be included in st_meta")

    distA = squareform(pdist(A, metric='euclidean'))
    if distance_threshold:
        distance_rank = np.sort(distA, axis=1)
        dis_f = np.percentile(distance_rank[:, 1:11].flatten(), 95)
        distA = np.where(distA <= dis_f, distA, 0)
    # distA[distA>distance_threshold] =float('-inf')
    if distA.sum() == 0:
        raise ValueError("invalid distance threshold")
    dist_data = pd.DataFrame(data=distA, index=st_meta["cell"].values.tolist(), columns=st_meta["cell"].values.tolist())
    return dist_data


def co_exp(matrix):
    co_exp_ratio = np.count_nonzero(matrix, axis=1) / matrix.shape[1]
    return co_exp_ratio


def co_exp_list(exp_list):
    co_exp_ratio = np.count_nonzero(exp_list) / len(exp_list)
    return co_exp_ratio


def get_cell_list(st_meta):
    cell_type = list(set(st_meta["cell_type"].values.tolist()))
    # print(cell_type)
    sender_list = []
    receiver_list = []
    for i in range(len(cell_type)):
        receiver_list += [cell_type[i]] * ((len(cell_type)))
        # sender_list += cell_type[0:[cell_type[i]]*(len(cell_type))]
        sender_list += cell_type[0:len(cell_type)]
    # print(sender_list)
    # print(receiver_list)
    return sender_list, receiver_list, cell_type


def get_cell_pair(st_meta, dist_data, cell_sender_name, cell_receiver_name, n_neighbor=10, min_pairs_ratio=0.001):
    cell_sender = st_meta["cell"][st_meta["cell_type"] == cell_sender_name].values.tolist()
    cell_receiver = st_meta["cell"][st_meta["cell_type"] == cell_receiver_name].values.tolist()
    pair_sender = []
    pair_receiver = []
    distance = []
    sender_type = []
    receiver_type = []

    for i in cell_sender:
        pairs = dist_data[i][dist_data[i] != 0].sort_values()[:n_neighbor].index.values.tolist()
        pair_sender += [i] * len(pairs)
        pair_receiver += pairs
        # print(dist_data[i][dist_data[i]!=0].sort_values()[:n_neighbor].values)
        distance += dist_data[i][dist_data[i] != 0].sort_values()[:n_neighbor].values.tolist()
        sender_type += [cell_sender_name] * len(pairs)
        receiver_type += [cell_receiver_name] * len(pairs)

    cell_pair = pd.DataFrame(
        {'cell_sender': pair_sender, 'cell_receiver': pair_receiver, 'distance': distance, "sender_type": sender_type,
         "receiver_type": receiver_type})
    cell_pair = cell_pair[(cell_pair['cell_receiver'].isin(cell_receiver))]
    all_pair_number = len(cell_sender) * len(cell_receiver)
    pair_number = cell_pair.shape[0]
    flag = 1
    if pair_number <= all_pair_number * min_pairs_ratio:
        print(f"Cell pairs found between {cell_sender_name} and {cell_receiver_name} less than min_pairs_ratio!")
        flag = 0
    return cell_pair, flag


def find_sig_lr(st_data, lr_pair, cell_pair, per_num=1000, pvalue=0.05):
    data_ligand = st_data.loc[lr_pair["ligand"].values.tolist(), cell_pair["cell_sender"].values.tolist()]
    data_receptor = st_data.loc[lr_pair["receptor"].values.tolist(), cell_pair["cell_receiver"].values.tolist()]
    lr_matrix = data_ligand.values * data_receptor.values
    if lr_matrix.shape[0] > 1:
        co_exp_value = co_exp(lr_matrix)

        co_exp_number = [x * len(lr_matrix[0]) for x in co_exp_value]
    else:
        co_exp_value = co_exp_list(lr_matrix[0])
        co_exp_number = co_exp_value * lr_matrix.shape[1]

    per_exp_ratio = []
    items = st_data.columns.values.tolist()
    replace_flag = False
    if cell_pair.shape[0] * 2 > len(items):
        replace_flag = True
    for i in range(per_num):
        random.seed(i)
        cell_id = np.random.choice(items, cell_pair.shape[0] * 2, replace=replace_flag)
        # cell_id = random.sample(items,cell_pair.shape[0]*2)
        per_ligand = st_data.loc[
            lr_pair["ligand"].values.tolist(), cell_id[0:int(len(cell_id) / 2)]]
        per_receptor = st_data.loc[
            lr_pair["receptor"].values.tolist(), cell_id[int(len(cell_id) / 2):int(len(cell_id))]]
        per_matrix = per_ligand.values * per_receptor.values
        per_exp_ratio.append(co_exp(per_matrix))
    per_data = np.mat(per_exp_ratio)
    co_exp_p = []
    if lr_matrix.shape[0] > 1:
        for j in range(len(co_exp_value)):
            co_exp_p.append((np.sum(per_data[:, j] >= co_exp_value[j])) / per_num)
    else:
        co_exp_p = (np.sum(per_data >= co_exp_value)) / per_num
    lr_pair = lr_pair.assign(co_exp_value=co_exp_value, co_exp_number=co_exp_number, co_exp_p=co_exp_p)
    lr_pair = lr_pair[lr_pair['co_exp_p'] < pvalue]

    return lr_pair


def find_high_exp_path(pathway, receiver_list, st_data):
    st_data = st_data.loc[:, receiver_list]
    per_src = st_data.loc[pathway['src'].values.tolist(), :]
    per_dest = st_data.loc[pathway['dest'].values.tolist(), :]
    per_matrix = per_src.values * per_dest.values
    per_exp = co_exp(per_matrix)

    pathway["co_exp_ratio"] = per_exp
    return pathway


def get_score(sig_lr_pair, tf):
    tf_score = tf.groupby(by=['receptor'])['score_rt'].sum()
    tf_score = tf_score * (-1)
    sig_lr_pair["lr_score"] = 1 - sig_lr_pair["co_exp_p"]
    rt_score = tf_score.map(lambda x: 1 / (1 + math.exp(x)))
    rt = pd.DataFrame({'receptor': rt_score.index, 'rt_score': rt_score.values})
    result = pd.merge(sig_lr_pair, rt, on=['receptor'])
    result["score"] = result.apply(lambda x: math.sqrt(x.lr_score * x.rt_score), axis=1)
    return result


def post_process(results):
    df_empty = pd.DataFrame(
        columns=["ligand", "receptor", "species", "cell_sender", "cell_receiver", "co_exp_value", "co_exp_number",
                 "co_exp_p", "lr_score", "rt_score", "score"])
    pair_empty = pd.DataFrame(columns=["cell_sender", "cell_receiver", "distance", "sender_type", "receiver_type"])
    obj = {}
    for result in results:
        if result is not None:
            df_empty = pd.concat([df_empty, result[0]], axis=0)
            pair_empty = pd.concat([pair_empty, result[1]], axis=0)
    obj['lr_score'] = df_empty
    pair_empty.rename(columns={"cell_sender": "cell_sender_id", "cell_receiver": "cell_receiver_id"}, inplace=True)
    obj['pair_distance'] = pair_empty

    return obj


def process_sender_receiver(i, lr_pair, dist_data, sender_list, receiver_list, st_meta, pathway, valid_st_data, st_data,
                            max_hop):
    cell_sender = sender_list[i]
    cell_receiver = receiver_list[i]
    # for each cell type receiver, find all edge with expression ration>0.1
    cell_pair, flag = get_cell_pair(st_meta, dist_data, cell_sender, cell_receiver)
    if flag == 0:
        return
    print(f"The cell pair number found between {cell_sender} and {cell_receiver} is {cell_pair.shape[0]}")
    f_path = find_high_exp_path(pathway, cell_pair["cell_receiver"].values.tolist(), valid_st_data)
    f_path = f_path[f_path['co_exp_ratio'] > 0.10]
    path_gene = list(set(f_path['src'].values).union(set(f_path['dest'].values)))
    pathway_graph = Cytograph.PathGraph(path_gene, max_hop)
    pathway_graph.built_edge(f_path)
    receptor_gene = pathway_graph.find_valid_lr_pair()
    # de_bug = lr_pair[lr_pair['receptor'].isin(f_path["src"])]
    lr_pair = lr_pair[lr_pair['receptor'].isin(receptor_gene)]

    if lr_pair.shape[0] == 0:
        print(
            f"No ligand-recepotor pairs found between {cell_sender} and {cell_receiver} because of no downstream transcriptional factors found for receptors!")
        return
    else:
        print(f"the number of valid pathway number between {cell_sender} and {cell_receiver} is: {lr_pair.shape[0]}")
        # find cell pair
        lr_pair.insert(0, 'cell_sender', cell_sender)
        lr_pair.insert(1, 'cell_receiver', cell_receiver)

        sig_lr_pair = find_sig_lr(st_data, lr_pair, cell_pair)

        if sig_lr_pair.shape[0] == 0:
            print(
                f"No ligand-recepotor pairs found between {cell_sender} and {cell_receiver} with significant expression")
            return
        print(
            f"The ligand-recepotor pairs found between {cell_sender} and {cell_receiver} with significant expression is {sig_lr_pair.shape[0]}")

        tf = pathway_graph.find_lr_tf(sig_lr_pair)

        path_score = get_score(sig_lr_pair, tf)

        print(f"{cell_receiver} and {cell_sender} done")
    return path_score, cell_pair
