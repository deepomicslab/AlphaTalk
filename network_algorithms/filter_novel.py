import pandas as pd
from itertools import product
import numpy as np
from joblib import Parallel, delayed
from utils import LRBipartite


def get_novel_once(batch, peer_num, pred_thd, topo_thd, lr_type):
    result_ls = []
    if lr_type == 'ligand':
        df_pred1 = df_pred.loc[df_pred['receptor'].isin(lrdb_receptors)]
        for test_ligand in batch:
            df_tmp = df_pred1.query('ligand == @test_ligand and pred > @pred_thd')
            if len(df_tmp) == 0:
                continue

            filtered = lrdb_B.filter_pred(df_tmp['receptor'], lrdb_B.r_jm, peer_num=peer_num, threshold=topo_thd)
            if filtered is None:
                continue
            else:
                couple = np.array([[test_ligand] * len(filtered), filtered])
                result_ls.append(couple)

        if len(result_ls) > 0:
            out = np.concatenate(result_ls, axis=-1)
        else:
            out = None
    else:
        df_pred1 = df_pred.loc[df_pred['ligand'].isin(lrdb_ligands)]
        for test_receptor in batch:
            df_tmp = df_pred1.query('receptor == @test_receptor and pred > @pred_thd')
            if len(df_tmp) == 0:
                continue

            filtered = lrdb_B.filter_pred(df_tmp['ligand'], lrdb_B.l_jm, peer_num=peer_num, threshold=topo_thd)
            if filtered is None:
                continue
            else:
                couple = np.array([filtered, [test_receptor] * len(filtered)])
                result_ls.append(couple)

        if len(result_ls) > 0:
            out = np.concatenate(result_ls, axis=-1)
        else:
            out = None

    return out


def get_l3_score(ligand, receptor, graph):
    N_l = list(set(graph.neighbors(ligand)))
    if receptor in N_l:
        N_l.remove(receptor)

    N_r = list(set(graph.neighbors(receptor)))
    if ligand in N_r:
        N_r.remove(ligand)

    degree = graph.degree()
    ls = []
    for x, y in product(N_l, N_r):
        if graph.has_edge(x, y):
            ls.append(1 / np.sqrt(degree(x) * degree(y)))
    l3 = np.sum(ls)
    return l3


def lrdb_analysis():
    # hypothesis validation
    sl_ls, sr_ls = lrdb_B.get_sl_sr()
    df_l = pd.DataFrame(sl_ls, columns=['LRDB', 'Random'])
    df_r = pd.DataFrame(sr_ls, columns=['LRDB', 'Random'])

    # l3
    l3_dic = {0: [], 1: []}
    for ligand, receptor in product(lrdb_ligands, lrdb_receptors):
        score = get_l3_score(ligand, receptor, lrdb_B.graph)
        if lrdb_B.graph.has_edge(ligand, receptor):
            l3_dic[1].append(score)
        else:
            l3_dic[0].append(score)
    df_l3 = pd.DataFrame.from_dict(l3_dic, orient='index').transpose()
    df_l3.columns = ['no_edge', 'edge']

    return df_l, df_r, df_l3


def threshold_analysis():
    df_train = df_dataset.query('set == "train" or set == "val"').reset_index(drop=True)
    df_test = df_dataset.query('set == "test"').reset_index(drop=True)

    train_ligands = np.sort(df_train['ligand'].unique())
    train_receptors = np.sort(df_train['receptor'].unique())
    train_edges = df_train[['ligand', 'receptor']].values
    B = LRBipartite(train_ligands, train_receptors, train_edges)

    test_ligands = np.sort(df_test['ligand'].unique())
    n_batch = 8
    ligand_batch_ls = np.array_split(test_ligands, n_batch)

    def get_once(batch, peer_num, pred_thd, topo_thd):
        result_ls = []
        for test_ligand in batch:
            df_tmp = df_pred_.query('ligand == @test_ligand and pred > @pred_thd')
            if len(df_tmp) == 0:
                continue
            filtered = B.filter_pred(df_tmp['receptor'], B.r_jm, peer_num=peer_num, threshold=topo_thd)
            if filtered is None:
                continue
            else:
                couple = np.array([[test_ligand] * len(filtered), filtered])
                result_ls.append(couple)

        if len(result_ls) > 0:
            out = np.concatenate(result_ls, axis=-1)
        else:
            out = None

        return out

    def get_density(df):
        # true_positive = len(df_test.query('receptor in @train_receptors'))  # 557
        true_positive = df_test['receptor'].isin(train_receptors).sum()
        df = df.merge(df_dataset[['ligand', 'receptor', 'set']], how='left').fillna(value='novel')
        return len(df.query('set != "novel"')) / true_positive, len(df.query('set != "novel"')) / len(df)

    df_pred_ = df_pred.query('receptor in @train_receptors').reset_index(drop=True)

    result_dic = {}
    pred_thd_ls = [0.5, 0.7, 0.9, 0.95, 0.98, 0.99]
    topo_thd_ls = [0.0, 0.1, 0.2, 0.3]
    peer_num = 5
    for pred_thd, topo_thd in product(pred_thd_ls, topo_thd_ls):
        if pred_thd == 0.99 and topo_thd == 0.1:
            break
        output = Parallel(n_jobs=n_batch)(
            delayed(get_once)(batch, peer_num, pred_thd, topo_thd) for batch in ligand_batch_ls)
        df_out = pd.DataFrame(
            np.concatenate([value for value in output if value is not None], axis=-1).T,
            columns=['ligand', 'receptor']
        )
        result_dic[(pred_thd, topo_thd)] = df_out

    result_ls = []
    for (pred_thd, topo_thd), df_i in result_dic.items():
        if len(df_i) != 0:
            value = get_density(df_i)
            result_ls.append([pred_thd, topo_thd, len(df_i), value[0], value[1]])
    df_result = pd.DataFrame(result_ls, columns=['pred_thd', 'topo_thd', 'num', 'recall', 'density'])
    df_result['density'] = df_result['density'] * 100
    df_result['recall'] = df_result['recall'] * 100

    return df_result


def get_new_graph():
    df_potlr = pd.read_csv(r'../gnn_predictor/data/df/df_potential_lr.csv')
    test_ligands = np.sort(df_potlr.query('lr_type == "ligand" and gene not in @lrdb_ligands')['gene'].unique())
    test_receptors = np.sort(df_potlr.query('lr_type == "receptor" and gene not in @lrdb_receptors')['gene'].unique())

    n_batch = 8
    ligand_batch_ls = np.array_split(test_ligands, n_batch)
    receptor_batch_ls = np.array_split(test_receptors, n_batch)

    # novel ligand
    pred_thd = 0.98
    topo_thd = 0.3
    peer_num = 5
    output = Parallel(n_jobs=n_batch)(
        delayed(get_novel_once)(batch, peer_num, pred_thd, topo_thd, 'ligand') for batch in ligand_batch_ls)
    df_novel_l = pd.DataFrame(
        np.concatenate([value for value in output if value is not None], axis=-1).T,
        columns=['ligand', 'receptor']
    )

    # novel receptor
    pred_thd = 0.95
    topo_thd = 0.3
    peer_num = 5
    output = Parallel(n_jobs=n_batch)(
        delayed(get_novel_once)(batch, peer_num, pred_thd, topo_thd, 'receptor') for batch in receptor_batch_ls)
    df_novel_r = pd.DataFrame(
        np.concatenate([value for value in output if value is not None], axis=-1).T,
        columns=['ligand', 'receptor']
    )

    # combine [lrdb, novel_ligand, novel_receptor], using l3 to get links between novel L/R
    novel_ligands = np.sort(df_novel_l['ligand'].unique())
    novel_receptors = np.sort(df_novel_r['receptor'].unique())
    all_ligands = np.concatenate((lrdb_ligands, novel_ligands))
    all_receptors = np.concatenate((lrdb_receptors, novel_receptors))
    all_edges = pd.concat((df_dataset[['ligand', 'receptor']], df_novel_l, df_novel_r), ignore_index=True).values
    allB = LRBipartite(all_ligands, all_receptors, all_edges)

    result_ls = []
    df_pred_ = df_pred.query('ligand in @novel_ligands and receptor in @novel_receptors').reset_index(drop=True)
    for ligand, receptor in product(novel_ligands, novel_receptors):
        pred = df_pred_.query('ligand == @ligand and receptor == @receptor')['pred'].item()
        l3_score = get_l3_score(ligand, receptor, allB.graph)
        result_ls.append([ligand, receptor, pred, l3_score])
    df_novel_l3 = pd.DataFrame(result_ls, columns=['ligand', 'receptor', 'pred', 'l3'])

    df_novel_l['type'] = 'L'
    df_novel_r['type'] = 'R'
    df_novel_l3['type'] = 'LR'

    return df_novel_l, df_novel_r, df_novel_l3


if __name__ == '__main__':
    # global variables
    df_pred = pd.read_csv('./lr_database_raw.csv')
    df_pred = df_pred.query('receptor != "GPR1"').reset_index(drop=True)  # duplicated GPR1 with CMKLR2
    df_dataset = pd.read_csv('../gnn_predictor/data/df/df_human_dataset_nfold1.csv').query('y == 1')
    lrdb_ligands = np.sort(df_dataset['ligand'].unique())
    lrdb_receptors = np.sort(df_dataset['receptor'].unique())
    lrdb_edges = df_dataset[['ligand', 'receptor']].values
    lrdb_B = LRBipartite(lrdb_ligands, lrdb_receptors, lrdb_edges)

    #
    df_l, df_r, df_l3 = lrdb_analysis()
    df_thd = threshold_analysis()
    df_novel_l, df_novel_r, df_novel_l3 = get_new_graph()

    df_l.to_csv('./result/df_l.csv', index=False)
    df_r.to_csv('./result/df_r.csv', index=False)
    df_l3.to_csv('./result/df_l3.csv', index=False)
    df_thd.to_csv('./result/df_thd.csv', index=False)
    df_novel_l3.to_csv('./result/df_novel_l3.csv', index=False)

    df_novel = pd.concat(
        (df_novel_l, df_novel_r, df_novel_l3.query('pred > 0.95 and l3 > 0.5')[['ligand', 'receptor', 'type']]),
        ignore_index=True)
    df_novel['species'] = 'Human'
    df_novel.to_csv('../st_filtering/data/lr_novel.csv', index=False)

