"""CytoTour
Usage:
    CytoTour.py <lr_pair> <pathwaydb> <st_file>... [--species=<sn>]  [--filtering=<fn>]  [--distance_threshold]  [--parallel]  [--max_hop=<mn>]  [--cell_sender=<cn>]  [--cell_receiver=<rn>]  [--out=<fn>]  [--core=<cn>]
    CytoTour.py (-h | --help)
    CytoTour.py --version

Options:
    -s --species=<sn>   Choose species Mouse or Human [default: Human].
    -f --filtering=<fn>   The thershold of valid expression count, choose median, mean or null [default: mean].
    -d --distance_threshold   Using 95% largest distance to filter cell neighbors.
    -p --parallel   Using parallel.
    -m --max_hop=<mn>   Set the max hop to find lr pairs with tf [default: 3].
    -c --cell_sender=<cn>   The cell type of sender [default: all].
    -r --cell_receiver=<rn>   The cell type of receiver [default: all].
    -o --out=<fn>   Outdir [default: .].
    -n --core=<cn>   Core number [default: 4].
    -h --help   Show this screen.
    -v --version    Show version.
"""

import pandas as pd
import numpy as np
from docopt import docopt
from utils import *
import Cytograph
import multiprocessing as mp
import datetime
import anndata as ad
import pickle


def main(arguments):
    print("start filtering LRIs with spatial data")
    st_files = arguments.get("<st_file>")
    lr_pair = arguments.get("<lr_pair>")
    pathwaydb = arguments.get("<pathwaydb>")
    cell_sender = str(arguments.get("--cell_sender"))
    cell_receiver = str(arguments.get("--cell_receiver"))
    parallel = arguments.get("--parallel")
    filtering = str(arguments.get("--filtering"))
    species = str(arguments.get("--species"))
    distance_threshold = arguments.get("--distance_threshold")
    out_dir = str(arguments.get("--out"))
    max_hop = int(arguments.get("--max_hop"))
    n_core = int(arguments.get("--core"))
    starttime = datetime.datetime.now()

    if max_hop is None:
        if species == "Mouse":
            max_hop = 4
        else:
            max_hop = 3
    print("reading data")
    if len(st_files) == 1:
        if st_files[0].endswith("h5ad"):
            adata = ad.read_h5ad(st_files[0])
            st_data = adata.to_df()

            st_data = st_data.transpose()
            st_meta = adata.obsm['meta']
            st_meta['cell'] = st_meta.index.values.tolist()
            # st_meta = st_meta[st_meta["cell_type"].isin(["Acinar_cells","pDCs"])]
            if 'cell_type' not in st_meta.columns:
                TypeError("There is no column named 'cell_type' in st_meta file")
        else:
            TypeError("st_file should be st_meta.csv and st_data.csv or xxx.h5ad")
    elif len(st_files) == 2:
        if st_files[0].endswith(".csv"):
            st_meta = pd.read_csv(st_files[0])
            if 'cell_type' not in st_meta.columns:
                TypeError("There is no column named 'cell_type'")
        else:
            TypeError("st_file should be st_meta.csv and st_data.csv or xxx.h5ad")
        st_data = pd.read_csv(st_files[1], index_col=0)
    print("reading data done")

    ##read data
    # st_meta = st_meta[st_meta["label"] != "less nFeatures"]
    st_data = st_data[st_data.apply(np.sum, axis=1) != 0]
    st_gene = st_data.index.values.tolist()
    lr_pair = pd.read_csv(lr_pair)
    lr_pair = lr_pair[lr_pair['species'] == species]
    pathway = pd.read_table(pathwaydb, delimiter='\t', encoding='unicode_escape')

    ##filtering
    pathway = pathway[["src", "dest", "src_tf", "dest_tf"]][pathway['species'] == species].drop_duplicates()
    pathway = pathway[(pathway['src'].isin(st_gene)) & (pathway['dest'].isin(st_gene))]

    valid_gene = list(set(pathway['src'].values).union(set(pathway['dest'].values)))
    st_data = preprocess_st(st_data, filtering)
    valid_st_data = st_data.loc[valid_gene, :]
    lr_pair = lr_pair[lr_pair['receptor'].isin(st_gene) & lr_pair['ligand'].isin(st_gene)]
    ##get cell list
    if cell_sender == "all" and cell_receiver == "all":
        sender_list, receiver_list, cell_type = get_cell_list(st_meta)
        print(f"The unique celltype list is {cell_type}")
    else:
        # print(f"The number of unique celltypes is {len(cell_type)}")
        sender_list = [cell_sender]
        receiver_list = [cell_receiver]
    dist_data = get_distance(st_meta, distance_threshold)
    lr_pair_all = lr_pair

    if not parallel:
        all_lr_score = pd.DataFrame(
            columns=["ligand", "receptor", "species", "cell_sender", "cell_receiver", "co_exp_value", "co_exp_number",
                     "co_exp_p", "lr_score", "rt_score", "score"])

        obj = {'cell_pair': {}}
        for i in range(len(sender_list)):
            cell_sender = sender_list[i]
            cell_receiver = receiver_list[i]
            # for each cell type receiver, find all edge with expression ration>0.1
            cell_pair, flag = get_cell_pair(st_meta, dist_data, cell_sender, cell_receiver)
            if flag == 0:
                continue
            print(f"The cell pair number found between {cell_sender} and {cell_receiver} is {cell_pair.shape[0]}")
            f_path = find_high_exp_path(pathway, cell_pair["cell_receiver"].values.tolist(), valid_st_data)
            f_path = f_path[f_path['co_exp_ratio'] > 0.10]
            path_gene = list(set(f_path['src'].values).union(set(f_path['dest'].values)))
            pathway_graph = Cytograph.PathGraph(path_gene, max_hop)
            pathway_graph.built_edge(f_path)
            receptor_gene = pathway_graph.find_valid_lr_pair()
            lr_pair_sub = lr_pair_all[lr_pair_all['receptor'].isin(receptor_gene)]

            if lr_pair_sub.shape[0] == 0:
                print(
                    f"No ligand-recepotor pairs found between {cell_sender} and {cell_receiver} because of no downstream transcriptional factors found for receptors!")
                continue
            else:
                print(
                    f"the number of valid pathway number between {cell_sender} and {cell_receiver} is: {lr_pair_sub.shape[0]}")
                # find cell pair
                lr_pair_sub.insert(0, 'cell_sender', cell_sender)
                lr_pair_sub.insert(1, 'cell_receiver', cell_receiver)

                sig_lr_pair = find_sig_lr(st_data, lr_pair_sub, cell_pair)
                if sig_lr_pair.shape[0] == 0:
                    print(
                        f"No ligand-recepotor pairs found between {cell_sender} and {cell_receiver} with significant expression")
                    continue
                print(
                    f"The ligand-recepotor pairs found between {cell_sender} and {cell_receiver} with significant expression is {sig_lr_pair.shape[0]}")

                tf = pathway_graph.find_lr_tf(sig_lr_pair)

                path_score = get_score(sig_lr_pair, tf)

                all_lr_score = pd.concat([all_lr_score, path_score], axis=0)
                obj['cell_pair'].update({f'{cell_sender}-{cell_receiver}': cell_pair})
                print(f"{cell_receiver} and {cell_sender} done")
        obj['lr_score'] = all_lr_score
    else:
        if n_core > mp.cpu_count():
            n_core = mp.cpu_count()
        print(f"parallel processing with {n_core} cores")
        with mp.Pool(processes=n_core) as pool:
            results = pool.starmap_async(process_sender_receiver, [
                (i, lr_pair, dist_data, sender_list, receiver_list, st_meta, pathway, valid_st_data, st_data, max_hop)
                for i in range(len(sender_list))])
            output = results.get()
            obj = post_process(output)

    with open(f'{out_dir}/cci_result.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    endtime = datetime.datetime.now()
    print(f"Total running time is {(endtime - starttime).seconds} seconds")


if __name__ == "__main__":
    arguments = docopt(__doc__, version="AlphaTalk-CytoTour 1.0.0")
    main(arguments)
