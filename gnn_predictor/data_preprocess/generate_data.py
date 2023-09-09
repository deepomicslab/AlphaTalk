import os
import pandas as pd
import torch
from Bio.PDB.MMCIFParser import MMCIFParser
from utils import MultiprocessData


if __name__ == '__main__':

    # alphafold human v4
    data_dir = '../data/source/human_v4/'

    #
    graph_dir = '../data/graph'
    parser = MMCIFParser()

    file_ls = os.listdir(data_dir)
    df = pd.DataFrame({
        'name': [file.replace('.cif', '') for file in file_ls],
        'path': [os.path.join(data_dir, file) for file in file_ls]
    })
    cutoff = 6
    c_type = 'alpha'
    mp = MultiprocessData(
        process_num=8,
        parser=parser,
        c_type=c_type,
        cutoff=cutoff
    )
    data_ls = mp.create(df)
    torch.save(data_ls, os.path.join(graph_dir, f'human_v4_{cutoff}_{c_type}.pt'))



