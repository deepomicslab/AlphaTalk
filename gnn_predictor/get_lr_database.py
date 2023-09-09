import os
import random
import torch
import numpy as np
import pandas as pd
from types import SimpleNamespace
import time
from utils import PairDataset
from torch_geometric.loader import DataLoader
from models import GNNPair

random.seed(42); np.random.seed(42)


class LRDataset(PairDataset):
    def __init__(self, pair_info_df, single_data_pool, reduce_form):
        super().__init__(pair_info_df, single_data_pool, reduce_form)

    def __getitem__(self, idx):
        row = self.info.iloc[idx, ]
        ligand_file = row['ligand_file']
        receptor_file = row['receptor_file']

        ligand_data = self.single_data_pool[ligand_file]
        receptor_data = self.single_data_pool[receptor_file]

        # emb
        ligand_data.emb = torch.tensor(
            np.load(os.path.join(self.emb_path, f'{ligand_file}.npy')), dtype=torch.float32)
        receptor_data.emb = torch.tensor(
            np.load(os.path.join(self.emb_path, f'{receptor_file}.npy')), dtype=torch.float32)
        if self.reduce_func is not None:
            ligand_data.emb = self.reduce_func(ligand_data.emb, dim=0, keepdim=True)
            receptor_data.emb = self.reduce_func(receptor_data.emb, dim=0, keepdim=True)

        return ligand_data, receptor_data, row['ligand'], row['receptor']


if __name__ == '__main__':
    species = "Human"
    args = {
        'device': 'cuda',
        'batch_size': 80
    }
    args = SimpleNamespace(**args)

    # dataloader
    df_potential_lr = pd.read_csv(r'./data/df/df_potential_lr.csv')
    df_ligand = df_potential_lr.query('lr_type == "ligand"').reset_index(drop=True)[['gene', 'name']]
    df_ligand.columns = ['ligand', 'ligand_file']
    df_receptor = df_potential_lr.query('lr_type == "receptor"').reset_index(drop=True)[['gene', 'name']]
    df_receptor.columns = ['receptor', 'receptor_file']
    df_comb = df_ligand.merge(df_receptor, how='cross')

    single_data_pool = {data['name']: data for data in torch.load('./data/graph/human_v4_6_alpha.pt')}

    comb_data = LRDataset(df_comb, single_data_pool, 'sum')
    dataloader = DataLoader(comb_data, batch_size=args.batch_size)
    print('total combinations:', len(dataloader.dataset))

    # model related
    model = GNNPair(input_dim=128, hidden_dim=128, num_layers=1, reduce_form='sum')
    weight_path = r'./result/human_v4_6_alpha_GNNPair/4_acc0.900_roc0.965_prc0.959_recall0.893.pt'
    model.load_state_dict(torch.load(weight_path))
    print('load weight:', weight_path)
    model.to(args.device)
    model.eval()
    print(model)

    # get result
    ligand_ls = []
    receptor_ls = []
    pred_ls = []
    result_file_path = f'../network_filtering/lr_database_raw.csv'
    for batch_idx, (ligand, receptor, l_gene, r_gene) in enumerate(dataloader):
        ligand = ligand.to(args.device)
        receptor = receptor.to(args.device)

        ligand_ls += l_gene
        receptor_ls += r_gene
        with torch.no_grad():
            pred = model(ligand, receptor).reshape(-1).detach().cpu().numpy().tolist()
            pred_ls += pred

        condition1 = len(ligand_ls) >= 20000
        condition2 = batch_idx == len(dataloader)-1  # final batch
        if condition1 or condition2:
            if os.path.exists(result_file_path):
                df0 = pd.read_csv(result_file_path)
            else:
                df0 = pd.DataFrame(columns=['ligand', 'receptor', 'pred'])
            df = pd.DataFrame({'ligand': ligand_ls, 'receptor': receptor_ls, 'pred': pred_ls})
            df = pd.concat((df0, df), ignore_index=True)
            df.to_csv(result_file_path, index=False)

            print(time.strftime('%H:%M:%S'), 'finished num: ', len(df))
            ligand_ls = []
            receptor_ls = []
            pred_ls = []
