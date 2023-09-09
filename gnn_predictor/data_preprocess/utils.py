from multiprocessing import Manager, Pool
import traceback
import numpy as np
from scipy.spatial import distance_matrix as DM
import time
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import os

res_to_id = {"ALA": 0,
             "ARG": 1,
             "ASN": 2,
             "ASP": 3,
             "CYS": 4,
             "GLU": 5,
             "GLN": 6,
             "GLY": 7,
             "HIS": 8,
             "ILE": 9,
             "LEU": 10,
             "LYS": 11,
             "MET": 12,
             "PHE": 13,
             "PRO": 14,
             "SER": 15,
             "THR": 16,
             "TRP": 17,
             "TYR": 18,
             "VAL": 19}

res_to_one = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


class MultiprocessData:
    def __init__(self, process_num, parser, c_type, cutoff):
        self.process_num = process_num
        self.parser = parser
        self.c_type = c_type
        self.cutoff = cutoff
        self.in_queue = Manager().Queue()
        self.out_queue = Manager().Queue()

    def _slice_raw(self, data, task_num):
        sub_df_ls = np.array_split(data, task_num)
        for sub_df in sub_df_ls:
            self.in_queue.put(sub_df.reset_index(drop=True))

    @staticmethod
    def get_distance_matrix(bio_structure, c_type):
        """
        generate the distance matrix of the structure's 1st model (all residues of different chains)
        :param bio_structure:
        :param c_type: alpha or beta
        :return:
        """
        model = bio_structure[0]  # all proteins have 1 model, 1 chain
        residue_ls = model.get_residues()
        coord_ls = []
        if c_type == 'alpha':
            for residue in residue_ls:
                c = residue['CA']
                coord_ls.append(c.get_coord())
        elif c_type == 'beta':
            for residue in residue_ls:
                if residue.get_resname() == "GLY":
                    c = residue['CA']
                else:
                    c = residue['CB']
                coord_ls.append(c.get_coord())
        else:
            raise Exception('Unknown carbon type')
        coords = np.array(coord_ls)

        distance_matrix = DM(coords, coords)
        return distance_matrix, coords

    @staticmethod
    def get_residue_name(bio_structure):
        model = bio_structure[0]  # all proteins have 1 model, 1 chain
        residue_ls = model.get_residues()
        name_ls = [residue.resname for residue in residue_ls]
        return name_ls

    # def _process(self, data):
    #     try:
    #         # residue_n_ls = []
    #         # distance_matrix_ls = []
    #         parser = MMCIFParser()
    #         for idx, content in data.iterrows():
    #             name = content['name']
    #             path = content['path']
    #             structure = parser.get_structure(name, path)
    #             # chain = structure[0].get_list()[0]
    #             # residue_n_ls.append(len(chain))
    #             #
    #             m = self.get_distance_matrix(structure)
    #             # distance_matrix_ls.append(m)
    #             np.save(f'./data/{name}.npy', m)
    #
    #         #
    #         # data['residue_n'] = residue_n_ls
    #         # data['distance_matrix'] = distance_matrix_ls
    #         # self.out_queue.put(data)
    #     except Exception:
    #         traceback.print_exc()

    # def _generate_matrix(self, data):
    #     try:
    #         parser = MMCIFParser()
    #         for idx, content in data.iterrows():
    #             name = content['name']
    #             path = content['path']
    #             structure = parser.get_structure(name, path)
    #             #
    #             m = self.get_distance_matrix(structure)
    #             np.save(f'./data/{name}.npy', m)
    #     except Exception:
    #         traceback.print_exc()

    def _generate_graph(self, data):
        try:
            data_ls = []
            for idx, content in data.iterrows():
                name = content['name']
                cif_path = content['path']

                #
                structure = self.parser.get_structure(name, cif_path)
                res_names = self.get_residue_name(structure)
                res_names = [res_names, [res_to_one[n] for n in res_names]]  # add one-code letter
                res_ids = [res_to_id[n] for n in res_names[0]]
                res_ids = torch.tensor(res_ids, dtype=torch.int32).unsqueeze(1)

                #
                dm, coords = self.get_distance_matrix(structure, self.c_type)
                coords = torch.tensor(coords, dtype=torch.float32)
                edge_index = torch.tensor(np.where(dm <= self.cutoff, 1, 0), dtype=torch.int64)
                edge_index = dense_to_sparse(edge_index)[0]

                #
                data = Data(
                    x=res_ids,
                    edge_index=edge_index,
                    coords=coords,
                    residues=res_names,
                    name=name
                )
                data_ls.append(data)
            # torch.save(data_ls, os.path.join(graph_dir, f'data_{cutoff}.pt'))
            self.out_queue.put(data_ls)
        except Exception:
            traceback.print_exc()

    def create(self, data, task_num=100):
        self._slice_raw(data, task_num)
        pool = Pool(self.process_num)
        for _ in range(task_num):
            pool.apply_async(
                func=self._generate_graph,
                args=(self.in_queue.get(), )
            )

        # get_data
        i = 0
        data_ls = []
        while i < task_num:
            if not self.out_queue.empty():
                one_data = self.out_queue.get()
                if i == 0:
                    data_ls = one_data
                else:
                    data_ls += one_data
                i += 1
                print(f'{time.strftime("%H:%M:%S")}: remaining task num', task_num - i)

        pool.close()
        pool.join()

        print(f'{len(data_ls)} graphs generated.')
        return data_ls
