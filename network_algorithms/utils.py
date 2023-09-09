import numpy as np
import networkx as nx
import copy
import pandas as pd


class LRBipartite:
    def __init__(self, ligands, receptors, edges):
        self.ligands = ligands
        self.receptors = receptors

        B = nx.Graph()
        B.add_nodes_from(ligands, bipartite='l')
        B.add_nodes_from(receptors, bipartite='r')
        B.add_edges_from(edges)
        self.graph = B

        self.l_jm = self.get_jaccard_matrix(self.ligands, self.graph)  # self.ligands x self.ligands
        self.r_jm = self.get_jaccard_matrix(self.receptors, self.graph)  # self.receptors x self.receptors

    @staticmethod
    def get_jaccard_matrix(nodes, graph):
        node_ls = [(u, v) for u in nodes for v in nodes]
        ls = [v[-1] for v in nx.jaccard_coefficient(graph, node_ls)]
        jm = np.array(ls).reshape((len(nodes), len(nodes)), order='C')
        return pd.DataFrame(jm, index=nodes, columns=nodes)

    @staticmethod
    def get_sim(node_1, counter_jm, graph):
        """
        compute the similarity of node1 with {node_i}
            node1
             |
             |
             |
        ...{node_i}...
        """
        graph = copy.deepcopy(graph)
        node_i = list(set(graph.neighbors(node_1)))
        n_i = len(node_i)
        if n_i == 1:  # node1 only has one neighbor
            return None

        graph.remove_node(node_1)
        node_ls = [(u, v) for u in node_i for v in node_i]
        ls = [v[-1] for v in nx.jaccard_coefficient(graph, node_ls)]
        jm_i = np.array(ls).reshape((n_i, n_i), order='C')

        # jm_i = counter_jm.loc[node_i, node_i].values
        np.fill_diagonal(jm_i, 0)  # set the diagonal to 0
        score = jm_i.sum() / ((n_i-1) * n_i)  # get the average score

        #
        rank_ls = []
        for i in range(100):
            np.random.seed(i)
            index = np.random.choice(len(counter_jm), size=n_i, replace=False)
            rank_jm_i = counter_jm.values[index, :][:, index]
            rank_ls.append((rank_jm_i.sum()-n_i) / ((n_i-1) * n_i))
        rank_score = np.mean(rank_ls)

        return score, rank_score

    def get_sl_sr(self):
        ls1 = []
        ls2 = []
        for ligand in self.ligands:
            result = self.get_sim(ligand, self.r_jm, self.graph)
            if result is not None:
                ls1.append(result)
        for receptor in self.receptors:
            result = self.get_sim(receptor, self.l_jm, self.graph)
            if result is not None:
                ls2.append(result)

        return ls1, ls2

    @staticmethod
    def filter_pred(pred_node, jm, peer_num, threshold):
        """
        compute the similarity of node1 with {node_i}
        """
        pred_node = np.array(pred_node)
        jm_i = jm.loc[pred_node, pred_node].values - np.eye(len(pred_node))
        jm_i = np.sort(jm_i, axis=-1)[:, -peer_num:]  # find the most similar peers

        index = np.where(jm_i.mean(axis=-1) > threshold)[0]
        if len(index) != 0:
            pred_node = pred_node[index]
        else:
            pred_node = None
        return pred_node

