import pandas as pd
import numpy as np


class PathGraph(object):

    def __init__(self, valid_gene, max_hop=4):
        self.max_hop = max_hop
        self.n_nodes = len(valid_gene)

        # Initialize the adjacency matrix
        # Create a matrix with `num_of_nodes` rows and columns
        self.lr_g = np.zeros((self.max_hop + 1, self.n_nodes, self.n_nodes))
        self.tf_g = np.zeros((self.max_hop + 1, self.n_nodes, self.n_nodes))
        self.hop_g = np.zeros((self.n_nodes, self.n_nodes))
        self.nodes = valid_gene

    def add_edge(self, node1, node2, weight=1):
        self.adj_matrix[node1][node2] = weight

    def built_edge(self, valid_pathway):
        for index, row in valid_pathway.iterrows():
            src = row['src']
            dest = row['dest']
            i = self.nodes.index(src)
            j = self.nodes.index(dest)
            self.lr_g[0, i, j] = 1
            if row["dest_tf"] == "YES":
                self.tf_g[0, i, j] = 1

    def find_hop(self):
        b = self.lr_g[0, :, :].nonzero()
        self.hop_g[b[0], b[1]] = 1
        for i in range(self.max_hop):
            c = self.lr_g[i + 1, :, :].nonzero()
            x = list(zip(b[0], b[1]))
            y = list(zip(c[0], c[1]))
            d = list(set(y).difference(set(x)))
            if (len(d) > 0):
                x_1, x_2 = [list(t) for t in zip(*d)]
                self.hop_g[x_1, x_2] = i + 2
                b = self.hop_g.nonzero()

    def find_valid_lr_pair(self, k_steps=4):
        for i in range(self.max_hop - 1):
            self.lr_g[i + 1, :, :] = np.dot(self.lr_g[i, :, :], self.lr_g[0, :, :])
            self.tf_g[i + 1, :, :] = np.dot(self.lr_g[i, :, :], self.tf_g[0, :, :])
        self.lr_g[self.max_hop, :, :] = np.sum(self.lr_g, 0)
        self.tf_g[self.max_hop, :, :] = np.sum(self.tf_g, 0)
        r_i = np.flatnonzero(np.sum(self.tf_g[self.max_hop, :, :], 1))
        receptor_list = np.array(self.nodes)[r_i]
        # self.find_hop()
        return receptor_list

    def find_lr_tf(self, lr_pair):
        receptor_list = lr_pair["receptor"].values.tolist()
        r_i_l = np.flatnonzero(np.sum(self.lr_g[0, :, :], 1))
        r_l = []
        t_l = []
        s_l = []
        h_l = []
        d_l = []
        for i in range(lr_pair.shape[0]):
            for j in range(self.max_hop):
                r_i = self.nodes.index(receptor_list[i])
                t_i = np.flatnonzero((self.tf_g[j, r_i, :]))
                t_i = np.intersect1d(r_i_l, t_i)
                r_l.extend([receptor_list[i]] * len(t_i))
                t_l.extend(np.array(self.nodes)[t_i])
                path_sum = np.sum(self.lr_g[j, r_i, :])
                tf_sum = self.lr_g[j, r_i, t_i]
                tf_score = tf_sum / path_sum
                s_l.extend(tf_score)
                h_l.extend([j + 1] * len(t_i))
                if len(t_i) > 0:
                    tf_reach_list = np.sum(self.lr_g[0, t_i, :], 1)
                    d_l.extend(tf_reach_list)

        score_rt = np.array(d_l) * np.array(s_l) / np.array(h_l)

        receptor_tf = pd.DataFrame(
            {'receptor': r_l, 'tf': t_l, "score": s_l, "hop": h_l, "target": d_l, "score_rt": score_rt})

        return receptor_tf
