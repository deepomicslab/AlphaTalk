import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, u_max=6.0, gamma=10.0, step=0.1):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        #
        self.u_max = u_max
        self.gamma = gamma
        self.step = step
        self.mlp_g = nn.Sequential(
            nn.Linear(int(u_max / step), out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(inplace=True)
        )

    def gaussian(self, r_i, r_j):
        d = torch.linalg.vector_norm(r_i - r_j, ord=2, dim=1, keepdim=True)
        u_k = torch.arange(0, self.u_max, self.step, device=r_i.device).unsqueeze(0)
        out = torch.exp(-self.gamma * torch.square(d - u_k))
        return out

    def forward(self, x, edge_index, coords):
        m = self.lin(x)
        m = self.propagate(edge_index, x=m, coords=coords)
        out = x + m
        return out

    def message(self, x_j, coords_j, coords_i):
        weight = self.mlp_g(self.gaussian(coords_i, coords_j))
        return weight * x_j


class GNNPair(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, reduce_form):
        super().__init__()
        self.num_layers = num_layers
        self.loss_func = nn.BCELoss()
        self.reduce = reduce_form
        self.embedding_residue = nn.Embedding(20, input_dim)  # residue ids are 0-19
        self.convs = torch.nn.ModuleList(
            [GCNConv(in_channels=input_dim, out_channels=hidden_dim)] +
            [GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
                for _ in range(num_layers-1)]
        )

        self.l_lin = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True))
        self.r_lin = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True))

        self.dropout = nn.Dropout(p=0.5)
        self.mlp = nn.Sequential(
            nn.Linear(2*hidden_dim + 2*1024, 5*hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(5*hidden_dim, 2*hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2*hidden_dim, 1),
            nn.Sigmoid()
        )

    def get_conv_result(self, x, edge_index, coords):
        for i in range(self.num_layers):
            x = self.convs[i](x=x, edge_index=edge_index, coords=coords)
            x = F.relu(x, inplace=True)
        return x

    def forward(self, l_data, r_data):
        l_x, l_edge_index, l_emb, l_batch = l_data.x, l_data.edge_index, l_data.emb, l_data.batch
        r_x, r_edge_index, r_emb, r_batch = r_data.x, r_data.edge_index, r_data.emb, r_data.batch

        #
        l_x = self.embedding_residue(l_x.reshape(-1))
        r_x = self.embedding_residue(r_x.reshape(-1))
        l_x = self.get_conv_result(l_x, l_edge_index, l_data.coords)
        r_x = self.get_conv_result(r_x, r_edge_index, r_data.coords)

        #
        l_x = scatter(l_x, l_batch, dim=-2, reduce=self.reduce)
        r_x = scatter(r_x, r_batch, dim=-2, reduce=self.reduce)
        l_x = self.l_lin(l_x)
        r_x = self.r_lin(r_x)

        #
        l_x = F.normalize(torch.concat((l_x, l_emb), dim=-1), dim=-1)
        r_x = F.normalize(torch.concat((r_x, r_emb), dim=1), dim=-1)
        x = self.dropout(torch.concat((l_x, r_x), dim=-1))
        out = self.mlp(x)

        return out

    def loss(self, pred, label):
        pred, label = pred.reshape(-1), label.reshape(-1)
        return self.loss_func(pred, label)

