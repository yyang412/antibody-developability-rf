import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GENConv, global_mean_pool


class AntibodyGNN(nn.Module):
    def __init__(self, input_dim=22, edge_dim=2, hidden_dim=64, num_conv_layers=3, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        in_dim = input_dim
        for _ in range(num_conv_layers):
            conv = GENConv(
                in_dim,
                hidden_dim,
                aggr="softmax",
                t=1.0,
                learn_t=True,
                learn_p=True,
                msg_norm=True,
                edge_dim=edge_dim,
            )
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, batch, edge_attr):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = bn(x)
            x = self.dropout(x)

        x = global_mean_pool(x, batch)
        out = self.mlp(x)
        return out