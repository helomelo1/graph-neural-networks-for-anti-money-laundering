import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, Linear


class GraphNeuralNet(nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, heads=0):
        super(GraphNeuralNet, self).__init__()

        self.lin_in = Linear(num_node_features, hidden_channels)
        self.gat1 = GATv2Conv(hidden_channels, hidden_channels, heads=heads, dropout=0.6)
        self.gat2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=0.6)
        self.lin_out = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_idx):
        x = self.lin_in(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.gat1(x, edge_idx)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)

        x = self.gat2(x, edge_idx)
        x = F.elu(x)

        x = self.lin_out(x)

        return F.log_softmax(x, dim=1)