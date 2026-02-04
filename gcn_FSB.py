import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class AFMBuilder:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        similarity = torch.mm(x_norm, x_norm.t())

        if x.shape[0] != 1:
            similarity = (similarity - similarity.min()) / (similarity.max() - similarity.min())

        mask = (similarity > self.threshold)
        edge_index = mask.nonzero().t()

        return edge_index, similarity


class FSB(nn.Module):
    def __init__(self, in_channels=2048, hidden_channels=512, out_channels=128):
        super(FSB, self).__init__()

        self.res_linear = torch.nn.Linear(hidden_channels, out_channels)

        self.projection = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.ln1 = nn.LayerNorm(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.ln2 = nn.LayerNorm(hidden_channels)

        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.ln3 = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index):
        x = self.projection(x)

        identity = x

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.ln1(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x += identity
        identity = x

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.ln2(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x += identity
        identity = x

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.ln3(x)

        x += self.res_linear(identity)

        return x