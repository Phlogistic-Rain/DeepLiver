import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models_gcn.gcn_FSB import FSB, AFMBuilder
from models_gcn.gcn_NCB import NCB
from models_gcn.gcn_GAB import GAB
from models_gcn.MSIM import MSIM


class TripleStreamGCN(nn.Module):
    def __init__(self, in_channels=2048, hidden_channels=256, out_channels=128,
                 momentum=0.3, momentum_decay=0.1, decay_steps=5,
                 afm_threshold=0.85, beta=0.5, gamma=0.3, num_classes=7,
                 tf_num_layers=2, tf_num_heads=4):
        super(TripleStreamGCN, self).__init__()

        self.afm_builder = AFMBuilder(afm_threshold)
        self.fsb = FSB(in_channels, hidden_channels, out_channels)
        self.ncb = NCB(in_channels, hidden_channels, out_channels,
                       momentum, momentum_decay, decay_steps)
        self.gab = GAB(in_channels, hidden_channels, out_channels,
                       num_layers=tf_num_layers, num_heads=tf_num_heads)
        self.msim = MSIM(beta, gamma)

        self.classifier = nn.Sequential(
            nn.Linear(3 * out_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, debug=False):
        edge_index, afm = self.afm_builder(x)
        fsb_features = self.fsb(x, edge_index)
        ncb_features, ncb_attention_weights, mam = self.ncb(x)
        gab_features, gab_attention_weights = self.gab(x)

        graph_representation = self.msim(
            fsb_features,
            ncb_features,
            gab_features,
            afm,
            ncb_attention_weights,
            gab_attention_weights
        )

        out = self.classifier(graph_representation)

        return out