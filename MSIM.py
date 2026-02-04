import torch
import torch.nn as nn
import torch.nn.functional as F


class MSIM(nn.Module):

    def __init__(self, beta=0.5, gamma=0.3):
        super(MSIM, self).__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, fsb_features, ncb_features, gab_features, afm, ncb_attention_weights, gab_attention_weights):
        num_nodes = fsb_features.shape[0]

        ncb_scaled_collection = torch.matmul(afm, ncb_attention_weights) / torch.sqrt(
            torch.tensor(num_nodes, dtype=torch.float))

        gab_scaled_collection = torch.matmul(afm, gab_attention_weights) / torch.sqrt(
            torch.tensor(num_nodes, dtype=torch.float))

        combined_attention = (1 - self.beta - self.gamma) * ncb_attention_weights + \
                             self.beta * ncb_scaled_collection + \
                             self.gamma * gab_scaled_collection

        aggregation_weights = F.softmax(combined_attention, dim=0)

        combined_features = torch.cat([fsb_features, ncb_features, gab_features], dim=1)

        graph_representation = torch.matmul(aggregation_weights.transpose(-2, -1), combined_features)

        return graph_representation
