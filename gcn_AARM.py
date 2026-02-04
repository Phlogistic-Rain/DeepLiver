import torch
import torch.nn as nn
import torch.nn.functional as F


class AARM(nn.Module):
    def __init__(self, beta=0.5):
        super(AARM, self).__init__()
        self.beta = beta

    def forward(self, fub_features, gab_features, afm, attention_weights):
        num_nodes = fub_features.shape[0]

        scaled_collection = torch.matmul(afm, attention_weights) / torch.sqrt(
            torch.tensor(num_nodes, dtype=torch.float))

        aggregation_weights = F.softmax(
            self.beta * scaled_collection + (1 - self.beta) * attention_weights,
            dim=0
        )

        combined_features = torch.cat([fub_features, gab_features], dim=1)

        graph_representation = torch.matmul(aggregation_weights.transpose(-2, -1), combined_features)

        return graph_representation