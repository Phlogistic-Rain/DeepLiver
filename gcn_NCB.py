import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class AttentionGenerator(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.linear1 = nn.Linear(in_channels, in_channels)
        self.linear2 = nn.Linear(in_channels, in_channels)
        self.linear3 = nn.Linear(in_channels, 1)

    def forward(self, x):
        h1 = torch.sigmoid(self.linear1(x))
        h2 = torch.tanh(self.linear2(x))
        attention = self.linear3(h1 * h2)
        return attention


class NCB(nn.Module):
    def __init__(self, in_channels=2048, hidden_channels=512, out_channels=128,
                 momentum=0.3, momentum_decay=0.1, decay_steps=5):
        super(NCB, self).__init__()

        self.momentum = momentum
        self.momentum_decay = momentum_decay
        self.decay_steps = decay_steps
        self.current_step = 0

        self.res_linear = torch.nn.Linear(hidden_channels, out_channels)

        self.projection = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.agm = AttentionGenerator(hidden_channels)

        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.ln1 = nn.LayerNorm(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.ln2 = nn.LayerNorm(hidden_channels)

        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.ln3 = nn.LayerNorm(out_channels)

    def get_momentum_attention_matrix(self, attention_weights):
        scaled_attention = torch.sigmoid(attention_weights)
        attention_matrix = torch.matmul(scaled_attention, scaled_attention.transpose(-2, -1))

        if self.training:
            mam = self.momentum + (1 - self.momentum) * attention_matrix

            if self.current_step != 0 and self.current_step % self.decay_steps == 0:
                self.momentum = max(0.0, self.momentum - self.momentum_decay)

            self.current_step += 1

            return mam

        else:
            return attention_matrix

    def forward(self, x):
        x = self.projection(x)

        attention_weights = self.agm(x)

        mam = self.get_momentum_attention_matrix(attention_weights)

        edge_index = (mam > 0).nonzero().t()
        edge_weight = mam[edge_index[0], edge_index[1]]

        identity = x

        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.ln1(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x += identity
        identity = x

        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.ln2(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x += identity
        identity = x

        x = self.conv3(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.ln3(x)

        x += self.res_linear(identity)

        return x, attention_weights, mam


if __name__ == "__main__":
    x = np.load('./features/resnet50.tv_in1k/0/Training_3_Pat 27.npy')
    x = torch.from_numpy(x)

    print(x.shape)

    ncb = NCB(in_channels=1024, hidden_channels=256, out_channels=128,
              momentum=0.5, momentum_decay=0.1, decay_steps=3)

    out, attention_weights, mam = ncb(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(ncb.momentum)

    for _ in range(20):
        out, attention_weights, mam = ncb(x)
        print(ncb.momentum)