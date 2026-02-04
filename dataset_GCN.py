import os
import numpy as np
import torch
import glob
from torch.utils.data import Dataset, DataLoader


class WSIFeatureDataset(Dataset):
    def __init__(self, feature_dir, split='Training'):

        super().__init__()

        self.feature_dir = feature_dir
        self.split = split

        all_feature_paths = glob.glob(os.path.join(feature_dir, "*.npy"))

        self.feature_paths = []
        self.labels = []
        self.positions = []

        for path in all_feature_paths:
            filename = os.path.basename(path)
            file_parts = filename.split('_')
            file_split = file_parts[0]

            if file_split == self.split:
                self.feature_paths.append(path)

                label = int(file_parts[1])
                self.labels.append(label)

                underscore_indices = [i for i, char in enumerate(filename) if char == '_']
                if len(underscore_indices) >= 2:
                    after_second_underscore = filename[underscore_indices[1] + 1:]
                    position = after_second_underscore.rsplit('.', 1)[0]
                else:
                    position = "unknown"
                self.positions.append(position)

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):
        features = np.load(self.feature_paths[idx])
        label = self.labels[idx]
        position = self.positions[idx]

        features = torch.FloatTensor(features)

        return features, torch.LongTensor([label]), position


if __name__ == '__main__':

    test_dataset = WSIFeatureDataset(feature_dir='./features/tf_efficientnetv2_b0.in1k_ft/0', split='Test')

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for data, target, pos in test_loader:
        print(data.shape)
        print(target.shape)

        pos = pos[0]

        data = data.squeeze(0)
        target = target.squeeze(0)

        print(data.shape)
        print(target.shape)

        print(pos)
