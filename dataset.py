# test_dct.py
import numpy as np
from scipy.fftpack import dct

def extract_dct_features_whole(file_path):
    data = np.loadtxt(file_path)  # shape: (N, 6)
    dct_feats = [dct(data[:, i], norm='ortho')[:80] for i in range(6)]
    return np.concatenate(dct_feats)  # shape: (480,)

# # Save fake IMU data
# np.savetxt("1.txt", np.random.randn(1000, 6), fmt="%.4f")

# # Test
# features = extract_dct_features_whole("1.txt")
# print(features.shape)  # (480,)


import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import numpy as np

# Dataset class
class TableTennisDCTDataset(Dataset):
    def __init__(self, info_csv, data_dir):
        # from extract_dct import extract_dct_features_whole  # or import if same file

        self.info = pd.read_csv(info_csv)
        self.data_dir = data_dir

        self.X = []
        self.y_gender = []
        self.y_handed = []
        self.y_years = []
        self.y_level = []

        for _, row in self.info.iterrows():
            file_path = os.path.join(data_dir, f"{row['unique_id']}.txt")
            x = extract_dct_features_whole(file_path)

            self.X.append(x)
            self.y_gender.append(row['gender'])
            self.y_handed.append(row['hold racket handed'])
            self.y_years.append(row['play years'])
            self.y_level.append(row['level'])

        self.X = torch.tensor(np.stack(self.X), dtype=torch.float32)
        self.y_gender = torch.tensor(self.y_gender, dtype=torch.float32)
        self.y_handed = torch.tensor(self.y_handed, dtype=torch.float32)
        self.y_years = torch.tensor(self.y_years, dtype=torch.long)
        self.y_level = torch.tensor(self.y_level, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_gender[idx], self.y_handed[idx], self.y_years[idx], self.y_level[idx]

# Test
dataset = TableTennisDCTDataset("train_info.csv", "./")
print(len(dataset[0][0]))  # (480,)


