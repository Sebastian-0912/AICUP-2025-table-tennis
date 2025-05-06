import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.fftpack import dct
import torch.nn as nn
from DCT_train import CNNClassifier as DCT_MLP
from DCT_train import extract_dct_features_jiugongge

class TableTennisCNN_Dataset(Dataset):
    def __init__(self, info_csv, data_dir):
        self.info = pd.read_csv(info_csv)
        self.data_dir = data_dir

        self.X = []
        self.ids = []

        for _, row in self.info.iterrows():
            file_path = os.path.join(data_dir, f"{row['unique_id']}.txt")
            x = extract_dct_features_jiugongge(file_path)
            self.X.append(x)
            self.ids.append(row['unique_id'])

        self.X = torch.tensor(np.stack(self.X), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.ids[idx]


# === Inference and Save CSV ===
def run_inference(train_info_path, train_data_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DCT_MLP()
    model.load_state_dict(torch.load("model_weights.pth", map_location=device))
    model.to(device)
    model.eval()

    test_dataset = TableTennisCNN_Dataset(train_info_path, train_data_path)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    results = []

    with torch.no_grad():
        for x, ids in test_loader:
            x = x.to(device)
            out = model(x)

            gender_preds = torch.sigmoid(out['gender']).cpu().numpy()
            hand_preds = torch.sigmoid(out['handed']).cpu().numpy()
            py_preds = torch.softmax(out['years'], dim=1).cpu().numpy()
            level_preds = torch.softmax(out['level'], dim=1).cpu().numpy()

            for i in range(len(ids)):
                # Normalize and round to 4 decimals
                py = py_preds[i] / py_preds[i].sum()
                level = level_preds[i] / level_preds[i].sum()

                result = {
                    "unique_id": int(ids[i].item()),
                    "gender": round(1 - float(gender_preds[i].item()), 4),
                    "hold racket handed": round(1 - float(hand_preds[i].item()), 4),
                    "play years_0": round(float(py[0]), 4),
                    "play years_1": round(float(py[1]), 4),
                    "play years_2": round(float(py[2]), 4),
                    "level_2": round(float(level[0]), 4),
                    "level_3": round(float(level[1]), 4),
                    "level_4": round(float(level[2]), 4),
                    "level_5": round(float(level[3]), 4),
                }
                results.append(result)

    df = pd.DataFrame(results)
    df.to_csv("submission.csv", index=False)
    print("✅ Saved submission.csv!")


if __name__ == "__main__":
    train_info_path = "39_Test_Dataset/test_info.csv"
    train_data_path = "39_Test_Dataset/test_data"  
    run_inference(train_info_path,train_data_path)
