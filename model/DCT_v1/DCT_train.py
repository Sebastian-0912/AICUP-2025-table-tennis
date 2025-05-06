import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from scipy.fftpack import dct
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.optim.lr_scheduler import StepLR


def evaluate_metrics(model, dataloader, device):
    model.eval()
    gender_preds, gender_trues = [], []
    handed_preds, handed_trues = [], []
    years_preds, years_trues = [], []
    level_preds, level_trues = [], []

    with torch.no_grad():
        for x, gender, handed, years, level in dataloader:
            x = x.to(device)
            out = model(x)

            # Binary outputs (sigmoid + threshold 0.5 for accuracy)
            gender_prob = torch.sigmoid(out['gender'].squeeze()).cpu().numpy()
            handed_prob = torch.sigmoid(out['handed'].squeeze()).cpu().numpy()
            gender_preds += list(gender_prob)
            handed_preds += list(handed_prob)
            gender_trues += list(gender.numpy())
            handed_trues += list(handed.numpy())

            # Multi-class outputs
            years_pred = torch.argmax(out['years'], dim=1).cpu().numpy()
            level_pred = torch.argmax(out['level'], dim=1).cpu().numpy()
            years_preds += list(years_pred)
            level_preds += list(level_pred)
            years_trues += list(years.numpy())
            level_trues += list(level.numpy())

    print("\n--- Evaluation Metrics ---")
    try:
        print(f"Gender ROC AUC: {roc_auc_score(gender_trues, gender_preds):.4f}")
    except:
        print("Gender ROC AUC: ERROR (only one class?)")
    try:
        print(f"Handed ROC AUC: {roc_auc_score(handed_trues, handed_preds):.4f}")
    except:
        print("Handed ROC AUC: ERROR (only one class?)")
    print(f"Years Accuracy: {accuracy_score(years_trues, years_preds):.4f}")
    print(f"Level Accuracy: {accuracy_score(level_trues, level_preds):.4f}")


def extract_dct_features_whole(file_path):
    data = np.loadtxt(file_path).transpose()  # shape: (6, N)
    dct_feats = dct(data, norm='ortho')[:,:100]
    return np.concatenate(dct_feats)  # shape: (6 * 80 = 480,)


class TableTennisDCTDataset(Dataset):
    def __init__(self, info_csv, data_dir):

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
            self.y_gender.append(row['gender']-1)
            self.y_handed.append(row['hold racket handed']-1)
            self.y_years.append(row['play years'])
            self.y_level.append(row['level']-2)

        self.X = torch.tensor(np.stack(self.X), dtype=torch.float32)
        self.y_gender = torch.tensor(self.y_gender, dtype=torch.float32)
        self.y_handed = torch.tensor(self.y_handed, dtype=torch.float32)
        self.y_years = torch.tensor(self.y_years, dtype=torch.long)
        self.y_level = torch.tensor(self.y_level, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # print(self.y_gender[idx])
        return self.X[idx], self.y_gender[idx], self.y_handed[idx], self.y_years[idx], self.y_level[idx]

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import roc_auc_score

# MLP Model
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=600):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.PReLU()
        )
        self.gender_head = nn.Linear(128, 1)
        self.handed_head = nn.Linear(128, 1)
        self.years_head = nn.Linear(128, 3)
        self.level_head = nn.Linear(128, 4)

    def forward(self, x):
        x = self.backbone(x)
        return {
            'gender': self.gender_head(x),
            'handed': self.handed_head(x),
            'years':  self.years_head(x),
            'level': self.level_head(x)
        }


# --- Training loop ---
def train_loop(model, dataloader, optimizer, criterion_bce, criterion_ce, device):
    model.train()
    total_loss = 0
    for x, gender, handed, years, level in dataloader:
        x, gender, handed, years, level = x.to(device), gender.to(device), handed.to(device), years.to(device), level.to(device)

        out = model(x)
        # print(out)
        loss = (
            criterion_bce(out['gender'].squeeze(), gender) +
            criterion_bce(out['handed'].squeeze(), handed) +
            criterion_ce(out['years'], years) +
            criterion_ce(out['level'], level)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# --- Validation loop ---
def val_loop(model, dataloader, criterion_bce, criterion_ce, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, gender, handed, years, level in dataloader:
            x, gender, handed, years, level = x.to(device), gender.to(device), handed.to(device), years.to(device), level.to(device)
            out = model(x)
            loss = (
                criterion_bce(out['gender'].squeeze(), gender) +
                criterion_bce(out['handed'].squeeze(), handed) +
                criterion_ce(out['years'], years) +
                criterion_ce(out['level'], level)
            )
            total_loss += loss.item()
    return total_loss / len(dataloader)

# --- Train + validate + plot ---
def run_training(dataset, batch_size=8, num_epochs=20, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier().to(device)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    print(len(val_ds))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_ce = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # reduce LR by half every 5 epochs

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = train_loop(model, train_loader, optimizer, criterion_bce, criterion_ce, device)
        val_loss = val_loop(model, val_loader, criterion_bce, criterion_ce, device)
        scheduler.step()
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    # --- Plot ---
    plt.plot(train_losses[10:], label="Train Loss")
    plt.plot(val_losses[10:], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")
    plt.grid()
    plt.savefig("DCT_loss_curve.png")
    plt.show()

    evaluate_metrics(model, val_loader, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    return model
  

train_info_path = "39_Training_Dataset/train_info.csv"
train_data_path = "39_Training_Dataset/train_data"  

if __name__ == "__main__":
    dataset = TableTennisDCTDataset(train_info_path, train_data_path)
    model = run_training(dataset, batch_size=32, num_epochs=80, lr=1e-3)
    torch.save(model.state_dict(), "model_weights.pth")
    # extract_dct_features_whole("39_Training_Dataset/train_data/1.txt")

