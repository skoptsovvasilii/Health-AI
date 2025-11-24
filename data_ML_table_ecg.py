import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import random
import os

# =========================
# 1. ФИКСАЦИЯ СИДОВ
# =========================
def set_seed(seed=42):
    """Фиксируем сиды для повторяемости результатов"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# =========================
# 2. НАСТРОЙКИ
# =========================
window_size = 5000      # длина сегмента (5 сек)
step_size = 500         # шаг (0.5 сек)
batch_size = 64
epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 3. КАСТОМНЫЙ DATASET
# =========================
class ECGDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.samples = []

        for _, row in self.data.iterrows():
            signal = np.array(row.iloc[:-1], dtype=np.float32)  # все кроме последнего столбца
            label = int(row.iloc[-1])                           # метка класса

            # режем сигнал на окна
            for start in range(0, len(signal) - window_size + 1, step_size):
                end = start + window_size
                segment = signal[start:end]
                self.samples.append((segment, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        segment, label = self.samples[idx]
        segment = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)  # [1, 5000]
        return segment, label

# =========================
# 4. СЕТЬ (1D CNN)
# =========================
class ECGNet(nn.Module):
    def __init__(self, num_classes=5):
        super(ECGNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# =========================
# 5. ЗАГРУЗКА ДАННЫХ
# =========================
train_dataset = ECGDataset("train.csv")
val_dataset   = ECGDataset("val.csv")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size)

# =========================
# 6. ОБУЧЕНИЕ
# =========================
model = ECGNet(num_classes=5).to(device)

# если классы не сбалансированы → добавь веса
# class_weights = torch.tensor([1.0, 2.0, 3.0, 1.0, 1.5]).to(device)
# criterion = nn.CrossEntropyLoss(weight=class_weights)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)

def train(model, optimizer, criterion, epochs=50):
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # === валидация ===
        model.eval()
        val_losses = []
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_losses.append(loss.item())
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}: Train Loss = {np.mean(train_losses):.4f}, "
              f"Val Loss = {np.mean(val_losses):.4f}, Acc = {acc:.4f}")

        if np.mean(val_losses) < best_val_loss:
            best_val_loss = np.mean(val_losses)
            torch.save(model.state_dict(), "ecg_model_2_0.pth")

train(model, optimizer, criterion, epochs)
