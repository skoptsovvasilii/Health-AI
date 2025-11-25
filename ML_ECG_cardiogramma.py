# file: train_resnet1d_two_channel.py
import os
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

# -------------------------
# 0. Настрой: поменяй путь к CSV
# -------------------------
csv_path   = "dataframe_ecg.csv"   # <- Укажи свой CSV (первый столбец = метка, потом 5000 чисел)
batch_size = 16
max_epochs = 60
valid_size = 0.2
lr_initial = 1e-4
l1_lambda  = 1e-6
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cache_npz = True    # если True — создаст/прочитает cache npz (ускоряет загрузку)

# -------------------------
# 1. Сиды
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# -------------------------
# 2. Читаем CSV (или кеш)
# -------------------------
if use_cache_npz:
    cache_file = os.path.splitext(csv_path)[0] + "_cache.npz"
else:
    cache_file = None

if cache_file and os.path.exists(cache_file):
    print("Loading cached npz:", cache_file)
    d = np.load(cache_file)
    X_all = d["X"]   # shape [N, L] float32
    y_all = d["y"]   # shape [N] int64
    classes = list(d["classes"])
    classes = [c.decode() if isinstance(c, bytes) else c for c in classes]
else:
    print("Reading CSV:", csv_path)
    df = pd.read_csv(csv_path)
    label_col = df.columns[0]
    labels_str = df[label_col].astype(str).values
    classes = sorted(df[label_col].astype(str).unique().tolist())
    class_to_id = {c: i for i, c in enumerate(classes)}
    id_to_class = {i: c for c, i in class_to_id.items()}
    print("Classes mapping:", class_to_id)

    # build y and X
    y_all = np.array([class_to_id[s] for s in labels_str], dtype=np.int64)
    X_all = df.iloc[:, 1:].values.astype(np.float32)  # [N, L]
    if cache_file:
        print("Saving cache:", cache_file)
        np.savez_compressed(cache_file, X=X_all, y=y_all, classes=np.array(classes, dtype=object))

num_classes = len(classes)
input_len = X_all.shape[1]
print(f"Total samples: {len(X_all)}, input_len: {input_len}, num_classes: {num_classes}")

# -------------------------
# 3. Train/Val split (stratified)
# -------------------------
sss = StratifiedShuffleSplit(n_splits=1, test_size=valid_size, random_state=42)
train_idx, val_idx = next(sss.split(X_all, y_all))
X_train, y_train = X_all[train_idx], y_all[train_idx]
X_val,   y_val   = X_all[val_idx],   y_all[val_idx]

# -------------------------
# 4. Class weights (train only)
# -------------------------
counts = np.bincount(y_train, minlength=num_classes)
class_weights = (len(y_train) / (num_classes * np.maximum(counts, 1))).astype(np.float32)
class_weights_t = torch.tensor(class_weights, device=device)
print("Train counts:", {classes[i]: int(counts[i]) for i in range(num_classes)})
print("Class weights:", {classes[i]: float(class_weights[i]) for i in range(num_classes)})

# -------------------------
# 5. Dataset: returns two-channel tensor [2, L]
#    Channel 0 = original signal
#    Channel 1 = derivative (np.diff with prepend)
# -------------------------
class TwoChannelECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X        # np.array [N, L]
        self.y = y        # np.array [N]
        self.L = X.shape[1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sig = self.X[idx].astype(np.float32)         # [L]
        # derivative channel: prepend first value so length stays L
        d = np.diff(sig, prepend=sig[0]).astype(np.float32)
        # Stack channels: shape [2, L]
        x = np.stack([sig, d], axis=0)
        x = torch.from_numpy(x)                      # float32 tensor
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

train_ds = TwoChannelECGDataset(X_train, y_train)
val_ds   = TwoChannelECGDataset(X_val,   y_val)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=(device.type=='cuda'))
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type=='cuda'))

# -------------------------
# 6. ResNet1D implementation (in_ch=2)
# -------------------------
class ResBlock(nn.Module):
    def __init__(self, C, kernel=9, dilation=1):
        super().__init__()
        pad = (kernel//2) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(C, C, kernel_size=kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(C),
            nn.ReLU(inplace=True),
            nn.Conv1d(C, C, kernel_size=kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(C),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))

class ResNet1D(nn.Module):
    def __init__(self, in_ch=2, num_classes=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.block1 = ResBlock(64, kernel=9, dilation=1)
        self.block2 = ResBlock.pthlock(64, kernel=9, dilation=2)
        self.block3 = ResBlock(64, kernel=9, dilation=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)   # [B, 64, L]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1)  # [B, 64]
        return self.head(x)

model = ResNet1D(in_ch=2, num_classes=num_classes).to(device)
print("Model params:", sum(p.numel() for p in model.parameters()))

# -------------------------
# 7. Loss, optimizer, scheduler, early stopping
# -------------------------
criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.05)
optimizer = Adam(model.parameters(), lr=lr_initial, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True, min_lr=1e-6)

class EarlyStopping:
    def __init__(self, patience=8, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = np.inf
        self.count = 0
    def step(self, metric):
        improved = (self.best - metric) > self.min_delta
        if improved:
            self.best = metric
            self.count = 0
        else:
            self.count += 1
        return self.count >= self.patience

early_stop = EarlyStopping(patience=8, min_delta=1e-4)

# -------------------------
# 8. Helpers: L1 penalty, evaluate
# -------------------------
def l1_penalty(model):
    l1 = torch.tensor(0., device=device)
    for p in model.parameters():
        if p.requires_grad:
            l1 = l1 + p.abs().sum()
    return l1

def evaluate(model, loader):
    model.eval()
    total, correct = 0, 0
    losses = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(loss.item())
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return np.mean(losses), correct / total

# -------------------------
# 9. Training loop
# -------------------------
best_val = np.inf
best_path = "best_resnet1d_two_channel.pth"

if __name__ == "__main__":
    for epoch in range(1, max_epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}")
        train_losses = []
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if l1_lambda > 0:
                loss = loss + l1_lambda * l1_penalty(model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
            pbar.set_postfix(loss=np.mean(train_losses))

        val_loss, val_acc = evaluate(model, val_loader)
        print(f"Val: loss={val_loss:.5f}, acc={val_acc:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}")
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)

        if early_stop.step(val_loss):
            print("Early stopping triggered.")
            break

    print("Training finished. Best model saved to:", best_path)
    print("Class mapping:", {i: classes[i] for i in range(num_classes)})
