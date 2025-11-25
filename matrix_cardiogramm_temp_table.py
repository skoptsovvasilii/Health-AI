import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tqdm
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
        self.block1 = ResBlock(64, kernel=3, dilation=1)
        self.block2 = ResBlock(64, kernel=3, dilation=2)
        self.block3 = ResBlock(64, kernel=3, dilation=4)
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
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "ml_cardiogram_resnet_3_0.pth"
csv_path   = "dataframe_ecg_4_0_500gc.csv"


df = pd.read_csv(csv_path)
label_col = df.columns[0]
labels_str = df[label_col].astype(str).values
classes = sorted(df[label_col].astype(str).unique().tolist())
class_to_id = {c: i for i, c in enumerate(classes)}
id_to_class = {i: c for c, i in class_to_id.items()}

X_all = df.iloc[:, 1:].values.astype(np.float32)  # [N, L]
y_all = np.array([class_to_id[s] for s in labels_str], dtype=np.int64)

print(f"Загружено {len(X_all)} сигналов, классы: {classes}")


model = ResNet1D(in_ch=2, num_classes=len(classes)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()



def predict_one(ecg_signal: np.ndarray):
    sig = ecg_signal.astype(np.float32)
    d = np.diff(sig, prepend=sig[0]).astype(np.float32)
    x = np.stack([sig, d], axis=0)
    x = torch.from_numpy(x).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        return classes[pred_idx], probs


x_answer = []
y_answer = []
if __name__ == "__main__":
   # idx = random.randint(0, len(X_all)-1)
   for i in tqdm.tqdm(range(0, len(X_all)-1)):
        signal = X_all[i]
        true_label = classes[y_all[i]]
        verdict, probabilities = predict_one(signal)
        x_answer.append(verdict)
        y_answer.append(true_label)

cm = confusion_matrix(y_answer, x_answer)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', annot_kws={'fontsize': 14})
plt.ylabel('Истинные значения')
plt.xlabel('Предсказанные значения')
plt.title('Матрица ошибок', pad=15)
plt.show()

