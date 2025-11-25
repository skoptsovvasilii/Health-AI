import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


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




class Get_ecg:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = "ml_cardiogram_resnet_3_0.pth"
        csv_path   = "dataframe_ecg_4_0_500gc.csv"
        self.cn = 0
        self.df = pd.read_csv(csv_path)
        self.label_col = self.df.columns[0]
        self.labels_str = self.df[self.label_col].astype(str).values
        self.classes = sorted(self.df[self.label_col].astype(str).unique().tolist())
        self.class_to_id = {c: i for i, c in enumerate(self.classes)}
        id_to_class = {i: c for c, i in self.class_to_id.items()}

        self.X_all = self.df.iloc[:, 1:].values.astype(np.float32)  # [N, L]
        self.y_all = np.array([self.class_to_id[s] for s in self.labels_str], dtype=np.int64)

        print(f"Загружено {len(self.X_all)} сигналов, классы: {self.classes}")


        self.model = ResNet1D(in_ch=2, num_classes=len(self.classes)).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()


    def predict_one(self, ecg_signal: np.ndarray):
        sig = ecg_signal.astype(np.float32)
        d = np.diff(sig, prepend=sig[0]).astype(np.float32)
        x = np.stack([sig, d], axis=0)
        x = torch.from_numpy(x).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            return self.classes[pred_idx], probs

    def ecg(self):

        idx = random.randint(0, len(self.X_all)-1)
        signal = self.X_all[idx]
        true_label = self.classes[self.y_all[idx]]

        verdict, probabilities = self.predict_one(signal)


        print(f"Пример #{idx}")
        print("Правильный класс:", true_label)
        print("Вердикт модели:", verdict)
        print("Вероятности:")
        lst = []
        for cls, p in zip(self.classes, probabilities):
            print(f"  {cls}: {p:.3f}")
            lst.append([p, cls])

        plt.figure(figsize=(4, 2))
        plt.plot(signal, label="ECG сигнал", linewidth=1)
        plt.title(f"ECG пример #{idx}")
        plt.xlabel("Время (отсчёты)")
        plt.ylabel("Амплитуда")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        file_path = f"{'card_for_code'}/plot_{self.cn}.png"
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()

        #plt.show()

        return lst


