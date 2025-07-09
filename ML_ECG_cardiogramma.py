import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# === Глобальные настройки ===
batch_size = 32
epochs = 100
num_classes = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Загрузка данных ===
train_file = 'mitbih_train.csv'
test_file = 'mitbih_test.csv'

df_train = pd.read_csv(train_file, header=None)
df_test = pd.read_csv(test_file, header=None)

X_train = df_train.iloc[:, :-1].values.astype(np.float32)
y_train = df_train.iloc[:, -1].values.astype(int)

X_test = df_test.iloc[:, :-1].values.astype(np.float32)
y_test = df_test.iloc[:, -1].values.astype(int)


# === Кастомный датасет ===
class ECGDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


# Создаем датасеты и загрузчики
train_dataset = ECGDataset(X_train, y_train)
test_dataset = ECGDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# === Исправленная модель ===
class ECGModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fc = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        # Добавляем размерность батча если ее нет
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc(out)
        return out


# Инициализация модели
input_size = X_train.shape[1]
hidden_size = 128  # Уменьшил для стабильности
model = ECGModel(input_size, hidden_size, num_classes).to(device)

# Функция потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

# === Обучение ===
best_accuracy = 0.0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Валидация
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Статистика
    train_loss = running_loss / len(train_loader)
    val_loss = val_loss / len(test_loader)
    accuracy = 100 * correct / total

    print(f'Epoch {epoch + 1}/{epochs}')
    print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
    print(f'Accuracy: {accuracy:.2f}%')

    # Сохранение лучшей модели
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'delete_me_it_is_bad.pth')
        print('Model saved!')

    scheduler.step(val_loss)

print(f'Best Accuracy: {best_accuracy:.2f}%')