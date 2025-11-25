import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Устройство (GPU или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Путь к данным
dataset_dir = 'путь/к/твоей/папке/Skin_Conditions'  # Замени на реальный путь

# Трансформации для данных (без аугментации)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Загрузка датасета
dataset = datasets.ImageFolder(dataset_dir, transform=transform)
class_names = dataset.classes
print("Порядок классов:", class_names)


# Разделение на train/val/test
def split_dataset(dataset, train_size=0.8, val_size=0.1):
    total_size = len(dataset)
    indices = list(range(total_size))
    np.random.shuffle(indices)

    train_end = int(total_size * train_size)
    val_end = train_end + int(total_size * val_size)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=16, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=16, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=16, sampler=test_sampler)

    return train_loader, val_loader, test_loader


train_loader, val_loader, test_loader = split_dataset(dataset)

# Модель ResNet50
model = models.resnet50(pretrained=True)

# Разморозка layer4 и части layer3 для fine-tuning
for name, param in model.named_parameters():
    if "layer4" in name or "layer3.5" in name:  # Размораживаем последние блоки
        param.requires_grad = True
    else:
        param.requires_grad = False

# Замена финального слоя
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),  # Уменьшил с 1024 до 256
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 5)  # 5 классов
)
model = model.to(device)

# Оптимизатор и лосс (учитываем размороженные слои)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
criterion = nn.CrossEntropyLoss()


# Early stopping
class EarlyStopping:
    def __init__(self, patience=3, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = model.state_dict()
        else:
            self.counter += 1
        return self.counter >= self.patience


early_stopping = EarlyStopping(patience=3)

# Обучение
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(60):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(
        f"Epoch {epoch + 1}/60, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    if early_stopping(val_loss, model):
        print("Early stopping triggered")
        model.load_state_dict(early_stopping.best_model)
        break

# Сохранение модели
torch.save(model.state_dict(), 'my_resnet_model.pth')
print("Модель сохранена в 'my_resnet_model.pth'")

# Графики
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Оценка на тесте
model.eval()
y_true, y_pred = [], []
test_loss, correct, total = 0.0, 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_loss = test_loss / len(test_loader)
test_acc = correct / total
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Метрики
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred,
                            target_names=['Cyanosis', 'Normal', 'Jugular Vein Distention', 'Allergic Reaction',
                                          'Other']))