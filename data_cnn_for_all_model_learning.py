import os
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# === Параметры ===
num_augmented = 10
input_dir = "../unets/dataset/вены"
augmented_dir = "../unets/dataset/вены"
batch_size = 8
epochs = 5
image_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Аугментация и сохранение ===
print(" Генерация аугментированных изображений...")
'''
augment = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
])

os.makedirs(augmented_dir, exist_ok=True)

for class_name in os.listdir(input_dir):
    input_class_path = os.path.join(input_dir, class_name)
    output_class_path = os.path.join(augmented_dir, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    for img_name in os.listdir(input_class_path):
        img_path = os.path.join(input_class_path, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except: 
            continue

        base_name = os.path.splitext(img_name)[0]
        image.resize((image_size, image_size)).save(os.path.join(output_class_path, f"{base_name}_orig.jpg"))

        for i in range(num_augmented):
            aug_img = augment(image)
            aug_img.save(os.path.join(output_class_path, f"{base_name}_aug{i}.jpg"))
'''
print("Аугментация завершена.")

# === Загрузка данных ===
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])



dataset = datasets.ImageFolder(augmented_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === Архитектура модели ===
class Model_cnn(nn.Module):
    def __init__(self):
        super(Model_cnn, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (image_size // 8) * (image_size // 8), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.classifier(x)
        return x

model = Model_cnn().to(device)

# === Обучение ===
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(" Обучение модели...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataset)
    print(f"Эпоха {epoch+1}/{epochs}, Потери: {epoch_loss:.4f}")

# === Сохранение модели ===
torch.save(model.state_dict(), "easy_big_data_vens.pth")
print(" Модель сохранена как defect_cnn_model.pth")