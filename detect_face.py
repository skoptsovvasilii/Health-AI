import torch
import torch.nn as nn
from tensorflow.python.keras.utils.generic_utils import to_list
from torchvision import transforms
from PIL import Image
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Архитектура должна совпадать с обучающей
class Model_cnn(nn.Module):
    def __init__(self):
        super(Model_cnn, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()

        )

        self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(32768, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.classifier(x)
        return x


# Загружаем модель
model = Model_cnn().to(device)
model.load_state_dict(torch.load('cnn_emotion_model_final2.pth', map_location=device))
model.eval()

# Подготовка изображения
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Путь к изображению
print(0)
cap = cv2.VideoCapture(0)


# Проверка, успешно ли открыта каме
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Камера активна. Нажмите 'q' для выхода.")
import numpy as np
while True:
    # Захват кадра: ret - булево значение (успех/неуспех), frame - сам кадр
    ret, frame = cap.read()

    if not ret:
        print("Ошибка: Не удалось прочитать кадр из потока. Выход.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Определяем положение лица на кадре
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Выделяем лицо в прямоугольнике
        roi_color = frame[y:y + h, x:x + w]

        # Уменьшаем размер исходного кадра до размера рамки
        resized_roi = cv2.resize(roi_color, (frame.shape[1], frame.shape[0]))

        # Создаем черное изображение размером оригинального кадра
        black_bg = np.zeros_like(frame)

        # Складываем лицо поверх черного фона
        output = black_bg.copy()
        output[:resized_roi.shape[0], :resized_roi.shape[1]] = resized_roi
        frame = output

        # Отображение результата
        #cv2.imshow('Cropped Face', output)

    # --- Наложение текста на кадр ---

    # 1. Формирование текста (например, текущее время)

        image = frame
        image = Image.fromarray(image)
        image = transform(image).unsqueeze(0).to(device)
        x = ["allergig", "vein", "norm", 'cianoz']
        with torch.no_grad():

            output = model(image)
            print(output)
            prediction = x[output[0].tolist().index(max(to_list(output[0])[0]))]
            print(prediction)
            probability = max(to_list(output[0])[0])
            print(probability)

        label = prediction
        print(f"Prediction: {label} (Probability: {probability})")


        text_to_display = f" Prediction: {label} (Probability: {probability:.4f}) "

        # 2. Параметры шрифта и позиции
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (10, 30)  # Позиция (X, Y) - нижний левый угол текста
        fontScale = 0.7
        color = (0, 255, 0)  # Цвет текста (B, G, R) - зеленый
        thickness = 2
        lineType = cv2.LINE_AA


        # 3. Наложение текста на изображение
        cv2.putText(frame, text_to_display, org, font, fontScale, color, thickness, lineType)

        # 4. Вывод кадра на экран в окне с именем 'Live Camera Feed'
        cv2.imshow('Live Camera Feed', frame)

        # 5. Ожидание нажатия клавиши 'q' для выхода из цикла
        # cv2.waitKey(1) ждет 1 миллисекунду
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# После завершения цикла освобождаем ресурсы камеры и закрываем окна
cap.release()
cv2.destroyAllWindows()

# Предсказание
with torch.no_grad():
    output = model(image)
    prediction = (output >= 0.5).float().item()
    probability = output.item()

label = "нет" if prediction == 1.0 else "есть"
print(f"Prediction: {label} (Probability: {probability:.4f})")