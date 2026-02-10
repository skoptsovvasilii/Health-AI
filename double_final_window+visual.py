import cv2
import numpy as np
import time
import threading
import random
import sys
import os
import serial # import Serial Library
import time
import numpy # Import numpy
from serial.tools import list_ports
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import resample
import matplotlib.pyplot as plt #import matplotlib library

from drawnow import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import resample
import matplotlib.pyplot as plt #import matplotlib library

from drawnow import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import shutil
import torch
import torch.nn as nn
import time
from serial.tools import list_ports
from scipy.signal import butter, filtfilt
import torch
import torch.nn as nn
from tensorflow.python.keras.utils.generic_utils import to_list
from torchvision import transforms
from PIL import Image
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np







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



model_cnn = Model_cnn().to(device)
model_cnn.load_state_dict(torch.load('cnn_emotion_model_final2.pth', map_location=device))
model_cnn.eval()

# Подготовка изображения
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
print("Камера активна. Нажмите 'q' для выхода.")





class ResBlock(nn.Module):
    def __init__(self, C, kernel=9, dilation=1):
        super().__init__()
        pad = (kernel // 2) * dilation
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


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(sig, fs, lowcut=0.3, highcut=10.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, sig)


classes = ['AV blocada', 'fibril', 'infarct', 'norm']
model_path = "ml_cardiogram_resnet_3_0.pth"

model = ResNet1D(in_ch=2, num_classes=len(classes)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


def predict_one(pulse_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = ['AV blocada', 'fibril', 'infarct', 'norm']
    # Берём последние 2500 отсчётов
    sig = np.array(pulse_data[-2500:])

    # Производная сигнала
    d = np.diff(sig, prepend=sig[0]).astype(np.float32)

    # Создаём 2 канала: [сигнал, производная]
    x = np.stack([sig, d], axis=0)  # (2, 2500)

    # Приводим к формату (1, 2, 2500)
    x = torch.from_numpy(x).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        print(f'answer: {classes}, - {probs}')
        ans = []
        for x, y in zip(classes, probs):
            ans.append([y, x])
        print(ans)
        return ans


shutil.rmtree('card_for_code', ignore_errors=True)

os.makedirs("card_for_code")
print(time.time())
print(time.time())

'''
import create_triangels
import vis
import ecgs
import datchik
import chek
'''
from get_dano_datch import *
from res_net_MLs import *
from datchiki_detect import *
from check_verdict_AI import *

# ECG = [(0.999, "IM"), (0.3000, "AV"), (0.004, "FIB"), (0.004, "NORM")]
vision = [(0.287, "cianoz"), (0.2000, "allergia"), (0.904, "vein"), (0.09, "NORM")]


# check — перепроверка ECG (label, prob)
# datchicky: main detected label + list of related labels + sensor values dict


# sample = {"SpO2":100.0, "MAP":58.0, "HR":80.0, "EtCO2":38.0, "CVP":93.0, "Urine":8.0, "Temp":36.0, "PI":0.7}
# datchicky = evaluate_reading(sample)
# print("Result:", datchicky)


# datchicky = ("Hypoxy", ["shok", "IM", "cianoz"], {"spo2": 85, "BP": 90, "pulse": 110})


def shoot(s1, s2, im, af, av):
    x = check1(im, af, av)
    print(s1)
    print(x)
    if x in [s1[1], s2[1]]:
        if x == s1[1]:
            return [s1[0], s1[1]]
        elif x == s2[1]:
            return [s2[0], s2[1]]
    return [s1[0], s1[1]]


def target(ecg, vis, datch, sample):
    dat = None
    norma_answer = False

    answer = [0, None]
    ecg_s = sorted(ecg)
    print()
    print(ecg)
    if ecg_s[-1][0] > 0.35:

        if (ecg_s[-1][0] - ecg_s[-2][0]) <= 0.08:
            im = [[[i[0] for i in ecg if i[1] == 'infarct'][0], sample['MAP'], sample["HR"],
                   max([i[0] for i in vis if i[1] in ['vein', 'cianoz']])]]
            af = [[[i[0] for i in ecg if i[1] == 'fibril'][0], sample['PI'], sample["SpO2"],
                   max([i[0] for i in vis if i[1] in ['vein', 'cianoz']])]]
            av = [[[i[0] for i in ecg if i[1] == 'AV blocada'][0], sample['HR'], sample["MAP"],
                   max([i[0] for i in vis if i[1] in ['vein', 'cianoz']])]]
            answer = shoot(ecg_s[-1], ecg_s[-2], im, af, av)
        else:
            answer = [ecg_s[-1][0], ecg_s[-1][1]]
        if answer[1] in datch['conflicts'] or answer[1] == datch['diagnosis']:
            print()
            print("datch conf")
            answer[0] = answer[0] * 1.1
        else:

            print()
            print("NO datch conf")
            answer[0] = answer[0] / 1.2
    if answer[1] in ["Normal", 'norm']:
        answer = [0, None]
    print()
    print(answer)
    print()
    vis_answer = [0, None]
    vis_s = sorted(vis)
    if vis_s[-1][0] > 0.45:
        if (vis_s[-1][0] - vis_s[-2][0]) <= 0.8:
            if (vis_s[-1] in datch['conflicts'] or vis_s[-1] == datch['diagnosis']) and (
                    vis_s[-2] in datch['conflicts'] or vis_s[-2] == datch['diagnosis']):
                vis_answer = [vis_s[-1][0], vis_s[-1][1]]
                if vis_s[-1] == datch[0]:
                    vis_answer[0] = vis_answer[0] * 1.5
                else:
                    vis_answer[0] = vis_answer[0] / 2
    if vis_answer[1] == 'NORM':
        vis_answer = [0, None]
    print(vis_answer)

    dat_answer = [0, None]
    if datchicky['diagnosis'] not in ["Myocardial infarction", "Third-degree AV block", "Cyanosis (isolated)",
                                      "Jugular venous distension (JVD)", 'Second-degree AV block',
                                      'First-degree AV block', "vein", "Atrial fibrillation (proxy)", 'No rule matched',
                                      'norm']:
        dat_answer = [0.5, datchicky['diagnosis']]
        dat = datchicky['explanation']
    if dat_answer[1] == "Normal":
        dat_answer = [0, None]

    if [dat_answer[0], vis_answer[0], answer[0]] in [[0, 0, 0], [0, 'norm', "Normal"], ['No rule matched', 'norm', 0],
                                                     ['No rule matched', 'norm', 'Normal']]:
        norma_answer = True

    print([dat_answer[0], vis_answer[0], answer[0]])
    print([dat_answer[1], vis_answer[1], answer[1]])

    return answer, vis_answer, dat_answer, norma_answer, dat


'''

# ---------- звук ----------
#try:
import winsound
HAVE_WINSOUND = True
#except ImportError:
 #   HAVE_WINSOUND = False

def beep_forever():
    """Постоянный фоновый звук в отдельном потоке."""
    while True:
        if HAVE_WINSOUND:
            winsound.Beep(800, 200)  # частота 800 Гц, длительность 200 мс
            time.sleep(0.2)
        else:
            sys.stdout.write('\a')
            sys.stdout.flush()
            time.sleep(0.5)
'''
import os

import time
import threading
import os
import platform
import numpy as np
import sounddevice as sd
import threading
import time

# Глобальный флаг для включения/выключения тревоги
alarm_active = False
alarm_thread = None


def play_alarm(frequency=1500, duration=0.3, period=0.8, volume=0.9):
    """
    Воспроизводит медицинский тревожный "beep" пока alarm_active = True

    frequency — частота тона (Гц), для медсигналов обычно 1200–1600
    duration  — длительность самого "пика" (сек)
    period    — период повтора (сек)
    volume    — громкость
    """

    global alarm_active

    fs = 44100  # частота дискретизации

    beep_samples = int(duration * fs)
    t = np.linspace(0, duration, beep_samples, False)

    # Резкий медицинский тон
    waveform = (np.sin(2 * np.pi * frequency * t)).astype(np.float32) * volume

    while alarm_active:
        sd.play(waveform, fs, blocking=True)  # воспроизведение beep
        time.sleep(period - duration)  # интервал между сигналами


def start_alarm():
    """Запускает тревожный сигнал в отдельном потоке."""
    global alarm_active, alarm_thread

    if alarm_active:
        return  # уже играет

    alarm_active = True
    alarm_thread = threading.Thread(target=play_alarm, daemon=True)
    alarm_thread.start()


def stop_alarm():
    """Останавливает тревожный сигнал."""
    global alarm_active
    alarm_active = False
    sd.stop()







"""
#####################
ЭКГ вставка кода 
#####################
"""




import serial
import struct
import threading
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

print(serial.tools.list_ports)

ports = list(serial.tools.list_ports.comports())

for port in ports:
    print(f"Порт: {port.device}")
    print(f"Описание: {port.description}")
    print(f"Производитель: {port.manufacturer}\n")
tempF= []
import csv
import os

PORT = '/dev/cu.usbmodem141301'   # <-- проверь порт
BAUD = 115200
FS = 333                          # частота дискретизации (Гц)
WINDOW_SEC = 3                    # сколько секунд показываем
WINDOW_SIZE = 2_500

# =======================
# КОЛЬЦЕВОЙ БУФЕР
# =======================

ecg_buffer = deque(maxlen=FS * 10)   # храним ~10 секунд сигнала

# =======================
# SERIAL
# =======================

ser = serial.Serial(PORT, BAUD, timeout=1)





def serial_reader():
    print(3)
    """
    Читает ВСЕ данные из Serial
    Каждая точка = 2 байта (uint16)
    """
    while True:
        data = ser.read(2)
        if len(data) == 2:
            value = struct.unpack('<H', data)[0]
            ecg_buffer.append(value)





# запуск фонового потока со звуком

# ---------- визуал ----------
W, H = 800, 500
cv2.namedWindow("Monitor", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Monitor", W, H)

alphaq = 0.0
dir_alpha = 1
cb1 = 0
times = time.time()
time.sleep(2)
flag = True
from scipy.signal import resample


def resample_ecg(signal, target_len=10000):
    """Ресемплинг ЭКГ до нужной длины"""
    return resample(signal, target_len)


def ecg1(signal):
    print(len(signal))

    plt.figure(figsize=(4, 2))
    plt.plot(signal, label="ECG сигнал", linewidth=1)
    plt.title(f"ECG пример")
    plt.xlabel("Время (отсчёты)")
    plt.ylabel("Амплитуда")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()  # вместо bbox_inches='tight' в savefig

    file_path = f"{'card_for_code'}/plot_{1}.png"
    plt.savefig(file_path)


    plt.close()


def meadly_text(text, coord, color):
    # setup text
    font = cv2.FONT_HERSHEY_SIMPLEX

    # get boundary of this text
    textsize = cv2.getTextSize(text, font, 1, 2)[0]

    # get coords based on boundary
    textX = int((frame.shape[1] - textsize[0]) / 2)

    # add text centered on image
    cv2.putText(frame, text, (textX, coord), font, 1, color, 2)



tempF = []

pressure = []

# arduinoData = serial.Serial('/dev/cu.usbmodem144201', 9600) #Creating our serial object named arduinoData

plt.ion()  # Tell matplotlib you want interactive mode to plot live data

cnt = 0
number_i = 0

pulse_data = np.array([float(0) for i in range(2508)]).astype(np.float32)

while True:


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



        image = frame
        image = Image.fromarray(image)
        image = transform(image).unsqueeze(0).to(device)

        x = ["allergig", "vein", "norm", 'cianoz']

        with torch.no_grad():
            #vision = [(0.287, "cianoz"), (0.2000, "allergia"), (0.904, "vein"), (0.09, "NORM")]

            output = model_cnn(image)
            print(output)
            prediction = x[output[0].tolist().index(max(to_list(output[0])[0]))]
            print(prediction)
            probability = max(to_list(output[0])[0])
            print(probability)
            vision = []
            for i in to_list(output[0])[0]:
                pred_for_vis = to_list(x[output[0].tolist().index(i)])[0]
                vision.append((i, pred_for_vis))
                #print(f"vision prediction - {vision}")

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
            exit()




    dataArray = [str(random.randint(-2, 4)) for i in range(30)]
    for i in range(25):
        pulse_data = np.append(pulse_data, float(dataArray[i]))
    print(dataArray)



    c = 0
    x = []


    #  pulse_data = resample_ecg(pulse_data, target_len=500)
    pulse_data = bandpass_filter(pulse_data[-2500:], 450)
    pulse_data.astype(np.float32)
    #   print(pulse_data)
    verdict = predict_one(pulse_data)
    #  print(len(pulse_data))
    ecg1(pulse_data[pulse_data.size - 2500:])
    print("1234", verdict)

    # plt.pause(.000001) #Pause Briefly. Important to keep drawnow from crashing

    cnt = cnt + 1

    if flag:
        # Тестовая вероятность (потом заменишь своей)\
        if time.time() - times > 1:
            times = time.time()
            ECG = verdict
            # print(ECG)

            sim = SensorSimulator()
            sample = sim.generate()
            datchicky = evaluate_reading(sample)
            print(sample)
            print(datchicky)
            # sample = {"SpO2":100.0, "MAP":58.0, "HR":80.0, "EtCO2":38.0, "CVP":93.0, "Urine":8.0, "Temp":36.0, "PI":0.7}
            # datchicky = evaluate_reading(sample)
            print(f"vision prediction - {vision}")

            answer_ecg, vis_answer, dat_answer, norma_answer, dat = target(ECG, vision, datchicky,
                                                                           sample)  # <-- вот сюда потом подставишь свою функцию

    frame = np.zeros((H, W, 3), dtype=np.uint8)

    # Цвет и текст по состоянию
    if norma_answer:

        text = "NORM"
        stop_alarm()

        color = (0, 255, 0)  # зелёный
        base_color = (10, 30, 10)
        verdict = []
    elif max([answer_ecg[0], vis_answer[0], dat_answer[0]]) < 0.55:
        text = "Warning"
        start_alarm()

        color = (0, 255, 255)  # жёлтый
        base_color = (30, 30, 0)
        verdict = [e for i, e in [answer_ecg, vis_answer, dat_answer] if e != None]

    else:
        text = "Dangerous!"
        start_alarm()

        color = (0, 0, 255)  # красный
        base_color = (20, 0, 0)
        verdict = [e for i, e in [answer_ecg, vis_answer, dat_answer] if e != None]

    # Мягкое мигание для предупреждения и опасности
    if max([answer_ecg[0], vis_answer[0], dat_answer[0]]) >= 0.6:
        alphaq += dir_alpha * 0.05
        if alphaq > 1 or alphaq < 0:
            dir_alpha *= -1
        overlay = frame.copy()
        overlay[:] = color
        cv2.addWeighted(overlay, alphaq * 0.5, frame, 1 - alphaq * 0.5, 0, frame)
    else:
        frame[:] = base_color

    # Отображение текста
    meadly_text(text, 150, color)
    # cv2.putText(frame, text, (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5, cv2.LINE_AA)
    meadly_text(f"probability: {float(max([answer_ecg[0], vis_answer[0], dat_answer[0]])):.2f}", 220, (200, 200, 200))
    # cv2.putText(frame, f"probability: {float(max([answer_ecg[0], vis_answer[0], dat_answer[0]])):.2f}", (250, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    # cv2.putText(frame, f"{verdict}", (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    meadly_text(f"{verdict}", 80, (200, 200, 200))
    print()
    print(dat)
    print()
    if dat_answer != None and dat != 'No rule variant satisfied. Consider more detailed analysis or clinician review.' and dat != None:
        cn = 0
        for x, y in dat.items():
            cv2.putText(frame, f'{x} - {y[0]} * {y[1]:.2f}', (50, 380 + cn), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            cn += 30
    print(answer_ecg)
    if answer_ecg[1] != None:
        # Загружаем изображение с альфа-каналом (PNG с прозрачностью)
        overlay = cv2.imread(f'card_for_code/plot_{1}.png', cv2.IMREAD_UNCHANGED)  # Важно: UNCHANGED!
        #  overlay = cv2.resize(overlay, (600, 300), interpolation=cv2.INTER_AREA)

        # Проверяем, что есть 4 канала (BGR + Alpha)
        if overlay.shape[2] == 4:
            # Разделяем каналы
            bgr = overlay[:, :, :3]  # BGR часть
            alpha = overlay[:, :, 3]  # Альфа-канал (0-255)

            # Нормализуем альфа-канал в диапазон [0, 1]
            alpha = alpha.astype(float) / 255.0

            # Размеры
            h, w = overlay.shape[:2]
            y, x = 250, 300  # Позиция, куда вставляем (можно менять)

            # Проверяем, не выходит ли за границы фона
            if y + h > frame.shape[0] or x + w > frame.shape[1]:
                print("Изображение выходит за границы фона!")
            else:
                # Область фона, куда будем вставлять
                roi = frame[y:y + h, x:x + w]

                # Альфа-блендинг: result = (alpha * foreground) + ((1 - alpha) * background)
                for c in range(3):  # по каждому каналу B, G, R
                    roi[:, :, c] = (alpha * bgr[:, :, c] + (1 - alpha) * roi[:, :, c])

                # Вставляем обратно в фон
                frame[y:y + h, x:x + w] = roi
        else:
            h, w = overlay.shape[:2]

            frame[100:100 + h, 300:300 + w] = overlay

    cv2.imshow("Monitor", frame)
    key = cv2.waitKey(100)
    if key == 27:  # ESC
        break
    if key == 32:
        if flag == False:
            flag = True
        else:
            flag = False
cap.release()
cv2.destroyAllWindows()

cv2.destroyAllWindows()