import cv2
import numpy as np
import time
import threading
import random
import sys
import os
import shutil
import torch
import torch.nn as nn
import time
from serial.tools import list_ports
from scipy.signal import butter, filtfilt




import serial
import struct
import threading
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(serial.tools.list_ports)

ports = list(serial.tools.list_ports.comports())

for port in ports:
    print(f"Порт: {port.device}")
    print(f"Описание: {port.description}")
    print(f"Производитель: {port.manufacturer}\n")
tempF= []
import csv
import os

PORT = '/dev/cu.usbmodem144201'   # <-- проверь порт
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


# Запускаем поток чтения
threading.Thread(target=serial_reader, daemon=True).start()

# =======================
# MATPLOTLIB
# =======================

plt.style.use('dark_background')
plt.ion()

fig, ax = plt.subplots(figsize=(10, 4))
line, = ax.plot(np.zeros(WINDOW_SIZE), lw=1, color='red')

ax.set_title("ECG realtime")
ax.set_xlabel("Samples")
ax.set_ylabel("ADC value")

ax.set_ylim(-100, 100)
ax.set_xlim(0, WINDOW_SIZE)
print(1)
plt.tight_layout()


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


'''
print("Включаю тревожный сигнал!")
start_alarm()

time.sleep(10)
'''

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


'''
ports = list(serial.tools.list_ports.comports())

for port in ports:
    print(f"Порт: {port.device}")
    print(f"Описание: {port.description}")
    print(f"Производитель: {port.manufacturer}\n")
    '''
tempF = []

pressure = []

# arduinoData = serial.Serial('/dev/cu.usbmodem144201', 9600) #Creating our serial object named arduinoData

plt.ion()  # Tell matplotlib you want interactive mode to plot live data

cnt = 0
number_i = 0

pulse_data = np.array([float(0) for i in range(2508)]).astype(np.float32)

while True:

    #if len(ecg_buffer) >= WINDOW_SIZE:

    data = np.array(list(ecg_buffer)[-WINDOW_SIZE:], dtype=np.float32)
    data -= 300
    data = bandpass_filter(data, 500)

        # ===== визуализация =====
    line.set_ydata(data)
    fig.canvas.draw()
    fig.canvas.flush_events()

    d = np.diff(data, prepend=data[0]).astype(np.float32)[1:]
    row = list(data) + list(d)

        # ===== ML (если нужно) =====
    verdict, probabilities, p = predict_one(row)
    print(verdict, probabilities)


    d = np.diff(data, prepend=data[0]).astype(np.float32)[1:]
    row = ['norm'] + list(data) + list(d)
    print(row)
    print(len(row))

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

cv2.destroyAllWindows()