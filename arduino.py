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



def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(sig, fs, lowcut=0.5, highcut=40.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, sig)

model_path = "ml_cardiogram_resnet_3_0.pth"

classes = ['AV blocada', 'fibril', 'infarct', 'norm']


model = ResNet1D(in_ch=2, num_classes=len(classes)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


def predict_one(pulse_data):
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
        return classes[pred_idx], probs







pulse_data = np.array([float(0) for i in range(2508)]).astype(np.float32)

# Фигуру и ось графика инициализируем вне главного цикла
fig, ax = plt.subplots()
line, = ax.plot(pulse_data, marker='.', color='red')

# Названия осей
ax.set_title("Динамика измерения пульса")
ax.set_xlabel("Измерение №")
ax.set_ylabel("Частота пульса (ударов/мин)")


# Функция обновления графика
def update_plot():
    """Обновляет график с последними измеренными данными"""
    global pulse_data
    if len(pulse_data) > 0:
        # Извлекаем индексы и сами значения
        x_values = list(range(len(pulse_data[-100:])))
        y_values = pulse_data[-100:]

        # Перерисовываем график с новыми данными
        line.set_data(x_values, y_values)

        # Авто-масштабируем оси, чтобы были видны все точки
        ax.relim()
        ax.autoscale_view()

        # Команда для немедленного обновления экрана
        fig.canvas.draw_idle()


print(serial.tools.list_ports)
ports = list(serial.tools.list_ports.comports())

for port in ports:
    print(f"Порт: {port.device}")
    print(f"Описание: {port.description}")
    print(f"Производитель: {port.manufacturer}\n")
tempF= []

pressure=[]

#arduinoData = serial.Serial('/dev/cu.usbmodem144201', 9600) #Creating our serial object named arduinoData

plt.ion() #Tell matplotlib you want interactive mode to plot live data

cnt=0




while True: # While loop that loops forever
    '''
    while (arduinoData.inWaiting()==0): #Wait here until there is data

        pass #do nothing

    arduinoString = arduinoData.readline() #read the line of text from the serial port

    dataArray = arduinoString.split() #Split it into an array called dataArray
    print(dataArray)
    '''
    dataArray = [str(i) for i in range(100, 2600)]


    c=0
    x = []
    try:
        np.append(pulse_data, float(dataArray[0][3:]))
    except:
        c+=1
    y = bandpass_filter(pulse_data[-100:], 500)

    pulse_data.astype(np.float32)
    verdict, probabilities = predict_one(pulse_data)
    print(verdict, probabilities)


    update_plot()
    plt.pause(.000001) #Pause Briefly. Important to keep drawnow from crashing

    cnt=cnt+1

plt.show(block=True)
