import numpy as np

def fibrilation(lst, time, fs):
    x = len(lst)*5//100
    lst = lst[x:len(lst)-x]
    one = 1/(len(lst)/time)
    QRS = []
    cn =[0, 0]
    bits_sec_QRS = []
    wave = []
    flagq = False
    last_val = 0
    flag, flag_wave = True, True
    cn_wave = 0
    last_val = 5.0
    flag_push = False
    cnt = 0
    k= 0

    for num, i in enumerate(lst):
        if i > 1.0:
            if flag:
                QRS.append(num * one)
                cn[0] += 1
                flag_push = True
            flagq = True
            flag =  False
            if num * one > cn[1]:
                cn[1] += 1
                bits_sec_QRS.append(cn[0])
                cn[0] = 0



        elif i < 1.0:
            flag = True



        if i > 0.0:
            if flag_wave:
                flag_wave = False
                cn_wave += 1
                if flag_push:
                    wave.append(cn_wave)
                    cn_wave = 0
                    flag_push=False
        elif i <= 0.0:
            flag_wave = True


    difference = []
    for i in range(1, len(QRS)):
        difference.append(abs(QRS[i-1]-QRS[i]))
    QRS_timing = max(difference) - min(difference)
    count = 0
    num = [0, 0, 0]
    if QRS_timing > 0.3:   # difference QRS time up
        count += 1
        num[0] = 1
    if max(wave) >3:   # f - waves
        count += 2
        num[1] = 2
    if  np.mean(bits_sec_QRS)>=1.2:    # media bits hearts
        count+=1
        num[2] = 1

    return count, num













class Detect_ECG:
    def __init__(self, time, fs):
        self.lst = []
        self.time = time
        self.fs = fs
        self.result = None
    def detect(self, lst):
        result, num =fibrilation(lst, self.time, self.fs)
        return num









'''
fibrilation








# Параметры сигнала
fs = 1000  # Частота дискретизации (Гц)
duration = 5  # Длительность записи (сек)
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# 1. Классическая ФП (хаотичные QRS, нет P, f-волны)
def generate_afib_classic(t, hr=100):
    ecg = np.zeros_like(t)
    heart_rate = hr / 60
    for i in range(int(duration * heart_rate)):
        cycle_start = i / heart_rate + np.random.uniform(-0.15, 0.15)  # Рандомные R-R
        cycle_t = t - cycle_start
        mask = (cycle_t >= 0) & (cycle_t < 0.6)
        # QRS (шире, чем в норме)
        qrs = 1.8 * np.exp(-300 * (cycle_t[mask] - 0.2)**2)
        # f-волны (мелкие осцилляции)
        f_waves = 0.1 * np.sin(2 * np.pi * 8 * cycle_t[mask])
        ecg[mask] += qrs + f_waves
    return ecg

# 2. ФП с "крупноволновой" фибрилляцией (четкие f-волны)
def generate_afib_coarse(t, hr=90):
    ecg = np.zeros_like(t)
    heart_rate = hr / 60
    for i in range(int(duration * heart_rate)):
        cycle_start = i / heart_rate + np.random.uniform(-0.1, 0.1)
        cycle_t = t - cycle_start
        mask = (cycle_t >= 0) & (cycle_t < 0.7)
        # Крупные f-волны (амплитуда ~0.3)
        f_waves = 0.3 * np.sin(2 * np.pi * 6 * cycle_t[mask])
        # QRS (менее выражены)
        qrs = 1.2 * np.exp(-250 * (cycle_t[mask] - 0.25)**2)
        ecg[mask] += qrs + f_waves
    return ecg

# 3. ФП с "мелковолновой" фибрилляцией (едва заметные f-волны)
def generate_afib_fine(t, hr=110):
    ecg = np.zeros_like(t)
    heart_rate = hr / 60
    for i in range(int(duration * heart_rate)):
        cycle_start = i / heart_rate + np.random.uniform(-0.2, 0.2)  # Сильная нерегулярность
        cycle_t = t - cycle_start
        mask = (cycle_t >= 0) & (cycle_t < 0.5)
        # Мелкие f-волны (амплитуда ~0.05)
        f_waves = 0.05 * np.sin(2 * np.pi * 12 * cycle_t[mask])
        # Узкие QRS (имитация желудочковых комплексов)
        qrs = 1.5 * np.exp(-400 * (cycle_t[mask] - 0.15)**2)
        ecg[mask] += qrs + f_waves
    return ecg

# Генерация сигналов
ecg_afib1 = generate_afib_classic(t)
ecg_afib2 = generate_afib_coarse(t)
ecg_afib3 = generate_afib_fine(t)


detect = Detect_ECG(5, fs)
print(detect.detect(ecg_afib1))
print(detect.detect(ecg_afib2))
print(detect.detect(ecg_afib3))'''

import numpy as np
import matplotlib.pyplot as plt

# Параметры сигнала
fs = 1000  # Частота дискретизации (Гц)
duration = 6  # Длительность записи (сек)
t = np.linspace(0, duration, int(fs * duration), endpoint=False)


# Функция генерации одного цикла P-QRS-T
def generate_normal_cycle(t_cycle, p_delay=0.0):
    p_wave = 0.25 * np.exp(-50 * (t_cycle - 0.1 - p_delay) ** 2)  # Зубец P
    qrs = 1.5 * np.exp(-500 * (t_cycle - 0.25) ** 2)  # QRS
    t_wave = 0.3 * np.exp(-50 * (t_cycle - 0.4) ** 2)  # Зубец T
    return p_wave + qrs + t_wave


# 1. АВ-блокада 1-й степени (удлинённый PQ > 0.2 сек)
def generate_av_block_1(t):
    ecg = np.zeros_like(t)
    heart_rate = 70 / 60  # ЧСС = 70 уд/мин
    for i in range(int(duration * heart_rate)):
        cycle_start = i / heart_rate
        cycle_t = t - cycle_start
        mask = (cycle_t >= 0) & (cycle_t < 0.8)
        ecg[mask] += generate_normal_cycle(cycle_t[mask], p_delay=0.15)  # PQ = 0.25 сек
    return ecg


# 2. АВ-блокада 2-й степени (Мобитц I: прогрессирующее удлинение PQ + выпадение QRS)
def generate_av_block_2_mobitz1(t):
    ecg = np.zeros_like(t)
    heart_rate = 60 / 60  # ЧСС = 60 уд/мин
    pq_delay = 0.1  # Начальное удлинение PQ
    for i in range(int(duration * heart_rate)):
        cycle_start = i / heart_rate
        cycle_t = t - cycle_start
        mask = (cycle_t >= 0) & (cycle_t < 0.9)
        if i % 4 != 3:  # Каждое 4-е сокращение выпадает
            ecg[mask] += generate_normal_cycle(cycle_t[mask], p_delay=pq_delay)
            pq_delay += 0.05  # PQ растёт
        else:
            pq_delay = 0.1  # Сброс после выпадения
    return ecg


# 3. АВ-блокада 3-й степени (полная: P и QRS независимы)
def generate_av_block_3(t):
    ecg = np.zeros_like(t)
    p_rate = 80 / 60  # ЧСС предсердий = 80 уд/мин
    qrs_rate = 40 / 60  # ЧСС желудочков = 40 уд/мин

    # Генерация P-зубцов
    for i in range(int(duration * p_rate)):
        cycle_start = i / p_rate
        cycle_t = t - cycle_start
        mask = (cycle_t >= 0) & (cycle_t < 0.7)
        ecg[mask] += 0.2 * np.exp(-50 * (cycle_t[mask] - 0.1) ** 2)  # Только P

    # Генерация QRS (желудочковые комплексы)
    for i in range(int(duration * qrs_rate)):
        cycle_start = i / qrs_rate
        cycle_t = t - cycle_start
        mask = (cycle_t >= 0) & (cycle_t < 0.7)
        ecg[mask] += 1.5 * np.exp(-500 * (cycle_t[mask] - 0.3) ** 2)  # Только QRS
    return ecg


# Генерация сигналов
ecg_av1 = generate_av_block_1(t)
ecg_av2 = generate_av_block_2_mobitz1(t)
ecg_av3 = generate_av_block_3(t)



detect = Detect_ECG(6, fs)
print(detect.detect(ecg_av1))
print(detect.detect(ecg_av2))
print(detect.detect(ecg_av3))