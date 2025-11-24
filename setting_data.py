import wfdb
import matplotlib.pyplot as plt

import os
import shutil

'''



import os
import shutil

# Указываем базовую директорию, где лежат папки 00000-21000
base_dir = "/Users/vasilii/Desktop/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records500"  # Замени на реальный путь, например, "C:/Users/Имя/PTB-XL"

# Создаем новую папку для всех файлов
new_folder = "cardiogramm_base_ptb"
new_folder_path = os.path.join(base_dir, new_folder)
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)
    print(f"Создана новая папка: {new_folder_path}")
else:
    print(f"Папка {new_folder_path} уже существует, файлы будут добавлены.")

# Диапазон папок
start_folder = 0
end_folder = 21000

# Перебираем папки
for folder_num in range(start_folder, end_folder + 1, 1000):
    folder_name = f"{folder_num:05d}"  # Форматируем как "00000", "00001" и т.д.
    folder_path = os.path.join(base_dir, folder_name)

    # Проверяем, существует ли папка
    if os.path.exists(folder_path):
        # Перебираем файлы в папке
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".dat"):
                source_path = os.path.join(folder_path, file_name)
                destination_path = os.path.join(new_folder_path, file_name)
                if os.path.exists(source_path):
                    shutil.move(source_path, destination_path)
            #        print(f"Перемещен файл: {file_name} в {new_folder}")
                else:
                    print(f"Файл {file_name}.dat не найден в {folder_path}")
            if file_name.endswith(".hea"):
                source_path = os.path.join(folder_path, file_name)
                destination_path = os.path.join(new_folder_path, file_name)
                if os.path.exists(source_path):
                    shutil.move(source_path, destination_path)
          #          print(f"Перемещен файл: {file_name} в {new_folder}")
                else:
                    print(f"Файл {file_name}.hea не найден в {folder_path}")

    else:
        print(f"Папка {folder_name} не найдена, пропускаем.")

print("Перемещение завершено!")












# Просто укажи полный путь к файлу БЕЗ pn_dir
record = wfdb.rdrecord("/Users/vasilii/Desktop/files/I01")

# Выводим названия отведений
print(record.sig_name)

# Ищем индекс II
lead_index = record.sig_name.index('II')

# Достаём II
lead_ii = record.p_signal[:, lead_index]

# Рисуем
plt.plot(lead_ii)
plt.title("Отведение II")
plt.show()

t = range(462600)

plt.figure(figsize=(15, 4))
plt.plot(t, lead_ii, 'b-', linewidth=1)

plt.title('ЭКГ: Инфаркт миокарда (II отвод, патологический Q)')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда (мВ)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3, label='Изолиния')
plt.legend()
plt.tight_layout()
plt.show()'''








import os
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import scipy.io
import tqdm


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import resample







def resample_ecg(signal, target_len=10000):
    """Ресемплинг ЭКГ до нужной длины"""
    return resample(signal, target_len)



# --- Функции фильтрации ---
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(sig, fs, lowcut=0.5, highcut=40.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, sig)

# --- Функция для отрисовки ---
def plot_signal(sig, fs, title="Сигнал"):
    t = np.arange(len(sig)) / fs
    plt.figure(figsize=(12, 4))
    plt.plot(t, sig, linewidth=1)
    plt.xlabel("Время (с)")
    plt.ylabel("Амплитуда (мВ)")
    plt.title(title)
    plt.grid(True)
    plt.show()





import os
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import scipy.io

def load_ecg(full_path, lead_name='II'):
    """
    Загружает ECG-сигнал из .mat или .hea/.dat по полному пути.
    """
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Файл не найден: {full_path}")

    _, ext = os.path.splitext(full_path)
    ext = ext.lower()

    if ext == '.mat':
  #      print(f"[MAT] Загружаю {full_path}")
        m = scipy.io.loadmat(full_path)
        if 'val' in m:
            arr = m['val']
            if arr.ndim == 2:
                sig = arr[0, :]
            else:
                sig = arr.flatten()
            fs = m.get('fs', None) or m.get('Fs', None) or None
            if fs is None:
                fs = float(input("Введите частоту дискретизации (Hz): "))
            return sig, fs, os.path.basename(full_path)
        else:
            raise ValueError(f"Нет ключа 'val' в {full_path}")

    elif ext == '.hea' or ext == '.dat':
        # Если дали .dat или .hea, убираем расширение и читаем через wfdb
        base = os.path.splitext(full_path)[0]
    #    print(f"[WFDB] Загружаю {base}")
        rec = wfdb.rdrecord(base)
        print(rec.sig_name)
        if lead_name in rec.sig_name:
            i = rec.sig_name.index(lead_name)
        else:
            print(f"Отведение {lead_name} не найдено, беру первый канал.")
            i = 0
        sig = rec.p_signal[:, i]
        fs = rec.fs
        return sig, fs, os.path.basename(base)

    else:
        raise ValueError("Поддерживаются только файлы .mat, .hea или .dat")




def load_ecg(full_path, lead_name='II'):
    """
    Загружает ECG-сигнал из .mat или .hea/.dat по полному пути.
    """
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Файл не найден: {full_path}")

    _, ext = os.path.splitext(full_path)
    ext = ext.lower()

    if ext == '.mat':
  #      print(f"[MAT] Загружаю {full_path}")
        m = scipy.io.loadmat(full_path)
        if 'val' in m:
            arr = m['val']
            if arr.ndim == 2:
                sig = arr[0, :]
            else:
                sig = arr.flatten()
            fs = m.get('fs', None) or m.get('Fs', None) or None
            if fs is None:
                fs = float(input("Введите частоту дискретизации (Hz): "))
            return sig, fs, os.path.basename(full_path)
        else:
            raise ValueError(f"Нет ключа 'val' в {full_path}")

    elif ext == '.hea' or ext == '.dat':
        # Если дали .dat или .hea, убираем расширение и читаем через wfdb
        base = os.path.splitext(full_path)[0]
      #  print(f"[WFDB] Загружаю {base}")
        rec = wfdb.rdrecord(base)
     #   print(rec.sig_name)
        if lead_name in rec.sig_name:
            i = rec.sig_name.index(lead_name)
        else:
            print(f"Отведение {lead_name} не найдено, беру первый канал.")
            i = 0
        sig = rec.p_signal[:, i]
        fs = rec.fs
        return sig, fs, os.path.basename(base)

    else:
        raise ValueError("Поддерживаются только файлы .mat, .hea или .dat")

def plot_signal(sig, fs, title='ECG Signal'):
    t = np.arange(len(sig)) / fs
    plt.figure(figsize=(10,4))
    plt.plot(t, sig, linewidth=1.0)
    plt.title(title)
    plt.xlabel("Время (с)")
    plt.ylabel("Амплитуда (мВ)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


'''
# ==== ПРИМЕР ИСПОЛЬЗОВАНИЯ ====
# Вставь сюда полный путь к файлу
file_path = "/Users/vasilii/Desktop/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/WFDBRecords/03/038/JS02925.hea"

try:
    sig, fs, name = load_ecg(file_path, lead_name='II')
    # Нормализация (опционально)
    sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
    plot_signal(sig, fs, title=f"{name} — Отведение II")
except Exception as e:
    print("Ошибка:", e)
'''













import os
import time
import shutil
import csv

a  = input("input 12345 and dont forgot setting answer to val")

if a == "12345":
    print("good")
    answer = {"norm":4000, "AV blocada":2000, "infarct":2000, "ishemia":2000, "fibril":2000}

    for i in range(5, 0, -1):
        print(f" до старта {i} секунд! заверши если случайно нажмешь")
        time.sleep(1)
else:
    exit()





import pandas as pd
import ast  # для безопасного преобразования строки в dict


count = {"norm":0, "AV blocada":0, "infarct":0,  "fibril":0}
# Загружаем CSV
files = ["norm", "AV blocada", "infarct", "fibril"]
# Список осложнений, которые нас интересуют
needed = [["NORM"], ["1AVB", "2AVB", "3AVB"], ["IMI", "ILMI", "AMI", "ALMI", "LMI", "IPLMI", "IPMI", "PMI", "ASMI"], ["AFIB"]]





x1 =[]


# Преобразуем строку в dict и фильтруем
def has_needed_condition(val, num):
    global x1
    d = ast.literal_eval(val)
    for i in range(len(needed)):
        for j in needed[i]:
            try:
                if d[j] != 0.0:
                    if answer[files[i]] >= count[files[i]]:
                        count[files[i]]+= 1
                        x1.append([df["filename_lr"][num], files[i]])
            except:
                pass


# Загружаем CSV
df = pd.read_csv("/Users/vasilii/Desktop/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv")


for i in range(0, 21790):
    filtered_df = has_needed_condition(df['scp_codes'][i], i)

for x, y in count.items():
    print(f"{x} - {y}")







csv_file = "dataframe_ecg_4_0_500gc.csv"
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    x2 = ["class"]
    [x2.append(str(i)) for i in range(1, 5_000)]
    writer.writerow(x2)  # Заголовок
#    writer.writerow(["JS00001", "II", 0.5])





    base_dir = "/Users/vasilii/Desktop/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records500"  # Замени на реальный путь, например, "C:/Users/Имя/PTB-XL"

    # Диапазон папок
    start_folder = 0
    end_folder = 21000



    for i in tqdm.tqdm(x1):
        url = f"/Users/vasilii/Desktop/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/{i[0]}.hea"
        sig, fs, name = load_ecg(url, lead_name='II')
        sig = resample_ecg(sig, target_len=5000)
        sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
        fs=500


        sig_filt = bandpass_filter(sig, fs, lowcut=0.5, highcut=40)

        for start in range(0, len(sig_filt)+1 - 2_500 + 1, 1_000):
            n = [i[1]]
            cardi = sig_filt[start:start+5_000]
            [n.append(i1) for i1 in sig_filt[start:start+2_500]]
            for i2 in range(1, len(sig_filt[start:start+2_500]-1)):
                n.append(cardi[i2]-cardi[i2-1])
            writer.writerow(n)


    print("Перемещение завершено! from PLT")
    print("got to next ->")


    base_dir = "/Users/vasilii/Desktop/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/WFDBRecords"  # Замени на свой путь, например, "/Users/vasilii/Health-AI"

    x123 = {"AV blocada": ["233917008", "270492004", "195042002", "54016002", "28189009", "27885002"], "infarct":["164865005"], "fibril":["164889003"]}
    check = {"AV blocada":[0, 2000], "infarct":[0, 2000], "ishemia":[0, 2000] ,"fibril":[0, 2000]}
    complications_count = {}
    flag = True
    for root, dirs, files in tqdm.tqdm(os.walk(base_dir)):


        if "RECORDS" in files:
            records_file = os.path.join(root, "RECORDS")
            with open(records_file, 'r', encoding='utf-8') as f:
                record_names = [line.strip() for line in f if line.strip()]
    #
            for record_name in record_names:
                hea_file = os.path.join(root, f"{record_name}.hea")
                if os.path.exists(hea_file):
                    with open(hea_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("#Dx:"):
                                dx_codes = line.replace("#Dx:", "").split(",")
                                for code in dx_codes:
                                    code = code.strip()
                                    if code:
                                        complications_count[code] = complications_count.get(code, 0) + 1
                                        for key, val in x123.items():
                                            for values in val:
                                                if code==values:
                                                    if check[key][0]<=check[key][1]:
                                                        check[key][0]+=1
                                                        dt = os.path.join(root, f"{record_name}.hea")
                                                        sig, fs, name = load_ecg(dt, lead_name='II')
                                                        sig = resample_ecg(sig, target_len=5_000)
                                                        sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
                                                        fs = 500

                                                        sig_filt = bandpass_filter(sig, fs, lowcut=0.5, highcut=40)
                                                        for i in range(0, 5_001-2_500, 1_000):
                                                            n = [key]
                                                            cardi = sig_filt[i:i + 5_000]
                                                            [n.append(i1) for i1 in sig_filt[i:i + 2_500]]
                                                            for i2 in range(1, len(sig_filt[i:i + 2_500] - 1)):
                                                                n.append(cardi[i2] - cardi[i2 - 1])
                                                            writer.writerow(n)



                else:
                    pass
        else:
            pass