

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, savgol_filter
import matplotlib.pyplot as plt

# try import pywt for wavelet denoising (optional)
try:
    import pywt
    _HAS_PYWT = True
except Exception:
    _HAS_PYWT = False

# -------------------------
# Базовые фильтры
# -------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(sig, fs, lowcut=0.5, highcut=40.0, order=4):
    """
    Zero-phase bandpass фильтр (filtfilt) — минимальное искажение формы зубцов.
    По умолчанию 0.5 - 40 Hz (хорошо для клинической ЭКГ).
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, sig)

def notch_filter(sig, fs, freq=50.0, q=30.0):
    """
    IIR notch (заглушение частоты питания).
    freq: 50 или 60 (Hz)
    q: добротность (чем больше - тем более узкая вырезка)
    """
    w0 = freq / (fs / 2)  # нормированная частота
    b, a = iirnotch(w0, q)
    return filtfilt(b, a, sig)

# -------------------------
# Baseline removal (robust)
# -------------------------
def median_baseline_remove(sig, fs, win1_sec=0.2, win2_sec=0.6):
    """
    Двухступенчатый медианный фильтр (согласно широко используемому подходу).
    win1_sec ~ небольшое окно для удаления QRS/T (например 0.2 s),
    win2_sec ~ большое окно для оценки baseline (например 0.6 s).
    Возвращает сигнал без baseline (mV).
    """
    # окна в сэмплах — округляем к нечётным числам для scipy.signal.medfilt
    import math
    def _odd(x):
        x = int(np.round(x))
        return x if x % 2 == 1 else x+1
    w1 = _odd(win1_sec * fs)
    w2 = _odd(win2_sec * fs)
    # первая медиана убирает узкие составляющие (QRS) -> сглаженный сигнал
    med1 = signal.medfilt(sig, kernel_size=w1)
    med2 = signal.medfilt(med1, kernel_size=w2)
    # baseline estimate = med2
    sig_detrended = sig - med2
    return sig_detrended

# -------------------------
# Wavelet denoising (optional)
# -------------------------
def wavelet_denoise(sig, wavelet='db6', level=None, method='soft'):
    """
    Простая вейвлет-денойзация: Thresholding на детальных коэффициентах.
    Требует pywt.
    """
    if not _HAS_PYWT:
        raise RuntimeError("pywt is required for wavelet_denoise. pip install pywt")
    # выбор уровня декомпозиции
    max_level = pywt.dwt_max_level(len(sig), pywt.Wavelet(wavelet).dec_len)
    if level is None:
        level = max(1, max_level // 2)
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    # универсальный порог (Донхоэ)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(sig)))
    # применим мягкое порогование ко всем детальным коэффициентам
    new_coeffs = [coeffs[0]] + [pywt.threshold(c, value=uthresh, mode=method) for c in coeffs[1:]]
    rec = pywt.waverec(new_coeffs, wavelet)
    # обрезаем до исходной длины
    return rec[:len(sig)]

# -------------------------
# Полный пайплайн
# -------------------------
def preprocess_ecg_pipeline(sig,
                            fs,
                            lowcut=0.5,
                            highcut=40.0,
                            notch_freq=50.0,
                            do_notch=True,
                            do_wavelet=True,
                            wavelet='db6'):
    """
    Полный pipeline:
      1) детрендинг через медианные фильтры (baseline removal)
      2) notch 50/60 Hz (опционально)
      3) bandpass 0.5-40 Hz (zero-phase)
      4) wavelet denoise (опционально)
      5) лёгкая Savitzky-Golay фильтрация для сглаживания (необязательна)
    Возвращает очищенный сигнал.
    """
    sig = np.asarray(sig, dtype=float)
    # 1) baseline removal robust
    sig = median_baseline_remove(sig, fs, win1_sec=0.2, win2_sec=0.6)

    # 2) notch
    if do_notch and notch_freq is not None:
        # если сеть или регион 60 Hz - укажи 60
        try:
            sig = notch_filter(sig, fs, freq=notch_freq, q=30.0)
        except Exception:
            # fallback: попытка со второй показанной нотацией
            sig = sig

    # 3) bandpass
    sig = bandpass_filter(sig, fs, lowcut=lowcut, highcut=highcut, order=4)

    # 4) wavelet denoising (optional but useful for muscle noise)
    if do_wavelet and _HAS_PYWT:
        try:
            sig = wavelet_denoise(sig, wavelet=wavelet)
        except Exception:
            pass

    # 5) лёгкая финальная сглаживающая обработка (не портит пики при небольших параметрах)
    # используем Savitzky-Golay для удаления мелких рывков (frame ~ 21 samples)
    window = 11 if int(0.02 * fs) < 11 else _odd(int(0.02 * fs))  # ~20 ms
    try:
        sig = savgol_filter(sig, window_length=window, polyorder=2)
    except Exception:
        pass

    return sig

# helper: ensure odd
def _odd(x):
    x = int(np.round(x))
    return x if x % 2 == 1 else x+1

# -------------------------
# Пример использования и визуализация
# -------------------------
if __name__ == "__main__":
    # Пример: сгенерируем "плохой" сигнал: синус + baseline drift + 60Hz + random spikes
    fs = 500

    import matplotlib.pyplot as plt
    import struct

    # Укажи путь к твоему файлу
    file_path = "00001_hr.dat"  # Замени на реальный путь, например, "C:/Users/Имя/файл.dat"

    try:
        # Открываем файл в бинарном режиме
        with open(file_path, 'rb') as file:
            binary_content = file.read()  # Читаем все байты

        # Декодируем бинарные данные как список int32
        numbers = []
        for i in range(0, len(binary_content) - 3, 4):  # Шаг 4 байта
            if i + 3 < len(binary_content):  # Проверяем, чтобы не выйти за пределы
                number = struct.unpack('i', binary_content[i:i + 4])[0]
                numbers.append(number)

        # Выводим первые несколько чисел для проверки
        print("Первые 10 чисел:", numbers[:10], len(numbers))



    except FileNotFoundError:
        print("Файл не найден. Проверь путь!")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

    ecg_noisy = numbers
    t = range(len(numbers))





    denoised = preprocess_ecg_pipeline(ecg_noisy, fs=fs, lowcut=0.5, highcut=40.0, notch_freq=50.0, do_wavelet=True)

    # Визуализация
    plt.figure(figsize=(12,6))
    plt.subplot(211)
    plt.plot(t, ecg_noisy, label='Noisy ECG', linewidth=0.8)
    plt.title("Noisy ECG (demo)")
    plt.legend()
    plt.subplot(212)
    plt.plot(t, denoised, label='Denoised ECG', linewidth=1.0)
    plt.title("Denoised ECG (pipeline)")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()



import matplotlib.pyplot as plt
import struct

# Укажи путь к твоему файлу
file_path = "00482_hr.dat"  # Замени на реальный путь, например, "C:/Users/Имя/файл.dat"

try:
    # Открываем файл в бинарном режиме
    with open(file_path, 'rb') as file:
        binary_content = file.read()  # Читаем все байты

    # Декодируем бинарные данные как список int32
    numbers = [1]
    for i in range(0, len(binary_content) - 3, 4):  # Шаг 4 байта
        if i + 3 < len(binary_content):  # Проверяем, чтобы не выйти за пределы
            number = struct.unpack('i', binary_content[i:i+4])[0]
            numbers.append(number)

    # Выводим первые несколько чисел для проверки
    print("Первые 10 чисел:", numbers[:10], len(numbers))

    # Строим график
    plt.figure(figsize=(10, 6))  # Устанавливаем размер графика
    plt.plot(numbers, label="Данные из .dat файла")  # Строим линейный график
    plt.title("График бинарных данных из файла")  # Заголовок
    plt.xlabel("Индекс")  # Ось X
    plt.ylabel("Значение")  # Ось Y
    plt.legend()  # Легенда
    plt.grid(True)  # Сетка для удобства
    plt.show()

except FileNotFoundError:
    print("Файл не найден. Проверь путь!")
except Exception as e:
    print(f"Произошла ошибка: {e}")