import wfdb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk

# Укажи путь к файлу и имя записи


# Читаем данные из .dat
record = wfdb.rdsamp("/Users/vasilii/Desktop/european-st-t-database-1.0.0/e0105", sampto=650000)  # Читаем до 650000 сэмплов

# Извлекаем сигналы
signals, fields = record  # Распаковываем кортеж
lead_ii = signals[:, 0]  # Первый канал (замени индекс, если II — другой)

# Читаем аннотации из .atr
annotation = wfdb.rdann("/Users/vasilii/Desktop/european-st-t-database-1.0.0/e0105", 'atr')

# Создаем временную шкалу
fs = fields['fs']  # Частота дискретизации (например, 500 Гц)
t = range(len(lead_ii))

# Создаем окно
root = tk.Tk()
root.title("ЭКГ с аннотациями")

fig, ax = plt.subplots(figsize=(15, 4))
ax.plot(t, lead_ii, 'b-', linewidth=0.5)

# Добавляем аннотации
for i in range(len(annotation.sample)):
    if annotation.sample[i] < len(lead_ii):  # Проверяем границы
        ax.plot(annotation.sample[i], lead_ii[annotation.sample[i]], 'ro')  # Красные точки
        ax.text(annotation.sample[i], lead_ii[annotation.sample[i]], annotation.symbol[i], fontsize=8)

ax.set_title('ЭКГ: Инфаркт миокарда (II отвод, патологический Q)')
ax.set_xlabel('Время (с)')
ax.set_ylabel('Амплитуда (мВ)')
ax.grid(True, linestyle='--', alpha=0.7)
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, label='Изолиния')
ax.legend()

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
toolbar = NavigationToolbar2Tk(canvas, root)
toolbar.update()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

tk.mainloop()