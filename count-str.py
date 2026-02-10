"""

import csv

# Открываем файл и считаем количество строк
with open('ecg_dataset_norm.csv', 'r') as file:
    reader = csv.reader(file)
    row_count = sum(1 for _ in reader)

print(f'Количество строк в файле: {row_count}')


"""


"""



import pandas as pd

# Чтение существующего CSV-файла
df = pd.read_csv('ecg_dataset_norm.csv')

# Новый список имен столбцов
new_columns = ["class"] + [i for i in range(1, 5_000)]

# Переименование столбцов
df.columns = new_columns

# Сохраняем обратно в CSV-файл
df.to_csv('updated_data.csv', index=False)



"""



import pandas as pd

# Читаем CSV-файл
df = pd.read_csv('updated_data.csv')

# Получаем уникальные значения столбца Gender
unique_values = df['class'].unique()

print(unique_values)
