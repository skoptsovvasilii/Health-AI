import os

# Базовая директория
base_dir = "/Users/vasilii/Desktop/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/WFDBRecords"  # Замени на свой путь, например, "/Users/vasilii/Health-AI"

x = {"AV blocada": ["233917008", "270492004", "195042002", "54016002", "28189009", "27885002"], "Миокарда":["164865005"], "Ишемия":["428750005", "164931005", "164930006", "429622005"], "фибриляция":["164889003"]}

# Словарь для подсчета осложнений
complications_count = {}
flag = True
# Рекурсивный перебор всех папок и подкаталогов
for root, dirs, files in os.walk(base_dir):
  #  print(f"Проверяется папка: {root}")
    # Ищем файл RECORDS в текущей папке
    if "RECORDS" in files:
        records_file = os.path.join(root, "RECORDS")
   #     print(f"Файл RECORDS найден: {records_file}")
        with open(records_file, 'r', encoding='utf-8') as f:
            record_names = [line.strip() for line in f if line.strip()]
 #       print(f"Имена записей из RECORDS: {record_names}")
#
        # Проходим по именам из RECORDS
        for record_name in record_names:
            hea_file = os.path.join(root, f"{record_name}.hea")
            if os.path.exists(hea_file):
    #            print(f"Файл .hea найден: {hea_file}")
                with open(hea_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("#Dx:"):
     #                       print(f"Найдена строка Dx: {line}")
                            # Извлекаем коды Dx
                            dx_codes = line.replace("#Dx:", "").split(",")
                            for code in dx_codes:
                                code = code.strip()
                                if code:
                                    if code == "164865005" and flag:
                                        print(record_name)
                                        print(code)
                                        flag = False
                                    complications_count[code] = complications_count.get(code, 0) + 1
      #                              print(f"Добавлен код: {code}, текущий счет: {complications_count[code]}")
            else:
                pass
       #         print(f"Файл .hea не найден: {hea_file}")
    else:
        pass
      #  print(f"Файл RECORDS не найден в {root}")

# Выводим результат
print("Количество осложнений по кодам:")
if complications_count:
    for i, j in x.items():
        print(f"{i} : ")
        for i in j:
            try:
                print(complications_count[i])
            except:
                pass

else:
    print("Нет данных об осложнениях.")