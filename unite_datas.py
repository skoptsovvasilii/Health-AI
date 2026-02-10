'''import pandas as pd

# =======================
# ПУТИ К ФАЙЛАМ
# =======================

BASE_DATASET_PATH = "updated_data.csv"          # твой НОВЫЙ датасет (норма)
EXTRA_DATASET_PATH = "dataframe_ecg_4_0_500gc.csv"   # старый датасет (АВ, ФП, ИМ)
OUTPUT_PATH = "ecg_dataset_full.csv"                # итоговый файл

# =======================
# ЗАГРУЗКА
# =======================

print("[INFO] loading base dataset...")
df_base = pd.read_csv(BASE_DATASET_PATH)

print("[INFO] loading extra dataset...")
df_extra = pd.read_csv(EXTRA_DATASET_PATH)

# =======================
# ПРОВЕРКИ (ОБЯЗАТЕЛЬНО)
# =======================

# 1. одинаковое число колонок
if df_base.shape[1] != df_extra.shape[1]:
    raise ValueError(
        f"Column count mismatch: "
        f"{df_base.shape[1]} vs {df_extra.shape[1]}"
    )

# 2. одинаковые имена колонок
if list(df_base.columns) != list(df_extra.columns):
    raise ValueError("Column names mismatch between datasets")

print("[OK] datasets are compatible")

# =======================
# ОБЪЕДИНЕНИЕ
# =======================

df_full = pd.concat(
    [df_base, df_extra],
    axis=0,
    ignore_index=True
)

# =======================
# СОХРАНЕНИЕ
# =======================

df_full.to_csv(OUTPUT_PATH, index=False)

print(f"[DONE] merged dataset saved to: {OUTPUT_PATH}")

# =======================
# СТАТИСТИКА (ОЧЕНЬ ПОЛЕЗНО)
# =======================

print("\nClass distribution:")
print(df_full["label"].value_counts())

print("\nTotal samples:", len(df_full))
'''







import pandas as pd

# =======================
# ФАЙЛЫ
# =======================

NEW_DATASET_PATH = "updated_data.csv"        # новый (чистая норма)
OLD_DATASET_PATH = "dataframe_ecg_4_0_500gc.csv"         # старый (все классы)
OUTPUT_PATH = "ecg_dataset_full.csv"             # итог

# =======================
# ЗАГРУЗКА
# =======================

print("[INFO] loading NEW dataset (norm only expected)...")
df_new = pd.read_csv(NEW_DATASET_PATH)

print("[INFO] loading OLD dataset...")
df_old = pd.read_csv(OLD_DATASET_PATH)

# =======================
# ЧИСТКА СЛУЖЕБНЫХ СТОЛБЦОВ
# =======================

def drop_junk_columns(df):
    junk = [c for c in df.columns if c.lower().startswith("unnamed")]
    return df.drop(columns=junk)


df_new = drop_junk_columns(df_new)
df_old = drop_junk_columns(df_old)


# =======================
# ПРОВЕРКА label
# =======================

if "class" not in df_new.columns or "class" not in df_old.columns:
    raise ValueError("Column 'class' not found in one of datasets")

# =======================
# ФИЛЬТРАЦИЯ ПО КЛАССАМ
# =======================

# В новом — оставляем ТОЛЬКО норму
df_new = df_new[df_new["class"] == "norm"]

# В старом — убираем норму
df_old = df_old[df_old["class"] != "norm"]

print("[INFO] class distribution after filtering:")
print("NEW:")
print(df_new["class"].value_counts())
print("OLD:")
print(df_old["class"].value_counts())

# =======================
# ПРИВЕДЕНИЕ К ОДИНАКОВЫМ СТОЛБЦАМ
# =======================

# Берём эталонные колонки из нового датасета
df_old = df_old[df_new.columns]

# =======================
# ОБЪЕДИНЕНИЕ
# =======================

df_full = pd.concat(
    [df_new, df_old],
    axis=0,
    ignore_index=True
)

# =======================
# СОХРАНЕНИЕ
# =======================

df_full.to_csv(OUTPUT_PATH, index=False)

print(f"\n[DONE] final dataset saved to {OUTPUT_PATH}")
print("\nFinal class distribution:")
print(df_full["label"].value_counts())
print("\nTotal samples:", len(df_full))
