# file: train_resnet1d_two_channel.py
import os
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

# -------------------------
# 0. Настрой: поменяй путь к CSV
# -------------------------
csv_path   = "dataframe_ecg_4_0_500gc.csv"   # <- Укажи свой CSV (первый столбец = метка, потом 5000 чисел)

df = pd.read_csv(csv_path)
label_col = df.columns[0]
print(label_col)
