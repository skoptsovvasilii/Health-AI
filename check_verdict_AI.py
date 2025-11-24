import matplotlib.pyplot as plt
import numpy as np
#import statsmodels.api as sm
import math


import numpy as np
from sklearn.linear_model import LogisticRegression

import math
import random






'''vertices = np.array([[1-p_hypoxia, 0], [1+p_cyanosis, 0], [cardiogramm, 1]])

vertices = np.vstack([vertices, vertices])

plt.plot(vertices[:, 0], vertices[:, 1], marker='o')

plt.xlim(0, 2)
plt.ylim(0, 2)

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Triangle in Matplotlib')

plt.show()'''




def triangls(typ, a, b, c, show=False):
    pass



def check1(lst1, lst2, lst3):
    mi_examples = [
     {"ecg_mi_prob":0.95, "systolic_bp":155, "pulse_rate":110, "cyanosis_or_jvd_prob":0.40, "label":1},
     {"ecg_mi_prob":0.90, "systolic_bp":148, "pulse_rate":105, "cyanosis_or_jvd_prob":0.35, "label":1},
     {"ecg_mi_prob":0.85, "systolic_bp":140, "pulse_rate":100, "cyanosis_or_jvd_prob":0.30, "label":1},
     {"ecg_mi_prob":0.78, "systolic_bp":132, "pulse_rate":95,  "cyanosis_or_jvd_prob":0.28, "label":1},
     {"ecg_mi_prob":0.82, "systolic_bp":138, "pulse_rate":98,  "cyanosis_or_jvd_prob":0.25, "label":1},
     {"ecg_mi_prob":0.88, "systolic_bp":150, "pulse_rate":108, "cyanosis_or_jvd_prob":0.33, "label":1},
     {"ecg_mi_prob":0.92, "systolic_bp":160, "pulse_rate":115, "cyanosis_or_jvd_prob":0.45, "label":1},
     {"ecg_mi_prob":0.81, "systolic_bp":135, "pulse_rate":97,  "cyanosis_or_jvd_prob":0.22, "label":1},
     {"ecg_mi_prob":0.75, "systolic_bp":128, "pulse_rate":90,  "cyanosis_or_jvd_prob":0.20, "label":1},
     {"ecg_mi_prob":0.70, "systolic_bp":125, "pulse_rate":88,  "cyanosis_or_jvd_prob":0.18, "label":1},
     {"ecg_mi_prob":0.67, "systolic_bp":122, "pulse_rate":86,  "cyanosis_or_jvd_prob":0.15, "label":1},
     {"ecg_mi_prob":0.60, "systolic_bp":118, "pulse_rate":82,  "cyanosis_or_jvd_prob":0.12, "label":1},
     {"ecg_mi_prob":0.55, "systolic_bp":120, "pulse_rate":84,  "cyanosis_or_jvd_prob":0.10, "label":1},
     {"ecg_mi_prob":0.20, "systolic_bp":150, "pulse_rate":100, "cyanosis_or_jvd_prob":0.05, "label":0},
     {"ecg_mi_prob":0.30, "systolic_bp":160, "pulse_rate":110, "cyanosis_or_jvd_prob":0.10, "label":0},
     {"ecg_mi_prob":0.25, "systolic_bp":145, "pulse_rate":95,  "cyanosis_or_jvd_prob":0.08, "label":0},
     {"ecg_mi_prob":0.10, "systolic_bp":135, "pulse_rate":85,  "cyanosis_or_jvd_prob":0.02, "label":0},
     {"ecg_mi_prob":0.05, "systolic_bp":120, "pulse_rate":78,  "cyanosis_or_jvd_prob":0.01, "label":0},
     {"ecg_mi_prob":0.93, "systolic_bp":158, "pulse_rate":118, "cyanosis_or_jvd_prob":0.47, "label":1},
     {"ecg_mi_prob":0.86, "systolic_bp":146, "pulse_rate":103, "cyanosis_or_jvd_prob":0.32, "label":1},
     {"ecg_mi_prob":0.79, "systolic_bp":130, "pulse_rate":92,  "cyanosis_or_jvd_prob":0.24, "label":1},
     {"ecg_mi_prob":0.74, "systolic_bp":126, "pulse_rate":87,  "cyanosis_or_jvd_prob":0.19, "label":1},
     {"ecg_mi_prob":0.68, "systolic_bp":121, "pulse_rate":83,  "cyanosis_or_jvd_prob":0.14, "label":1},
     {"ecg_mi_prob":0.83, "systolic_bp":142, "pulse_rate":99,  "cyanosis_or_jvd_prob":0.27, "label":1},
     {"ecg_mi_prob":0.77, "systolic_bp":132, "pulse_rate":91,  "cyanosis_or_jvd_prob":0.21, "label":1},
     {"ecg_mi_prob":0.62, "systolic_bp":119, "pulse_rate":80,  "cyanosis_or_jvd_prob":0.11, "label":1},
     {"ecg_mi_prob":0.48, "systolic_bp":115, "pulse_rate":76,  "cyanosis_or_jvd_prob":0.07, "label":0},
     {"ecg_mi_prob":0.40, "systolic_bp":110, "pulse_rate":72,  "cyanosis_or_jvd_prob":0.05, "label":0},
     {"ecg_mi_prob":0.52, "systolic_bp":125, "pulse_rate":82,  "cyanosis_or_jvd_prob":0.09, "label":0},
     {"ecg_mi_prob":0.99, "systolic_bp":162, "pulse_rate":122, "cyanosis_or_jvd_prob":0.50, "label":1},
    ]




    af_examples = [
     {"ecg_af_prob":0.97, "pulse_irregularity":0.96, "spo2":92, "cyanosis_or_jvd_prob":0.30, "label":1},
     {"ecg_af_prob":0.92, "pulse_irregularity":0.90, "spo2":94, "cyanosis_or_jvd_prob":0.25, "label":1},
     {"ecg_af_prob":0.88, "pulse_irregularity":0.86, "spo2":95, "cyanosis_or_jvd_prob":0.20, "label":1},
     {"ecg_af_prob":0.85, "pulse_irregularity":0.82, "spo2":96, "cyanosis_or_jvd_prob":0.18, "label":1},
     {"ecg_af_prob":0.80, "pulse_irregularity":0.78, "spo2":97, "cyanosis_or_jvd_prob":0.12, "label":1},
     {"ecg_af_prob":0.94, "pulse_irregularity":0.92, "spo2":91, "cyanosis_or_jvd_prob":0.35, "label":1},
     {"ecg_af_prob":0.76, "pulse_irregularity":0.70, "spo2":98, "cyanosis_or_jvd_prob":0.10, "label":1},
     {"ecg_af_prob":0.70, "pulse_irregularity":0.65, "spo2":99, "cyanosis_or_jvd_prob":0.05, "label":1},
     {"ecg_af_prob":0.66, "pulse_irregularity":0.60, "spo2":98, "cyanosis_or_jvd_prob":0.07, "label":1},
     {"ecg_af_prob":0.60, "pulse_irregularity":0.55, "spo2":97, "cyanosis_or_jvd_prob":0.04, "label":0},
     {"ecg_af_prob":0.58, "pulse_irregularity":0.50, "spo2":97, "cyanosis_or_jvd_prob":0.03, "label":0},
     {"ecg_af_prob":0.95, "pulse_irregularity":0.94, "spo2":90, "cyanosis_or_jvd_prob":0.40, "label":1},
     {"ecg_af_prob":0.82, "pulse_irregularity":0.79, "spo2":95, "cyanosis_or_jvd_prob":0.22, "label":1},
     {"ecg_af_prob":0.74, "pulse_irregularity":0.68, "spo2":96, "cyanosis_or_jvd_prob":0.11, "label":1},
     {"ecg_af_prob":0.50, "pulse_irregularity":0.30, "spo2":99, "cyanosis_or_jvd_prob":0.02, "label":0},
     {"ecg_af_prob":0.25, "pulse_irregularity":0.40, "spo2":98, "cyanosis_or_jvd_prob":0.05, "label":0},
     {"ecg_af_prob":0.30, "pulse_irregularity":0.45, "spo2":97, "cyanosis_or_jvd_prob":0.06, "label":0},
     {"ecg_af_prob":0.87, "pulse_irregularity":0.84, "spo2":93, "cyanosis_or_jvd_prob":0.28, "label":1},
     {"ecg_af_prob":0.79, "pulse_irregularity":0.76, "spo2":95, "cyanosis_or_jvd_prob":0.15, "label":1},
     {"ecg_af_prob":0.69, "pulse_irregularity":0.63, "spo2":96, "cyanosis_or_jvd_prob":0.09, "label":1},
     {"ecg_af_prob":0.98, "pulse_irregularity":0.97, "spo2":89, "cyanosis_or_jvd_prob":0.45, "label":1},
     {"ecg_af_prob":0.65, "pulse_irregularity":0.58, "spo2":97, "cyanosis_or_jvd_prob":0.08, "label":0},
     {"ecg_af_prob":0.91, "pulse_irregularity":0.89, "spo2":92, "cyanosis_or_jvd_prob":0.30, "label":1},
     {"ecg_af_prob":0.77, "pulse_irregularity":0.74, "spo2":95, "cyanosis_or_jvd_prob":0.13, "label":1},
     {"ecg_af_prob":0.83, "pulse_irregularity":0.80, "spo2":94, "cyanosis_or_jvd_prob":0.21, "label":1},
     {"ecg_af_prob":0.55, "pulse_irregularity":0.48, "spo2":98, "cyanosis_or_jvd_prob":0.05, "label":0},
     {"ecg_af_prob":0.72, "pulse_irregularity":0.69, "spo2":96, "cyanosis_or_jvd_prob":0.12, "label":1},
     {"ecg_af_prob":0.59, "pulse_irregularity":0.52, "spo2":97, "cyanosis_or_jvd_prob":0.06, "label":0},
     {"ecg_af_prob":0.89, "pulse_irregularity":0.87, "spo2":93, "cyanosis_or_jvd_prob":0.27, "label":1},
     {"ecg_af_prob":0.46, "pulse_irregularity":0.35, "spo2":99, "cyanosis_or_jvd_prob":0.02, "label":0},
     {"ecg_af_prob":0.99, "pulse_irregularity":0.98, "spo2":88, "cyanosis_or_jvd_prob":0.50, "label":1},
    ]


    av_examples = [
     {"ecg_av_prob":0.98, "pulse_rate":28, "systolic_bp":70,  "cyanosis_or_jvd_prob":0.50, "label":1},
     {"ecg_av_prob":0.95, "pulse_rate":32, "systolic_bp":75,  "cyanosis_or_jvd_prob":0.45, "label":1},
     {"ecg_av_prob":0.94, "pulse_rate":34, "systolic_bp":78,  "cyanosis_or_jvd_prob":0.40, "label":1},
     {"ecg_av_prob":0.92, "pulse_rate":36, "systolic_bp":82,  "cyanosis_or_jvd_prob":0.35, "label":1},
     {"ecg_av_prob":0.90, "pulse_rate":38, "systolic_bp":85,  "cyanosis_or_jvd_prob":0.30, "label":1},
     {"ecg_av_prob":0.88, "pulse_rate":40, "systolic_bp":88,  "cyanosis_or_jvd_prob":0.25, "label":1},
     {"ecg_av_prob":0.85, "pulse_rate":45, "systolic_bp":92,  "cyanosis_or_jvd_prob":0.20, "label":1},
     {"ecg_av_prob":0.80, "pulse_rate":50, "systolic_bp":98,  "cyanosis_or_jvd_prob":0.12, "label":1},
     {"ecg_av_prob":0.78, "pulse_rate":52, "systolic_bp":100, "cyanosis_or_jvd_prob":0.10, "label":1},
     {"ecg_av_prob":0.75, "pulse_rate":55, "systolic_bp":104, "cyanosis_or_jvd_prob":0.08, "label":0},
     {"ecg_av_prob":0.20, "pulse_rate":48, "systolic_bp":95,  "cyanosis_or_jvd_prob":0.05, "label":0},
     {"ecg_av_prob":0.30, "pulse_rate":42, "systolic_bp":90,  "cyanosis_or_jvd_prob":0.07, "label":0},
     {"ecg_av_prob":0.10, "pulse_rate":60, "systolic_bp":110, "cyanosis_or_jvd_prob":0.02, "label":0},
     {"ecg_av_prob":0.40, "pulse_rate":58, "systolic_bp":108, "cyanosis_or_jvd_prob":0.03, "label":0},
     {"ecg_av_prob":0.82, "pulse_rate":46, "systolic_bp":96,  "cyanosis_or_jvd_prob":0.18, "label":1},
     {"ecg_av_prob":0.86, "pulse_rate":44, "systolic_bp":94,  "cyanosis_or_jvd_prob":0.22, "label":1},
     {"ecg_av_prob":0.91, "pulse_rate":37, "systolic_bp":84,  "cyanosis_or_jvd_prob":0.28, "label":1},
     {"ecg_av_prob":0.97, "pulse_rate":30, "systolic_bp":72,  "cyanosis_or_jvd_prob":0.48, "label":1},
     {"ecg_av_prob":0.74, "pulse_rate":56, "systolic_bp":106, "cyanosis_or_jvd_prob":0.09, "label":0},
     {"ecg_av_prob":0.69, "pulse_rate":60, "systolic_bp":112, "cyanosis_or_jvd_prob":0.06, "label":0},
     {"ecg_av_prob":0.93, "pulse_rate":35, "systolic_bp":80,  "cyanosis_or_jvd_prob":0.33, "label":1},
     {"ecg_av_prob":0.87, "pulse_rate":41, "systolic_bp":89,  "cyanosis_or_jvd_prob":0.24, "label":1},
     {"ecg_av_prob":0.76, "pulse_rate":53, "systolic_bp":102, "cyanosis_or_jvd_prob":0.11, "label":0},
     {"ecg_av_prob":0.99, "pulse_rate":26, "systolic_bp":65,  "cyanosis_or_jvd_prob":0.55, "label":1},
     {"ecg_av_prob":0.84, "pulse_rate":47, "systolic_bp":93,  "cyanosis_or_jvd_prob":0.19, "label":1},
     {"ecg_av_prob":0.71, "pulse_rate":57, "systolic_bp":105, "cyanosis_or_jvd_prob":0.07, "label":0},
     {"ecg_av_prob":0.79, "pulse_rate":49, "systolic_bp":97,  "cyanosis_or_jvd_prob":0.13, "label":1},
     {"ecg_av_prob":0.88, "pulse_rate":39, "systolic_bp":86,  "cyanosis_or_jvd_prob":0.26, "label":1},
     {"ecg_av_prob":0.61, "pulse_rate":62, "systolic_bp":115, "cyanosis_or_jvd_prob":0.05, "label":0},
     {"ecg_av_prob":0.66, "pulse_rate":59, "systolic_bp":109, "cyanosis_or_jvd_prob":0.06, "label":0},
     {"ecg_av_prob":0.83, "pulse_rate":43, "systolic_bp":91,  "cyanosis_or_jvd_prob":0.21, "label":1},
    ]






    av_examples_data = []
    av_examples_label = []
    for x in av_examples:
        c = []
        for i in x:
            if i!="label":
                c.append(x[i])
        av_examples_data.append(c)
        av_examples_label.append([x["label"]])

    af_examples_data = []
    af_examples_label = []
    for x in af_examples:
        c = []
        for i in x:
            if i!="label":
                c.append(x[i])
        af_examples_data.append(c)
        af_examples_label.append([x["label"]])


    mi_examples_data = []
    mi_examples_label = []
    for x in mi_examples:
        c = []
        for i in x:
            if i!="label":
                c.append(x[i])
        mi_examples_data.append(c)
        mi_examples_label.append([x["label"]])





    for i, y in [[av_examples_data, av_examples_label], [af_examples_data, af_examples_label], [mi_examples_data, mi_examples_label]]:
        print(len(i))
        print(len(y))
        model = LogisticRegression()
        model.fit(i, y)
        print(model.coef_)


    datas = [[av_examples_data, av_examples_label], [af_examples_data, af_examples_label], [mi_examples_data, mi_examples_label]]

    model_av = LogisticRegression()
    model_av.fit(datas[0][0], datas[0][1])
    x = model_av.predict(lst3)

    model_mi = LogisticRegression()
    model_mi.fit(datas[2][0], datas[2][1])
    y = model_mi.predict(lst1)

    model_af = LogisticRegression()
    model_af.fit(datas[1][0], datas[1][1])
    z = model_af.predict(lst2)

    if max([x, y, z])==x:
        return "AV"
    elif max([x, y, z])==y:
        return "IM"
    elif max([x, y, z])==z:
        return 'FIB'



