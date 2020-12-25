import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

_samplerate = 125
_skiprate = 5
rate = _samplerate / _skiprate  # 25
SUBJECTS = ["T1_1109", "S1_1112", "H1_1202", "E1_1203", "N1_1008", "A1_1217", "Y1_1217"]
TASKS = ["1_1", "1_2", "2_1", "2_2", "3_1", "3_2", "4_1", "4_2"]
result = {}

for sub in SUBJECTS:
    temp = {}
    for task in TASKS:
        try:
            with open("resource/" + sub + "/ascore_task" + task + ".pkl", "rb") as f:  # 異常スコア読み込み
                score = pickle.load(f)
                score = [cp.asnumpy(s) for s in score]
        except Exception as e:
            continue

        gt_path = os.environ["ONEDRIVE"] + "/研究/2020実験データ/ELAN/" + sub + "/task" + task + ".csv"  # ビデオ分析データ読み込み
        groundtruth = pd.read_csv(gt_path, header=None).to_numpy()[:, 2:4]

        gt = np.zeros_like(score)
        for g in groundtruth:
            gt[int(g[0] * rate):int(g[0] * rate + g[1] * rate)] = 1  # ビデオ分析データ変形

        auc = roc_auc_score(gt, score)
        temp[task] = auc

    result[sub] = temp

result = pd.DataFrame(result)
print(result)
# subject = "T1_1109"
# task = "3_2"
#
# with open("resource/" + subject + "/ascore_task" + task + ".pkl", "rb") as f:
#     score = pickle.load(f)
#
# groundtruth = pd.read_csv(gt_path, header=None).to_numpy()[:, 2:4]
#
# gt = np.zeros_like(score)
# for g in groundtruth:
#     gt[int(g[0] * rate):int(g[0] * rate + g[1] * rate)] = 1
#
# fpr, tpr, thresholds = roc_curve(gt, score)
# print(roc_auc_score(gt, score))
#
# plt.plot(fpr, tpr, marker='o')
# plt.xlabel('FPR: False positive rate')
# plt.ylabel('TPR: True positive rate')
# plt.grid()
# plt.show()
