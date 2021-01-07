import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import os
import pandas as pd
import pickle
import cupy as cp

subject = "A1_1217"
task = "4_1"

gt_path = os.environ["ONEDRIVE"] + "/研究/2020実験データ/ELAN_forFalseAlarmTEST/" + subject + "/task" + task + ".csv"
groundtruth = pd.read_csv(gt_path, header=None).to_numpy()[:, 2:]

with open("draw_graph/resource/" + subject + "/ascore_task" + task + ".pkl", "rb") as f:  # 異常スコア読み込み
    score = pickle.load(f)
    score = [cp.asnumpy(s) for s in score]

print(len(score) / 25)
print(groundtruth)
