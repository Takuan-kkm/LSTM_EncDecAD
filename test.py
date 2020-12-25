import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import os
import pandas as pd

subject = "T1_1109"
task = "4_1"

gt_path = os.environ["ONEDRIVE"] + "/研究/2020実験データ/ELAN/" + subject + "/task" + task + ".csv"
groundtruth = pd.read_csv(gt_path, header=None).to_numpy()[:, 2:]

print(groundtruth)
