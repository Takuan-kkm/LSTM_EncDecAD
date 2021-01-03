from draw_graph.calc_hitrate import calc_hitrate
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
_samplerate = 125
_skiprate = 5
rate = _samplerate / _skiprate  # 25
SUBJECTS = ["T1_1109", "S1_1112", "H1_1202", "E1_1203", "N1_1008", "A1_1217", "Y1_1217"]
TASKS = ["1_1", "1_2", "2_1", "2_2", "3_1", "3_2", "4_1", "4_2"]


def main():
    thresholds = list(range(400, 3200, 200))
    result = {}

    for th in thresholds:
        confusion_matrix = []
        print("threshold:", th)
        for sub in SUBJECTS:
            for task in TASKS:
                try:
                    with open("resource/" + sub + "/ascore_task" + task + ".pkl", "rb") as f:  # 異常スコア読み込み
                        score = pickle.load(f)
                        score = [cp.asnumpy(s) for s in score]
                except Exception as e:
                    continue

                gt_path = os.environ["ONEDRIVE"] + "/研究/2020実験データ/ELAN/" + sub + "/task" + task + ".csv"  # ビデオ分析データ読み込み
                # print(gt_path)
                groundtruth = pd.read_csv(gt_path, header=None).to_numpy()[:, 2:]
                cal = calc_hitrate(groundtruth, score)
                confusion_matrix.append(cal.get_confusion_matrix(th))

        TP = sum([i[0] for i in confusion_matrix])
        TN = sum([i[1] for i in confusion_matrix])
        FP = sum([i[2] for i in confusion_matrix])
        FN = sum([i[3] for i in confusion_matrix])
        recall = TP / (TP + FN)
        presicion = TP / (TP + FP)
        SavedTimeRate = (FN + TN) / (TP + FP + TN + FN)
        # print(TP, FP, TN, FN, TP + FP + TN + FN, SavedTimeRate)
        f_score = 2 * recall * presicion / (recall + presicion)

        result["threshold:" + str(th)] = {"f score": f_score, "HR": TP / (FN + TP), "FAR": FP / (FP + TN),
                                          "STR": SavedTimeRate}
    result = pd.DataFrame(result)
    print(result.T)


if __name__ == "__main__":
    main()
