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


class calc_hitrate():
    window = 3
    _samplerate = 125
    _skiprate = 5
    rate = _skiprate / _samplerate

    def __init__(self, groundtruth, score):
        self.GT = groundtruth

        w = np.ones(self.window) / self.window
        self.score = np.convolve(score, w, mode="same")[::]
        self.ls_confusion = []
        self.ls_fa = []
        self.ls_hits = []

    def confusions(self, threshold):
        ls = []
        flag = False
        st = 0

        if np.min(self.score) >= threshold:
            ls.append((st, self.score.shape[0]))
        elif np.max(self.score) > threshold:
            for index, s in enumerate(self.score >= threshold):
                if not s:
                    if flag is True:
                        ls.append((st, index - st))
                        flag = False
                else:
                    if flag is False:
                        st = index
                        flag = True

        self.ls_confusion = np.array(ls) * self.rate

    def hits(self, result=None):
        self.ls_hits = []
        if len(self.ls_confusion) == 0:
            return result

        it = iter(self.ls_confusion)
        c = it.__next__()

        for gt in self.GT:
            while True:
                if c[0] >= gt[0] + gt[1]:  # とまどいの開始時刻が、現在参照しているground truthの終了時刻より遅ければbreak
                    break

                if c[0] + c[1] <= gt[0]:  # とまどいの終了時刻が、現在参照しているground truthの開始時刻より遅ければcontinue
                    try:
                        c = it.__next__()
                    except StopIteration:
                        break
                    continue

                self.ls_hits.append(gt)
                break

        if result is not None:
            for h in self.ls_hits:
                if h[2] == "turn around":
                    result["turn_around"]["count"] += 1
                    result["turn_around"]["length"] += h[1]
                elif h[2] == "経験したことのある誤った操作":
                    result["experienced_error"]["count"] += 1
                    result["experienced_error"]["length"] += h[1]
                elif h[2] == "動作の停止":
                    result["inactivity"]["count"] += 1
                    result["inactivity"]["length"] += h[1]
                elif h[2] == "闇雲なボタン押下":
                    result["blind_press"]["count"] += 1
                    result["blind_press"]["length"] += h[1]
                elif h[2] == "question":
                    result["question"]["count"] += 1
                    result["question"]["length"] += h[1]
                elif h[2] == "手のさまよい":
                    result["wandering_hands"]["count"] += 1
                    result["wandering_hands"]["length"] += h[1]
                elif h[2] == "other":
                    result["other"]["count"] += 1
                    result["other"]["length"] += h[1]

            return result

    def hits_true(self):  # セコくないほう
        self.ls_hits = []
        if len(self.ls_confusion) == 0:
            return None

        for c in self.ls_confusion:
            for gt in self.GT:
                if c is None:
                    break
                if c[0] >= gt[0] + gt[1]:
                    continue
                if c[0] + c[1] <= gt[0]:
                    continue

                if c[0] >= gt[0]:
                    if c[0] + c[1] <= gt[0] + gt[1]:
                        self.ls_hits.append(c)
                    else:
                        self.ls_hits.append([c[0], gt[0] + gt[1] - c[0]])
                else:
                    if c[0] + c[1] <= gt[0] + gt[1]:
                        self.ls_hits.append([gt[0], c[0] + c[1] - gt[0]])
                    else:
                        self.ls_hits.append(gt)

    def false_alarm(self):
        self.ls_fa = []

        for c in self.ls_confusion:
            for gt in self.GT:
                if c is None:
                    break
                if c[0] >= gt[0] + gt[1]:
                    continue
                if c[0] + c[1] <= gt[0]:
                    continue

                if c[0] >= gt[0]:
                    if c[0] + c[1] <= gt[0] + gt[1]:
                        c = None
                    else:
                        c = [gt[0] + gt[1], c[0] + c[1] - gt[0] - gt[1]]
                else:
                    if c[0] + c[1] <= gt[0] + gt[1]:
                        c = [c[0], gt[0] - c[0]]
                    else:
                        self.ls_fa.append([c[0], gt[0] - c[0]])
                        c = [gt[0] + gt[1], c[0] + c[1] - gt[0] - gt[1]]

            if c is not None:
                self.ls_fa.append(c)

    def confusion_matrix(self):
        # 異常:Positive
        TP = sum([i[1] for i in self.ls_hits])
        FP = sum([i[1] for i in self.ls_fa])
        FN = sum(self.GT[:, 1]) - TP
        TN = self.score.shape[0] * self.rate - TP - FN - FP
        # print(TP, TN, FP, FN)

        return TP, TN, FP, FN

    def get_confusion_matrix(self, threshold):
        self.confusions(threshold)
        self.hits()
        self.false_alarm()

        return self.confusion_matrix()


rslt = {"turn_around": {"count": 0, "length": 0}, "experienced_error": {"count": 0, "length": 0},
          "inactivity": {"count": 0, "length": 0}, "blind_press": {"count": 0, "length": 0},
          "question": {"count": 0, "length": 0}, "wandering_hands": {"count": 0, "length": 0},
          "other": {"count": 0, "length": 0}}


def main():
    global rslt
    confusion_matrix = []
    TH = 1600

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
            cal.confusions(TH)
            cal.false_alarm()
            rslt = cal.hits(rslt)

            confusion_matrix.append(cal.confusion_matrix())

    TP = sum([i[0] for i in confusion_matrix])
    TN = sum([i[1] for i in confusion_matrix])
    FP = sum([i[2] for i in confusion_matrix])
    FN = sum([i[3] for i in confusion_matrix])
    print("FAR:", FP / (FP + TN))
    rslt = pd.DataFrame(rslt)
    print(rslt)


if __name__ == "__main__":
    main()
