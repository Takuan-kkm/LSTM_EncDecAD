import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import os


class plot_TL():
    window = 3
    _samplerate = 125
    _skiprate = 5
    rate = _skiprate / _samplerate * 3

    def __init__(self, groundtruth, score):
        self.GT = groundtruth

        w = np.ones(self.window) / self.window
        self.score = np.convolve(score, w, mode="same")[::3]
        xl = [i * self.rate for i in range(self.score.shape[0])]
        self.fig = plt.figure(figsize=(15, 8))
        self.ax = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)

        self.init_fig()

        self.ax.plot(xl, self.score)
        self.ax2.broken_barh(self.GT, (4, 2), facecolors='tab:red')

        self.ln_threshold = self.ax.axhline(400, color='red')
        self.conf_barh = self.ax2.broken_barh([], (9, 2), facecolors='tab:red')
        self.hits_barh = self.ax2.broken_barh([], (14, 2), facecolors='tab:red')
        self.fa_barh = self.ax2.broken_barh([], (19, 2), facecolors='tab:red')

        self.ls_confusion = []
        self.ls_fa = []
        self.ls_hits = []

        plt.connect('motion_notify_event', self.motion)

    def init_fig(self):
        self.ax.set_ylim([0, 4000])
        self.ax.set_xlim([0, len(self.score) * self.rate])
        self.ax.set_xlabel("time[sec]")
        self.ax.set_ylabel("anomary score")

        self.ax2.set_ylim(2, 22)
        self.ax2.set_xlim([0, len(self.score) * self.rate])
        self.ax2.set_xlabel('time(sec)', fontdict={"size": 12})
        self.ax2.set_yticks([5, 10, 15, 20])
        self.ax2.set_yticklabels(['Ground truth', 'Confusions', 'Hits', 'False alarms'])
        self.ax2.grid(True)

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
        self.conf_barh.remove()
        self.conf_barh = self.ax2.broken_barh(self.ls_confusion, (9, 2), facecolors='tab:red')

    def motion(self, event):
        if event.button == 1:
            y = event.ydata
            self.ln_threshold.set_ydata(y)
            self.confusions(y)
            self.hits()
            self.false_alarm()
            print(self.calc_leeliu_metric(), "\n")

        plt.draw()

    def hits(self):
        self.ls_hits = []
        if len(self.ls_confusion) == 0:
            return None

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

        self.hits_barh.remove()
        self.hits_barh = self.ax2.broken_barh(self.ls_hits, (14, 2), facecolors='blue')

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

        self.fa_barh.remove()
        self.fa_barh = self.ax2.broken_barh(self.ls_fa, (19, 2), facecolors='red')

    def calc_leeliu_metric(self):
        TP, TN, FP, FN = self.confusion_matrix()
        # recall = TP/(TP+FN)
        # prfx_1 = (TP+FP)/(TP+FP+FN+TN)
        # lm = (recall**2)/prfx_1
        lm = (TP * (TP + TN + FP + FN)) / ((TP + FP) * (TP + FN) ** 2)
        return lm

    def confusion_matrix(self):
        # TN = sum([i[1] for i in self.ls_hits])
        # FN = sum([i[1] for i in self.ls_fa])
        # FP = sum(self.GT[:, 1]) - TN
        # TP = self.score.shape[0] * self.rate - TN - FN - FP
        TP = sum([i[1] for i in self.ls_hits])
        FP = sum([i[1] for i in self.ls_fa])
        FN = sum(self.GT[:, 1]) - TP
        TN = self.score.shape[0] * self.rate - TP - FN - FP
        print(TP, TN, FP, FN)

        return TP, TN, FP, FN

    def show(self):
        plt.show()

    def get_confusion_matrix(self, threshold):
        self.confusions(threshold)
        self.hits()
        self.false_alarm()

        return self.confusion_matrix()


def sss(subject_id, task, coordinate):
    with open("resource/" + coordinate + "/ascore_task" + task + ".pkl", "rb") as f:
        score = pickle.load(f)

    gt_path = os.environ["ONEDRIVE"] + "/研究/2020実験データ/ELAN/" + subject_id + "/task" + task + ".csv"
    try:
        groundtruth = pd.read_csv(gt_path, header=None).to_numpy()[:, 2:4]
    except Exception as e:
        groundtruth = None

    ptl = plot_TL(groundtruth=groundtruth, score=score)
    return ptl.get_confusion_matrix(2300)


def main():
    coordinate = "POLAR"
    subject = "E1_1203"

    with open("../ascore_task4_1.pkl", "rb") as f:
        score = pickle.load(f)

    gt_path = os.environ["ONEDRIVE"] + "/研究/2020実験データ/ELAN/" + "S1_1112" + "/task3_1.csv"
    groundtruth = pd.read_csv(gt_path, header=None).to_numpy()[:, 2:4]

    ptl = plot_TL(groundtruth=groundtruth, score=score)
    ptl.show()

    # result = []
    # result.append(sss(subject, "1_1", coordinate))
    # result.append(sss(subject, "2_1", coordinate))
    # result.append(sss(subject, "2_2", coordinate))
    # result.append(sss(subject, "3_1", coordinate))
    # result.append(sss(subject, "3_2", coordinate))
    # result.append(sss(subject, "4_1", coordinate))
    # result.append(sss(subject, "4_2, coordinate))
    #
    # TP = sum([i[0] for i in result])
    # TN = sum([i[1] for i in result])
    # FP = sum([i[2] for i in result])
    # FN = sum([i[3] for i in result])
    #
    # lm = (TP * (TP + TN + FP + FN)) / ((TP + FP) * (TP + FN) ** 2)
    # print(TP,TN,FP,FN)
    # print(lm)


if __name__ == "__main__":
    main()
