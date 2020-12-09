import matplotlib.pyplot as plt
import numpy as np
import pickle


class plot_TL():
    window = 10
    _samplerate = 125
    _skiprate = 5
    rate = _skiprate / _samplerate

    TP = 0
    TN = 0
    FN = 0
    FP = 0

    def __init__(self, groundtruth, score):
        self.GT = groundtruth

        w = np.ones(self.window) / self.window
        self.score = np.convolve(score, w, mode="same")
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
        self.ax.set_ylim([0, 1000])
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

        plt.draw()

    def hits(self):
        self.ls_hits = []

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
        it = iter(self.GT)
        gt = it.__next__()


        for c in self.ls_confusion:
            while True:
                # print("\nloop:", loop)
                # print("GT:", gt)
                # print("C:", c)
                # loop += 1

                if c[0] >= gt[0] + gt[1]:
                    try:
                        gt = it.__next__()
                    except StopIteration:
                        break
                    continue

                if c[0] + c[1] <= gt[0]:
                    self.ls_fa.append(c)
                    break

                if c[0] < gt[0]:
                    self.ls_fa.append((c[0], gt[0] - c[0]))

                if c[0] + c[1] > gt[0] + gt[1]:
                    self.ls_fa.append((gt[0] + gt[1], c[0] + c[1] - gt[0] - gt[1]))

                break

        self.fa_barh.remove()
        self.fa_barh = self.ax2.broken_barh(self.ls_fa, (19, 2), facecolors='red')

    def calc_leeliu_metric(self):
        return 0

    def show(self):
        plt.show()


def main():
    with open("resource/ascore_normal.pkl", "rb") as f:
        score = pickle.load(f)

    groundtruth = np.array([(10, 4), (17, 2), (20, 3), (31, 9), (50, 2)])
    ptl = plot_TL(groundtruth=groundtruth, score=score)
    ptl.show()


if __name__ == "__main__":
    main()
