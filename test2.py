import pickle

import matplotlib.pyplot as plt
from adfunc import AnomaryDetector
import numpy as np
import cupy as cp

from sklearn.preprocessing import StandardScaler

import os

SUBJECT_ID = "TEST_NOAKI_1008"
ptask1_1 = os.environ["ONEDRIVE"] + "/研究/2020実験データ/BIN/" + SUBJECT_ID + "/task1_1.pkl"
ptask1_2 = os.environ["ONEDRIVE"] + "/研究/2020実験データ/BIN/" + SUBJECT_ID + "/task1_2.pkl"
ptask2_1 = os.environ["ONEDRIVE"] + "/研究/2020実験データ/BIN/" + SUBJECT_ID + "/task2_1.pkl"
ptask2_2 = os.environ["ONEDRIVE"] + "/研究/2020実験データ/BIN/" + SUBJECT_ID + "/task2_2.pkl"
ptask3_1 = os.environ["ONEDRIVE"] + "/研究/2020実験データ/BIN/" + SUBJECT_ID + "/task3_1.pkl"
ptask3_2 = os.environ["ONEDRIVE"] + "/研究/2020実験データ/BIN/" + SUBJECT_ID + "/task3_2.pkl"
normal_path = os.environ["ONEDRIVE"] + "/研究/2020実験データ/BIN/" + "TEST_NOAKI_1008_TRAIN.pkl"


def score_plot(score, label="score"):
    score = np.array(score)

    window = 1
    samplerate = 125
    skiprate = 10

    w = np.ones(window) / window
    score = np.convolve(score, w, mode="same")
    xl = [2 * i * skiprate / samplerate for i in range(len(score))]

    fig = plt.figure(figsize=[15, 5])
    ax = fig.add_subplot(111)
    ax.plot(xl, score, label=label)
    ax.set_ylim([0, 1600])
    ax.set_xlim([0, 2 * len(score) * skiprate / samplerate])
    ax.set_xlabel("time[sec]")
    ax.set_ylabel("anomary score")
    ax.legend()
    plt.show()


def main():
    # Load dataset
    with open(normal_path, "rb") as f:
        test = pickle.load(f)

    # with open(ptask1_2, "rb") as f:
    #     task1_2 = pickle.load(f)

    with open(ptask2_1, "rb") as f:
        task2_1 = pickle.load(f)

    # with open(ptask2_2, "rb") as f:
    #     task2_2 = pickle.load(f)
    #
    # with open(ptask3_1, "rb") as f:
    #     task3_1 = pickle.load(f)
    #
    # with open(ptask3_2, "rb") as f:
    #     task3_2 = pickle.load(f)

    with open(ptask1_1, "rb") as f:
        task1_1 = pickle.load(f)

    # Load network
    with open("result/model.pkl", "rb") as f:
        net = pickle.load(f)

    # net.train = True
    detector = AnomaryDetector(net, seq_length=50, dim=78, calc_length=25)
    scaler = StandardScaler()

    data = task1_1[10]
    # scaler.fit(data)
    rc_data = detector.reconstruct(data)

    data = cp.asnumpy(data.T[80])
    rc_data = [i[0][80].data for i in rc_data]

    fig = plt.figure(figsize=[15, 10])
    ax = fig.add_subplot(211)
    # ax.set_ylim([0.9475, 0.9525])
    ax.set_ylim([-1.8, 2.8])
    ax.plot(data, label="data")

    ax_rc = fig.add_subplot(212)
    ax_rc.set_ylim([-1.8, 2.8])
    ax_rc.plot(rc_data, label="rc_data")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
