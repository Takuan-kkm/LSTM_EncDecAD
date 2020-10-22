import pickle

import matplotlib.pyplot as plt
from adfunc import AnomaryDetector
import numpy as np

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

    window = 10
    samplerate = 250
    skiprate = 15

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

    with open(ptask1_2, "rb") as f:
        task1_2 = pickle.load(f)

    with open(ptask2_1, "rb") as f:
        task2_1 = pickle.load(f)

    with open(ptask2_2, "rb") as f:
        task2_2 = pickle.load(f)

    with open(ptask3_1, "rb") as f:
        task3_1 = pickle.load(f)

    with open(ptask3_2, "rb") as f:
        task3_2 = pickle.load(f)

    with open(ptask1_1, "rb") as f:
        task1_1 = pickle.load(f)

    # Load network
    with open("result/model_epoch25.pkl", "rb") as f:
        net = pickle.load(f)

    detector = AnomaryDetector(net, seq_length=50, dim=78, calc_length=25)
    print("Fitting...")
    index = 0
    for seq in test:
        print(index, " ", end="")
        index += 1
        detector.fit(seq)

    # save the model
    with open("detector.pkl", "wb") as f:
        pickle.dump(detector, f)

    score_normal = []
    for i in range(0, len(test), 19):
        score_normal = score_normal + detector.calc_anomary_score(test[i])

    score_task1_1 = []
    for i in task1_1:
        score_task1_1 = score_task1_1 + detector.calc_anomary_score(i)

    score_task1_2 = []
    for i in task1_2:
        score_task1_2 = score_task1_2 + detector.calc_anomary_score(i)

    score_task2_1 = []
    for i in task2_1:
        score_task2_1 = score_task2_1 + detector.calc_anomary_score(i)

    score_task2_2 = []
    for i in task2_2:
        score_task2_2 = score_task2_2 + detector.calc_anomary_score(i)

    score_task3_1 = []
    for i in task3_1:
        score_task3_1 = score_task3_1 + detector.calc_anomary_score(i)

    score_task3_2 = []
    for i in task3_2:
        score_task3_2 = score_task3_2 + detector.calc_anomary_score(i)

    print("normal:", len(score_normal))
    # print("confuse:", len(score_confuse))

    score_plot(score_normal, label="normal")
    score_plot(score_task1_1, label="task1_1")
    score_plot(score_task1_2, label="task1_2")
    score_plot(score_task2_1, label="task2_1")
    score_plot(score_task2_2, label="task2_2")
    score_plot(score_task3_1, label="task3_1")
    score_plot(score_task3_2, label="task3_2")

    # save result
    # with open("ascore_normal", "wb") as f:
    #     pickle.dump(score_normal, f)
    # with open("ascore_confuse.pkl", "wb") as f:
    #     pickle.dump(score_confuse, f)


if __name__ == "__main__":
    main()
