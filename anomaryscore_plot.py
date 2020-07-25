import pickle

import matplotlib.pyplot as plt
from adfunc import AnomaryDetector
import numpy as np


def main():
    # Load dataset
    with open("test.sav", "rb") as f:
        test = pickle.load(f)

    with open("confuse.sav", "rb") as f:
        confuse = pickle.load(f)

    # Load network
    with open("result/model.pkl", "rb") as f:
        net = pickle.load(f)

    detector = AnomaryDetector(net, seq_length=50, dim=50, calc_length=25)
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
    for i in range(0, len(test), 5):
        score_normal = score_normal + detector.calc_anomary_score(test[i])

    score_confuse = []
    for i in range(0, len(confuse), 5):
        score_confuse = score_confuse + detector.calc_anomary_score(confuse[i])

    print("normal:", len(score_normal))
    print("confuse:", len(score_confuse))

    score_normal = np.array(score_normal)
    score_confuse = np.array(score_confuse)
    score_normal = (score_normal[:-2]+score_normal[1:-1]+score_normal[2:])/3
    score_confuse = (score_confuse[:-2] + score_confuse[1:-1] + score_confuse[2:])/3

    # save result
    with open("ascore_normal", "wb") as f:
        pickle.dump(score_normal, f)
    with open("ascore_confuse.pkl", "wb") as f:
        pickle.dump(score_confuse, f)

    fig = plt.figure(figsize=[15, 5])
    ax = fig.add_subplot(111)
    ax.plot(score_normal, label="normal behavior/anomary score")
    ax.set_ylim([0, 400])
    ax.set_xlim([0, len(score_normal)])
    ax.set_xlabel("time[*0.1sec]")
    ax.set_ylabel("anomary score")
    ax.legend()
    plt.show()

    fig = plt.figure(figsize=[15, 5])
    ax = fig.add_subplot(111)
    ax.plot(score_confuse, label="confusing behavior/anomary score")
    ax.set_ylim([0, 400])
    ax.set_xlim([0, len(score_confuse)])
    ax.set_xlabel("time[*0.1sec]")
    ax.set_ylabel("anomary score")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
