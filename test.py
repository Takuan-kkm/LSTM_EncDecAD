import pickle

import matplotlib.pyplot as plt
from adfunc import AnomaryDetector

STEP_SIZE = 5
SEQ_LEN = 50

def main():
    # Load dataset
    with open("test.sav", "rb") as f:
        test = pickle.load(f)

    with open("confuse.sav", "rb") as f:
        confuse = pickle.load(f)

    # Load detector
    with open("detector.pkl", "rb") as f:
        detector = pickle.load(f)

    score_normal = []
    for i in range(0, len(test), int(SEQ_LEN/STEP_SIZE)):
        score_normal = score_normal + detector.calc_anomary_score(test[i])

    score_confuse = []
    for i in range(0, len(confuse), int(SEQ_LEN/STEP_SIZE)):
        score_confuse = score_confuse + detector.calc_anomary_score(confuse[i])

    print("normal:",len(score_normal))
    print("confuse:",len(score_confuse))

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
    ax.set_xlim([1250, 1300])
    ax.set_xlabel("time[*0.1sec]")
    ax.set_ylabel("anomary score")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
