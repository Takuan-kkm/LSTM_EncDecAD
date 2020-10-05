import pickle
import os
import matplotlib.pyplot as plt
from adfunc import AnomaryDetector

STEP_SIZE = 5
SEQ_LEN = 50


def main():
    OUT_PATH = os.environ["ONEDRIVE"] + "/研究/2020実験データ/Take 2020-09-29 07.18.47 PM.pkl"
    # Load dataset
    with open(OUT_PATH, "rb") as f:
        test = pickle.load(f)

    print(test.shape)
    print(test[0])


if __name__ == "__main__":
    main()
