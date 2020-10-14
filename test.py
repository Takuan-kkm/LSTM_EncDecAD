import pickle
import os

SUBJECT_ID = "TEST_NOAKI_1008"
path = os.environ["ONEDRIVE"] + "/研究/2020実験データ/BIN/" + SUBJECT_ID + "_TRAIN.pkl"

with open(path, "rb") as f:
    arr = pickle.load(f)

print(arr.shape)
print(arr[0])
print(arr[1])
