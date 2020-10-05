import os
import glob
SUBJECT_ID = "00"
DATA_DIR = os.environ["ONEDRIVE"] + "/研究/2020実験データ/CSV/" + SUBJECT_ID + "/"

path = glob.glob(DATA_DIR+"*.csv")
print(DATA_DIR+"*.csv")
print(path)
