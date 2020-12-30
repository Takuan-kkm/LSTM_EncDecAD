#########################
# データの量をユーザ毎に計算 #
#########################

import glob
import os
import pandas as pd

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

SAMPLE_RATE = 125
SUBJECTS = ["T1_1109", "S1_1112", "H1_1202", "E1_1203", "N1_1008", "A1_1217", "Y1_1217"]

result = {}
for sub in SUBJECTS:
    csv = os.environ["ONEDRIVE"] + "/研究/2020実験データ/CSV_BIN/" + sub + "/TRAIN/*.csv"
    csv_train = glob.glob(csv)
    csv = os.environ["ONEDRIVE"] + "/研究/2020実験データ/CSV_BIN/" + sub + "/TEST/*.csv"
    csv_test = glob.glob(csv)

    train = 0
    test = 0
    total = 0
    no_train = 0
    no_test = 0
    for c in csv_train:
        no_train += 1
        df = pd.read_csv(c, header=0, skiprows=7)
        train += df.shape[0] / SAMPLE_RATE

    for c in csv_test:
        no_test += 1
        df = pd.read_csv(c, header=0, skiprows=7)
        test += df.shape[0] / SAMPLE_RATE

    result[sub] = {"train(sec)": train, "test(sec)": test, "No. of TrainTasks": no_train, "No. of TestTasks": no_test,
                   "total(sec)": train + test}

total_train = sum([result[r]["train(sec)"] for r in result])
total_test = sum([result[r]["test(sec)"] for r in result])
total_notrain = sum([result[r]["No. of TrainTasks"] for r in result])
total_notest = sum([result[r]["No. of TestTasks"] for r in result])
result["all"] = {"train(sec)": total_train, "test(sec)": total_test, "No. of TrainTasks": total_notrain,
                 "No. of TestTasks": total_notest, "total(sec)": total_test + total_train}

result = pd.DataFrame(result)
print(result)
