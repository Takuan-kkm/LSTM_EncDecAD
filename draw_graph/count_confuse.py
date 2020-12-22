########################################
# とまどいの種類別の秒数・回数をユーザ毎に計算 #
########################################

import glob
import pandas as pd
import os

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

SUBJECTS = ["T1_1109", "S1_1112", "H1_1202", "E1_1203", "N1_1008", "A1_1217", "Y1_1217"]

for sub in SUBJECTS:
    result = {"turn_around": {"count": 0, "length": 0}, "experienced_error": {"count": 0, "length": 0},
              "inactivity": {"count": 0, "length": 0}, "blind_press": {"count": 0, "length": 0},
              "question": {"count": 0, "length": 0}, "wandering_hands": {"count": 0, "length": 0},
              "other": {"count": 0, "length": 0}}

    csv = os.environ["ONEDRIVE"] + "/研究/2020実験データ/ELAN/" + sub + "/*.csv"
    csv = glob.glob(csv)

    for c in csv:
        df = pd.read_csv(c, header=None, index_col=None)
        for confuse in df.itertuples():
            if confuse[5] == "turn around":
                result["turn_around"]["count"] += 1
                result["turn_around"]["length"] += confuse[4]
            elif confuse[5] == "経験したことのある誤った操作":
                result["experienced_error"]["count"] += 1
                result["experienced_error"]["length"] += confuse[4]
            elif confuse[5] == "動作の停止":
                result["inactivity"]["count"] += 1
                result["inactivity"]["length"] += confuse[4]
            elif confuse[5] == "闇雲なボタン押下":
                result["blind_press"]["count"] += 1
                result["blind_press"]["length"] += confuse[4]
            elif confuse[5] == "question":
                result["question"]["count"] += 1
                result["question"]["length"] += confuse[4]
            elif confuse[5] == "手のさまよい":
                result["wandering_hands"]["count"] += 1
                result["wandering_hands"]["length"] += confuse[4]
            elif confuse[5] == "other":
                result["other"]["count"] += 1
                result["other"]["length"] += confuse[4]
            else:
                print("Invalid confusion:", c, confuse)

    total_l = 0
    total_c = 0
    for key in result:
        total_c += result[key]["count"]
        total_l += result[key]["length"]
    result["total"] = {"count": total_c, "length": total_l}

    result = pd.DataFrame(result)
    print("#################", sub, "#####################")
    print(result, "\n")
