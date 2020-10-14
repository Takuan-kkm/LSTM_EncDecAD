import pandas as pd
import cupy as cp
import pickle
import math
import os
import glob

pd.set_option('display.max_columns', 150)
#######################################################################################
# OptiTrackからの出力CSVを読み込み、EncDecADに入力できる形(Cupy Seqences)に変換するスクリプト   #
#######################################################################################

TRAIN = True
DATA_PATH = os.environ["ONEDRIVE"] + "/研究/2020実験データ/Take 2020-09-29 07.18.47 PM.csv"  # Temporary

SUBJECT_ID = "TEST_NOAKI_1008"
SKELETON_NAME = "Skeleton 002:"

if TRAIN is True:
    DATA_DIR = os.environ["ONEDRIVE"] + "/研究/2020実験データ/CSV/" + SUBJECT_ID + "/TRAIN/"
    OUT_PATH = os.environ["ONEDRIVE"] + "/研究/2020実験データ/BIN/" + SUBJECT_ID + "_TRAIN.pkl"
else:
    DATA_DIR = os.environ["ONEDRIVE"] + "/研究/2020実験データ/CSV/" + SUBJECT_ID + "/TEST/"
    OUT_PATH = os.environ["ONEDRIVE"] + "/研究/2020実験データ/BIN/" + SUBJECT_ID + "_TEST.pkl"

SEQ_LEN = 50  # 時系列データの長さ
STEP_SIZE = 20  # 1時系列データ間の開始フレームの差
SKIP = 15  # 飛ばすフレーム数 ex)OptiTrackの記録レートが250fps,SKIP=10 → 25fpsにダウンサンプリングしてデータ化される

Markers_to_use = ["Hip", "WaistLFront", "WaistRFront", "WaistLBack", "WaistRBack", "Chest",
                  "BackTop", "BackLeft", "BackRight", "HeadTop", "HeadFront", "HeadSide",
                  "LShoulderBack", "LShoulderTop", "LElbowOut", "LUArmHigh", "LHandOut", "LWristOut", "LWristIn",
                  "RShoulderBack", "RShoulderTop", "RElbowOut", "RUArmHigh", "RHandOut", "RWristOut", "RWristIn"]
Markers_to_drop = ["Hip", "Ab", "Chest", "Neck", "Head", "LShoulder", "LUArm", "LFArm", "LHand", "RShoulder", "RUArm",
                   "RFArm", "RHand", "LThigh", "LShin", "LFoot", "RThigh", "RShin", "RFoot", "LToe", "RToe"]
Markers_to_use = [SKELETON_NAME + name for name in Markers_to_use]
Markers_to_drop = [SKELETON_NAME + name for name in Markers_to_drop]


def df_to_cp(df):
    # 使わない列は削除
    df = df.drop(("Name", "Unnamed: 1_level_1", "Time (Seconds)"), axis=1)
    for m in Markers_to_drop:
        df = df.drop((m, "Rotation"), axis=1)

    # ndarrayに変換
    ndarr = df.to_numpy()
    if SKIP == 0:
        ndarr = [ndarr[i:i + SEQ_LEN] for i in range(0, ndarr.shape[0], STEP_SIZE)]
        out = cp.array(ndarr[1:-math.floor(SEQ_LEN / STEP_SIZE)], dtype="float32")
    else:
        ndarr = [ndarr[i:i + SEQ_LEN * SKIP:SKIP] for i in range(0, ndarr.shape[0] - SKIP * (SEQ_LEN-1), STEP_SIZE)]
        out = cp.array(ndarr, dtype="float32")

    return out


def main():
    # CSVファイル読み込み
    if DATA_DIR == "":
        df = pd.read_csv(DATA_PATH, skiprows=3, header=[0, 2, 3], index_col=0)
        out = df_to_cp(df)
    else:
        path = glob.glob(DATA_DIR + "*.csv")
        print(path)

        for index, csv in enumerate(path):
            print(csv)
            df = pd.read_csv(csv, skiprows=3, header=[0, 2, 3], index_col=0)
            if index == 0:
                out = df_to_cp(df)
            else:
                out = cp.concatenate([out, df_to_cp(df)])

    pickle.dump(out, open(OUT_PATH, "wb"))


if __name__ == "__main__":
    main()
