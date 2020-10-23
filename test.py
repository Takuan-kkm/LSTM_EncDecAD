import pandas as pd
import numpy as cp
import pickle
import math
import os
import glob
import matplotlib.pyplot as plt
from scipy import signal

pd.set_option('display.max_columns', 150)
#######################################################################################
# OptiTrackからの出力CSVを読み込み、EncDecADに入力できる形(Cupy Seqences)に変換するスクリプト   #
#######################################################################################

TRAIN = False  # TRAIN == Falseの場合、csvファイル毎にpklファイルを作る。Trueのときは、ディレクトリ内のcsvをまとめてpkl化する。
DATA_PATH = os.environ["ONEDRIVE"] + "/研究/2020実験データ/Take 2020-09-29 07.18.47 PM.csv"  # Temporary

SUBJECT_ID = "TEST_NOAKI_1008"
SKELETON_NAME = "Skeleton 002:"

SEQ_LEN = 50  # 時系列データの長さ
STEP_SIZE = 20  # 1時系列データ間の開始フレームの差
SKIP = 0  # 飛ばすフレーム数 ex)OptiTrackの記録レートが250fps,SKIP=10 → 25fpsにダウンサンプリングしてデータ化される

if TRAIN is True:
    DATA_DIR = os.environ["ONEDRIVE"] + "/研究/2020実験データ/CSV/" + SUBJECT_ID + "/TRAIN/"
    OUT_PATH = os.environ["ONEDRIVE"] + "/研究/2020実験データ/BIN/" + SUBJECT_ID + "_TRAIN.pkl"
else:
    DATA_DIR = os.environ["ONEDRIVE"] + "/研究/2020実験データ/CSV/" + SUBJECT_ID + "/TEST/"
    OUT_PATH = os.environ["ONEDRIVE"] + "/研究/2020実験データ/BIN/" + SUBJECT_ID + "_TEST.pkl"
    OUT_DIR = os.environ["ONEDRIVE"] + "/研究/2020実験データ/BIN/" + SUBJECT_ID + "/"
    STEP_SIZE = int(SEQ_LEN * SKIP / 2)

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
        # ndarr = [ndarr[i:i + SEQ_LEN] for i in range(0, ndarr.shape[0], STEP_SIZE)]
        out = cp.array(ndarr, dtype="float32")
    else:
        # ndarr = [ndarr[i:i + SEQ_LEN * SKIP:SKIP] for i in range(0, ndarr.shape[0] - SKIP * (SEQ_LEN - 1), STEP_SIZE)]
        out = cp.array(ndarr, dtype="float32")

    return out


def main():
    # CSVファイル読み込み
    path = glob.glob(DATA_DIR + "*.csv")
    print(path)

    if TRAIN is False:
        for index, csv in enumerate(path):
            print(csv, " ", end="")
            df = pd.read_csv(csv, skiprows=3, header=[0, 2, 3], index_col=0)
            out = df_to_cp(df)
            break

    if TRAIN is True:
        for index, csv in enumerate(path):
            print(csv, " ", end="")
            df = pd.read_csv(csv, skiprows=3, header=[0, 2, 3], index_col=0)
            out = df_to_cp(df)
            break

    pos = out
    # vel = pos[1:]-pos[:-1]
    # sos = signal.butter(3, 2.5, btype="lowpass", output="sos", fs=125)
    # vel_smoothed = signal.sosfilt(sos, vel)
    #
    # # pos = [i[42] for i in pos]
    # # vel = [i[42] for i in vel]
    # # vel_smoothed = [i[42] for i in vel_smoothed]
    #
    # pos = cp.array([i[42] for i in out])
    # vel = pos[1:] - pos[:-1]
    # sos = signal.butter(3, 2.5, btype="lowpass", output="sos", fs=125)
    # vel_smoothed = signal.sosfilt(sos, vel)

    sos = signal.butter(3, 2.5, btype="lowpass", output="sos", fs=125)
    pos = pos.T
    vel = cp.empty_like(pos)
    for index, p in enumerate(pos):
        v = p[1:] - p[:-1]
        v_smoothed = signal.sosfilt(sos, v)
        vel[index] = cp.append(v_smoothed, cp.empty(1))
    pos = cp.concatenate([pos, vel]).T[:-1]

    position = [i[42] for i in pos]
    vel_smoothed = [i[120] for i in pos]
    # vel_smoothed = [i[42] for i in vel]

    fig = plt.figure(figsize=[20, 15])
    ax_pos = fig.add_subplot(311)
    ax_pos.plot(position)
    ax_pos.set_xlabel("frame")
    ax_pos.set_ylabel("Hip x position")
    # ax_pos.legend()

    ax_vel = fig.add_subplot(312)
    ax_vel.plot(vel_smoothed)
    ax_pos.set_xlabel("frame")
    ax_pos.set_ylabel("Hip x velocity")

    ax_vel_smoothed = fig.add_subplot(313)
    ax_vel_smoothed.plot(vel_smoothed)
    ax_vel_smoothed.set_xlabel("frame")
    ax_vel_smoothed.set_ylabel("Hip x velocity")
    plt.show()


if __name__ == "__main__":
    main()
