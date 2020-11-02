import pandas as pd
import cupy as cp
from scipy.spatial.transform import Rotation
import numpy as np
import pickle
import math
import os
import glob
from scipy import signal
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', 150)
#######################################################################################
# OptiTrackからの出力CSVを読み込み、EncDecADに入力できる形(Cupy Seqences)に変換するスクリプト   #
#######################################################################################

# TRAIN = True  # TRAIN == Falseの場合、csvファイル毎にpklファイルを作る。Trueのときは、ディレクトリ内のcsvをまとめてpkl化する。
# VALID = True  # Trueで学習のvalidationに使うデータセットを作る
DATASET = "TEST"  # TRAIN,TEST,VALIDのどれか

SUBJECT_ID = "TEST_NOAKI_1008"
SKELETON_NAME = "Skeleton 002:"

SEQ_LEN = 125  # 時系列データの長さ
STEP_SIZE = 12  # 1時系列データ間の開始フレームの差
SKIP = 5  # 飛ばすフレーム数 ex)OptiTrackの記録レートが125fps,SKIP=5 → 25fpsにダウンサンプリングしてデータ化される

SCALER_PATH = os.environ["ONEDRIVE"] + "/研究/2020実験データ/BIN/" + SUBJECT_ID + "_scaler.pkl"

if DATASET in ["TRAIN", "TEST", "VALID"]:
    if DATASET == "TRAIN":
        DATA_DIR = os.environ["ONEDRIVE"] + "/研究/2020実験データ/CSV/" + SUBJECT_ID + "/TRAIN/"
        OUT_PATH = os.environ["ONEDRIVE"] + "/研究/2020実験データ/BIN/" + SUBJECT_ID + "_TRAIN.pkl"
    if DATASET == "TEST":
        DATA_DIR = os.environ["ONEDRIVE"] + "/研究/2020実験データ/CSV/" + SUBJECT_ID + "/TEST/"
        OUT_DIR = os.environ["ONEDRIVE"] + "/研究/2020実験データ/BIN/" + SUBJECT_ID + "/"
        STEP_SIZE = int(SEQ_LEN * SKIP / 2)
    if DATASET == "VALID":
        DATA_DIR = os.environ["ONEDRIVE"] + "/研究/2020実験データ/CSV/" + SUBJECT_ID + "/TEST/"
        OUT_PATH = os.environ["ONEDRIVE"] + "/研究/2020実験データ/BIN/" + SUBJECT_ID + "_VALID.pkl"
else:
    print("DATASETの値を確認してください")
    exit(-1)

# Markers_to_use = ["Hip", "WaistLFront", "WaistRFront", "WaistLBack", "WaistRBack", "Chest",
#                   "BackTop", "BackLeft", "BackRight", "HeadTop", "HeadFront", "HeadSide",
#                   "LShoulderBack", "LShoulderTop", "LElbowOut", "LUArmHigh", "LHandOut", "LWristOut", "LWristIn",
#                   "RShoulderBack", "RShoulderTop", "RElbowOut", "RUArmHigh", "RHandOut", "RWristOut", "RWristIn"]
# Markers_to_drop = ["Hip", "Ab", "Chest", "Neck", "Head", "LShoulder", "LUArm", "LFArm", "LHand", "RShoulder", "RUArm",
#                    "RFArm", "RHand", "LThigh", "LShin", "LFoot", "RThigh", "RShin", "RFoot", "LToe", "RToe"]
Markers_to_use = ["Hip", "Ab", "Chest", "Neck", "Head", "LShoulder", "LUArm", "LFArm", "LHand", "RShoulder", "RUArm",
                  "RFArm", "RHand"]
Markers_to_drop = ["LThigh", "LShin", "LFoot", "RThigh", "RShin", "RFoot", "LToe", "RToe"]
Markers_to_use = [SKELETON_NAME + name for name in Markers_to_use]
Markers_to_drop = [SKELETON_NAME + name for name in Markers_to_drop]


def df_to_cp(df, scaler=None):
    # 使わない列は削除
    df = df.drop(("Name", "Unnamed: 1_level_1", "Time (Seconds)"), axis=1)
    for m in Markers_to_drop:
        df = df.drop((m, "Rotation"), axis=1)

    # 座標系の変換
    df = world_to_local(df)

    # ndarrayに変換
    ndarr = df.to_numpy()

    # 速度を計算
    pos_t = ndarr.T
    vel_t = calc_vel(pos_t)
    ndarr = np.concatenate([pos_t, vel_t]).T[:-1]

    # 標準化
    if scaler is not None:
        ndarr = scaler.transform(ndarr)

    if SKIP == 0:
        ndarr = [ndarr[i:i + SEQ_LEN] for i in range(0, ndarr.shape[0], STEP_SIZE)]
        out = cp.array(ndarr[1:-math.floor(SEQ_LEN / STEP_SIZE)], dtype="float32")
    else:
        ndarr = [ndarr[i:i + SEQ_LEN * SKIP:SKIP] for i in range(0, ndarr.shape[0] - SKIP * (SEQ_LEN - 1), STEP_SIZE)]
        out = cp.array(ndarr, dtype="float32")

    return out


def world_to_local(df):
    # Convert Global Coordinates to Local Coordinate System
    # Rotation
    for m in Markers_to_use:
        if m == "Skeleton 002:Hip":
            continue
        df[(m, "Rotation")] = df[(m, "Rotation")] - df[("Skeleton 002:Hip", "Rotation")]
    print("Rotation Coordinates Convert: Done!")

    # Position
    len_df = df.shape[0]
    percent = 0
    print("  Position Coordinates Convert")
    print("         0%|         |         |         |          |100%")
    print("  Progress: ", end="")
    for idx, d in df.iterrows():
        if idx / len_df > percent:
            print("#", end="")
            percent += 0.025

        for m in Markers_to_use:
            if m == "Skeleton 002:Hip":
                rotate_array = d[m, "Rotation"].values
                hip_origin = d[m, "Position"].values
                r = Rotation.from_euler('XYZ', [rotate_array[0], rotate_array[1], rotate_array[2]], degrees=True)
                continue

            body_part_origin = d[(m, "Position")].values
            new_position = r.inv().as_matrix() @ (body_part_origin - hip_origin)
            df.at[idx, (m, "Position", "X")] = new_position[0]
            df.at[idx, (m, "Position", "Y")] = new_position[1]
            df.at[idx, (m, "Position", "Z")] = new_position[2]
    print("  Done!")

    return df


def calc_vel(pos_t):
    sos = signal.butter(3, 2.5, btype="lowpass", output="sos", fs=125)
    vel = np.empty_like(pos_t)
    for index, p in enumerate(pos_t):
        v = p[1:] - p[:-1]
        v_smoothed = signal.sosfilt(sos, v)
        vel[index] = np.append(v_smoothed, np.empty(1))

    return vel


def create_scaler(path):
    for index, csv in enumerate(path):
        df = pd.read_csv(csv, skiprows=3, header=[0, 2, 3], index_col=0)
        df = df.drop(("Name", "Unnamed: 1_level_1", "Time (Seconds)"), axis=1)
        for m in Markers_to_drop:
            df = df.drop((m, "Rotation"), axis=1)

        # 座標系の変換
        df = world_to_local(df)

        # ndarrayに変換
        ndarr = df.to_numpy()

        # 速度を計算
        pos_t = ndarr.T
        vel_t = calc_vel(pos_t)
        ndarr = np.concatenate([pos_t, vel_t]).T[:-1]

        if index == 0:
            tofit = ndarr
        else:
            tofit = np.concatenate([tofit, ndarr])

    scaler = StandardScaler()
    scaler.fit(tofit)

    return scaler


def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        s = pickle.load(f)

    return s


def main():
    # CSVファイル読み込み
    path = glob.glob(DATA_DIR + "*.csv")
    # print(path)

    if DATASET == "TEST":
        scaler = load_scaler()
        for index, csv in enumerate(path):
            print(csv, " ", end="")
            df = pd.read_csv(csv, skiprows=3, header=[0, 2, 3], index_col=0)
            out_path = OUT_DIR + os.path.splitext(os.path.basename(csv))[0] + ".pkl"
            out = df_to_cp(df, scaler)
            print(out.shape)
            pickle.dump(out, open(out_path, "wb"))

    if DATASET == "TRAIN":
        scaler = create_scaler(path)
        for index, csv in enumerate(path):
            print(csv, " ", end="")
            df = pd.read_csv(csv, skiprows=3, header=[0, 2, 3], index_col=0)
            if index == 0:
                out = df_to_cp(df, scaler)
            else:
                out = cp.concatenate([out, df_to_cp(df, scaler)])
            print(out.shape)

        pickle.dump(out, open(OUT_PATH, "wb"))
        pickle.dump(scaler, open(SCALER_PATH, "wb"))

    if DATASET == "VALID":
        scaler = load_scaler()
        for index, csv in enumerate(path):
            print(csv, " ", end="")
            df = pd.read_csv(csv, skiprows=3, header=[0, 2, 3], index_col=0)
            if index == 0:
                out = df_to_cp(df, scaler)
            else:
                out = cp.concatenate([out, df_to_cp(df, scaler)])
            print(out.shape)

        pickle.dump(out, open(OUT_PATH, "wb"))


if __name__ == "__main__":
    main()
