import pandas as pd
import cupy as cp
from scipy.spatial.transform import Rotation
import numpy as np
import pickle
import math
import os
import glob
import warnings
from scipy import signal
from sklearn.preprocessing import StandardScaler
import argparse

warnings.simplefilter("ignore")

SKELETON_NAME = "Skeleton 002:"

SEQ_LEN = 156  # 時系列データの長さ
STEP_SIZE = 12  # 1時系列データ間の開始フレームの差
SKIP = 5  # 間引くフレーム数 ex)OptiTrackの記録レートが125fps,SKIP=5 → 25fpsにダウンサンプリングしてデータ化

Markers_to_use = ["Hip", "Ab", "Chest", "Neck", "Head", "LShoulder", "LUArm", "LFArm", "LHand", "RShoulder", "RUArm",
                  "RFArm", "RHand"]
Markers_to_drop = ["LThigh", "LShin", "LFoot", "RThigh", "RShin", "RFoot", "LToe", "RToe"]
Markers_to_use = [SKELETON_NAME + name for name in Markers_to_use]
Markers_to_drop = [SKELETON_NAME + name for name in Markers_to_drop]


def df_to_cp(df, scaler=None, coordinate=None):
    # 使わない列は削除
    df = df.drop(("Name", "Unnamed: 1_level_1", "Time (Seconds)"), axis=1)
    for m in Markers_to_drop:
        df = df.drop(m, axis=1)

    # ローカル座標系への変換
    if coordinate == "LOCAL":
        df = world_to_local(df)

    # ndarrayに変換
    ndarr = df.to_numpy()

    # 極座標への変換
    if coordinate == "POLAR":
        ndarr = convert_logscaled_polar(ndarr)

    # 速度を計算
    pos_t = ndarr.T
    vel_t = calc_vel(pos_t)
    ndarr = np.concatenate([pos_t, vel_t]).T[:-1]

    # 標準化
    if scaler is not None:
        ndarr = scaler.transform(ndarr)

    if SKIP == 0:
        ndarr = [[ndarr[i:i + SEQ_LEN]] for i in range(0, ndarr.shape[0], STEP_SIZE)]
        out = cp.array(ndarr[1:-math.floor(SEQ_LEN / STEP_SIZE)], dtype="float32")
    else:
        ndarr = [[ndarr[i:i + SEQ_LEN * SKIP:SKIP]] for i in range(0, ndarr.shape[0] - SKIP * (SEQ_LEN - 1), STEP_SIZE)]
        out = cp.array(ndarr, dtype="float32")

    return out


def world_to_local(df):
    # Convert Global Coordinates to Local Coordinate System
    # Rotation
    hip_name = SKELETON_NAME + "Hip"
    for m in Markers_to_use:
        if m == hip_name:
            continue
        df[(m, "Rotation")] = df[(m, "Rotation")] - df[(SKELETON_NAME + "Hip", "Rotation")]
    print("  Rotation Coordinates Convert: Done!")

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
            if m == hip_name:
                rotate_array = d[m, "Rotation"].values
                hip_origin = d[m, "Position"].values
                r = Rotation.from_euler('XYZ', [rotate_array[0], rotate_array[1], rotate_array[2]], degrees=True)
                continue

            body_part_origin = d[(m, "Position")].values
            new_position = r.inv().as_matrix() @ (body_part_origin - hip_origin)
            df.at[idx, (m, "Position", "X")] = new_position[0]
            df.at[idx, (m, "Position", "Y")] = new_position[1]
            df.at[idx, (m, "Position", "Z")] = new_position[2]
    print("  Done!\n")

    return df


def calc_vel(pos_t):
    sos = signal.butter(3, 2.5, btype="lowpass", output="sos", fs=125)
    vel = np.empty_like(pos_t)
    for index, p in enumerate(pos_t):
        v = p[1:] - p[:-1]
        v_smoothed = signal.sosfilt(sos, v)
        vel[index] = np.append(v_smoothed, np.empty(1))

    return vel


def create_scaler(path, coordinate):
    print("###############################  Scaler Creation  ###############################")
    for index, csv in enumerate(path):
        print(csv)
        df = pd.read_csv(csv, skiprows=3, header=[0, 2, 3], index_col=0)
        df = df.drop(("Name", "Unnamed: 1_level_1", "Time (Seconds)"), axis=1)
        for m in Markers_to_drop:
            df = df.drop(m, axis=1)

        # ローカル座標系への変換
        if coordinate == "LOCAL":
             df = world_to_local(df)

        # ndarrayに変換
        ndarr = df.to_numpy()

        # 極座標への変換
        if coordinate == "POLAR":
            ndarr = convert_logscaled_polar(ndarr)

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
    print("############################ Scaler Creation Done #############################")

    return scaler


def load_scaler(scaler_path):
    with open(scaler_path, "rb") as f:
        s = pickle.load(f)

    return s


def convert_logscaled_polar(data, origin=(1.4107, 1.0109, 0.061)):
    # 3次元直交座標系を引数座標originを中心とする極座標系に変換
    # 距離軸rは自然対数でスケールされる
    # 回転はそのまま
    x = origin[0] - data[:, 3::6]
    y = origin[1] - data[:, 4::6]
    z = origin[2] - data[:, 5::6]

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan(y / x)

    data[:, 3::6] = np.log(r)
    data[:, 4::6] = theta
    data[:, 5::6] = phi

    return data


def create_dataset(sub_id, dataset_type, coordinate):
    scaler_path = os.environ["ONEDRIVE"] + "/研究/2020実験データ/BIN/" + sub_id + "/" + sub_id + "_scaler.pkl"
    global SEQ_LEN
    global STEP_SIZE
    global SKIP

    if "TRAIN" in dataset_type:
        print("-TRAIN-")
        data_dir = os.environ["ONEDRIVE"] + "/研究/2020実験データ/CSV_BIN/" + sub_id + "/TRAIN/"
        out_path = os.environ["ONEDRIVE"] + "/研究/2020実験データ/CSV_BIN/" + sub_id + "/" + sub_id + "_TRAIN.pkl"

        path = glob.glob(data_dir + "*.csv")

        scaler = create_scaler(path, coordinate)
        for index, csv in enumerate(path):
            print(csv)
            df = pd.read_csv(csv, skiprows=3, header=[0, 2, 3], index_col=0)
            if index == 0:
                out = df_to_cp(df, scaler, coordinate)
            else:
                out = cp.concatenate([out, df_to_cp(df, scaler, coordinate)])

        print(out_path, out.shape)
        pickle.dump(out, open(out_path, "wb"))
        pickle.dump(scaler, open(scaler_path, "wb"))

    if "TEST" in dataset_type:
        print("-TEST-")
        data_dir = os.environ["ONEDRIVE"] + "/研究/2020実験データ/CSV_BIN/" + sub_id + "/TEST/"
        out_dir = os.environ["ONEDRIVE"] + "/研究/2020実験データ/CSV_BIN/" + sub_id + "/"

        step_temp = STEP_SIZE
        STEP_SIZE = int((SEQ_LEN*SKIP)/2)
        path = glob.glob(data_dir + "*.csv")

        scaler = load_scaler(scaler_path)
        for index, csv in enumerate(path):
            print(csv, " ", end="")
            df = pd.read_csv(csv, skiprows=3, header=[0, 2, 3], index_col=0)
            out_path = out_dir + os.path.splitext(os.path.basename(csv))[0] + ".pkl"
            out = df_to_cp(df, scaler, coordinate)
            print(out.shape)
            pickle.dump(out, open(out_path, "wb"))

        STEP_SIZE = step_temp

    if "VALID" in dataset_type:
        print("-VALID-")
        data_dir = os.environ["ONEDRIVE"] + "/研究/2020実験データ/CSV_BIN/" + sub_id + "/TEST/"
        out_path = os.environ["ONEDRIVE"] + "/研究/2020実験データ/CSV_BIN/" + sub_id + "/" + sub_id + "_VALID.pkl"

        STEP_SIZE = int(STEP_SIZE * 15)

        scaler = load_scaler(scaler_path)
        path = glob.glob(data_dir + "*.csv")

        for index, csv in enumerate(path):
            # print(csv, " ", end="")
            df = pd.read_csv(csv, skiprows=3, header=[0, 2, 3], index_col=0)
            if index == 0:
                out = df_to_cp(df, scaler, coordinate)
            else:
                out = cp.concatenate([out, df_to_cp(df, scaler, coordinate)])
        print(out_path, out.shape)
        STEP_SIZE = int(STEP_SIZE / 15)

        pickle.dump(out, open(out_path, "wb"))


def main():
    parser = argparse.ArgumentParser(description='OptiTrackからの出力CSVを読み込み、EncDecADへの入力(Cupy Seqences)に変換')
    parser.add_argument('-dataset_type', default=["TRAIN", "VALID", "TEST"], help='TRAIN/VALID/TEST　のいずれかのリスト')
    # parser.add_argument('-subject_id', default=["T1_1109", "S1_1112", "H1_1202", "E1_1203"], help='被験者IDのリスト')
    parser.add_argument('-subject_id', default=["N1_1008"], help='被験者IDのリスト')
    parser.add_argument('-coordinate', default="LOCAL", help='POLAR(レンジを中心としたlog-scaled 極座標) or LOCAL(腰を中心にした座標系)',
                        choices=["POLAR", "LOCAL"])
    args = parser.parse_args()

    print("DATASET TYPE:", args.dataset_type)
    print("SUBJECTS:", args.subject_id)
    print("COORDINATE TYPE:", args.coordinate)

    for sub_id in args.subject_id:
        print("\n******************************************************")
        print("********************   "+sub_id+"   *********************")
        print("******************************************************")
        create_dataset(sub_id, args.dataset_type, args.coordinate)


if __name__ == "__main__":
    main()
