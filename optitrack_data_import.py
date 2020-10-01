import pandas as pd
import numpy as cp
import pickle
import math
import os

pd.set_option('display.max_columns', 150)
#######################################################################################
# OptiTrackからの出力CSVを読み込み、EncDecADに入力できる形(Cupy Seqences)に変換するスクリプト   #
#######################################################################################

DATA_PATH = os.environ["ONEDRIVE"] + "/研究/2020実験データ/Take 2020-09-29 07.18.47 PM.csv"
OUT_PATH = ""
SEQ_LEN = 50
STEP_SIZE = 10
SKELTON_NAME = "Skeleton 002:"
Markers = ["Hip", "WaistLFront", "WaistRFront", "WaistLBack", "WaistRBack", "Chest",
           "BackTop", "BackLeft", "BackRight", "HeadTop", "HeadFront", "HeadSide",
           "LShoulderBack", "LShoulderTop", "LElbowOut", "LUArmHigh", "LHandOut", "LWristOut", "LWristIn",
           "RShoulderBack", "RShoulderTop", "RElbowOut", "RUArmHigh", "RHandOut", "RWristOut", "RWristIn"]
Markers_to_drop = ["Hip", "Ab", "Chest", "Neck", "Head", "LShoulder", "LUArm", "LFArm", "LHand", "RShoulder", "RUArm",
                   "RFArm", "RHand", "LThigh", "LShin", "LFoot", "RThigh", "RShin", "RFoot", "LToe", "RToe"]
Markers = [SKELTON_NAME + name for name in Markers]
Markers_to_drop = [SKELTON_NAME + name for name in Markers_to_drop]

# CSVファイル読み込み
df = pd.read_csv(DATA_PATH, skiprows=3, header=[0, 2, 3], index_col=0)

# 使わない列は削除
df = df.drop(("Name", "Unnamed: 1_level_1", "Time (Seconds)"), axis=1)
for m in Markers_to_drop:
    df = df.drop((m, "Rotation"), axis=1)

ndarr = df.to_numpy()
ndarr = [ndarr[i:i + SEQ_LEN] for i in range(0, ndarr.shape[0], STEP_SIZE)]
out = cp.array(ndarr[1:-math.floor(SEQ_LEN/STEP_SIZE)], dtype="float32")
pickle.dump(out, open(OUT_PATH, "wb"))
