import numpy as np
import pandas as pd
from scipy.spatial.transform import *
import os
import glob
import time

# Global 5.30389	-8.0065	-5.60693	0.774617	0.881389	0.271791
hipRot = [5.30389, -8.0065, -5.60693]
hipPos = [0.774617, 0.881389, 0.271791]

chestRot = [8.552199, -6.29635, -3.388422]
chestPos = [0.792099, 1.134489, 0.231872]

# Local Chest Rot   22.002422	-0.007569	1.247368
hipRot_L = [5.30389, -8.0065, -5.60693]
hipPos_L = [0.774617, 0.881389, 0.271791]

# abRot_L = [-19.750607, 0.322335, 4.693333]
chestRot_L = [22.855778, -0.113972, -2.440463]

SUBJECT_ID = "TEST_NOAKI_1008"
SKELETON_NAME = "Skeleton 002:"

Markers_to_use = ["Hip", "Ab", "Chest", "Neck", "Head", "LShoulder", "LUArm", "LFArm", "LHand", "RShoulder", "RUArm",
                  "RFArm", "RHand"]
Markers_to_use = [SKELETON_NAME + name for name in Markers_to_use]
Markers_to_drop = ["LThigh", "LShin", "LFoot", "RThigh", "RShin", "RFoot", "LToe", "RToe"]
Markers_to_drop = [SKELETON_NAME + name for name in Markers_to_drop]

DATA_DIR = os.environ["ONEDRIVE"] + "/研究/2020実験データ/CSV/" + SUBJECT_ID + "/csv_world_boneonly/"
path = glob.glob(DATA_DIR + "*.csv")

for index, csv in enumerate(path):
    print(csv)
    df = pd.read_csv(csv, skiprows=3, header=[0, 2, 3], index_col=0)
    df = df.drop(("Name", "Unnamed: 1_level_1", "Time (Seconds)"), axis=1)
    for m in Markers_to_drop:
        df = df.drop(m, axis=1)

    # Convert Global Coordinate to Local Coordinate
    # Rotation
    for m in Markers_to_use:
        if m == "Skeleton 002:Hip":
            continue
        df[(m, "Rotation")] = df[(m, "Rotation")] - df[("Skeleton 002:Hip", "Rotation")]
    print("Rotation Coordinates Convert: Done!")

    # Position
    len_df = df.shape[0]
    percent = 0
    print("Position Coordinates Convert")
    print("       0%|         |         |         |          |100%")
    print("Progress: ", end="")
    for idx, d in df.iterrows():
        if idx/len_df > percent:
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
    print("Done!")

exit()

rotate_array = np.array(hipRot)
hip_origin = np.array(hipPos)
r = Rotation.from_euler('XYZ', [rotate_array[0], rotate_array[1], rotate_array[2]], degrees=True)

body_part_origin = np.array(chestPos)
new_position = r.inv().as_matrix() @ (body_part_origin - hip_origin)
# new_position = r.inv() @ (body_part_origin - hip_origin)
print(new_position)
# temp_body_part = ":".join(body_part.split(":")[0:1] + body_part.split(":")[2:])
# new_df[temp_body_part + "Position:X"][index] = new_position[0]
# new_df[temp_body_part + "Position:Y"][index] = new_position[1]
# new_df[temp_body_part + "Position:Z"][index] = new_position[2]
