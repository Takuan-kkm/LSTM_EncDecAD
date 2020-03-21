import cupy as cp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import glob
import json
import argparse
import pickle
import sys


def extraction_serialnum(filename):  # ファイルパスlist.sort()で使う
    return int(filename.lstrip(args[0].Body_keypoints_dir).rstrip("_keypoints.json"))


# ファイルpath等指定部分 flags ###################################################################
parser = argparse.ArgumentParser()

# OpenPoseで抽出したBodykeypoints.jsonファイル群のディレクトリ
# Body_keypoints_dirのpathはバックスラッシュ+エスケープ使用必須 (globの仕様上)
parser.add_argument("--Body_keypoints_dir", default="output_json\\",
                    help="Process a directory of Body keypoints files(.json).")
parser.add_argument("--train_path", default="train.sav")
parser.add_argument("--test_path", default="test.sav")
parser.add_argument("-r", default=10, help="Set frame rate used when extracting keypoints with OpenPose")

args = parser.parse_known_args()
##################################################################################################

Body_keypoints_path = glob.glob(args[0].Body_keypoints_dir + "*.json")
Body_keypoints_path.sort(key=extraction_serialnum)

keypoints = []
X = []

# Body_keypoints(.json)の読み込み
for path in Body_keypoints_path:
    print(path)
    with open(path, 'r') as f:
        try:
            Bk = json.load(f)
            X.append(Bk["people"][0]["pose_keypoints_2d"])
        except Exception as e:
            X.append(X[-1])
            print(e)

l = len(X)
X_train, X_test = train_test_split(X, test_size=0.2)


# 標準化
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_train = cp.array(X_train, dtype="float32")
X_test = cp.array(X_test, dtype="float32")

#print(X_train.shape)
pickle.dump(X_train, open(args[0].train_path, "wb"))
pickle.dump(X_test, open(args[0].test_path, "wb"))
