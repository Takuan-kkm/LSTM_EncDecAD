import cupy as cp
from sklearn.model_selection import train_test_split
import glob
import json
import argparse
import pickle
import math


def extraction_serialnum(filename):  # ファイルパスlist.sort()で使う
    return int(filename.lstrip(args[0].Body_keypoints_dir).rstrip("_keypoints.json"))


# ファイルpath等指定部分 flags ###################################################################
parser = argparse.ArgumentParser()

# OpenPoseで抽出したBodykeypoints.jsonファイル群のディレクトリ
# Body_keypoints_dirのpathはバックスラッシュ+エスケープ使用必須 (globの仕様上)
parser.add_argument("--Body_keypoints_dir", default="output_json_confuse\\",
                    help="Process a directory of Body keypoints files(.json).")
parser.add_argument("--Body_keypoints_dir_confuse", default="output_json_confuse\\",
                    help="Process a directory of Body keypoints files(.json).")
parser.add_argument("--train_path", default="train.sav")
parser.add_argument("--test_path", default="test.sav")
parser.add_argument("--confuse_path", default="confuse.sav")
parser.add_argument("--test_train_split", default=False)
parser.add_argument("--seq_len", default=50)
parser.add_argument("--step_size", default=5)
parser.add_argument("-r", default=10, help="Set frame rate used when extracting keypoints with OpenPose")

args = parser.parse_known_args()
##################################################################################################

Body_keypoints_path = glob.glob(args[0].Body_keypoints_dir + "*.json")
Body_keypoints_path.sort(key=extraction_serialnum)

X = []
X_temp = []

# Body_keypoints(.json)の読み込み
for path in Body_keypoints_path:
    print(path)
    with open(path, 'r') as f:
        try:
            Bk = json.load(f)
            # keypoint列中の3の倍数個目の要素（推定確率）は使わないため、スキップしてlistにappend
            X_temp.append([Bk["people"][0]["pose_keypoints_2d"][i] for i in range(75) if i % 3 != 2])

            # 腰・腹部・両腕の内、検出されていない部位があった場合には前フレームと同位置とする
            if len(X_temp) > 1:  # 最初のフレームは飛ばす
                for i in list(range(20)) + [24, 25]:
                    if X_temp[-1][i] == 0:
                        X_temp[-1][i] = X_temp[-2][i]

            # 各部位の座標を腰(x_temp[16], x_temp[17])からの相対位置とする, y座標は正負反転
            temp = []
            temp_angle = []
            for i in list(range(10)) + [12] + list(range(15, 19)):
                temp.append(X_temp[-1][i * 2] - X_temp[-1][8 * 2])  # i番目の関節のx座標
                temp.append(X_temp[-1][8 * 2 + 1] - X_temp[-1][i * 2 + 1])  # i番目の関節のy座標

            # 関節角を計算
            for angle in [[0, 1, 5], [0, 1, 2], [1, 2, 3], [2, 3, 4], [1, 5, 6],
                          [5, 6, 7], [1, 8, 10], [1, 0, 11], [0, 11, 13]]:
                ax = temp[2 * angle[0]]
                ay = temp[2 * angle[0] + 1]
                bx = temp[2 * angle[1]]
                by = temp[2 * angle[1] + 1]
                cx = temp[2 * angle[2]]
                cy = temp[2 * angle[2] + 1]

                if [(ax - bx), (ay - by)] == [0, 0] or [(cx - bx), (cy - by)] == [0, 0]:  # 内積が0になるときには角度は0とする
                    temp_angle.append(0)
                else:
                    cos = ((ax - bx) * (cx - bx) + (ay - by) * (cy - by)) / (
                            math.sqrt((ax - bx) ** 2 + (ay - by) ** 2) * math.sqrt((cx - bx) ** 2 + (cy - by) ** 2))
                    temp_angle.append(math.acos(cos))

            X.append(temp)
            # print(X_temp[-1][15:19])
            # print(temp[15:19])

        except Exception as e:
            X.append(X[-1])
            print(e)

if args[0].test_train_split is True:
    X_train, X_test = train_test_split(X, test_size=0.2, shuffle=False)

    # 時系列長とステップ幅にしたがって切り出し
    train = [X_train[i:i + args[0].seq_len] for i in range(0, len(X_train), args[0].step_size)]
    test = [X_test[i:i + args[0].seq_len] for i in range(0, len(X_test), args[0].step_size)]

    # 先頭と末尾の要素を除いてndarrayに変換
    X_train = cp.array(train[1:-math.floor(args[0].seq_len / args[0].step_size)], dtype="float32")
    X_test = cp.array(test[1:-math.floor(args[0].seq_len / args[0].step_size)], dtype="float32")

    pickle.dump(X_train, open(args[0].train_path, "wb"))
    pickle.dump(X_test, open(args[0].test_path, "wb"))
else:
    confuse = [X[i:i + args[0].seq_len] for i in range(0, len(X), args[0].step_size)]
    X_confuse = cp.array(confuse[1:-math.floor(args[0].seq_len / args[0].step_size)], dtype="float32")
    pickle.dump(X_confuse, open(args[0].confuse_path, "wb"))
