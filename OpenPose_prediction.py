import pickle
from pose_animation import make_gifanime
from pose_animation import seq_to_png

SEQUENCE_LEN = 30

def main():
    # Load dataset
    with open("test.sav", "rb") as f:
        test = pickle.load(f)

    # Load network
    with open("result/model.pkl", "rb") as f:
        net = pickle.load(f)

    index = 41  # 参照するデータ番号

    # 元データをアニメgif化
    seq_to_png(test[index])
    make_gifanime(SEQUENCE_LEN, out="test[41]a.gif")

    datum = test[index]
    shape = datum.shape
    pred = net(datum.reshape([shape[0], 1, shape[1]]))

    pred_skelton = [pred[i][0].data for i in range(SEQUENCE_LEN)]

    seq_to_png(pred_skelton)
    make_gifanime(SEQUENCE_LEN, out="test[41]b.gif")


if __name__ == "__main__":
    main()
