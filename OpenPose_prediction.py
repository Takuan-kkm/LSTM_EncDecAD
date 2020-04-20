import pickle
from pose_animation import make_gifanime
from pose_animation import seq_to_png

def main():
    # Load dataset
    with open("test.sav", "rb") as f:
        test = pickle.load(f)

    # Load network
    with open("result/model.pkl", "rb") as f:
        net = pickle.load(f)

    # 元データをアニメgif化
    seq_to_png(test[1])
    make_gifanime(100, out="test[1].gif")

    pred = net(test[1])

    seq_to_png(pred)
    make_gifanime(100, out="pred.gif")


if __name__ == "__main__":
    main()
