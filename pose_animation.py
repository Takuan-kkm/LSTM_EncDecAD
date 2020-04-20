import matplotlib.pyplot as plt
import matplotlib.collections as mc
from matplotlib.animation import ArtistAnimation
import pickle
from PIL import Image


def keypoints_to_img(keypoints, fname):
    x = [keypoints[i] for i in range(0, 50, 2)]
    y = [keypoints[i + 1] for i in range(0, 50, 2)]

    skelton = (
        (0, 1), (0, 15), (0, 16), (1, 2), (1, 5), (1, 8), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (8, 12), (9, 10),
        (10, 11), (11, 22), (11, 24), (12, 13), (13, 14), (14, 19), (14, 21), (15, 17), (16, 18), (19, 20), (22, 23)
    )

    ln = [[(x[s[0]], y[s[0]]), (x[s[1]], y[s[1]])] for s in skelton]
    cl = ["crimson", "orchid", "fuchsia", "orange", "greenyellow", "firebrick", "gold", "yellow", "lime", "limegreen",
          "mediumseagreen", "skyblue", "mediumaquamarine", "mediumturquoise", "lightseagreen", "darkturquoise",
          "deepskyblue", "steelblue", "royalblue", "midnightblue", "darkorchid", "darkmagenta", "darkblue", "turquoise"]

    lines = []
    colors = []
    for i in range(len(ln)):
        if not (ln[i][0][0] <= 0.1 or ln[i][1][0] <= 0.1):  # 線分の始点、終点の座標が0だったら(keypointを検出できていなかったら)線分を削除
            lines.append(ln[i])
            colors.append(cl[i])

    lc = mc.LineCollection(lines, color=colors, linewidths=3)

    fig = plt.figure(figsize=(6.4, 3.6))
    ax = fig.add_subplot(aspect=9 / 16)

    ax.add_collection(lc)
    ax.scatter(x, y)
    ax.autoscale()

    ax.set_xlim([0, 1])
    ax.set_ylim([1, 0])

    plt.savefig(fname)
    plt.close(fig)


def seq_to_png(seq, temp="temp"):
    i = 0
    for keypoints in seq:
        fname = temp + "/" + str(i).rjust(3, "0") + ".png"
        keypoints_to_img(keypoints, fname=fname)
        i += 1


def make_gifanime(n_img, dir="temp", out="out.gif"):
    imgs = []
    for i in range(n_img):
        fname = dir + "/" + str(i).rjust(3, "0") + ".png"
        im = Image.open(fname)
        imgs.append(im)

    imgs[0].save(out, save_all=True, append_images=imgs[1:], loop=1, duration=100)


def main():
    # データの読み込み
    with open("train.sav", "rb") as f:
        datum = pickle.load(f)

    points = datum[1]
    seq_to_png(points)
    make_gifanime(100)


if __name__ == '__main__':
    main()
