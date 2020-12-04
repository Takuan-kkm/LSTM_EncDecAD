import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as a3d
import pickle
import os
from PIL import Image
import LSTM_func


SUBJECT_ID = "TEST_SHINCHAN_1112"
# CSV_PATH = "C:/Users/Kuzlab-VR4/PycharmProjects/CNN_AnomaryDetection/temp/" + SUBJECT_ID + "/task4_1.pkl"
CSV_PATH = os.environ["ONEDRIVE"] + "/研究/2020実験データ/BIN/" + SUBJECT_ID + "/task2_1.pkl"
SCALER_PATH = os.environ["ONEDRIVE"] + "/研究/2020実験データ/BIN/" + SUBJECT_ID + "/" + SUBJECT_ID + "_scaler.pkl"
UNIT_VECTOR_LENGTH = 0.0001
LIM_X = 0.4
LIM_Y = 0.4
LIM_Z = 0.5


def data2pic(data, fname):
    fig = plt.figure(figsize=[7, 7])
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim([-LIM_X, LIM_X])
    ax.set_ylim([-LIM_Y, LIM_Y])
    ax.set_zlim([-LIM_Z / 2, LIM_Z])
    ax.view_init(elev=40, azim=40)

    for i in range(0, int(data.shape[0] / 2), 6):
        if i == 0:
            continue

        if i == 6:  # Ab
            col = "red"
        elif i == 12:  # Chest
            col = "blue"
        elif i == 30:  # LShoulder
            col = "green"
        elif i == 54:  # RShoulder
            col = "black"
        elif i == 60:
            col = "grey"
        elif i == 66:
            col = "lightgrey"
        elif i == 72:
            col = "silver"
        else:
            col = None

        line = a3d.Line3D([data[i + 5], data[i + 5] + data[i + 2] * UNIT_VECTOR_LENGTH],
                          [data[i + 3], data[i + 3] + data[i] * UNIT_VECTOR_LENGTH],
                          [data[i + 4], data[i + 4] - data[i + 1] * UNIT_VECTOR_LENGTH],
                          color=col, linewidth=12)
        ax.add_line(line)
    # plt.show()
    # exit()
    plt.savefig(fname)
    plt.close(fig)


def data2skeleton(data, fname, ex=False):
    # 1:Ab, 2:Chest 3:Neck 4:Head 5:LShoulder 6:LUArm 7:LFArm 8:LHand 9:RShoulder 10:RUArm 11:RFArm 12:RHand
    skeleton = ((1, 2), (2, 3), (3, 4),
                (2, 5), (3, 5), (5, 6), (6, 7), (7, 8),
                (2, 9), (3, 9), (9, 10), (10, 11), (11, 12))

    fig = plt.figure(figsize=[7, 7])
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim([-LIM_X, LIM_X])
    ax.set_ylim([-LIM_Y, LIM_Y])
    ax.set_zlim([-LIM_Z / 2, LIM_Z])
    ax.view_init(elev=40, azim=40)

    for i, s in enumerate(skeleton):
        if i in (0, 1, 2):  # taikan
            col = "red"
        elif i in (3, 4, 5, 6, 7):  # L
            col = "blue"
        elif i in (8, 9, 10, 11, 12):  # R
            col = "green"
        line = a3d.Line3D([data[s[0] * 6 + 5], data[s[1] * 6 + 5]],  # Z
                          [data[s[0] * 6 + 3], data[s[1] * 6 + 3]],  # X
                          [data[s[0] * 6 + 4], data[s[1] * 6 + 4]],  # Y
                          color=col, linewidth=2)
        ax.add_line(line)

    # plt.show()
    if ex is True:
        exit()

    plt.savefig(fname)
    plt.close(fig)


def data2skeletons(data1, data2, fname, ex=False):
    # 1:Ab, 2:Chest 3:Neck 4:Head 5:LShoulder 6:LUArm 7:LFArm 8:LHand 9:RShoulder 10:RUArm 11:RFArm 12:RHand
    skeleton = ((1, 2), (2, 3), (3, 4),
                (2, 5), (3, 5), (5, 6), (6, 7), (7, 8),
                (2, 9), (3, 9), (9, 10), (10, 11), (11, 12))

    fig = plt.figure(figsize=[7, 7])
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim([-LIM_X, LIM_X])
    ax.set_ylim([-LIM_Y, LIM_Y])
    ax.set_zlim([-LIM_Z / 2, LIM_Z])
    ax.view_init(elev=40, azim=40)

    col = "red"
    for s in skeleton:
        line = a3d.Line3D([data1[s[0] * 6 + 5], data1[s[1] * 6 + 5]],  # Z
                          [data1[s[0] * 6 + 3], data1[s[1] * 6 + 3]],  # X
                          [data1[s[0] * 6 + 4], data1[s[1] * 6 + 4]],  # Y
                          color=col, linewidth=2)
        ax.add_line(line)

    col = "blue"
    for s in skeleton:
        line = a3d.Line3D([data2[s[0] * 6 + 5], data2[s[1] * 6 + 5]],  # Z
                          [data2[s[0] * 6 + 3], data2[s[1] * 6 + 3]],  # X
                          [data2[s[0] * 6 + 4], data2[s[1] * 6 + 4]],  # Y
                          color=col, linewidth=2)
        ax.add_line(line)

    # plt.show()
    if ex is True:
        exit()

    plt.savefig(fname)
    plt.close(fig)


def seq_to_png(data1, data2, temp="temp"):
    index = 0
    for d1, d2 in zip(data1, data2):
        print(index, end=" ")
        fname = temp + "/" + str(index).rjust(3, "0") + ".png"
        data2skeletons(d1, d2, fname=fname)
        index += 1

    return index


def make_gifanime(n_img, dir="temp", out="out.gif"):
    imgs = []
    for i in range(n_img):
        fname = dir + "/" + str(i).rjust(3, "0") + ".png"
        im = Image.open(fname)
        imgs.append(im)

    imgs[0].save(out, save_all=True, append_images=imgs[1:], loop=1, duration=40)


def main():
    with open(CSV_PATH, "rb") as f:
        ndarr = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # Load network
    with open("../result_3000units/model.pkl", "rb") as f:
        net = pickle.load(f)

    data = ndarr[8]
    redata = net(data)
    data = scaler.inverse_transform([data])
    data = data[0][0]
    redata = scaler.inverse_transform([redata])
    redata = [[r.data for r in re] for re in redata[0][0]]

    n_img = seq_to_png(data, redata)
    make_gifanime(n_img)


if __name__ == '__main__':
    main()
