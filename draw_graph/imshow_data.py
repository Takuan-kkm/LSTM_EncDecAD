import chainer
from chainer import Variable
import chainer.functions as F
from chainer.dataset.convert import concat_examples
import numpy as np
import pickle
import cupy as cp
from matplotlib import pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

np.set_printoptions(threshold=1000000, linewidth=10000)

SUBJECT_ID = "TEST_SHINCHAN_1112"
with open("../result_2048units/model.pkl", "rb") as f:
    net = pickle.load(f)

for i in range(10,20):
    path = os.environ["ONEDRIVE"] + "/研究/2020実験データ/BIN/" + SUBJECT_ID + "/task4_1.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)

    # 再構成データ描画
    INDEX = i
    in_data = data[INDEX]

    reconst_data = net(in_data)
    reconst_data = cp.asnumpy(reconst_data[0].data).T
    data = cp.asnumpy(in_data[0]).T

    # xyz軸統合
    data = (data[::3] ** 2 + data[1::3] ** 2 + data[2::3] ** 2) ** 0.5
    reconst_data = (reconst_data[::3] ** 2 + reconst_data[1::3] ** 2 + reconst_data[2::3] ** 2) ** 0.5
    diff = np.abs(data - reconst_data)
    # 描画 上段が入力データ、下段が出力データ
    fig = plt.figure(figsize=[15, 15])
    ax = fig.add_subplot(311)
    ax.imshow(data, cmap="jet", vmin=-1, vmax=4, aspect="auto", interpolation="none",
              extent=[INDEX * 3.12, data.shape[1] / 25 + INDEX * 3.12, data.shape[0], 0])

    ax = fig.add_subplot(312)
    ax.imshow(reconst_data, cmap="jet", vmin=-1, vmax=4, aspect="auto", interpolation="none",
              extent=[INDEX * 3.12, data.shape[1] / 25 + INDEX * 3.12, data.shape[0], 0])

    ax = fig.add_subplot(313)
    ax.imshow(diff, cmap="jet", vmin=-1, vmax=4, aspect="auto", interpolation="none",
              extent=[INDEX * 3.12, data.shape[1] / 25 + INDEX * 3.12, data.shape[0], 0])
    plt.show()
    plt.clf()

    print((reconst_data - data).shape)

exit()
