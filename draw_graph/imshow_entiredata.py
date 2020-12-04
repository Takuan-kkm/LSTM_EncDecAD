import numpy as np
import pickle
import cupy as cp
from matplotlib import pyplot as plt

np.set_printoptions(threshold=1000000, linewidth=10000)

SUBJECT_ID = "TEST_SHINCHAN_1112"
with open("../result_3000units/model.pkl", "rb") as f:
    net = pickle.load(f)

path = "C:/Users/Kuzlab-VR4/PycharmProjects/CNN_AnomaryDetection/temp/" + SUBJECT_ID + "/task1_1.pkl"
with open(path, "rb") as f:
    data = pickle.load(f)

# 全体描画
data = cp.asnumpy(data).T
xyz = (data[::3] ** 2 + data[1::3] ** 2 + data[2::3] ** 2) ** 0.5
x_axis = [0, data.shape[0], 1 / 25]

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(111)
img = ax.imshow(xyz, cmap="jet", vmin=-1, vmax=4, aspect="auto", interpolation="none",
                extent=[0, xyz.shape[1] / 25, xyz.shape[0], 0])
ax.grid(1, fillstyle="full")

plt.show()
