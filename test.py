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
# path = "C:/Users/Kuzlab-VR4/PycharmProjects/CNN_AnomaryDetection/temp/" + SUBJECT_ID + "/task1_1.pkl"
path = os.environ["ONEDRIVE"] + "/研究/2020実験データ/BIN/" + SUBJECT_ID + "/" + "task3_1.pkl"
with open(path, "rb") as f:
    data = pickle.load(f)

with open("result/model.pkl", "rb") as f:
    net = pickle.load(f)

in_data = data[3]

reconst_data = net(in_data)
reconst_data = cp.asnumpy(reconst_data[0].data).T
data = cp.asnumpy(in_data[0]).T
data = (data[::3] ** 2 + data[1::3] ** 2 + data[2::3] ** 2) ** 0.5
reconst_data = (reconst_data[::3] ** 2 + reconst_data[1::3] ** 2 + reconst_data[2::3] ** 2) ** 0.5

fig = plt.figure(figsize=[15,10])
ax = fig.add_subplot(211)
ax.imshow(data, cmap="jet", vmin=-1, vmax=4, aspect="auto", interpolation="none",
                extent=[0, data.shape[1] / 25, data.shape[0],0 ])

ax = fig.add_subplot(212)
ax.imshow(reconst_data, cmap="jet", vmin=-1, vmax=4, aspect="auto", interpolation="none",
                extent=[0, data.shape[1] / 25, data.shape[0],0 ])
plt.show()
exit()

data = cp.asnumpy(data).T
xyz = (data[::3] ** 2 + data[1::3] ** 2 + data[2::3] ** 2) ** 0.5
# print(np.var(data, axis=1))
# print(np.average(data, axis=1))
# print(xyz.shape)
x_axis = [0, data.shape[0], 1 / 25]

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(111)
img = ax.imshow(xyz, cmap="jet", vmin=-1, vmax=4, aspect="auto", interpolation="none",
                extent=[0, xyz.shape[1] / 25, xyz.shape[0],0 ])
ax.grid(1, fillstyle="full")

# fig.colorbar(img)

# ax2 = fig.add_subplot(212)
# ax2.set_ylim([0, 20])
# ax2.set_xlim([0, xyz.shape[1]])
# ax2.plot(xyz[:26].T)


plt.show()
