import chainer
from chainer import Variable
import chainer.functions as F
from chainer.dataset.convert import concat_examples
import numpy as np
import pickle
import cupy as cp
from matplotlib import pyplot as plt
import os
from adfunc import AnomaryDetector

np.set_printoptions(threshold=100000, linewidth=10000)

SUBJECT_ID = "TEST_NOAKI_1008"
TEST_PATH = os.environ["ONEDRIVE"] + "/研究/2020実験データ/BIN/" + SUBJECT_ID + "_VALID.pkl"
TRAIN_PATH = os.environ["ONEDRIVE"] + "/研究/2020実験データ/BIN/" + SUBJECT_ID + "_TRAIN.pkl"

with open("result/model.pkl", "rb") as f:
    net = pickle.load(f)

with open(TRAIN_PATH, "rb") as f:
    VALID = pickle.load(f)

print(sum(p.data.size for p in net.params()))

in_arr = VALID[10]
net.train = True
detector = AnomaryDetector(net, seq_length=50, dim=78, calc_length=25)
rc_data = detector.reconstruct(in_arr)

in_arr = cp.asnumpy(VALID[10])
out_arr = [[cp.asnumpy(j.data) for j in i[0]] for i in rc_data]
#print(in_arr)
print(out_arr)
#plt.imshow(in_arr)
plt.imshow(out_arr)
plt.show()
