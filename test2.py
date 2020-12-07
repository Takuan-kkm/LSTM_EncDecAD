import pickle

import matplotlib.pyplot as plt
from adfunc import AnomaryDetector
import numpy as np
import cupy as cp
import os
import time

N = 1024 * 1024 * 100
ls = [i + 1 for i in range(N)]
x = cp.array(ls)
ls.reverse()
y = cp.array(ls)
z = cp.ones(shape=[N])

st = time.time()
r = cp.sqrt(x ** 2 + y ** 2 + z ** 2)
theta = cp.arccos(z / r)
phi = cp.arctan(y / x)
print("cp:", time.time() - st)
print(r.shape, theta.shape, phi.shape)


x = np.array(ls)
ls.reverse()
y = np.array(ls)
z = np.ones(shape=[N])

st = time.time()
r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
theta = np.arccos(z / r)
phi = np.arctan(y / x)
print("np:", time.time() - st)
print(r.shape, theta.shape, phi.shape)
exit()

# fig = plt.figure()
# ax = fig.add_subplot(211, projection="3d")
# # ax.scatter(cp.asnumpy(x), cp.asnumpy(y), cp.asnumpy(z))
# # ax.scatter(x, y, z)
# ax.set_zlim([0, 2])
#
# ax2 = fig.add_subplot(212, polar=True)
# # ax2.scatter(cp.asnumpy(r), cp.asnumpy(theta))
# # ax2.scatter(theta, r)
# ax2.grid(True)
# # plt.show()
