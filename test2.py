import pickle

import matplotlib.pyplot as plt
from adfunc import AnomaryDetector
import numpy as np
import cupy as cp
import os
import time

np.set_printoptions(threshold=10000, linewidth=250)

col = 4800
row = 1000
ls = [i for i in range(row * col)]
data = np.array(ls).reshape([row, col])

st = time.time()
x = data[:, 3::6] - 1024
y = data[:, 4::6] - 512
z = data[:, 5::6] - 256

r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
theta = np.arccos(z / r)
phi = np.arctan(y / x)

data[:, 3::6] = np.log(r)
data[:, 4::6] = theta
data[:, 5::6] = phi

print(r.shape)
print(data.shape)

print(time.time()-st)
exit()
fig = plt.figure()
ax = fig.add_subplot(211, projection="3d")
# ax.scatter(cp.asnumpy(x), cp.asnumpy(y), cp.asnumpy(z))
ax.scatter(x, y, z)
ax.set_zlim([0, 10000])

ax2 = fig.add_subplot(212, polar=True)
ax2.scatter(phi, r)
ax2.grid(True)
plt.show()
