import numpy as np

ls = []
d = np.array([1, 2, 3])
print(np.append(d, np.empty(1)))
print(np.append(d, np.empty(1)).shape)
