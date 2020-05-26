import numpy as np

a = np.array([2, 2, 2])
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(np.dot(a, b))
print(np.dot(b, a))
c = np.dot(a, b)
print(type(np.dot(np.dot(a, b), a)))

l = []
ls = [1,2,3,4]
ls2 = [5,6,7,8]
l = l+ls
l = l+ls2
print(l)
