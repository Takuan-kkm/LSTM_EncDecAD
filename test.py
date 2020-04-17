import cupy as cp

print(list(range(0, 140, 25)))
a = cp.arange(16)
print(list(range(0, a.shape[0], 5)))

b = [a[i:i+5] for i in range(0, a.shape[0], 5)]
c = cp.array(b[1:-1], dtype="float32")
print(b)
print(c)
