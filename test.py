import matplotlib.pyplot as plt
import matplotlib.collections as mc
import pickle

# データの読み込み
with open("test.sav", "rb") as f:
    datum = pickle.load(f)

skelton = datum[0][0]

points = [(skelton[i], skelton[i + 1]) for i in range(0, 50, 2)]
print(points)
exit()
lines = [[(0.1, 0.1), (0.3, 0.1)], [(0.2, 0.5), (0.4, 0.5)]]
colors = ["chartreuse", "cyan"]

lc = mc.LineCollection(lines, color=colors, linewidths=2)

fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(aspect=9 / 16)
ax.add_collection(lc)
ax.autoscale()

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()
