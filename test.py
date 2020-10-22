import numpy as np
import matplotlib.pyplot as plt

score = [32, 2, 5, 6, 5, 7, 676, 57, 65, 34, 52, 45, 2, 43, 6, 567, 657, 456, 3, 63, 543, 6, 457, 4, 567]
score = np.array(score)

window = 10
samplerate = 250
skiprate = 15

w = np.ones(window) / window
score = np.convolve(score, w, mode="same")
xl = [i * skiprate / samplerate for i in range(len(score))]

fig = plt.figure(figsize=[15, 5])
ax = fig.add_subplot(111)
ax.plot(xl, score)
ax.set_ylim([0, 600])
ax.set_xlim([0, len(score)* skiprate / samplerate])
ax.set_xlabel("time[*sec]")
ax.set_ylabel("anomary score")
ax.legend()
plt.show()