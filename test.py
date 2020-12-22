import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 1000)
y = np.exp(x)-1
z = 10000*np.exp(-x)*np.sin(x*10)
plt.plot(x, z+y)
plt.show()
