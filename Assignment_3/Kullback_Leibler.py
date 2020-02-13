import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 5, 2000)

plt.plot(x, np.log(x),label = r'y = ln x')
plt.plot(x, x-1, label = r'y = x-1')
plt.xlim(0, 2)
plt.ylim(-2, 2)
plt.scatter(1,0, c = "r", marker = "*", label='x=1')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()