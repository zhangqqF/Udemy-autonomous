import os
import numpy as np
import matplotlib.pyplot as plt



Ad = np.zeros((4, 4))
Bd = np.ones((4,1))
c = np.ones((1,5))
# c = np.concatenate((a, b), axis=1)

temp1 = np.zeros((np.size(Bd, 1), np.size(Ad, 1)))
temp2 = np.identity(np.size(Bd, 1))
temp = np.concatenate((temp1 ,temp2), axis=1)


a = [1,2,3,4]
print(np.linspace(0, 5, 4))

xx = np.linspace(-1000, 1000, 5000)
yy = np.arctan(xx)
plt.plot(xx, yy)
# plt.plot(xx, np.linspace(np.pi/2, np.pi/2, 5000))
# plt.plot(xx, np.linspace(-np.pi/2, -np.pi/2, 5000))
plt.yticks([-np.pi/2, 0, np.pi/2], [r'$-\frac{\pi}{2}$', 0, 'np.pi/2'])
# plt.show()

a = 4
a -= 1
print(a)