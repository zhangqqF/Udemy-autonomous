import os
import numpy as np



Ad = np.zeros((4, 4))
Bd = np.ones((4,1))
c = np.ones((1,5))
# c = np.concatenate((a, b), axis=1)

temp1 = np.zeros((np.size(Bd, 1), np.size(Ad, 1)))
temp2 = np.identity(np.size(Bd, 1))
temp = np.concatenate((temp1 ,temp2), axis=1)



print(Bd.shape[1])