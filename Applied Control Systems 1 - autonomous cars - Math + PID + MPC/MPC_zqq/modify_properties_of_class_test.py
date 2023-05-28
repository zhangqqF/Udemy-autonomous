import class_test
import numpy as np


b = class_test.a
b.zqq = 2
c = class_test.a

# 实例b修改zqq后，后面的实例也修改了
print(b.zqq, c.zqq)


Q = np.matrix('1 0; 0 1')
print(Q)


# 修改私有变量
f = class_test.a
