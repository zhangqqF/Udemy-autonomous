# ============================================
#   zhangqq
#   May-4. 2023
#   Chongqing
# ============================================


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import numpy as np
import support_file_car_zqq as sfc



support = sfc.SupportFilesCar()     # 实例化类
Ts = support.Ts                     # 
outputs = support.outputs           # number of outputs (psi, y)，状态量个数
hz = support.hz                     # horizon prediction period，预测周期
x_dot = support.x_dot               # constant longitudinal velocity @ 20m/s
time_length = support.time_lenght   # duration of the manoeuver



# Build up the reference signal vector, [psi y]

t = np.arange(0, time_length+Ts, Ts)    # 事件向量
r = support.r
f = support.f
Psi_ref, X_ref, Y_ref = support.trajectory_generator(t, r, f)



plt.plot(X_ref, Y_ref)
plt.show()