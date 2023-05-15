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
Num_state = support.outputs           # number of outputs (psi, y)，状态量个数
hz = support.hz                     # horizon prediction period，预测周期
x_dot = support.x_dot               # constant longitudinal velocity @ 20m/s
time_length = support.time_lenght   # duration of the manoeuver


t = np.arange(0, time_length+Ts, Ts)    # 时间向量
r = support.r
f = support.f
PHI, X, Y = support.trajectory_generator(t, r, f)
sin_length = len(t)                 # Number of control loop iterations

# Build up the reference signal vector:
# refSignals = [PHI0, Y0, PHIi, Yi, ..., etc.]
refSignals = np.zeros(len(X)*Num_state)
k = 0
for i in range(0, len(refSignals), Num_state):
    refSignals[i] = PHI[k]
    refSignals[i+1] = Y[k]
    k += 1

# Load the initial states
# float and not integer
y = 0.
y_dot = 0.
phi = 0.
phi_dot = 0.
states = np.array([y_dot, phi, phi_dot, y])
statesTotal = np.zeros((len(t), len(states)))   # 行位时间，列为状态量
statesTotal[0][:] = states                      # 赋予初始值


# 控制量U，此处U只有delta一个参数
U = np.zeros(len(t))
U[0] = 0


Ad, Bd, Cd, Dd = support.state_space()
H_bbar, F_bbar, C_bbar, A_hhat = support.mpc_simplification(Ad, Bd, Cd, Dd, hz)


# 預測
k = 0
for i in range(0, sin_length-1):
    x_aug_t = np.transpose([np.concatenate((states, [U[0]]), axis=0)])
    print(x_aug_t)

    k = k + Num_state
    if k + Num_state*hz < len(refSignals):
        r = refSignals[k: k + Num_state*hz]
    else:
        r = refSignals[k: len(refSignals)]
        hz -= 1

    if hz < support.hz:
        H_bb, F_bb, C_bb, A_hh = support.mpc_simplification(Ad, Bd, Cd, Dd, hz)

    ft = 0





################################ ANIMATION LOOP ###############################


plt.plot(X, Y)
plt.show()