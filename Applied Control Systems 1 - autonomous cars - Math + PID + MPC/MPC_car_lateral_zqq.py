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




psi_opt_total = np.zeros((len(t), hz))
Y_opt_total = np.zeros((len(t), hz))

C_psi_opt = np.zeros((hz, (len(states)+np.size(U[0])) * hz))
for i in range(1, hz+1):
    C_psi_opt[i-1][i+4*(i-1)] = 1

C_Y_opt = np.zeros((hz, (len(states)+np.size(U[0])) * hz))
for i in range(3, hz+3):
    C_Y_opt[i-3][i + 4*(i-3)] = 1






Ad, Bd, Cd, Dd = support.state_space()
H_bbar, F_bbar, C_bbar, A_hhat = support.mpc_simplification(Ad, Bd, Cd, Dd, hz)


# 預測
k = 0
for i in range(0, sin_length-1):
    x_aug_t = np.transpose([np.concatenate((states, [U[0]]), axis=0)])  # [dy φ dφ y δ]_0
    # print(x_aug_t)

    k = k + Num_state # Num_state=2
    if k + Num_state*hz < len(refSignals): # refSinals=[φ_ref_0, y_ref_0, φ_ref_0.02, y_ref_0.02, ...]
        r = refSignals[k: k + Num_state*hz]
    else:
        r = refSignals[k: len(refSignals)]
        hz -= 1

    if hz < support.hz:
        H_bb, F_bb, C_bb, A_hh = support.mpc_simplification(Ad, Bd, Cd, Dd, hz)

    F_bbt = np.transpose(F_bb)
    ft = np.matmul(np.concatenate((np.transpose(x_aug_t)[0][0:len(x_aug_t)], r), axis=0), F_bbt)
    du = -np.matmul(np.linalg.inv(H_bb), np.transpose([ft]))
    x_aug_opt = np.matmul(C_bb, du) + np.matmul(A_hh, x_aug_t)
    psi_opt = np.matmul(C_psi_opt[0:hz,0:(len(states) + np.size(U[0]))*hz], x_aug_opt)
    Y_opt = np.matmul(C_Y_opt[0:hz, 0:(len(states) + np.size(U[0]))*hz], x_aug_opt)

    psi_opt = np.transpose((psi_opt))[0]
    psi_opt_total[i+1][0:hz] = psi_opt
    Y_opt = np.transpose((Y_opt))[0]
    Y_opt_total[i+1][0:hz] = Y_opt



################################ ANIMATION LOOP ###############################


plt.plot(X, Y)
plt.show()