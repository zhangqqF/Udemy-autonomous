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
U0 = 0
U = np.zeros(len(t))




psi_opt_total = np.zeros((len(t), hz))
Y_opt_total = np.zeros((len(t), hz))

# C_bb矩阵，分别提取phi和Y（分别在各自位置值1）
# 此处不用原代码，不好理解
C_phi_opt = np.zeros((hz, (len(states)+np.size(U0)) *hz))
C_Y_opt = np.zeros((hz, (len(states)+np.size(U0)) *hz))
for i in range(0, hz):
    C_phi_opt[i][i*5 + 1] = 1   # [dy phi dphi y delta] phi索引为1
    C_Y_opt[i][i*5 + 3] = 1     # # [dy phi dphi y delta] y索引为3



Ad, Bd, Cd, Dd = support.state_space()
H_bb, F_bbt, C_bb, A_hh = support.mpc_simplification2(Ad, Bd, Cd, Dd, hz)



# 預測
k = 0
for i in range(0, sin_length-1):
    X_tilde_k = np.transpose([np.concatenate((states, [U0]), axis=0)])  # [dy φ dφ y δ]_0 注意行向量只能axis=0，並擴展列，與矩陣連接并不相同
    

    # 从参考信号中取出一小段（长度为hz）信号
    # 儅hz=1時，則每次取一組參考值，剛好與i同時取完
    k = k + Num_state # Num_state=2，不取φ_ref_0, y_ref_0
    if k + Num_state*hz <= len(refSignals): # refSinals=[φ_ref_0, y_ref_0, φ_ref_0.02, y_ref_0.02, ...]
        r = refSignals[k: k + Num_state*hz]
    else:       # 最後如果不夠取，就取到末尾即可，這樣hz會少一項，的重新計算bb矩陣
        r = refSignals[k: len(refSignals)]
        hz -= 1

    if hz < support.hz:
        H_bb, F_bbt, C_bb, A_hh = support.mpc_simplification(Ad, Bd, Cd, Dd, hz)


    # 通過梯度公式求Δδ
    X_r = np.matmul(np.concatenate((np.transpose(X_tilde_k)[0][0: len(X_tilde_k)], r), axis=0), F_bbt) # len(矩陣)=行數
    du = -np.matmul(np.linalg.inv(H_bb), np.transpose([X_r]))

    # 帶入預測公式計算預測值
    x_aug_opt = np.matmul(C_bb, du) + np.matmul(A_hh, X_tilde_k)

    # 提取預測值中的φ和y
    phi_opt = np.matmul(C_phi_opt[0:hz, 0:(len(states)+np.size(U[0])) *hz], x_aug_opt)
    Y_opt = np.matmul(C_Y_opt[0:hz, 0:(len(states)+np.size(U[0])) *hz], x_aug_opt)


    # 儲存所有預測值
    psi_opt = np.transpose((phi_opt))[0]
    psi_opt_total[i+1][0:hz] = np.transpose(phi_opt) #######???????????????????????????????????????????????????
    Y_opt = np.transpose((Y_opt))[0]
    Y_opt_total[i+1][0:hz] = Y_opt

    # 更新輸入
    U0 = U0 + du[0][0]



    
    # ---------------------- PID 控制 ---------------------- #
    PID_switch = support.PID_switch
    if PID_switch == 1:
        if i == 0:
            E_phi = 0
            E_y = 0
        else:
            pass
    # ---------------------- PID End ---------------------- #





################################ ANIMATION LOOP ###############################


frames = int(time_length/Ts)
Lf = support.lf
Lr = support.lr


def update_plot(frame):
    pass


fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=(0.8, 0.8, 0.8))
gs = gridspec.GridSpec(3, 3)


lane_width = support.lane_width
ax0 = fig.add_subplot(gs[0, :], facecolor=(0.9, 0.9, 0.9))
ref_trajectory, = ax0.plot(X, Y, 'b', lw=2)
lane_1, = ax0.plot([X[0], X[-1]], [lane_width/2, lane_width/2], 'k', lw=0.2)
lane_2, = ax0.plot([X[0], X[-1]], [-lane_width/2, -lane_width/2], 'k', lw=0.2)
lane_3, = ax0.plot([X[0], X[-1]], [lane_width/2 + lane_width, lane_width/2 + lane_width], 'k', lw=0.2)
lane_4, = ax0.plot([X[0], X[-1]], [-lane_width/2 - lane_width, -lane_width/2 - lane_width], 'k', lw=0.2)
lane_5, = ax0.plot([X[0], X[-1]], [lane_width/2 + lane_width*2, lane_width/2 + lane_width*2], 'k', lw=0.2)
lane_6, = ax0.plot([X[0], X[-1]], [-lane_width/2 - lane_width*2, -lane_width/2 - lane_width*2], 'k', lw=0.2)
plt.xlim(X[0], X[-1])
plt.ylim()
# plt.ylim(-X_ref[frame_amount]/(n*(fig_x/fig_y)*2),X_ref[frame_amount]/(n*(fig_x/fig_y)*2))
plt.ylabel('Y-distance [m]', size=12)
copyright=ax0.text(0, 20,'© Mark Misin Engineering@zhangqq', size=15)
car_1, = ax0.plot([], [], 'r', lw=3)
car_predicted, = ax0.plot([], [], '-m', lw=1)
car_determined, = ax0.plot([], [], '-r', lw=1)



ax1 = fig.add_subplot(gs[1, :], facecolor=(0.9, 0.9, 0.9))


ax2 = fig.add_subplot(gs[2, 0], facecolor=(0.9, 0.9, 0.9))
steering_angle, = plt.plot([], [], '-r', lw=1, label='steering angle [deg]')
plt.xlabel('Time [s]', size=12)
plt.xlim(t[0], t[-1])
plt.grid(True)
plt.legend(loc='upper right', fontsize='small')


ax3 = fig.add_subplot(gs[2, 1], facecolor=(0.9, 0.9, 0.9))
yaw_ref, = plt.plot([], [], '-b', lw=1, label='steering angle [deg]')
yaw, = plt.plot([], [], '-r', lw=1, label='yaw angle [deg]')
phi_predicted, = plt.plot([], [], '-m', lw=3, label='φ predicted [deg]')
plt.xlabel('Time [s]', size=12)
plt.xlim(t[0], t[-1])
plt.grid(True)
plt.legend(loc='upper right', fontsize='small')


ax4 = fig.add_subplot(gs[2, 2], facecolor=(0.9, 0.9, 0.9))
Y_ref, = plt.plot([], [], '-b', lw=1, label='steering angle [deg]')
y_pos, = plt.plot([], [], '-r', lw=1, label='yaw angle [deg]')
phi_predicted, = plt.plot([], [], '-m', lw=3, label='φ predicted [deg]')
plt.xlabel('Time [s]', size=12)
plt.xlim(t[0], t[-1])
plt.grid(True)
plt.legend(loc='upper right', fontsize='small')



plt.plot(X, Y)
plt.show()