# ============================================
#   zhangqq
#   May-4. 2023
#   Chongqing
# ============================================


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import numpy as np
# import support_file_car_zqq as sfc
import support_files_car as sfc



support = sfc.SupportFilesCar()     # 实例化类
Ts = support.Ts                     # 
outputs = support.outputs           # number of outputs (psi, y)，状态量个数
hz = support.hz                     # horizon prediction period，预测周期
x_dot = support.x_dot               # constant longitudinal velocity @ 20m/s
time_length = support.time_length   # duration of the manoeuver


t = np.arange(0, time_length+Ts, Ts)    # 时间向量
r = support.r
f = support.f
phi_ref, X_ref, Y_ref = support.trajectory_generator(t, r, f)
sin_length = len(t)                 # Number of control loop iterations

# Build up the reference signal vector:
# refSignals = [PHI0, Y0, PHIi, Yi, ..., etc.]
refSignals = np.zeros(len(X_ref)*outputs)
k = 0
for i in range(0, len(refSignals), outputs):
    refSignals[i] = phi_ref[k]
    refSignals[i+1] = Y_ref[k]
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
H_bb, F_bbt, C_bb, A_hh = support.mpc_simplification(Ad, Bd, Cd, Dd, hz)



# 預測
k = 0
for i in range(0, sin_length-1):
    X_tilde_k = np.transpose([np.concatenate((states, [U0]), axis=0)])  # [dy φ dφ y δ]_0 注意行向量只能axis=0，並擴展列，與矩陣連接并不相同
    

    # 从参考信号中取出一小段（长度为hz）信号
    # 儅hz=1時，則每次取一組參考值，剛好與i同時取完
    k = k + outputs # Num_state=2，不取φ_ref_0, y_ref_0
    if (k + outputs*hz) <= len(refSignals): # refSinals=[φ_ref_0, y_ref_0, φ_ref_0.02, y_ref_0.02, ...]
        r = refSignals[k: k + outputs*hz]
    else:       # 最後如果不夠取，就取到末尾即可，這樣hz會少一項，的重新計算bb矩陣
        r = refSignals[k: len(refSignals)]
        hz -= 1

    if hz < support.hz:
        H_bb, F_bbt, C_bb, A_hh = support.mpc_simplification(Ad, Bd, Cd, Dd, hz)


    # 通過梯度公式求Δδ
    X_r = np.concatenate((np.transpose(X_tilde_k)[0][:], r), axis=0) # len(矩陣)=行數
    ft = np.matmul([X_r], F_bbt)
    du = -np.matmul(np.linalg.inv(H_bb), np.transpose(ft))


    # 帶入預測公式計算預測值
    # x_aug_opt = np.matmul(C_bb, du) + np.matmul(A_hh, X_tilde_k)
    # # 提取預測值中的φ和y
    # phi_opt = np.matmul(C_phi_opt[0:hz, 0:(len(states)+np.size(U0)) *hz], x_aug_opt)
    # Y_opt = np.matmul(C_Y_opt[0:hz, 0:(len(states)+np.size(U0)) *hz], x_aug_opt)


    # # 儲存所有預測值
    # phi_opt = np.transpose((phi_opt))[0]
    # psi_opt_total[i+1][0:hz] = phi_opt
    # Y_opt = np.transpose((Y_opt))[0]
    # Y_opt_total[i+1][0:hz] = Y_opt


    # 更新輸入
    U0 = U0 + du[0][0]


    # 控制量約束
    if U0 < -np.pi/6:
        U0 = -np.pi/6
    elif U0 > np.pi/6:
        U0 = np.pi/6
    else:
        pass

    U[i+1] = U0


    states = support.open_loop_new_states(states, U0)
    statesTotal[i+1][:] = states


################################ ANIMATION LOOP ###############################


fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=(0.8, 0.8, 0.8))
gs = gridspec.GridSpec(3, 3)


# 軌跡
lane_width = support.lane_width
ax0 = fig.add_subplot(gs[0, :], facecolor=(0.9, 0.9, 0.9))
ref_trajectory = ax0.plot(X_ref, Y_ref, 'b', lw=2)
lane_1, = ax0.plot([X_ref[0], X_ref[-1]], [lane_width/2, lane_width/2], 'k', lw=0.2)
lane_2, = ax0.plot([X_ref[0], X_ref[-1]], [-lane_width/2, -lane_width/2], 'k', lw=0.2)
lane_3, = ax0.plot([X_ref[0], X_ref[-1]], [lane_width/2 + lane_width, lane_width/2 + lane_width], 'k', lw=0.2)
lane_4, = ax0.plot([X_ref[0], X_ref[-1]], [-lane_width/2 - lane_width, -lane_width/2 - lane_width], 'k', lw=0.2)
lane_5, = ax0.plot([X_ref[0], X_ref[-1]], [lane_width/2 + lane_width*2, lane_width/2 + lane_width*2], 'k', lw=3)
lane_6, = ax0.plot([X_ref[0], X_ref[-1]], [-lane_width/2 - lane_width*2, -lane_width/2 - lane_width*2], 'k', lw=3)
plt.xlim(X_ref[0], X_ref[-1])
plt.ylim()
# plt.ylim(-X_ref[frame_amount]/(n*(fig_x/fig_y)*2),X_ref[frame_amount]/(n*(fig_x/fig_y)*2))
plt.ylabel('Y-distance [m]', size=12)
copyright=ax0.text(0, 20,'© Mark Misin Engineering@zhangqq', size=15)
car_1, = ax0.plot([], [], 'r', lw=3)
car_predicted, = ax0.plot([], [], '-m', lw=1)
car_determined, = ax0.plot([], [], '-r', lw=1)


# 汽車局部運動圖
ax1 = fig.add_subplot(gs[1, :], facecolor=(0.9, 0.9, 0.9))
neutral_line, = ax1.plot([-50,50], [0,0], 'k', lw=1)
car_1_body, = ax1.plot([], [], 'k', lw=3)
car_1_axle_f, = ax1.plot([], [], 'k', lw=3)
car_1_axle_r, = ax1.plot([], [], 'k', lw=3)
car_1_wheel_fl, = ax1.plot([], [], 'r', lw=8)
car_1_wheel_fr, = ax1.plot([], [], 'r', lw=8)
car_1_wheel_rl, = ax1.plot([], [], 'k', lw=8)
car_1_wheel_rr, = ax1.plot([], [], 'k', lw=8)
car_1_extension_yaw, = ax1.plot([], [], '--k', lw=1)
car_1_extension_steer, = ax1.plot([], [], '--r', lw=1)
xmin = -5
xmax = 30
plt.xlim(xmin, xmax)
plt.ylim(-(xmax-xmin)/(3*(16/9)*2), (xmax-xmin)/(3*(16/9)*2))
plt.ylabel('Y-distance [m]', fontsize=15)
bbox_props_angle = dict(boxstyle='square', fc=(0.9,0.9,0.9), ec='k', lw=1)
bbox_props_steer_angle = dict(boxstyle='square', fc=(0.9,0.9,0.9), ec='r', lw=1)
yaw_angle_text = ax1.text(22, 1.5, '', size='20', color='k', bbox=bbox_props_angle)
steer_angle_text = ax1.text(22, -2, '', size='20', color='r', bbox=bbox_props_steer_angle)


# 方向盤轉角曲綫（即δ）
ax2 = fig.add_subplot(gs[2, 0], facecolor=(0.9, 0.9, 0.9))
steering_angle, = plt.plot([], [], '-r', lw=1, label='steering angle [deg]')
plt.xlabel('Time [s]', size=12)
plt.xlim(t[0], t[-1])
plt.grid(True)
plt.legend(loc='upper right', fontsize='small')


# 航向角曲綫（即yaw或φ）
ax3 = fig.add_subplot(gs[2, 1], facecolor=(0.9, 0.9, 0.9))
yaw_ref, = plt.plot(t, phi_ref, '-b', lw=1, label='yaw reference [deg]')
yaw_ang, = plt.plot([], [], '-r', lw=1, label='yaw angle [deg]')
yaw_pre, = plt.plot([], [], '-m', lw=3, label='yaw predicted [deg]')
plt.xlabel('Time [s]', size=12)
plt.xlim(t[0], t[-1])
plt.grid(True)
plt.legend(loc='upper right', fontsize='small')


# Y位置曲綫
ax4 = fig.add_subplot(gs[2, 2], facecolor=(0.9, 0.9, 0.9))
Y_ref, = plt.plot(t, Y_ref, '-b', lw=1, label='Y - reference [m]')
Y_pos, = plt.plot([], [], '-r', lw=1, label='Y - position [m]')
Y_pre, = plt.plot([], [], '-m', lw=3, label='Y - predicted [m]]')
plt.xlabel('Time [s]', size=12)
plt.xlim(t[0], t[-1])
plt.grid(True)
plt.legend(loc='upper right', fontsize='small')


wheel_base_half = 1.5
wheel_radius = 0.4
Lf = support.lf
Lr = support.lr
# update函数可以调用全局变量
def update_plot(frame):

    x = X_ref[frame]
    y = statesTotal[frame, 3]
    phi = statesTotal[frame,1]
    delta = U[frame]

    '''
    轨迹跟踪图
    '''
    car_1.set_data([x - Lr*np.cos(phi), x + Lf*np.cos(phi)],
                   [y - Lr*np.sin(phi), y + Lf*np.sin(phi)])

    '''
    汽车转向细节图
    '''
    # 车身
    car_1_body.set_data([-Lr*np.cos(phi), Lf*np.cos(phi)],
                        [-Lr*np.sin(phi), Lf*np.sin(phi)])
    # 前轴
    car_1_axle_f.set_data([Lf*np.cos(phi) - wheel_base_half*np.sin(phi),
                           Lf*np.cos(phi) + wheel_base_half*np.sin(phi)],
                          [Lf*np.sin(phi) + wheel_base_half*np.cos(phi),
                           Lf*np.sin(phi) - wheel_base_half*np.cos(phi)])
    # 左前轮
    car_1_wheel_fl.set_data([Lf*np.cos(phi) - wheel_base_half*np.sin(phi) - wheel_radius*np.cos(phi+delta),
                             Lf*np.cos(phi) - wheel_base_half*np.sin(phi) + wheel_radius*np.cos(phi+delta)],
                            [Lf*np.sin(phi) + wheel_base_half*np.cos(phi) - wheel_radius*np.sin(phi+delta),
                             Lf*np.sin(phi) + wheel_base_half*np.cos(phi) + wheel_radius*np.sin(phi+delta)])
    # 右前轮
    car_1_wheel_fr.set_data([Lf*np.cos(phi) + wheel_base_half*np.sin(phi) - wheel_radius*np.cos(phi+delta),
                             Lf*np.cos(phi) + wheel_base_half*np.sin(phi) + wheel_radius*np.cos(phi+delta)],
                            [Lf*np.sin(phi) - wheel_base_half*np.cos(phi) - wheel_radius*np.sin(phi+delta),
                             Lf*np.sin(phi) - wheel_base_half*np.cos(phi) + wheel_radius*np.sin(phi+delta)])
    # 后轴
    car_1_axle_r.set_data([-(Lr*np.cos(phi) - wheel_base_half*np.sin(phi)),
                           -(Lr*np.cos(phi) + wheel_base_half*np.sin(phi))],
                          [-(Lr*np.sin(phi) + wheel_base_half*np.cos(phi)),
                           -(Lr*np.sin(phi) - wheel_base_half*np.cos(phi))])
    # 左后轮
    car_1_wheel_rl.set_data([-(Lr*np.cos(phi) - wheel_base_half*np.sin(phi) - wheel_radius*np.cos(phi)),
                             -(Lr*np.cos(phi) - wheel_base_half*np.sin(phi) + wheel_radius*np.cos(phi))],
                            [-(Lr*np.sin(phi) + wheel_base_half*np.cos(phi) - wheel_radius*np.sin(phi)),
                             -(Lr*np.sin(phi) + wheel_base_half*np.cos(phi) + wheel_radius*np.sin(phi))])
    # 右后轮
    car_1_wheel_rr.set_data([-(Lr*np.cos(phi) + wheel_base_half*np.sin(phi) - wheel_radius*np.cos(phi)),
                             -(Lr*np.cos(phi) + wheel_base_half*np.sin(phi) + wheel_radius*np.cos(phi))],
                            [-(Lr*np.sin(phi) - wheel_base_half*np.cos(phi) - wheel_radius*np.sin(phi)),
                             -(Lr*np.sin(phi) - wheel_base_half*np.cos(phi) + wheel_radius*np.sin(phi))])
    # steering角度延长线
    car_1_extension_yaw.set_data([0, (Lf+40)*np.cos(phi)],
                                 [0, (Lf+40)*np.sin(phi)])
    # 汽车航向角延长线
    car_1_extension_steer.set_data([Lf*np.cos(phi), Lf*np.cos(phi) + (0.5+40)*np.cos(phi + delta)],
                                   [Lf*np.sin(phi), Lf*np.sin(phi) + (0.5+40)*np.sin(phi + delta)])
    # steering和φ角度值显示
    yaw_angle_text.set_text('yaw: %s rad' %str(round(phi, 2)))
    steer_angle_text.set_text('steer: %s rad' %str(round(delta, 2)))


    ret = [car_1,
           car_1_body,
           car_1_axle_f,
           car_1_wheel_fl,
           car_1_wheel_fr,
           car_1_axle_r,
           car_1_wheel_rl,
           car_1_wheel_rr,
           car_1_extension_yaw,
           car_1_extension_steer,
           yaw_angle_text,
           steer_angle_text,
    ]
    return ret


frames = int(time_length/Ts)
ani = animation.FuncAnimation(fig, update_plot,
                              frames=frames,          # frames這裏不支持表達式計算
                              interval=20,
                              repeat=False,
                              blit=True)
plt.show()