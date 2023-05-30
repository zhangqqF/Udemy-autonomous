# ============================================
#   zhangqq
#   May-28. 2023
#   Chongqing
# ============================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from support_file_car_zqq import MPC_Controller


'''
trajectory
'''
mpc = MPC_Controller()              # 实例化类
Ts = mpc.Ts                         # 只有实例化类才能调用init中的变量
time = mpc.sim_time
t = np.arange(0, time+Ts, Ts)

trajectory = 3                      # MPC_Controller定义了三种轨迹，选择其中一条

X_ref, Y_ref, PHI_ref = mpc.trajectory_generator(t, trajectory)
ref = mpc.combine_ref([Y_ref, PHI_ref])


'''
matrix
'''
hz = 20         # horizon period, also known as prediction horizon
a, b, c, d, e, f = mpc.dynamic_func_abcdef()
A, B, C, D = mpc.state_ABCD(a, b, c, d, e, f)
Ad, Bd, Cd, Dd = mpc.discrete(A, B, C, D)
A_tilde, B_tilde, C_tilde, D_tilde = mpc.augmented_tilde(Ad, Bd, Cd, Dd)
Q_bb, T_bb, R_bb, C_bb, A_hh = mpc.bbar(A_tilde, B_tilde, C_tilde, D_tilde, hz)
H_bb, F_bbt = mpc.gradient(Q_bb, T_bb, R_bb, C_bb, A_hh)


'''
predictive
'''
# 构造预测状态矩阵
init_dy = 0
init_dphi = 0
init_phi = 0
init_y = Y_ref[0] + 10   # 初始y在参考y的10上面位置
init_states = np.array([init_dy, init_phi, init_dphi, init_y])
states = np.zeros((len(t), len(init_states)))
states[0] = init_states

# 构造预测控制量矩阵
init_U = 0
U = np.zeros(len(t))
U[0] = init_U

# 预测
num_ref = len(ref)/len(t)
sim_steps = len(t)
hz_temp = hz
for i in range(sim_steps-1):

    # 状态量和控制量的增广矩阵
    X_aug_T = np.concatenate((init_states, [init_U]), axis=0)
    X_tilde = np.transpose(np.concatenate((init_states, [init_U]), axis=0))
    
    # 从参考矩阵中提取r矩阵
    if (i+hz)*num_ref > len(ref):   # 如果不够取，那就取到最后的值，而此时hz将变小，须重新计算系数矩阵
        r = ref[int(i*num_ref):]
        hz_temp -= 1

        [Q_bb, T_bb, R_bb, C_bb, A_hh] = mpc.bbar(A_tilde, B_tilde, C_tilde, D_tilde, hz_temp)
        [H_bb, F_bbt] = mpc.gradient(Q_bb, T_bb, R_bb, C_bb, A_hh)
        print(hz_temp)
    else:
        r = ref[int(i*num_ref): int((i+hz)*num_ref)]
    
    X_r = np.concatenate((X_aug_T, r), axis=0)   # 行向量，不是矩阵，因此下面矩阵计算时须加[]使其变为矩阵
    ft = np.transpose(np.matmul([X_r], F_bbt))   # 为矩阵乘法m x n · n x k故，须转置，但公式推导却不转置
    du = -np.matmul(np.linalg.inv(H_bb), ft)
    
    init_U = init_U + du[0][0]  # 取预测的第一个ΔU
    
    # 约束
    if init_U < -np.pi/6:
        init_U = -np.pi/6
    elif init_U > np.pi/6:
        init_U = np.pi/6
    else:
        pass
    
    U[i+1] = init_U
    
    # 用预测的U来计算状态量
    init_states = mpc.update_states(init_states, init_U)
    states[i+1][:] = init_states



'''
animation
'''
fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=(0.8, 0.8, 0.8))
gs = gridspec.GridSpec(3, 3)


# 軌跡
lane_width = mpc.lane_width
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
yaw_ref, = plt.plot(t, PHI_ref, '-b', lw=1, label='yaw reference [deg]')
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
Lf = mpc.lf
Lr = mpc.lr
# update函数可以调用全局变量
def update_plot(frame):

    x = X_ref[frame]
    y = states[frame, 3]
    phi = states[frame,1]
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

ani = animation.FuncAnimation(fig, update_plot,
                              frames=sim_steps,          # frames這裏不支持表達式計算
                              interval=20,
                              repeat=False,
                              blit=True)
plt.show()