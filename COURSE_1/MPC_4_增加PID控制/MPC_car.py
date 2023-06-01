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

x_ref, Y_ref, phi_ref = mpc.trajectory_generator(t, trajectory)
# ref = mpc.combine_ref([PHI_ref, Y_ref])
ref = mpc.combine_ref([phi_ref, Y_ref])


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
init_Y = Y_ref[0] + 10
init_states = np.array([init_dy, init_phi, init_dphi, init_Y])
states = np.zeros((len(t), len(init_states)))
states[0][:] = init_states

# 构造预测控制量矩阵
init_U = 0
U = np.zeros(len(t))
U[0] = init_U


'''
储存每一步的所有预测数据
'''
params = len(init_states)+len([init_U])
# X_aug_hz_ALL = np.zeros((hz*params, len(t)))  # hz是行，时间是列
phi_hz = np.zeros((len(t), hz))
Y_hz = np.zeros((len(t), hz))   # 注意这里不能用phi_hz, Y_hz或phi_hz = Y_hz的写法
phi_extra = np.zeros((hz, hz*params))
Y_extra = np.zeros((hz, hz*params))
for i in range(phi_extra.shape[0]):
    phi_extra[i, i*params + 1] = 1
    Y_extra[i, i*params + 3] = 1


# 预测
num_ref = int(len(ref)/len(t))  # 注意，一旦用了除法计算，int会变成float
sim_steps = len(t)
hz_temp = hz
PID_switch = mpc.PID_switch
for i in range(sim_steps):

    # 状态量和控制量的增广矩阵
    X_aug_T = np.concatenate((init_states, [init_U]), axis=0)
    X_tilde = np.transpose([X_aug_T])

    # 从参考矩阵中提取r矩阵
    if (i+hz)*num_ref > len(ref):   # 如果不够取，那就取到最后的值，而此时hz将变小，须重新计算系数矩阵
        r = ref[i*num_ref:]
        hz_temp -= 1
        
        [Q_bb, T_bb, R_bb, C_bb, A_hh] = mpc.bbar(A_tilde, B_tilde, C_tilde, D_tilde, hz_temp)
        [H_bb, F_bbt] = mpc.gradient(Q_bb, T_bb, R_bb, C_bb, A_hh)
        
    else:
        r = ref[i*num_ref: (i+hz)*num_ref]
    
    X_r = np.concatenate((X_aug_T, r), axis=0)   # 行向量，不是矩阵，因此下面矩阵计算时须加[]使其变为矩阵
    ft = np.transpose(np.matmul([X_r], F_bbt))   # 为矩阵乘法m x n · n x k故，须转置，但公式推导却不转置
    du = -np.matmul(np.linalg.inv(H_bb), ft)

    
    init_U = init_U + du[0][0]  # 取预测的第一个ΔU


    # ---------------------  PID控制φ、Y ----------------------
    if PID_switch == 1:
        old_states = states     # 保留上一步的states，用以下面计算积分和微分项
        if i == 0:
            pid_e_phi = 0
            pid_e_Y = 0
        else:
            pid_e_phi = phi_ref[i] - init_states[1]                         # 常数项
            pid_e_phi_old = phi_ref[i-1] - old_states[1]                    # 上一步常数项
            pid_e_phi_i = pid_e_phi + (pid_e_phi + pid_e_phi_old) / 2*Ts    # 积分项
            pid_e_phi_d = (pid_e_phi - pid_e_phi_old) / Ts                  # 微分项

            Kp_phi = mpc.Kp_yaw
            Ki_phi = mpc.Ki_yaw
            Kd_phi = mpc.Kd_yaw
            
            pid_phi = Kp_phi*pid_e_phi + Ki_phi*pid_e_phi_i + Kd_phi*pid_e_phi_d


            # PID control for Y
            pid_e_Y = Y_ref[i] - init_states[1]
            pid_e_Y_old = Y_ref[i-1] - old_states[1] 
            pid_e_Y_i = pid_e_Y + (pid_e_Y + pid_e_Y_old) / 2*Ts
            pid_e_Y_d = (pid_e_Y - pid_e_Y_old) / Ts

            Kp_Y = mpc.Kp_Y
            Ki_Y = mpc.Ki_Y
            Kd_Y = mpc.Kd_Y
            
            pid_Y = Kp_Y*pid_e_Y + Ki_Y*pid_e_Y_i + Kd_Y*pid_e_Y_d


            init_U = pid_phi

    # ---------------------  PID end ----------------------
    
    # 约束
    if init_U < -np.pi/6:
        init_U = -np.pi/6
    elif init_U > np.pi/6:
        init_U = np.pi/6
    else:
        pass
    
    U[i] = init_U

    
    # 用预测的U来计算状态量
    init_states = mpc.update_states(init_states, init_U)
    states[i][:] = init_states


    # 计算和存储预测的所有φ、Y的值
    X_aug_hz = np.matmul(C_bb, du) + np.matmul(A_hh, X_tilde)
    # X_aug_hz_ALL[:,i] = X_aug_hz[:,0]
    phi_hz[i, :hz_temp] = np.matmul(phi_extra[:hz_temp, :hz_temp*params], X_aug_hz)[:, 0]
    Y_hz[i, :hz_temp] = np.matmul(Y_extra[:hz_temp, :hz_temp*params], X_aug_hz)[:, 0]



'''
=============================================== animation
'''
fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=(0.8, 0.8, 0.8))
gs = gridspec.GridSpec(3, 3)
fig.suptitle('MPC Lateral Control @ Udemy Autonomous car', color='black', size=20)


# 軌跡
lane_width = mpc.lane_width
ax0 = fig.add_subplot(gs[0, :], facecolor=(0.9, 0.9, 0.9))
ref_trajectory = ax0.plot(x_ref, Y_ref, 'b', lw=2)
lane_1, = ax0.plot([x_ref[0], x_ref[-1]], [lane_width/2, lane_width/2], 'k', lw=0.2)
lane_2, = ax0.plot([x_ref[0], x_ref[-1]], [-lane_width/2, -lane_width/2], 'k', lw=0.2)
lane_3, = ax0.plot([x_ref[0], x_ref[-1]], [lane_width/2 + lane_width, lane_width/2 + lane_width], 'k', lw=0.2)
lane_4, = ax0.plot([x_ref[0], x_ref[-1]], [-lane_width/2 - lane_width, -lane_width/2 - lane_width], 'k', lw=0.2)
lane_5, = ax0.plot([x_ref[0], x_ref[-1]], [lane_width/2 + lane_width*2, lane_width/2 + lane_width*2], 'k', lw=3)
lane_6, = ax0.plot([x_ref[0], x_ref[-1]], [-lane_width/2 - lane_width*2, -lane_width/2 - lane_width*2], 'k', lw=3)
plt.xlim(x_ref[0], x_ref[-1])
plt.ylim()
# plt.ylim(-X_ref[frame_amount]/(n*(fig_x/fig_y)*2),X_ref[frame_amount]/(n*(fig_x/fig_y)*2))
plt.ylabel('Y-distance [m]', size=12)
copyright=ax0.text(0, 20,'© Mark Misin Engineering@zhangqq', size=15)
car_1, = ax0.plot([], [], 'k', lw=3)
car_predicted, = ax0.plot([], [], '-m', lw=2)
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
plt.ylabel('Y-distance [m]', fontsize=12)
bbox_props_angle = dict(boxstyle='square', fc=(0.9,0.9,0.9), ec='k', lw=1)
bbox_props_steer_angle = dict(boxstyle='square', fc=(0.9,0.9,0.9), ec='r', lw=1)
yaw_angle_text = ax1.text(23, 2, '', size='20', color='k', bbox=bbox_props_angle)
steer_angle_text = ax1.text(23, -2.5, '', size='20', color='r', bbox=bbox_props_steer_angle)


# 方向盤轉角曲綫（即δ）
ax2 = fig.add_subplot(gs[2, 0], facecolor=(0.9, 0.9, 0.9))
steering_angle, = plt.plot([], [], '-r', lw=1, label='steering - angle [rad]')
plt.xlabel('Time [s]', size=12)
plt.xlim(t[0], t[-1])
plt.ylim(np.min(U)-0.1, np.max(U)+0.1)
plt.grid(True)
plt.legend(loc='upper right', fontsize='small')


# 航向角曲綫（即yaw或φ）
ax3 = fig.add_subplot(gs[2, 1], facecolor=(0.9, 0.9, 0.9))
yaw_ref, = plt.plot(t, phi_ref, '-b', lw=1, label='yaw - reference [rad]')
yaw_determined, = plt.plot([], [], '-r', lw=1, label='yaw - determined [rad]')
yaw_predicted, = plt.plot([], [], '-m', lw=3, label='yaw - predicted [rad]')
plt.xlabel('Time [s]', size=12)
plt.xlim(t[0], t[-1])
plt.ylim(np.min(states[:,1])-0.1, np.max(states[:,1])+0.1)
# plt.ylim(np.min(phi_hz)-0.1, np.max(phi_hz)+0.1)
plt.grid(True)
plt.legend(loc='upper right', fontsize='small')


# Y位置曲綫
ax4 = fig.add_subplot(gs[2, 2], facecolor=(0.9, 0.9, 0.9))
Y_reference, = plt.plot(t, Y_ref, '-b', lw=1, label='Y - reference [m]')
Y_determined, = plt.plot([], [], '-r', lw=1, label='Y - determined [m]')
Y_predicted, = plt.plot([], [], '-m', lw=3, label='Y - predicted [m]]')
plt.xlabel('Time [s]', size=12)
plt.xlim(t[0], t[-1])
plt.grid(True)
plt.legend(loc='upper right', fontsize='small')


wheel_base_half = 1.5
wheel_radius = 0.4
Lf = mpc.lf
Lr = mpc.lr
# set_...内可以调用update_plot函数外的全局变量，其它不行
def update_plot(frame):

    x = x_ref[frame]
    Y = states[frame, 3]      # [frame][3]亦可
    phi = states[frame, 1]
    delta = U[frame]

    '''
    轨迹跟踪图
    '''
    car_1.set_data([x - Lr*np.cos(phi), x + Lf*np.cos(phi)],
                   [Y - Lr*np.sin(phi), Y + Lf*np.sin(phi)])

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

    # 方向盘转角
    steering_angle.set_data(t[:frame], U[:frame])


    # 增加图1的预测轨迹和行驶轨迹
    hz = 20
    if len(t)-frame < hz:
        hz = len(t) - frame 
    car_predicted.set_data(x_ref[frame: frame+hz], Y_hz[frame, :hz])
    car_determined.set_data(x_ref[:frame], states[:frame, 3])


    # yaw angle
    yaw_determined.set_data(t[:frame], states[:frame, 1])
    yaw_predicted.set_data(t[frame: frame+hz], phi_hz[frame,: hz])


    # Y
    Y_determined.set_data(t[:frame], states[:frame, 3])
    Y_predicted.set_data(t[frame: frame+hz], Y_hz[frame, :hz])



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
           steering_angle,
           # 预测轨迹
           car_predicted,
           car_determined,
           #
           yaw_determined,
           yaw_predicted,
           # 
           Y_determined,
           Y_predicted,
    ]
    return ret

ani = animation.FuncAnimation(fig, update_plot,
                              frames=sim_steps,          # frames這裏不支持表達式計算
                              interval=20,
                              repeat=False,
                              blit=True)
plt.show()
ani.save('MPC_car.gif', writer='pillow', fps=30)