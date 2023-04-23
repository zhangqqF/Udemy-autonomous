# ============================================
#   zhangqq
#   Apr-23, 2023
#   Chongqing
# ============================================


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import numpy as np
import random


# 仿真时间
t0 = 0
dt = 0.02
t_end = 5
t = np.arange(t0, t_end+dt, dt)

# 仿真次数
trials = 3

# PID控制参数
Kp = 300
Ki = 10
Kd = 300

# 车子基本信息
mass_train = 100
g = 10
G = mass_train*g
height_half_train = 10
width_half_train = 5

# cube尺寸
leng_cube = 5

# 轨道坡度角
angle = np.pi/6  # 即30deg

# 重力切向分力
F_t = G*np.sin(angle)

# 初始化矩阵
pos_x_train = np.zeros((trials, len(t)))
pos_y_train = np.zeros((trials, len(t)))
disp_train = np.zeros((trials, len(t)))
v_train = np.zeros((trials, len(t)))
a_train = np.zeros((trials, len(t)))
pos_x_cube = np.zeros((trials, len(t)))
pos_y_cube = np.zeros((trials, len(t)))
e_p = np.zeros((trials, len(t)))
e_i = np.zeros((trials, len(t)))
e_d = np.zeros((trials, len(t)))

# 轨道位置
x0 = 0
y0 = 5
xn = 120
yn = y0 + 120*np.tan(angle)

# 车子的初始状态，此处不定义位移，初始位移在while trials里面更新
init_pos_x_train = xn
init_pos_y_train = yn
init_v_train = 0
init_a_train = 0

# 函数: 随机生成cube的x、y坐标
def get_pos_cube(yn, height_half_train):
    x = random.uniform(0, 120)
    y = random.uniform(yn+200+height_half_train, yn+400+height_half_train)
    return x, y


global_trials = trials
while(trials > 0):
    # 获取cube的初始位置
    pos_x_cube_ref = get_pos_cube(yn, height_half_train)[0]
    pos_y_cube_ref = get_pos_cube(yn, height_half_train)[1]

    # 第几次trial
    trial = global_trials - trials

    # 完成cube的矩阵，x始终不变，y为初始位置减去下坠位移
    pos_x_cube[trial] = pos_x_cube_ref
    pos_y_cube[trial] = pos_y_cube_ref - g*t**2/2


    # 车子的初始位移
    init_disp_train = (init_pos_x_train**2 + init_pos_y_train**2)**0.5

    # 赋予车子坐标、位移、速度、加速度矩阵初始值
    pos_x_train[trial][0] = init_pos_x_train
    pos_y_train[trial][0] = init_pos_y_train
    disp_train[trial][0] = init_disp_train
    v_train[trial][0] = 0
    a_train[trial][0] = 0

    # 通过误差e，推算车子相关矩阵在每个t_i的值
    for i in range(1, len(t)):

        # 车子与cube的x坐标误差
        e_p[trial][i] = pos_x_cube_ref - pos_x_train[trial][i]
        e_i[trial][i] = (e_p[trial][i] + e_p[trial][i-1])/2*dt
        e_d[trial][i] = (e_p[trial][i] - e_p[trial][i-1])/dt

        # PID控制公式
        F_pid = Kp*e_p[trial][i] + Ki*e_i[trial][i] + Kd*e_d[trial][i]

        # 计算车子驱动合力：F_pid + 重力切向分力
        F_d = F_pid + F_t

        # 用牛顿第二定律F=ma，推算车子的加速度、速度、位移、坐标矩阵
        a_train[trial][i] = F_d/mass_train
        v_train[trial][i] = (a_train[trial][i] + a_train[trial][i-1])/2*dt
        disp_train[trial][i] = (v_train[trial][i] + v_train[trial][i-1])/2*dt
        pos_x_train[trial][i] = disp_train[trial][i]*np.cos(angle)
        pos_y_train[trial][i] = disp_train[trial][i]*np.sin(angle)

        # # 判断车子是否接住了cube
        # if pos_y_cube[trial][i]-pos_x_train[trial][i] < width_half_cube+width_half_train:

    # 更新下一个trial车子的位置初始值，定为这个trial结束时的位置
    init_pos_x_train = pos_x_train[trial][-1]
    init_pos_y_train = pos_y_train[trial][-1]

    trials = trials - 1


# -------------------------------- 作图区 --------------------------------

# 创建figure
fig = plt.figure(figsize=(16, 9), facecolor=(0.8,0.8,0.8))

# plot布局：4行3列
gs = gridspec.GridSpec(4, 3)

# 主plot，合并前三行前二列
ax_main = fig.add_subplot(gs[0:3, 0:2], facecolor=(0.9,0.9,0.9))
copyright = ax_main.text(0, 122, '© Refer Mark Misin Engineering', size=12)  # 在主plot头上添加版权声明
rail, = ax_main.plot([x0, xn], [y0, yn], 'k', lw=8)
train, = ax_main.plot([60-(width_half_train-height_half_train), 60+(width_half_train-height_half_train)], [10, 10], 'b', lw=2*height_half_train)
# train, = ax_main.plot([55, 65], [10, 10], 'b', lw=50)
# train, = ax_main.plot([55, 65], [10, 10], 'b', lw=2)
cube, = ax_main.plot([], [], 'k', lw=leng_cube)
plt.xlim(0, 120)
plt.ylim(0, 120)

ax1 = fig.add_subplot(gs[0, 2], facecolor=(0.9,0.9,0.9))
plt.xlim(t0, t_end)
plt.legend(loc='lower left',fontsize='small')





plt.show()
