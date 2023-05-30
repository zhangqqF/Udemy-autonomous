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

# 几何尺寸
half_h_train = 5   # 该参数用来识别车子接住cube，但是lw的尺寸跟这里的尺寸不是一个意思，因此需要调节这个高度来适应线宽
half_w_train = 3
half_w_cube = 1
half_h_cube = 1

# 轨道坡度角
angle = np.pi/6  # 即30deg

# 重力切向分力
F_t = G*np.sin(angle)

# 初始化矩阵
len_t = len(t)
pos_x_train = np.zeros((trials, len_t))
pos_y_train = np.zeros((trials, len_t))
disp_train = np.zeros((trials, len_t))
v_train = np.zeros((trials, len_t))
a_train = np.zeros((trials, len_t))
pos_x_cube = np.zeros((trials, len_t))
pos_y_cube = np.zeros((trials, len_t))
e_p = np.zeros((trials, len_t))
e_i = np.zeros((trials, len_t))
e_d = np.zeros((trials, len_t))

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
def get_pos_cube(yn, half_h_train):
    x = random.uniform(0, 120)
    y = random.uniform(yn+20+half_h_train, yn+40+half_h_train)
    return x, y


trial = 0
catch = np.zeros((trials, len_t))   # 记录是否接住cube，以便画图
exchange = [1, -9, 1]   # 使在成功与失败之间切换，-9是随意的数字，意在使PID控制失去作用
while(trial < trials):
    # 获取cube的初始位置
    pos_x_cube_ref = get_pos_cube(yn, half_h_train)[0]
    pos_y_cube_ref = get_pos_cube(yn, half_h_train)[1]

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

    isCatched = False
    # 通过误差e，推算车子相关矩阵在每个t_i的值
    for i in range(1, len_t):

        # 上一步的误差
        e_p[trial][i-1] = pos_x_cube_ref - pos_x_train[trial][i-1]

        # 上一步的误差积分和微分
        if i > 1:
            e_i[trial][i-1] = e_i[trial][i-2] + (e_p[trial][i-1] + e_p[trial][i-2])/2*dt
            e_d[trial][i-1] = (e_p[trial][i-1] - e_p[trial][i-2])/dt

        # PID控制公式，用上一步的误差推算驱动力，进而推算出车子的运动矩阵!!!
        F_pid = Kp*e_p[trial][i-1] + Ki*e_i[trial][i-1] + Kd*e_d[trial][i-1]

        # 计算车子驱动合力：F_pid + 重力切向分力
        F_d = F_pid + F_t * exchange[trial]

        # 用牛顿第二定律F=ma，推算车子的加速度、速度、位移、坐标矩阵
        a_train[trial][i] = F_d/mass_train
        v_train[trial][i] = v_train[trial][i-1] + (a_train[trial][i] + a_train[trial][i-1])/2*dt
        disp_train[trial][i] = disp_train[trial][i-1] + (v_train[trial][i] + v_train[trial][i-1])/2*dt
        pos_x_train[trial][i] = disp_train[trial][i]*np.cos(angle)
        pos_y_train[trial][i] = disp_train[trial][i]*np.sin(angle) + y0     # 需要加上y0

        # 判断车子是否接住了cube
        # if (pos_y_cube[trial][i]-pos_y_train[trial][i]) - (half_h_cube+half_h_train) < (g*t[i]**2/2 - g*t[i-1]**2/2):
        if (isCatched == True) or (pos_y_cube[trial][i]-pos_y_train[trial][i]) - (half_h_cube+half_h_train) < 0.1:
            if (isCatched == True) or abs(pos_x_cube[trial][i] - pos_x_train[trial][i]) < (half_w_cube + half_w_train):
                isCatched = True
                catch[trial][i] = 1   # 后面动图显示用
                pos_y_cube[trial][i] = pos_y_train[trial][i] + (half_h_cube+half_h_train)
            else:
                catch[trial][i] = 2
       

    # 更新下一个trial车子的位置初始值，定为这个trial结束时的位置
    init_pos_x_train = pos_x_train[trial][-1]
    init_pos_y_train = pos_y_train[trial][-1]

    trial += 1
print(catch)
# -------------------------------- 作图区 --------------------------------

# 创建figure
fig = plt.figure(figsize=(16, 9), facecolor=(0.8,0.8,0.8))
fig.suptitle('PID Control @ Udemy Autonomous car', color='black', size=20)

# plot布局：4行3列
gs = gridspec.GridSpec(4, 3)

# 主plot，合并前三行前二列
ax_main = fig.add_subplot(gs[0:3, 0:2], facecolor=(0.9,0.9,0.9))
copyright = ax_main.text(0, 122, '© Refer Mark Misin Engineering @ zhangqq', size=12)  # 在主plot头上添加版权声明
rail, = ax_main.plot([x0, xn], [y0, yn], 'k', lw=8)                          # .plot返回的是一个list，"list,"表示该列表中的一个元素
train, = ax_main.plot([], [], 'b', lw=20)
cube, = ax_main.plot([], [], 'g', lw=14)
plt.xlim(0, xn)
plt.ylim(0, xn)
# 是否接住cube的弹出文字框
# box_succe = dict(boxstyle='square', fc=(0.9, 0.9, 0.9), ec='g', lw=1)
# success = ax_main.text(20, 60, '', size=20, color='g', bbox=box_succe)
success = ax_main.text(30, 80, '', size=20, color='g')
# box_fail = dict(boxstyle='square', fc=(0.9, 0.9, 0.9), ec='r', lw=1)
# fail = ax_main.text(20, 60, '', size=20, color='r', bbox=box_fail)
fail = ax_main.text(30, 80, '', size=20, color='r')

# 位移图
ax1v = fig.add_subplot(gs[0, 2], facecolor=(0.9,0.9,0.9))
disp_plot, = ax1v.plot([], [], 'b', lw=2, label='disp. on rails [m]')
plt.xlim(t0, t_end)
plt.legend(loc='lower left',fontsize='small')
plt.xlim(0, t_end)
lower = np.min(disp_train)      # np.min可以取到嵌套列表的最小值，纵坐标上下留了一些边距
upper = np.max(disp_train)
margin_low = abs(lower)*0.1
margin_upp = abs(upper)*0.1
plt.ylim(lower-margin_low, upper+margin_upp)

# 速度图
ax2v = fig.add_subplot(gs[1, 2], facecolor=(0.9,0.9,0.9))
v_plot, = ax2v.plot([], [], 'b', lw=2, label='velocity on rails [m/s]')
plt.xlim(t0, t_end)
plt.legend(loc='lower left',fontsize='small')
plt.xlim(0, t_end)
lower = np.min(v_train)
upper = np.max(v_train)
margin_low = abs(lower)*0.1
margin_upp = abs(upper)*0.1
plt.ylim(lower-margin_low, upper+margin_upp)

# 加速度图
ax3v = fig.add_subplot(gs[2, 2], facecolor=(0.9,0.9,0.9))
a_plot, = ax3v.plot([], [], 'b', lw=2, label='accel. om rails [m/s^2]')
plt.xlim(t0, t_end)
plt.legend(loc='lower left',fontsize='small')
plt.xlim(0, t_end)
lower = np.min(a_train)
upper = np.max(a_train)
margin_low = abs(lower)*0.1
margin_upp = abs(upper)*0.1
plt.ylim(lower-margin_low, upper+margin_upp)

# 误差
ax1h = fig.add_subplot(gs[3, 0], facecolor=(0.9,0.9,0.9))
ep_plot, = ax1h.plot([], [], 'b', lw=2, label='pos. x error [m]')
plt.xlim(t0, t_end)
plt.legend(loc='lower left',fontsize='small')
plt.xlim(0, t_end)
lower = np.min(e_p)
upper = np.max(e_p)
margin_low = abs(lower)*0.1
margin_upp = abs(upper)*0.1
plt.ylim(lower-margin_low, upper+margin_upp)

# 误差积分
ax2h = fig.add_subplot(gs[3, 1], facecolor=(0.9,0.9,0.9))
ei_plot, = ax2h.plot([], [], 'b', lw=2, label='sum of x error [m*s]')
plt.xlim(t0, t_end)
plt.legend(loc='lower left',fontsize='small')
plt.xlim(0, t_end)
lower = np.min(e_i)
upper = np.max(e_i)
margin_low = abs(lower)*0.1
margin_upp = abs(upper)*0.1
plt.ylim(lower-margin_low, upper+margin_upp)

# 误差微分
ax3h = fig.add_subplot(gs[3, 2], facecolor=(0.9,0.9,0.9))
ed_plot, = ax3h.plot([], [], 'b', lw=2, label='change of x error [m/s]')
plt.xlim(t0, t_end)
plt.legend(loc='lower left',fontsize='small')
plt.xlim(0, t_end)
lower = np.min(e_d)
upper = np.max(e_d)
margin_low = abs(lower)*0.1
margin_upp = abs(upper)*0.1
plt.ylim(lower-margin_low, upper+margin_upp)


# 动画更新函数
def update_plot(frame):
    train.set_data([pos_x_train[int(frame/len_t)][frame%len_t]-half_w_train,
                   pos_x_train[int(frame/len_t)][frame%len_t]+half_w_train],
                   [pos_y_train[int(frame/len_t)][frame%len_t], 
                    pos_y_train[int(frame/len_t)][frame%len_t]])
    
    cube.set_data([pos_x_cube[int(frame/len_t)][frame%len_t]-half_w_cube,
                  pos_x_cube[int(frame/len_t)][frame%len_t]+half_w_cube],
                  [pos_y_cube[int(frame/len_t)][frame%len_t],
                   pos_y_cube[int(frame/len_t)][frame%len_t]])
    
    if catch[int(frame/len_t)][frame%len_t] == 1:
        success.set_text('Yeah, I caught you bitch')
    elif catch[int(frame/len_t)][frame%len_t] == 2:
        fail.set_text('Fuck, Let you bitch slipped away')
    else:
        success.set_text('')
        fail.set_text('')

    disp_plot.set_data(t[0:frame%len_t], disp_train[int(frame/len_t)][0:frame%len_t])   
    v_plot.set_data(t[0:frame%len_t], v_train[int(frame/len_t)][0:frame%len_t])
    a_plot.set_data(t[0:frame%len_t], a_train[int(frame/len_t)][0:frame%len_t])

    ep_plot.set_data(t[0:frame%len_t], e_p[int(frame/len_t)][0:frame%len_t])
    ei_plot.set_data(t[0:frame%len_t], e_i[int(frame/len_t)][0:frame%len_t])
    ed_plot.set_data(t[0:frame%len_t], e_d[int(frame/len_t)][0:frame%len_t])

    return train, cube, disp_plot, v_plot, a_plot, ep_plot, ei_plot, ed_plot, success, fail


# 动态图
frames = len(t)*trials
ani = animation.FuncAnimation(fig,
                              update_plot,
                              frames=frames,
                              interval=20,      # 每一帧间隔时间 [ms]
                              repeat=False,     # 不循环播放
                              blit=True)        # 全部更新，而非只更新变化的部分


plt.show()