# ============================================
#   zhangqq
#   Apr 18, 2023
#   Chongqing
# ============================================

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import numpy as np
import random


# 初始参数
trials = 3
incl_angle = np.pi/6*1
g = 10
mass_cart = 100


K_p = 300
K_d = 300
K_i = 10


trials_global = trials


def set_x_ref(incl_angle):
    '''
    随机生成掉落快的位置
    '''
    rand_h = random.uniform(0, 120) # 随机均布0到120内的数
    rand_v = random.uniform(20 + 120*np.tan(incl_angle) + 6.5, 40 + 120*np.tan(incl_angle) + 6.5)
    return rand_h, rand_v


dt = 0.02
t0 = 0
t_end = 5
t = np.arange(t0, t_end, dt)

grav = mass_cart*g


# 预定义数组：
#     1. 可以先定义空数组，然后随着迭代再append，但是这样的程序运行很慢
#     2. 如果预先知道最终数组的大小，那么可以先预定一个相同大小的空数组，再随着迭代逐步替换对应的数值
rail_disp = np.zeros((trials, len(t)))
rail_velo = np.zeros((trials, len(t)))
rail_acce = np.zeros((trials, len(t)))
train_pos_x = np.zeros((trials, len(t)))
train_pos_y = np.zeros((trials, len(t)))
e = np.zeros((trials, len(t)))
e_dot = np.zeros((trials, len(t)))
e_int = np.zeros((trials, len(t)))

cube_pos_x = np.zeros((trials, len(t)))
cube_pos_y = np.zeros((trials, len(t)))


F_ga_t = grav*np.sin(incl_angle)
init_pos_x = 120
init_pos_y = 120*np.tan(incl_angle) + 6.5
init_rail_disp = (init_pos_x**2 + init_pos_y**2)**0.5
init_rail_velo = 0
init_rail_acce = 0

init_global_pos_x = init_pos_x

trials_magn = trials
history = np.ones(trials)
while(trials > 0):
    cube_pos_x_ref = set_x_ref(incl_angle)[0]
    cube_pos_y_ref = set_x_ref(incl_angle)[1]
    times = trials_magn - trials    # remain times
    cube_pos_x[times] = cube_pos_x_ref
    cube_pos_y[times] = cube_pos_y_ref - g/2*t**2
    win = False
    delta = 1

    for i in range(1, len(t)):
        if i == 1:
            # 更新数组
            rail_disp[times][0] = init_rail_disp
            rail_velo[times][0] = init_rail_velo
            rail_acce[times][0] = init_rail_acce

            train_pos_x[times][0] = init_pos_x
            train_pos_y[times][0] = init_pos_y

        e[times][i-1] = cube_pos_x_ref - train_pos_x[times][i-1]

        # if cube_pos_y[times][i-1] < 10 or cube_pos_y[times][i-1] > -10:
        #     e
        


        if i > 1:
            e_dot[times][i-1] = (e[times][i-1] - e[times][i-2]) / dt
            e_int[times][i-1] = (e[times][i-1] - e[times][i-2]) / dt + e_int[times][i-2]
        if i == len(t)-1:
            e[times][-1] = e[times][-2]
            e_dot[times][-1] = e_dot[times][-2]
            e_int[times][-1] = e_int[times][-2]