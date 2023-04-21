# ============================================
#   zhangqq
#   Apr-20, 2023
#   Chongqing
# ============================================


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import numpy as np
import random



dt = 0.02
t = 5
T = np.arange(0, t+dt, dt)

angle = 30


# matrixs of initial
trial = 1
train_pos_x = np.zeros((trial, T))
train_pos_y = np.zeros((trial, T))
train_disp = np.zeros((trial, T))
train_v = np.zeros((trial, T))
train_a = np.zeros((trial, T))
cube_pos_x = np.zeros((trial, T))
cube_pos_y = np.zeros((trial, T))
Ep = np.zeros((trial, T))
Ei = np.zeros((trial, T))
Ed = np.zeros((trial, T))




# initial position of the train
train_pos_x[0][0] = 200
train_pos_y[0][0] = 200*np.sin(angle)
train_disp[0][0] = (train_pos_x[0][0]**2 + train_pos_y[0][0]**2)**0.5
train_v[0][0] = 0
train_a[0][0] = 0
Ep[0][0] = 0
Ei[0][0] = 0
Ed[0][0] = 0


def get_x_ref(angle):
    x = random.uniform(0, 120)
    y = random.uniform(200 + 120*np.sin(angle) + 6.5, 400 + 120*np.sin(angle) + 6.5)
    return x, y

# initial coordnates of the cube
cube_pos_x[0][0] = get_x_ref(angle)[0]
cube_pos_y[0][0] = get_x_ref(angle)[1]



Kp = 300
Ki = 10
Kd = 300
train_mass = 100
g = 10
Fgt = train_mass*g*np.sin(angle)


for i in range(1, len(T)):
    Ep[0][i] = Ep[0][i] - Ep[0][i-1]
    Ei[0][i] = Ei[0][i-1] + (Ei[0][i] + Ei[0][i-1])*dt/2
    Ed[0][i] = (Ed[0][i] - Ed[0][i-1]) / dt

    Fa = Kp*Ep + Ki*Ei + Kd*Ed
    Ft = Fa + Fgt


    train_a[0][i] = Ft/train_mass
    train_v[0][i] = train_v[0][i-1] + (train_a[0][i-1] + train_a[0][i])/2*dt
    train_disp[0][i] = train_disp[0][i-1] + (train_v[0][i-1] + train_v[0][i])/2*dt
    train_pos_x[0][i] = train_pos_x[0][i-1] + train_disp[0][i]*np.cos(angle)
    train_pos_y[0][i] = train_pos_y[0][i-1] + train_disp[0][i]*np.sin(angle)