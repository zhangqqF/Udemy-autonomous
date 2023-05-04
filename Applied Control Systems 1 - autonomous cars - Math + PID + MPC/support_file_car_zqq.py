# ============================================
#   zhangqq
#   May-4. 2023
#   Chongqing
# ============================================


'''
Functions:
    trajectory_generator()



'''
import numpy as np
import matplotlib.pyplot as plt


class SupportFilesCar:

    def __init__(self):
        '''
        Load the constants that do not change
        '''

        self.g = 9.81
        self.m = 1500
        self.Iz = 3000
        self.Caf = 19000     # front wheel cornering stiffness
        self.Car = 30000     # rear wheel cornering stiffness
        self.lf = 2          # distance of the front wheel from the C.M
        self.lr = 3          # distance of the rear wheel from the C.M
        self.Ts = 0.02

        self.Q = np.matrix('100 0; 0 1')     # weight for outputs
        self.S = np.matrix('100 0; 0 1')     # 
        self.R = np.matrix('100')            # weight for inputs

        self.outputs = 2
        self.hz = 20         # horizon period
        self.x_dot = 20      # car's longitudinal velocity
        self.lane_width = 7
        self.nr_lanes = 5

        self.r = 4
        self.f = 0.01
        self.time_lenght = 10
        self.car_distance = 30

        # --------------------------- PID ---------------------------
        self.PID_switch = 0

        self.Kp_yaw = 7
        self.Ki_yaw = 5
        self.Kd_yaw = 3

        self.Kp_Y = 7
        self.Ki_Y = 5
        self.Kd_Y = 3
        # ------------------------- PID End -------------------------

        self.trajectory = 3

        return None
    

    def trajectory_generator(self, t, r, f):
        '''
        This method creates the trajectory for a car to follow
        '''

        Ts = self.Ts
        x_dot = self.x_dot
        trajectory = self.trajectory

        # Define the x length, depends on the car's longitudinal velocity
        x = np.linspace(0, x_dot*t[-1], len(t))

        # Define trajectories, There has three
        if trajectory == 1:
            y = -9 * np.ones(len(t))
        elif trajectory == 2:
            y = 9 * np.tanh(t - t[-1]/2)
        elif trajectory == 3:
            aaa = -28 / 100**2
            aaa = aaa / 1.1
            if aaa < 0:
                bbb = 14
            else:
                bbb = -14
            y_1 = aaa * (x + self.f - 100)**2 + bbb     # parabolic function
            y_2 = 2 * r * np.sin(2 * np.pi * f * x)     # sinsoidal function
            y = (y_1 + y_2) / 2
        else:
            print(' For trajectory, only choose 1, 2, or 3 as an integer vallue')
            exit()

        dx = x[1:len(x)] - x[0:len(x)-1]
        dy = y[1:len(y)] - y[0:len(y)-1]

        # Define the reference yaw angles
        psi = np.zeros(len(x))
        psiInt = psi
        psi[0] = np.arctan2(dy[0], dx[0])
        psi[1:len(psi)] = np.arctan2(dy[0:len(dy)], dx[0:len(dx)])

        dpsi = psi[1:len(psi)] - psi[0:len(psi)-1]
        psiInt[0] = psi[0]
        for i in range(1, len(psiInt)):
            if dpsi[i-1] < -np.pi:
                psiInt[i] = psiInt[i-1] + (dpsi[i-1] + 2*np.pi)
            elif dpsi[i-1] > np.pi:
                psiInt[i] = psiInt[i-1] + (dpsi[i-1] - 2*np.pi)
            else:
                psiInt[i] = psiInt[i-1] + psi[i-1]

        return psiInt, x, y
    

    def state_space(self):
        '''
        This function forms the state space matrices and transforms them in the discrete form
        计算矩阵 A, B, C, D
        '''
        m = self.m
        Iz = self.Iz
        Caf = self.Caf
        Car = self.Car
        lf = self.lf
        lr = self.lr
        Ts = self.Ts
        x_dot = self.x_dot

        A1 = -(2*self.Caf + 2*self.Car) / (self.m * self.x_dot)

