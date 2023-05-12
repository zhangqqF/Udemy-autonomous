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
            y_1 = aaa * (x + f - 100)**2 + bbb     # parabolic function
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
            1. 动力学方程矩阵 a, b, c, d
            2. 加入φ_dot和y_dot的增广矩阵 A, B, C, D
            3. 预测矩阵 Ad, Bd, Cd, Dd
        '''
        m = self.m
        Iz = self.Iz
        Caf = self.Caf
        Car = self.Car
        lf = self.lf
        lr = self.lr
        Ts = self.Ts
        x_dot = self.x_dot

        # 动力学方程矩阵
        a = -(2*self.Caf + 2*self.Car) / (self.m * self.x_dot)
        b = ((-2*Caf*lf + 2*Car*lr) / x_dot*m) - x_dot
        c = (-2*Caf*lf + 2*Car*lr) / x_dot*Iz
        d = (-2*Caf*lf**2 - 2*Car*lr**2) / x_dot*m
        e = 2*Caf / m
        f = 2*Caf*lf / Iz

        # 增加psi_dot和y_dot状态量后的矩阵
        A = np.array([a, 0, b, 0],
                     [0, 0, 1, 0],
                     [c, 0, d, 0],
                     [1, x_dot, 0, 0])
        B = np.array([e, 0, f, 0])
        C = np.array([0, 1, 0, 0],
                     [0, 0, 0, 1])
        D = 0

        # 离散化后的矩阵
        Ad = np.identity(np.size(A, 1)) + Ts*A
        Bd = Ts*B
        Cd = C
        Dd = D

        return Ad, Bd, Cd, Dd
    

    def mpc_simplification(self, Ad, Bd, Cd, Dd, Hz):
        '''
        This function create the compact matrices for Model Predictive Control
            4. 用Δδ代替δ后的预测增广矩阵 A_tilde, B_tilde, C_tilde
            5. 状态量误差代替状态量后的预测矩阵 Q_bbar, T_bbar, R_bbar
            6. 摆脱状态量后的min矩阵 H_bbar, F_bbar
        '''

        # 构建增广矩阵A_tilde, B_tilde, C_tilde, D_tilde
        # |Ad  Bd|
        # |0   I|
        temp_A_B = np.concatenate((Ad, Bd), axis=1)              # 构造[Ad Bd]矩阵，Ad=4x4, Bd=4x1, temp_A_B=4x5，axis=1列叠加
        temp_0 = np.zeros((1, np.size(Ad, 1)))                   # 构造0矩阵，0=1xA的列数
        temp_I = np.identity(np.size(Bd, 1))                     # 构造单位矩阵I, I=B的列数
        temp_0_I = np.concatenate((temp_0, temp_I), axis=1)      # 构造[0 I]矩阵
        A_tilde = np.concatenate((temp_A_B, temp_0_I), axis=0)   # 构造A_tilde矩阵，axis=0行叠加
        B_tilde = np.concatenate((Bd, np.identity(np.size(Bd, 1))), axis=0)
        C_tilde = np.concatenate((Cd, np.zeros(np.size(Cd, 0), np.size(Bd, 1))))  # 列叠加，所以0矩阵的行数与Cd相同，为与A_tilde列数相同，0矩阵列数等同Bd列数
        D_tilde = Dd

        Q = self.Q
        S = self.S
        R = self.R

        # 误差min矩阵
        CQC = np.matmul(np.matmul(np.transpose(C_tilde), Q), C_tilde)       # matmul一次只能算两个吗？
        CSC = np.matmul(np.matmul(np.transpose(C_tilde), S), C_tilde)
        QC = np.matmul(Q, C_tilde)
        SC = np.matmul(S, C_tilde)

        Q_bbar = np.zeros((np.size(CQC, 0)*Hz), np.size(CQC, 1)*Hz)         # CQC行列均乘时间Hz
        T_bbar = np.zeros((np.size(QC, 0)*Hz), np.size(QC, 1)*Hz)
        R_bbar = np.zeros((np.size(R, 0)*Hz), np.size(R, 1)*Hz)
        C_bbar = np.zeros((np.size(B_tilde, 0)*Hz), np.size(B_tilde, 1)*Hz)
        A_hhat = np.zeros((np.size(A_tilde, 0)*Hz), np.size(A_tilde, 1))    # 注意，列不 x hz
        for i in range(0, Hz):
            if i == Hz-1: # 最后一项是SC
                Q_bbar[CSC.shape[0]*i:CSC.shape[0]*(i+1), CSC.shape[1]*i:CSC.shape[1]*(i+1)] = CSC
                T_bbar[SC.shape[0]*i:SC.shape[0]*(i+1), SC.shape[1]*i:SC.shape[1]*(i+1)] = SC
            else:
                Q_bbar[CQC.shape[0]*i:CQC.shape[0]*(i+1), CQC.shape[1]*i:CQC.shape[1]*(i+1)] = CQC
                T_bbar[QC.shape[0]*i:QC.shape[0]*(i+1), QC.shape[1]*i:QC.shape[1]*(i+1)] = QC
                
            R_bbar[np.size(R,0)*i:np.size(R,0)*i+R.shape[0], np.size(R,1)*i:np.size(R,1)*i+R.shape[1]] = R

            for j in range(0, Hz):
                if j <= i:
                    AB = np.matmul(np.linalg.matrix_power(A_tilde, i-j), B_tilde)
                    C_bbar[B_tilde.shape[0]*j:B_tilde.shape[0]*(i+1), B_tilde.shape[1]*i:B_tilde.shape[1]*(i+1)] = AB

            A_hhat[A_tilde.shape[0]*i:A_tilde.shape[0]*(i+1), :] = np.linalg.matrix_power(A_tilde, i+1)

        H_bbar = np.matmul(np.matmul(np.transpose(C_bbar), Q_bbar), C_bbar) + R_bbar
        AQC = np.matmul(np.matmul(np.transpose(A_hhat), Q), C_bbar)
        TC = -np.matmul(T_bbar, C_bbar)
        F_bbar = np.concatenate(AQC, TC, axis=1)
        
        return H_bbar, F_bbar, C_bbar, A_hhat
    
    def open_loop_new_states(self, states, delta):
        '''
        This function computes the new state vector for one sample time later
        '''
        m=self.m
        Iz=self.Iz
        Caf=self.Caf
        Car=self.Car
        lf=self.lf
        lr=self.lr
        Ts=self.Ts
        x_dot=self.x_dot

        current_states = states
        new_states = current_states
        y_dot = new_states[0]
        psi = new_states[1]
        psi_dot = new_states[2]
        y = new_states[3]

        loop = 30 # 将采样时间再分成30份，更加细化
        for i in range(0, loop):
            a = -(2*self.Caf + 2*self.Car) / (self.m * self.x_dot)
            b = ((-2*Caf*lf + 2*Car*lr) / x_dot*m) - x_dot
            c = (-2*Caf*lf + 2*Car*lr) / x_dot*Iz
            d = (-2*Caf*lf**2 - 2*Car*lr**2) / x_dot*m
            e = 2*Caf / m
            f = 2*Caf*lf / Iz
            
            y_ddot = a*y_dot + b*psi_dot + e*delta
            psi_dot = psi_dot
            psi_ddot = c*y_ddot + d*psi_dot + f*delta
            y_dot = y_dot*np.cos(psi) + x_dot*np.sin(psi)

            # 更新
            y_dot = y_dot + y_ddot*Ts/loop
            psi = psi + psi_dot*Ts/loop
            psi_ddot = psi_ddot + psi_dot*Ts/loop
            y = y + y_dot*Ts/loop

        # Take the last states
        new_states[0] = y_dot
        new_states[1] = psi
        new_states[2] = psi_dot
        new_states[3] = y

        return new_states
    
