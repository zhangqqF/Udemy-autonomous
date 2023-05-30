# ============================================
#   zhangqq
#   May-4. 2023
#   Chongqing
#   May 27 在完成jupyter notebook后重写
# ============================================


import numpy as np



class MPC_Controller:

    def __init__(self):
        '''
        Load the constants that do not change
        '''

        self.g = 9.81
        self.m = 1500        # mass of the car
        self.Iz = 3000       # inertial moment about axile Z
        self.Cf = 19000      # front wheel cornering stiffness
        self.Cr = 33000      # rear wheel cornering stiffness
        self.lf = 2          # distance of the front wheel from the C.M
        self.lr = 3          # distance of the rear wheel from the C.M
        self.Ts = 0.02       # sample time

        self.Q = np.matrix('1 0; 0 1')     # weight for outputs
        self.S = np.matrix('1 0; 0 1')     # 
        self.R = np.matrix('1')            # weight for inputs

        self.outputs = 2     # numbers of the output. [y φ]
        # self.hz = 20         # horizon period, also known as prediction horizon
        self.dx = 20         # car's longitudinal velocity
        self.lane_width = 7
        self.lanes = 5

        self.r = 4                  # use for create the trajectory
        self.f = 0.01               # use for create the trajectory
        self.sim_time = 10          # duration time of the simulation
        self.car_distance = 30

        # self.trajectory = 3         # what trajectory you choose

        # --------------------------- PID ---------------------------
        self.PID_switch = 0

        self.Kp_yaw = 7
        self.Ki_yaw = 5
        self.Kd_yaw = 3

        self.Kp_Y = 7
        self.Ki_Y = 5
        self.Kd_Y = 3
        # ------------------------- PID End -------------------------

        return None
    

    def trajectory_generator(self, t, trajectory):
        '''
        This method creates the trajectory for a car to follow
        '''

        dx = self.dx
        # trajectory = self.trajectory
        r = self.r
        f = self.f

        # Define the x length, depends on the car's longitudinal velocity
        x = np.linspace(0, dx*t[-1], len(t))

        # Define trajectories, There has three
        if trajectory == 1:
            y = -9 * np.ones(len(t))
        elif trajectory == 2:
            y = 9 * np.tanh(t - t[-1]/2)
        elif trajectory == 3:
            a = -28 / 100**2
            a = a / 1.1
            if a < 0:
                b = 14
            else:
                b = -14
            y_1 = a * (x + f - 100)**2 + b              # parabolic function
            y_2 = 2 * r * np.sin(2 * np.pi * f * x)     # sinsoidal function
            y = (y_1 + y_2) / 2
        else:
            print(' For trajectory, only choose 1, 2, or 3 as an integer vallue')
            exit()

        # calculate the phi by x and y
        dx = x[1:len(x)] - x[0:len(x)-1]
        dy = y[1:len(y)] - y[0:len(y)-1]

        phi = np.zeros(len(x))
        phi[0] = np.arctan2(dy[0], dx[0])
        phi[1:len(phi)] = np.arctan2(dy[0:len(dy)], dx[0:len(dx)])
        dphi = phi[1:len(phi)] - phi[0:len(phi)-1]

        for i in range(1, len(phi)):
            if dphi[i-1] < -np.pi:
                phi[i] = phi[i-1] + (dphi[i-1] + 2*np.pi)
            elif dphi[i-1] > np.pi:
                phi[i] = phi[i-1] + (dphi[i-1] - 2*np.pi)
            else:
                phi[i] = phi[i-1] + dphi[i-1]

        return x, y, phi
    

    def combine_ref(self, reflist):
        '''
        合并参考信号，此前参考信号已离散化为行向量
        '''
        num_ref = len(reflist)
        sim_steps = len(reflist[0])
        ref = np.zeros(num_ref*sim_steps)

        for i in range(sim_steps):
            for j in range(num_ref):
                ref[i*num_ref + j] = reflist[j][i]

        return ref
    

    def dynamic_func_abcdef(self):
        '''
        dynamic function of the car
        '''
        m = self.m
        Iz = self.Iz
        Cf = self.Cf
        Cr = self.Cr
        lf = self.lf
        lr = self.lr
        dx = self.dx

        # 动力学方程矩阵
        a = -(2*Cf + 2*Cr) / (m*dx)
        b = (-2*Cf*lf + 2*Cr*lr) / (dx*m) - dx
        c = (-2*Cf*lf + 2*Cr*lr) / (dx*Iz)
        d = (-2*Cf*lf**2 - 2*Cr*lr**2) / (dx*Iz)
        e = 2*Cf / m
        f = 2*Cf*lf / Iz

        return a,b,c,d,e,f
    

    def state_ABCD(self, a, b, c, d, e, f):
        '''
        增加dφ和dy状态量后的系数矩阵
        '''
        dx = self.dx

        A = np.array([[a, 0, b, 0],
                      [0, 0, 1, 0],
                      [c, 0, d, 0],
                      [1, dx, 0, 0]])
        B = np.transpose(np.array([[e, 0, f, 0]]))
        C = np.array([[0, 1, 0, 0],
                      [0, 0, 0, 1]])
        D = 0

        return A, B, C, D
    

    def discrete(self, A, B, C, D):
        '''
        离散化后的系数矩阵
        '''
        Ts = self.Ts

        Ad = np.identity(np.size(A, 1)) + Ts*A
        Bd = Ts*B
        Cd = C
        Dd = D

        return Ad, Bd, Cd, Dd
    

    def augmented_tilde(self, Ad, Bd, Cd, Dd):
        '''
        用Δδ代替δ的cost function预测系数矩阵
        '''

        temp_AB = np.concatenate((Ad, Bd), axis=1)
        temp_0 = np.zeros((Bd.shape[1], Ad.shape[1]))
        temp_I = np.identity(Bd.shape[1])
        temp_0I = np.concatenate((temp_0, temp_I), axis=1)
        A_tilde = np.concatenate((temp_AB, temp_0I), axis=0)

        B_tilde = np.concatenate((Bd, np.identity(Bd.shape[1])), axis=0)
        C_tilde = np.concatenate((Cd, np.zeros((Cd.shape[0], Bd.shape[1]))), axis=1)
        D_tilde = Dd

        return A_tilde, B_tilde, C_tilde, D_tilde


    def bbar(self, A_tilde, B_tilde, C_tilde, D_tilde, hz):
        '''
        全误差优化矩阵
        '''
        Q = self.Q
        S = self.S
        R = self.R

        CQC = np.matmul(np.matmul(np.transpose(C_tilde), Q), C_tilde)
        CSC = np.matmul(np.matmul(np.transpose(C_tilde), S), C_tilde)
        QC = np.matmul(Q, C_tilde)
        SC = np.matmul(S, C_tilde)
        
        Q_bb = np.zeros((CQC.shape[0]*hz, CQC.shape[1]*hz))
        T_bb = np.zeros((QC.shape[0]*hz, QC.shape[1]*hz))
        R_bb = np.zeros((R.shape[0]*hz, R.shape[1]*hz))
        C_bb = np.zeros((B_tilde.shape[0]*hz, B_tilde.shape[1]*hz))
        A_hh = np.zeros((A_tilde.shape[0]*hz, A_tilde.shape[1]))
        for i in range(0, hz):
            if i == hz-1:
                Q_bb[CQC.shape[0]*i:CQC.shape[0]*i+CQC.shape[0], CQC.shape[1]*i:CQC.shape[1]*i+CQC.shape[1]] = CSC
                T_bb[QC.shape[0]*i:QC.shape[0]*i+QC.shape[0], QC.shape[1]*i:QC.shape[1]*i+QC.shape[1]] = SC
            else:
                Q_bb[CQC.shape[0]*i:CQC.shape[0]*i+CQC.shape[0], CQC.shape[1]*i:CQC.shape[1]*i+CQC.shape[1]] = CQC
                T_bb[QC.shape[0]*i:QC.shape[0]*i+QC.shape[0], QC.shape[1]*i:QC.shape[1]*i+QC.shape[1]] = QC

            R_bb[R.shape[0]*i:R.shape[0]*i+R.shape[0], R.shape[1]*i:R.shape[1]*i+R.shape[1]] = R
            A_hh[A_tilde.shape[0]*i:A_tilde.shape[0]*i+A_tilde.shape[0], :] = np.linalg.matrix_power(A_tilde, i+1)

            for j in range(0, hz):
                if j <= i:
                    AB = np.matmul(np.linalg.matrix_power(A_tilde, i-j), B_tilde)
                    C_bb[B_tilde.shape[0]*i:B_tilde.shape[0]*i+B_tilde.shape[0], B_tilde.shape[1]*j:B_tilde.shape[1]*j+B_tilde.shape[1]] = AB
        
        return Q_bb, T_bb, R_bb, C_bb, A_hh
    

    def gradient(self, Q_bb, T_bb, R_bb, C_bb, A_hh):
        '''
        求Δδ的梯度矩阵
        '''

        H_bb = np.matmul(np.matmul(np.transpose(C_bb), Q_bb), C_bb) + R_bb
    
        AQC = np.matmul(np.matmul(np.transpose(A_hh), Q_bb), C_bb)
        TC = np.matmul(-T_bb, C_bb)
        F_bbt = np.concatenate((AQC, TC), axis=0)
        
        return H_bb, F_bbt


    def update_states(self, init_states, init_U):
        '''
        通过ΔU计算状态量，以实现预测
        '''
        a,b,c,d,e,f = self.dynamic_func_abcdef()
        dx = self.dx
        Ts = self.Ts
        dy = init_states[0]
        phi = init_states[1]
        dphi = init_states[2]
        Y = init_states[3] 
        
        loop = 30 # 将采样时间再分成30份，更加细化
        for i in range(0, loop):
            ddy = a*dy + b*dphi + e*init_U
            
            dphi = dphi
            ddphi = c*dy + d*dphi + f*init_U
            dY = dy*np.cos(phi) + dx*np.sin(phi)
            
            # 通过积分计算并更新状态矩阵
            dy = dy + ddy*Ts/loop
            
            phi = phi + dphi*Ts/loop
            dphi = dphi + ddphi*Ts/loop
            Y = Y + dY*Ts/loop
        
        return [dy, phi, dphi, Y]       # 注意，dy的y是车辆坐标系，大Y是道路坐标系
    