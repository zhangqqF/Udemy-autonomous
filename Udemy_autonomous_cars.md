# Applied Control Systems 1: autonomous cars: Math + PID + MPC

> zhangqq  
> Apr 18, 2023  
> Chongqing

---

The course on [Udemy](https://www.udemy.com/course/applied-systems-control-for-engineers-modelling-pid-mpc/) or [学术Fun](https://xueshu.fun/1511/)

## Intro to Control - PID Controller

A sensor measures the volume of the water every 0.02 seconds or @ 50 Hz.
$$
f = 50 Hz
$$
Output: something that we can observe (measure). RELEVANT

Input: action that causes output to change.

If error = 0, then reference volume = real volume.
$$
\dot m = 
\left\{
    \begin{array}{cc}
        -10 & [kg/s],\quad e<0 \\
        0 & [kg/s],\quad e=0 \\
        10 & [kg/s],\quad e>0 \\
    \end{array}
\right.
$$
If you want it to happen faster than the controller could be something like this same logic, instead of 
$$
\dot m = 
\left\{
    \begin{array}{cc}
        -100 & [kg/s],\quad e<0 \\
        0 & [kg/s],\quad e=0 \\
        100 & [kg/s],\quad e>0 \\
    \end{array}
\right.
$$

### Modeling the water tank

$$
\dot m = \frac{\mathrm{d}(V\cdot\rho)}{\mathrm{d}t}
=\frac{\mathrm{d}(V)}{\mathrm{d}t}*\rho
\quad\Rightarrow\quad
\frac{\dot{m}}{\rho}=\frac{\mathrm{d}(V)}{\mathrm{d}t}
$$

Where $V$ refer to volume [m^3^].

so,
$$
V=V_0+\frac{1}{\rho} \int_0^t \dot{m} \, \mathrm{d}t
$$
Discretion
$$
V_{t(i)}=V_{t(i-1)}+(\frac{ \frac{1}{\rho}\dot{m}_{t(i-1)} + \frac{1}{\rho}\dot{m}_{t(i)} } {2})\Delta{t}
$$

### PID Controller

[卡曼滤波](https://engineeringmedia.com/controlblog/the-kalman-filter)

## MPC控制代码分析

1. 车辆横向运动方程

   旋转部分造成的侧向：

$$
\left\{
    \begin{array}{cc}
		F_{yf}+F_{yf}=m(\ddot{y}+\dot{x}\dot{\varphi}) \\
		F_{yf}\cdot{L_f}-F_{yr}\cdot{L_r}=I\ddot{\varphi}
    \end{array}
\right.
$$

​		考虑轮胎侧偏（轮胎小侧偏角假设）：
$$
\left\{
    \begin{array}{ll}
		F_{yf}=2C_{Lf}(\delta-\theta_{vf}) \\
		F_{yr}=2C_{Lr}(-\theta_{vr})
    \end{array}
\right.
$$
​		转向角小角度假设：
$$
\left\{
	\begin{array}{ll}
		\tan{\theta_{vf}} \approx \theta_{vf}=\frac{\dot{y}+L_{f}\dot{\varphi}}{\dot{x}} \\
		\tan{\theta_{vr}} \approx \theta_{vr}=\frac{\dot{y}-L_{r}\dot{\varphi}}{\dot{x}}
	\end{array}
\right.
$$
​		将上面方程联立，写成矩阵：
$$
\begin{bmatrix}
    \ddot{y} \\
    \ddot{\varphi}
\end{bmatrix}
=
\begin{bmatrix}
    a & b \\
    c & d
\end{bmatrix}
\begin{bmatrix}
    \dot{y} \\
    \dot{\varphi}
\end{bmatrix}
+
\begin{bmatrix}
    e \\
    f
\end{bmatrix} \delta
$$
​		系数矩阵中$a$、$b$、$c$、$d$、$e$、$f$的值如下：(**代码部分1**)
$$
\begin{aligned}
    a&=-\frac{2C_{Lf}+2C_{Lr}}{\dot{x}m} \\ \\
    b&=\frac{-2C_{Lf}L_{f}+2C_{Lr}L_r}{\dot{x}m}-\dot{x} \\ \\
    c&=\frac{-2C_{Lf}L_{f}+2C_{Lr}L_{r}}{\dot{x}J} \\ \\
    d&=\frac{-2C_{Lf}L_{f}^2-2C_{Lr}L_{r}^2}{\dot{x}J} \\ \\
    e&=\frac{2C_{Lf}}{m} \\ \\
    f&=\frac{2C_{Lf}L_{f}}{I}
\end{aligned}
$$
​		增加$\dot\varphi$和$\dot{y}$两个状态量，写成矩阵：
$$
\begin{bmatrix}
	\ddot{y} \\
	\dot{\varphi} \\
	\ddot{\varphi} \\
	\dot{y} \\
\end{bmatrix}
=
\begin{bmatrix}
	a & 0 & b & 0 \\
	0 & 0 & 1 & 0 \\
	c & 0 & d & 0 \\
	1 & \dot{x} & 0 & 0 \\
\end{bmatrix}
\begin{bmatrix}
	\dot{y} \\
	\varphi \\
	\dot{\varphi} \\
	y \\
\end{bmatrix}
+
\begin{bmatrix}
	e \\
	0 \\
	f \\
	0 \\
\end{bmatrix}\delta
$$
​		即：（$\boldsymbol{A}$、$\boldsymbol{B}$矩阵为**代码部分2**）
$$
\boldsymbol{\dot{X}}=\boldsymbol{A}\boldsymbol{X}+\boldsymbol{B}\delta
$$

2. 离散化

   离散化方程：（**代码部分3**，$Ts$为采样时间）
   $$
   \boldsymbol{X_{k+1}}=(\boldsymbol{I}+\boldsymbol{A}Ts)\boldsymbol{X}_{k}+\boldsymbol{B}Ts\delta_{k}
   $$
   即：
   $$
   \boldsymbol{X_{k+1}}=\boldsymbol{A_{d}}\boldsymbol{X}_{k}+\boldsymbol{B_{d}}\delta_{k}
   $$
   

​		用误差来代替状态量和控制量，先看min关系式（上式离散方程是符合以下优化公式的）：
$$
\newcommand{\e}{\boldsymbol{e}}
\newcommand{\S}{\boldsymbol{S}}
\newcommand{\Q}{\boldsymbol{Q}}
\newcommand{\R}{\boldsymbol{R}}
\newcommand{\dt}{\boldsymbol{\delta}}
J=\frac{1}{2}\e_{k+N}^T\S\e_{k+N}+\frac{1}{2}\sum\limits_{i=0}^{N-1}[\e_{k+i}^T\Q\e_{k+i}+\dt_{k+i}^T{\R}\dt_{k+i}]
$$
​		令$\boldsymbol{e}=\boldsymbol{r}-\boldsymbol{CX}$，$\boldsymbol{r}$为参考值，代入上式展开并写成矩阵，得：（**代码部分4**）
$$
J^{'}=\frac{1}{2}\boldsymbol{X}^T
\begin{bmatrix}
	\boldsymbol{C}^T\boldsymbol{TQC} & & & & \\
	 & \boldsymbol{C}^T\boldsymbol{TQC} & & & \\
	 & & \boldsymbol{C}^T\boldsymbol{TQC} & & \\
	 & & & \boldsymbol{C}^T\boldsymbol{TQC} & \\
	 & & & & \boldsymbol{C}^T\boldsymbol{TSC}
\end{bmatrix}\boldsymbol{X}-
\boldsymbol{r}^T
\begin{bmatrix}
	\boldsymbol{QC} & & & & \\
	 & \boldsymbol{QC} & & & \\
	 & & \boldsymbol{QC} & & \\
	 & & & \boldsymbol{QC} & \\
	 & & & & \boldsymbol{SC}
\end{bmatrix}\boldsymbol{X}+
\frac{1}{2}\boldsymbol{\Delta{\delta}}^T
\begin{bmatrix}
	\boldsymbol{R} & & & & \\
	 & \boldsymbol{R} & & & \\
	 & & \boldsymbol{R} & & \\
	 & & & \boldsymbol{R} & \\
	 & & & & \boldsymbol{R}
\end{bmatrix}\boldsymbol{\Delta\delta}
$$

$$
即：J^{'}=\frac{1}{2}\boldsymbol{X}^T\boldsymbol{\bar{\bar{Q}}}\boldsymbol{X}-\boldsymbol{r}^T\boldsymbol{\bar{\bar{T}}}\boldsymbol{X}+\frac{1}{2}\boldsymbol{\Delta\delta}\boldsymbol{\bar{\bar{R}}}\boldsymbol{\delta}
$$

​		重写$J$函数，使其摆脱$\boldsymbol{X}$，得：
$$
J^{''}=\frac{1}{2}\boldsymbol{\Delta\delta}^T(\boldsymbol{C}^T\boldsymbol{QC}+\boldsymbol{R})\boldsymbol{\Delta\delta}+
\begin{bmatrix}
	\boldsymbol{X}_k^T & \boldsymbol{R}^T
\end{bmatrix}
\begin{bmatrix}
	\boldsymbol{A}^T\boldsymbol{QC} \\
	-\boldsymbol{TC}
\end{bmatrix}\boldsymbol{\Delta\delta}
$$

## Numpy

```python
import numpy as np

# size
# np.size(A)  行x列
# np.size(A, 0) 行
# np.size(A, 1) 列
```

