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

