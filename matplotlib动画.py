import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

 

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)



x = np.linspace(0, 2 * np.pi, 5000)
y = np.exp(-x) * np.cos(2 * np.pi * x)
line, = ax.plot(x, y, color="cornflowerblue", lw=3)
ax.set_ylim(-1.1, 1.1)



# 清空当前帧
def init():
    line.set_ydata([np.nan] * len(x))
    return line,


# 更新新一帧的数据
def update(frame):
    line.set_ydata(np.exp(-x) * np.cos(2 * np.pi * x + float(frame)/100))
    print(frame)
    return line,


# 调用 FuncAnimation
ani = FuncAnimation(fig
                   ,update
                   ,init_func=init
                   ,frames=200
                   ,interval=2
                   ,blit=True
                   )


# ani.save("animation.gif", fps=25, writer="pillow")
plt.show()


