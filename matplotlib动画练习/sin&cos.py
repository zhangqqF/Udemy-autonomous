import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np



x = np.linspace(0, 10, 300)
y1 = np.sin(x)
y2 = np.cos(x)


# 创建画布和子图
fig, ax = plt.subplots()


line1, = ax.plot(x, y1, lw=2, color='cornflowerblue')
line2, = ax.plot(x, y2, ls='-.', lw=2, color='crimson')


def update_anim(frame):
    line1.set_data(x[:frame], y1[:frame])
    line2.set_data(x[:frame], y2[:frame])
    return line1, line2

ani = animation.FuncAnimation(fig,           # 需要赋予给变量
                        update_anim,
                        frames=len(x),
                        interval=10,
                        blit=True)
plt.title('sin & cos Fuction')
plt.show()