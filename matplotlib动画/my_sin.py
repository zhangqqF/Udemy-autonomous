import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)



x = np.linspace(-np.pi, np.pi, 100)
y = np.sin(x)
line, = ax.plot(x, y, color='cornflowerblue', lw=2)
plt.xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],[r'$-\pi$',r'$-\pi/2$',r'$0$',r'$\pi/2$',r'$\pi$'])
plt.yticks([-1,0,1])



def anim_upt(frame):
    line.set_data(x[:frame], y[:frame])
    return line,

anim = animation.FuncAnimation(fig, anim_upt,
                               frames=len(x),
                               interval=20,
                               repeat=False,
                               blit=True)
plt.show()