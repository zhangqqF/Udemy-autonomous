import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

fig = plt.figure(figsize=(16,9),dpi=120,facecolor=(0.8,0.8,0.8))
gs = gridspec.GridSpec(4,3)


init_pos_x_global = 120
ax_main=fig.add_subplot(gs[0:3,0:2],facecolor=(0.9,0.9,0.9))
plt.xlim(0,init_pos_x_global)
plt.ylim(0,init_pos_x_global)
plt.xticks(np.arange(0,init_pos_x_global+1,10))
plt.yticks(np.arange(0,init_pos_x_global+1,10))
plt.grid(True)


# plt.show()


a = [2, 4]
print(a-3)