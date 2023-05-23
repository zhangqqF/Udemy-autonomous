'''
LICENSE AGREEMENT

In relation to this Python file:

1. Copyright of this Python file is owned by the author: Mark Misin
2. This Python code can be freely used and distributed
3. The copyright label in this Python file such as

copyright=ax_main.text(x,y,'© Mark Misin Engineering',size=z)
that indicate that the Copyright is owned by Mark Misin MUST NOT be removed.

WARRANTY DISCLAIMER!

This Python file comes with absolutely NO WARRANTY! In no event can the author
of this Python file be held responsible for whatever happens in relation to this Python file.
For example, if there is a bug in the code and because of that a project, invention,
or anything else it was used for fails - the author is NOT RESPONSIBLE!

'''

import numpy as np
import matplotlib.pyplot as plt
import support_files_car_2 as sfc
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation


# Create an object for the support functions.
support=sfc.SupportFilesCar()



# Generate the discrete state space matrices
A1,A2,A3,A4,A,B,C,D,Ad, Bd, Cd, Dd=support.state_space()

# UPDATE FROM THE VIDEO EXPLANATIONS:
# Generate the compact simplification matrices for the cost function
# The matrices (Hdb,Fdbt,Cdb,Adc) stay mostly constant during the simulation.
# Therefore, it is more efficient to generate them here before you start the simulation loop.
# However, in the end of the simulation, the horizon period (hz) will start decreasing.
# That is when the matrices need to be regenerated (done inside the simulation loop)
Hdb,Fdbt,Cdb,Adc, A_aug,B_aug,C_aug,D_aug,Qdb,Tdb,Rdb=support.mpc_simplification(Ad,Bd,Cd,Dd,20)
print(Fdbt)