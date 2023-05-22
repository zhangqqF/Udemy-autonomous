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
import support_files_car as sfc
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation


# Create an object for the support functions.
support=sfc.SupportFilesCar()
constants=support.constants

# Load the constant values needed in the main file
Ts=constants[6]
outputs=constants[10] # number of outputs (psi, Y)
hz = constants[11] # horizon prediction period
x_dot=constants[12] # constant longitudinal velocity
time_length=constants[15] # duration of the manoeuvre

# Generate the refence signals
t=np.arange(0,time_length+Ts,Ts) # time from 0 to 10 seconds, sample time (Ts=0.1 second)
r=constants[13]
f=constants[14]
psi_ref,X_ref,Y_ref=support.trajectory_generator(t,r,f)

print(psi_ref)
sim_length=len(t) # Number of control loop iterations
refSignals=np.zeros(len(X_ref)*outputs)

# Build up the reference signal vector:
# refSignal = [psi_ref_0, Y_ref_0, psi_ref_1, Y_ref_1, psi_ref_2, Y_ref_2, ... etc.]
k=0
for i in range(0,len(refSignals),outputs):
    refSignals[i]=psi_ref[k]
    refSignals[i+1]=Y_ref[k]
    k=k+1

# Load the initial states
# If you want to put numbers here, please make sure that they are float and not
# integers. It means that you should add a point there.
# Example: Please write 0. in stead of 0 (Please add the point to make it float)
y_dot=0.
psi=0.
psi_dot=0.
Y=Y_ref[0]+10.

states=np.array([y_dot,psi,psi_dot,Y])
statesTotal=np.zeros((len(t),len(states))) # It will keep track of all your states during the entire manoeuvre
statesTotal[0][0:len(states)]=states
psi_opt_total=np.zeros((len(t),hz))
Y_opt_total=np.zeros((len(t),hz))

# Load the initial input
U1=0 # Input at t = -1 s (steering wheel angle in rad (delta))
UTotal=np.zeros(len(t)) # To keep track all your inputs over time
UTotal[0]=U1

# To extract psi_opt from predicted x_aug_opt
C_psi_opt=np.zeros((hz,(len(states)+np.size(U1))*hz))
for i in range(1,hz+1):
    C_psi_opt[i-1][i+4*(i-1)]=1

# To extract Y_opt from predicted x_aug_opt
C_Y_opt=np.zeros((hz,(len(states)+np.size(U1))*hz))
for i in range(3,hz+3):
    C_Y_opt[i-3][i+4*(i-3)]=1

# Generate the discrete state space matrices
Ad,Bd,Cd,Dd=support.state_space()

# UPDATE FROM THE VIDEO EXPLANATIONS:
# Generate the compact simplification matrices for the cost function
# The matrices (Hdb,Fdbt,Cdb,Adc) stay mostly constant during the simulation.
# Therefore, it is more efficient to generate them here before you start the simulation loop.
# However, in the end of the simulation, the horizon period (hz) will start decreasing.
# That is when the matrices need to be regenerated (done inside the simulation loop)
Hdb,Fdbt,Cdb,Adc=support.mpc_simplification(Ad,Bd,Cd,Dd,hz)
print(Fdbt)