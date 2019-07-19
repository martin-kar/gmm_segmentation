'''
=================================
Gaussian Mixture Model Segmentation
=================================
'''
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import one_d_kalman_module

# User-defined parameters
force_dof = 3 # Degrees of freedom of force
frequency = 200



# Load trajectory data:
data = np.genfromtxt('../tecnalia_data/Trial_1.csv', delimiter=';',skip_header=1)
time = data[:,0]/frequency
force = data[:,13:13 + force_dof]

force = force[:,2]


R1 = 1
R2 = 1
P = 1
Phi = 1
xhat = force[0]

P_log = np.zeros_like(force)
P_log[0] = P
xhat_log = np.zeros_like(force)
xhat_log[0] = xhat

for time_step, y in enumerate(force[1:], start = 1):
    xhat, P = one_d_kalman_module.one_d_kalman(R1, R2, xhat, y, Phi, P)
    print(time_step)
    P_log[time_step] = P
    xhat_log[time_step] = xhat

estimation_error = xhat_log - force
epsilon = estimation_error/P_log*estimation_error


plt.plot(time, epsilon)
plt.show()
