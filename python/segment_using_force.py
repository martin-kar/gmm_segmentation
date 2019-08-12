'''
=================================
Gaussian Mixture Model Segmentation
=================================
'''
import numpy as np
import matplotlib.pyplot as plt
import kalman_filter

def force_segment(

R1 = 0.025*np.identity(force_dof)
R2 = 0.025*np.identity(force_dof)
P = 0.025*np.identity(force_dof)
Phi = np.identity(force_dof)
xhat = force[0]
xhat = xhat.reshape(force_dof, 1)

print(xhat)
xhat_log = np.zeros_like(force)
xhat_log[0] = xhat.reshape(force_dof)


normalized_filter_error = np.zeros(nbr_time_steps)
normalized_filter_error[0] = 0

for time_step, y in enumerate(force[1:], start = 1):
    y = y.reshape(force_dof,1)
    xhat, P = kalman_filter.kf(R1, R2, xhat, y, Phi, P)
    #xhat = xhat.reshape(force_dof)
    #xhat_log[time_step] = xhat
    estimation_error = xhat - y
    normalized_filter_error[time_step] = np.transpose(estimation_error).dot(np.linalg.inv(P)).dot(estimation_error)

#estimation_error = xhat_log - force

plt.figure()
plt.plot(time, normalized_filter_error)
print("HEJ")

plt.figure()
plt.plot(time, force)
plt.show()
