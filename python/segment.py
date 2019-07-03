'''
=================================
Gaussian Mixture Model Segmentation
=================================
'''
import numpy as np
from scipy import linalg
import matplotlib as mpl
from sklearn import mixture
from plot_gmm import plot_gmm_segments

# User-defined parameters
dof = 3 # Degrees of freedom of configuration
frequency = 200
nbr_of_components = 4 # Could also be determined by criteria such as
                        # Bayesian information criterion.

# Load trajectory data:
data = np.genfromtxt('../tecnalia_data/Trial_1.csv', delimiter=';',skip_header=1)
time = data[:,0]/frequency
pos = data[:,1:dof + 1]

traj = np.column_stack((time, pos))

#Fit a Gaussian mixture with EM
gmm = mixture.GaussianMixture(n_components=nbr_of_components, covariance_type='full', reg_covar=10**-5).fit(traj)

plot_gmm_segments(dof, traj, gmm)


import force_segment
