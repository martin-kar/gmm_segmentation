'''
=================================
Gaussian Mixture Model Segmentation
=================================
'''
import matplotlib.pyplot as plt

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

# Get segmentation dividers.


def gmm_to_dividers(gmm):
    std_boundaries = []
    GMMmeans = gmm.means_[:,[0,1]]
    covariances = gmm.covariances_[:,[[0,0],[1,1]],[[0,1],[0,1]]]
    for i, (mean, covar) in enumerate(zip(GMMmeans, covariances)):
        v, _ = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        std_boundaries.append(mean[0] - v[1]/2)
        std_boundaries.append(mean[0] + v[1]/2)
    std_boundaries = sorted(std_boundaries)
    dividers = []
    for index, _ in enumerate(std_boundaries):
        if index%2==1 and index+1 < len(std_boundaries):
            dividers.append((std_boundaries[index] + std_boundaries[index+1])/2)
    return dividers
    
gmm_dividers = gmm_to_dividers(gmm)
print(gmm_dividers)





plt.show()
