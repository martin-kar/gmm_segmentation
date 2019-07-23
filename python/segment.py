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
import matplotlib.pyplot as plt
import kalman_filter


def gmm_segment(time, pos, nbr_of_components):
    dof = pos.shape[1]
    traj = np.column_stack((time, pos))

    #Fit a Gaussian mixture with EM
    gmm = mixture.GaussianMixture(n_components=nbr_of_components, covariance_type='full', reg_covar=10**-2).fit(traj)

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


def force_segment(time, force):
    force_dof = force.shape[1]
    nbr_time_steps = force.shape[0]
    noise_std = 0.5
    R1 = noise_std*np.identity(force_dof)
    R2 = noise_std*np.identity(force_dof)
    P = noise_std*np.identity(force_dof)
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


    plt.figure()
    plt.plot(time, force)
