import itertools
import matplotlib.pyplot as plt
from scipy import linalg
import numpy as np
import matplotlib as mpl
from sklearn import mixture


inch_rate = 2.54

def plot_results(X, Y_, means, covariances, index, title, gmm):
    
    color_list = ['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange','red', 'green']

    color_list = color_list[0:gmm.n_components]
    splot = plt.subplot()
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_list)):
        v, w = linalg.eigh(covar)
        
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    #plt.title(title)
    #plt.xlabel("Time [s]")
    plt.ylabel("Position [mm]")
    plt.xlabel("Time [s]")
    
    

def plot_gmm_segments(dof, traj, gmm):
    #scaler = 3
    plt.figure()#(figsize=(scaler*7/inch_rate, scaler*3/inch_rate))
    for i in range(1, dof + 1):
        # print(gmm.covariances_[:,[[0,0],[i,i]],[[0,i],[0,i]]])
        plot_results(traj[:,[0,i]], gmm.predict(traj), gmm.means_[:,[0,i]], gmm.covariances_[:,[[0,0],[i,i]],[[0,i],[0,i]]], 0,
                 'Trajectory Segmented into Gaussian Mixture Model',gmm)

    #plt.show()
