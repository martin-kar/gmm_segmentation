import itertools
import matplotlib.pyplot as plt
from scipy import linalg
import numpy as np
import matplotlib as mpl
from sklearn import mixture



color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])
def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot()
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
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

    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Position")


def plot_gmm_segments(dof, traj, gmm):
    for i in range(1, dof + 1):
        # print(gmm.covariances_[:,[[0,0],[i,i]],[[0,i],[0,i]]])
        plot_results(traj[:,[0,i]], gmm.predict(traj), gmm.means_[:,[0,i]], gmm.covariances_[:,[[0,0],[i,i]],[[0,i],[0,i]]], 0,
                 'Trajectory Segmented into Gaussian Mixture Model')
                 
    plt.show()
