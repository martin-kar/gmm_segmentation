'''
=================================
Gaussian Mixture Model Segmentation
=================================
'''
import itertools
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture

                              
# Load trajectory data:
data = np.genfromtxt('../tecnalia_data/Trial_1.csv', delimiter=';',skip_header=2)
frequency = 200
time = data[:,0]/frequency

#pos = data[:,1:4]
pos = data[:,1]

traj = np.column_stack((time, pos))
print(traj[:,0])
print(traj[:,0:1])

plt.plot(time,pos)
plt.title("Trajectory")
plt.xlabel("Time [s]")
plt.ylabel("Position")
plt.show()

#Fit a Gaussian mixture with EM
gmm = mixture.GaussianMixture(n_components=3, covariance_type='full', reg_covar=10**-5).fit(traj)

print(gmm)




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

    #plt.xlim(0., 50)
    #plt.ylim(-1., 1)
    #plt.xticks(())
    #plt.yticks(())
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Position")
    

#print(gmm.predict(traj))   
plot_results(traj, gmm.predict(traj), gmm.means_, gmm.covariances_, 0,
             'Trajectory Segmented into Gaussian Mixture Model')
             
plt.show()
