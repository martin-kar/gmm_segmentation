
import numpy as np


def kf(R1, R2, xhat, y, Phi, P):
    Kf = P.dot(np.linalg.inv(P + R2))
    K = Phi.dot(P).dot(np.linalg.inv(P + R2))
    P = Phi.dot(P).dot(np.transpose(Phi)) + R1 - K.dot(P + R2).dot(np.transpose(K))
    xhat = xhat + Kf.dot(y - xhat)
    xhat = Phi.dot(xhat) + K.dot(y-xhat)
    return xhat, P
