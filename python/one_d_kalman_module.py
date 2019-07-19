



def one_d_kalman(R1, R2, xhat, y, Phi, P):
    Kf = P/(P + R2)
    K = Phi*P/(P + R2)
    P = Phi*P*Phi + R1 - K*(P + R2)*K
    xhat = xhat + Kf*(y - xhat)
    xhat = Phi*xhat + K*(y-xhat)
    return xhat, P
