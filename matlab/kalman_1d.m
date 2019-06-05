function [theta, P, e, yp] = kalman_1d(R1,R2,theta, y, A, P)
    ATP = A'*P;
    K = (P*A)/(R2+ATP*A);
    P = P - (P*A*ATP)./(R2 + ATP*A) + R1;
    yp = (A'*theta);
    e = y-yp;
    theta = theta + K*e;
end

