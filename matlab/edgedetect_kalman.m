load trials;
dt = 0.005;
trial = Trial1(:,2:end);
f = trial(:,13:15);
%plot(f)

% f = sum(f.^2,2);
% plot(f)
f = f(:,1);

T = length(f);
A = 1;
R1 = 1;
R2 = 1;
theta = f(1);


e = zeros(size(f));
e(1) = 0;
P = zeros(size(f));
P(1) = 1;

for t = 2:T
    [theta, P(t), e(t), yp] = kalman_1d(R1,R2,theta, f(t), A, P(t-1));
end

g = e./P.*e;


figure
plot(g)

figure
plot(f)