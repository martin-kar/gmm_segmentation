% This Matlab script segments an example trajectory using GMM.

load t1

p = Trial1(:,2:4);
% figure
% plot(p)
P = size(p,1);
X = p;

t = Trial1(:,1)/1000;

% Project trajectory onto 1D using PCA:
[A2,score_red] = pca(X, 'NumComponents',1);

% figure
plot(t,score_red)
Psi = [t score_red];

gmm = fitgmdist(Psi,3,'RegularizationValue',10^-4);

% Plot segmented trajectory:
%figure
%ezcontour(@(x1,x2)pdf(gmm,[x1 x2]),2*[min(t) max(t) min(score_red) max(score_red)])

figure
fsurf(@(x,y)reshape(pdf(gmm,[x,y]),size(x)),1.2*[min(t) max(t) min(score_red) max(score_red)])

