clear, clc

n0=500;
n1=n0;

T=5;
d=3;
mu0=zeros(T,1);
mu1=mu0; mu1(1)=0.5;

Sigma=eye(T);
Sigma(1)=0.01;

x0=mvnrnd(mu0, Sigma, n0);
x1=mvnrnd(mu1, Sigma, n1);

figure(1), clf, 
subplot(141), hold all, plot3(x0(:,1),x0(:,2),x0(:,3),'r.')
subplot(141), plot3(x1(:,1),x1(:,2),x1(:,3),'b+')
title('original'), grid on

% PCA 
x0means=repmat(mean(x0),n0,1);
x1means=repmat(mean(x1),n1,1);
x0centered=x0-x0means;
x1centered=x1-x0means;
[U D V] = svd([x0centered; x1centered]);
x0_PCA = x0centered*V(:,1:d)*sqrt(D(1:d,1:d))+x0means(:,1:d);
x1_PCA = x1centered*V(:,1:d)*sqrt(D(1:d,1:d))+x1means(:,1:d);
subplot(142), hold all,  plot3(x0_PCA(:,1), x0_PCA(:,2), x0_PCA(:,3),'r.')
subplot(142), plot3(x1_PCA(:,1), x1_PCA(:,2), x1_PCA(:,3),'b+')
title('pca'), grid on

% RP
[RP,~] = qr(randn(T,d),0);
x0_RP = x0*RP;
x1_RP = x1*RP;
subplot(143), hold all, plot3(x0_RP(:,1), x0_RP(:,2), x0_RP(:,3),'r.')
subplot(143), plot3(x1_RP(:,1), x1_RP(:,2), x1_RP(:,3),'b+')
title('rp'), grid on

% SDA
delta=mean(x0)-mean(x1);
[SP, R, E]=qr([delta' V],0);
x0_SP = x0*SP(:,1:d);
x1_SP = x1*SP(:,1:d);
subplot(144), hold all, plot3(x0_SP(:,1), x0_SP(:,2), x0_SP(:,3),'r.')
subplot(144), plot3(x1_SP(:,1), x1_SP(:,2), x1_SP(:,3),'b+')
title('sp'), grid on


wh=[4 2];   %set paper width and height
set(gcf,'PaperSize',wh,'PaperPosition',[0 0 wh],'Color','w');
print('PC_fail_demo','-dpdf')