% Waveform alignment and reconstruction example:
load bitzle_180_drg
N=1000;
wf=wf{6}(:,1:N);
Xaligned=alignSpikes(wf);
[Xreconstructed, Xhat]=BayesianFA_MCMC(Xaligned,8);
%% Example Plot 1
[minval,minloc]=min(wf);
figure(1)
subplot(2,1,1)
plot(wf(:,minloc>8))
title('Unaligned Spikes','FontSize',16)
ylabel('Amplitude, \it units','FontSize',14)
xlabel('Sample Index','FontSize',14)
axis([1 22 -40 40])
subplot(2,1,2)
plot(Xaligned(:,minloc>8))
axis([1 22 -40 40])
title('Aligned Spikes','FontSize',16)
ylabel('Amplitude, \it units','FontSize',14)
xlabel('Sample Index','FontSize',14)
%% Example Plot 2
figure(2);clf
subplot(2,1,1)
plot(Xaligned(:,minloc>8))
axis([1 22 -40 40])
title('Aligned Spikes','FontSize',18)
ylabel('Amplitude, \it units','FontSize',16)
xlabel('Sample Index','FontSize',16)
%pdfcrop(gcf)
%print -dpdf im/alignment
subplot(2,1,2)
ndx=minloc>8;
plot(Xaligned(:,ndx))
B=isnan(Xaligned(:,ndx));
hold on
tmp=Xreconstructed(:,ndx);
tmp(~B)=nan;
plot(tmp,'r')
title('Completed Spikes','FontSize',18)
ylabel('Amplitude, \it units','FontSize',16)
xlabel('Sample Index','FontSize',16)
axis([1 22 -40 40])
%% Example Plot 3
figure(3);clf
t=30;
subplot(2,1,1)
plot(Xreconstructed(:,1:t))
title('Xreconstructed')
subplot(2,1,2)
plot(Xhat(:,1:t))
title('Xhat')

