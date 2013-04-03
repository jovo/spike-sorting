if 1
clear all;close all
load HarrisData
 X=[];
numgroup=4;
for mm=1:numgroup
    YY=[];
    YY=ec_spikes(mm,:,:);
    YY=reshape(YY,size(YY,2),size(YY,3));
    meanmaxYY(mm)=mean(max(YY));
    datass_train{mm}=YY/mean(max(YY));
%     datass_train{mm}=YY;
end
[p,n]=size(YY);
end
%-----------------------------------------------------
TuningParameter1=10^6;
 burnin =200; num =200; space = 1;Ncentres=20; K=32;
 spl = DPBPFA_TIME_v6(datass_train,K,Ncentres,TuningParameter1,burnin,num,space);
 I=findI_1(spl,label);
 figure;plot(spl.H_z{I},'r+')
 for mm=1:numgroup
    point=[spl.S{1}(1,:)/100;spl.S{1}(2,:)/100].';
    figure;plot(point(:,1),point(:,2),'o');

    z=spl.H_z{I};

    scatterMixture(point, z);
    set(gca,'fontsize',25);
    xlabel('Learned feature 1(a.u.)','fontsize',25);ylabel('Learned feature 2 (a.u.)','fontsize',15);
 end
     %--------------------------------------
