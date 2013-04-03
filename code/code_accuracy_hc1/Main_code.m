%%=====================================================
%Abbrevation of method names are consistent in the paper of
% Multichannel Electrophysiological Spike Sorting via
%Joint Dictionary Learning & Mixture Modeling
%%=========================================================================================
%-----------------------%Hc-1 data-----------------------
clear all;close all
load d533101
flag=6;
if flag==0
    method_name='MDP_DL';
elseif flag==1
    method_name='FMM_DL';
elseif flag==2
     method_name='HDP_DL';
elseif flag==3
     method_name='DPBPFA_DL';
elseif flag==4
     method_name='FMM_PC';
elseif flag==5
     method_name='MDP_PC';
elseif flag==6
     method_name='HDP_PC';
end
switch method_name
    case 'MDP_DL'
%---------------main code-------------------------------------
numgroup=4;
for mm=1:numgroup
    YY=[];
    YY=ec_spikes(mm,:,:);
    YY=reshape(YY,size(YY,2),size(YY,3));
    datass_train{mm}=YY;
    datass{mm}=YY;
end
 burnin =20000; num = 20000; space = 1;Ncentres=20; K=40;
 spl =MDP_DL(datass_train,K,Ncentres,burnin,num,space);
case 'FMM_DL'
numgroup=4;
for mm=1:numgroup
    YY=[];
    YY=ec_spikes(mm,:,:);
    YY=reshape(YY,size(YY,2),size(YY,3));
    datass_train{1}(:,:,mm)=YY;
end
burnin =20000; num = 20000; space = 1;Ncentres=20; K=40;
spl = FMM_HC1(datass_train,K,Ncentres,burnin,num,space);    
case 'HDP_DL'
numgroup=4;
for mm=1:numgroup
    YY=[];
    YY=ec_spikes(mm,:,:);
    YY=reshape(YY,size(YY,2),size(YY,3));
    datass_train{mm}=YY;
end
burnin =20000; num = 20000; space = 1;Ncentres=20; Local_num=40;K=40;
spl = HDP_DL(datass_train,K,Ncentres,Local_num,burnin,num,space);
 case 'DP_DL'
YY=ec_spikes(1,:,:);
YY=reshape(YY,size(YY,2),size(YY,3));
datass{1}=YY;
burnin =20000; num = 20000; space = 1;Ncentres=20; K=40;
spl =Testing_DPBPSVD_Time_depedent_model(datass,K,Ncentres,burnin,num,space);
 case 'FMM_PC'
numgroup=4;
for mm=1:numgroup
    YY=[];
    YY=ec_spikes(mm,:,:);
    YY=reshape(YY,size(YY,2),size(YY,3));
    dataXX=YY;
    [UU0,VV0]=princomp(dataXX.');
     datass{1}(:,:,mm)=VV0(:,1:2).';
end
burnin =20000; num =20000; space = 1;Ncentres=20; K=2;
spl =No_Dict_FMM(datass,K,Ncentres,burnin,num,space);
 case 'MDP_PC'
 numgroup=4;
for mm=1:numgroup
    YY=[];
    YY=ec_spikes(mm,:,:);
    YY=reshape(YY,size(YY,2),size(YY,3));
    dataXX=YY;
    [UU0,VV0]=princomp(dataXX.');
     datass{mm}=VV0(:,1:2).';
end
      burnin =20000; num =20000; space = 1;Ncentres=20; K=2;
      spl = No_Dict_DPBPSVD(datass,K,Ncentres,burnin,num,space);
case 'HDP_PC'
    numgroup=1;
for mm=1:numgroup
    YY=[];
    YY=ec_spikes(mm,:,:);
    YY=reshape(YY,size(YY,2),size(YY,3));
    dataXX=YY;
      [UU0,VV0]=princomp(dataXX.');
     datass{mm}=VV0(:,1:2).';

end
      burnin =20000; num =20000; space = 1;Ncentres=20;Local_num=40; K=2;
      spl = No_Dict_HDP(datass,K,Ncentres,Local_num,burnin,num,space);
end
%----------------HC-1-data--------------------
% save Time_dependent_model_HC_1_Perfect_Result
[minnum,I]=findI(spl,label,flag);
     for mm=1:numgroup
        dataxx(:,:,mm)=ec_spikes(mm,:,:);
        dataXX=reshape(dataxx(:,:,mm),size(dataxx,1),size(dataxx,2));
        [coeff{mm},score{mm}]=princomp(dataXX.');
        point=[score{mm}(:,1),score{mm}(:,2)];
        figure;plot(score{mm}(:,1),score{mm}(:,2),'o');
        z=spl.H_z{I};
        scatterMixture(point, z);
        xlabel('Pc-1','fontsize',25);ylabel('Pc-2','fontsize',25);
     end
     for mm=1:numgroup
        dataxx(:,:,mm)=ec_spikes(mm,:,:);
        dataXX=reshape(dataxx(:,:,mm),size(dataxx,1),size(dataxx,2));
        [coeff{mm},score{mm}]=princomp(dataXX.');
        point=[score{mm}(:,1)/100,score{mm}(:,2)/100];
        figure;plot(score{mm}(:,1)/100,score{mm}(:,2)/100,'o');
        z=spl.H_z{I};
        scatterMixture(point, z);
            set(gca,'fontsize',25);
        xlabel(' 1st PC (a.u.)','fontsize',25);ylabel('2nd PC (a.u.)','fontsize',25);
        xlim([-4,4])
        ylim([-4.5,4.5])
     end
    for mm=1:numgroup
        point=[spl.S{mm}(1,:)/100;spl.S{mm}(2,:)/100].';
        z=spl.H_z{I};
        figure;plot(point(:,1),point(:,2),'o');
        scatterMixture(point, z);
        set(gca,'fontsize',25);
        xlabel('1st learned feature (a.u.)','fontsize',25);ylabel('2nd learned feature (a.u.)','fontsize',25);
            xlim([-4,4])
        ylim([-4.5,4.5])
     end
     
     
     
     