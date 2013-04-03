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
%     datass_train{mm}=YY/mean(max(YY));
    datass_train{mm}=YY;
end
[p,n]=size(YY);
%-------------10% spike missing data---------------
missingRate=300;
Bmatrix=ones(p,n);
for mm=1:numgroup
    tmp0=datass_train{mm}(:,1:missingRate);
    tmp0(1:10,:)=0;
    tmp0(40-15+1:40,:)=0;
    Bmatrix(1:10,1:missingRate)=0;
    Bmatrix(40-15+1:40,1:missingRate)=0; 
    datass_train{mm}(:,1:missingRate)=tmp0;
end
end
 burnin =5000; num =5000; space = 1;Ncentres=20; K=32;
spl = TSPLIT_DPBPSVD_Time_depedent_model(datass_train,Bmatrix,K,Ncentres,burnin,num,space);
% save missing_HC_1_for_paper_FullChannel.mat
%%===================
cc=0;
ReData3=0;
YY=ec_spikes(1,:,:);
YY=reshape(YY,size(YY,2),size(YY,3));
MeanX=mean(datass_train{1},2);
for I=1:10:200
      cc=cc+1;
      MeanX=mean(datass_train{1},2);
      ReData=spl.A{I}*diag(spl.z{I})*spl.S{I}{1}(:,1:300);
%       ReData=spl.A{I}*diag(spl.z{I})*spl.S{I}{1}(:,151:300);
      ReData2=ReData+MeanX*ones(1,size(ReData,2));
      ReData3= ReData3+ReData2;
      mse0 = YY(:,1:300)-ReData2;
      Mse0(:,cc) =(sqrt(sum(mse0.^2,2)))./(sum(abs(YY(:,1:300)),2));
      Examplewaveform(:,cc)=ReData2(:,100);
%       figure;plot( ReData3,'b')
%       set(gca,'fontsize',15);
%       xlabel('Feature','fontsize',15);ylabel('Amplitude ','fontsize',15);
%       hold on;plot(datass_train{1}(:,1:150),'r-.')
%       set(gca,'fontsize',15);
%       xlabel('Feature','fontsize',15);ylabel('Amplitude ','fontsize',15);
%       mse(:,cc) =   mean((ReData3- datass_train{1}(:,1:300)),2);
%        mse(:,I) =   mean((ReData3- datass_train{1}(:,151:300)),2);

end
%====================figure-2-a==================================
XConstruction=ReData3/cc;
mse = YY(:,1:300)-XConstruction(:,1:300);
Stdwaveform =std(mse,0,2);
figure;errorbar([1:40]/40,XConstruction(:,100),Stdwaveform,'k','linewidth',4)
% plot([1:40]/40,XConstruction(:,100),'k','linewidth',3);
hold on;plot([1:40]/40,YY(:,100),'-.','color',[0.8,0.8,0.8],'linewidth',4);
idx=find(datass_train{1}(:,100));
hold on;plot(idx/40,datass_train{1}(idx,100),'--','linewidth',4);
set(gca,'fontsize',20);
xlim([0,1]);ylim([-300,500]);
legend('Recovered waveform','Original waveform', 'Clipped waveform','Northeast')
xlabel('Time (msec)','fontsize',20);ylabel('Amplitude (a.u.)','fontsize',20);
%====================figure-2-b==================================
mse = YY(:,1:300)-XConstruction(:,1:300);
Mse =(sqrt(sum(mse.^2,2)))./(sum(abs(YY(:,1:300)),2));Stdwaveform =std(mse,0,2);
figure;hold on;plot([1:40]/40,Mse,'k*','linewidth',4);
%  bb=get(gca,'Yticklabel');
% nbb=size(bb,1);bb2=[num2str(str2num(bb)*100),repmat('%',nbb,1)];
% set(gca,'Yticklabel',bb2); 
set(gca,'fontsize',20);
xlabel('Time (msec)','fontsize',20);ylabel('Relative error of recovered signal ','fontsize',20);
 bb=get(gca,'Yticklabel');
nbb=size(bb,1);bb2=[num2str(str2num(bb)*100),repmat('%',nbb,1)];
set(gca,'Yticklabel',bb2); 
% title('Relative recovery errors','fontsize',15)
% hold on;errorbar([1:1:40],Mse(1:1:40),Stdwaveform(1:1:40),'r.');
subplot(2,1,2);hold on;plot(sqrt(sum(mse.^2,2))/300,'b*','linewidth',3)
set(gca,'fontsize',15);
xlabel('Samples','fontsize',15);ylabel('Errors ','fontsize',15);title('Absolute recovery errors','fontsize',15)

xlabel('Samples','fontsize',15);ylabel('Amplitude ','fontsize',15);
figure(10);plot(mean((mse),2),'r^','linewidth',2);ylim([-1,25])

