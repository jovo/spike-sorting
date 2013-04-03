%code for sparsely-firing problem%-----------------------------------------------------
%% Please Note: For ISOMAP method please refer to the paper
%% Adamos DA, Laskaris NA, Kosmidis EK, Theophilidis G
%%In quest of the missing neuron: spike sorting based on dominant-sets clustering.Comput Methods Programs Biomed.
%% For waveclus software; please go to the website,http://www.vis.caltech.edu/~rodri/Wave_clus/Wave_clus_home.htm
%%%===============================================================================================================================
%%============================================
%% please note:
%  Refine_original_waveforms % original waveforms dataset;
%vnoisewaveforms_1 %added real background noise in the original
%%waveforms, SNR is about 2.5;
%%noisewaveforms_2 added real background noise 
%%in the original waveforms, SNR is about 1.5
%%==============================================
 clear all;close all
flag=0;
if flag==0
      file_name='Original_dataset';
elseif flag==1
     file_name='NoiseLevel_1';
elseif flag==2
     file_name='NoiseLevel_2';
end
switch file_name
    case 'Original_dataset'
        load Refine_original_waveforms
        datass_train=Spike; burnin =20000; num = 20000; space = 1;Ncentres=20; K=32; option=1;
        spl = Testing_sparse_neuron(datass_train,K,Ncentres,burnin,num,space,option);
    case 'NoiseLevel_1'
        load noisewaveforms_1
         datass_train=Spike;burnin =20000; num = 20000; space = 1;Ncentres=20; K=32; option=2;
         spl = Testing_sparse_neuron(datass_train,K,Ncentres,burnin,num,space,option);
    case 'NoiseLevel_2'
        load noisewaveforms_2
         datass_train=Spike; burnin =20000; num = 20000; space = 1;Ncentres=20; K=32;option=3;
         spl = Testing_sparse_neuron(datass_train,K,Ncentres,burnin,num,space,option);   
end
 
 %%==========plot part=========================
 for mm=1:1
        DataXX=spl.S{num};
        datacc{mm}=spl.H_z{num};
        z=datacc{mm};
        Point=[DataXX(1,:);DataXX(2,:)].';
        figure;scatterMixture(Point, z);
        set(gca,'fontsize',15);
        xlabel('Learned feature-1','fontsize',15);ylabel('Learned feature-2','fontsize',15);
        box
 end
 figure(1000);
 YY=Spike;
 color=['b','r','k','c'];
     for tt=1:1
        uniqueIdex=unique(datacc{tt});
        figure(5000+1)
    for ii=1:length( uniqueIdex)
        idx=find(datacc{tt}==uniqueIdex(ii));
           hold on;plot([1:32]/40,mean(YY(:,idx)/100,2),color(ii),'linewidth',2);
    end
     set(gca,'fontsize',25);
     xlabel('Time (msec)','fontsize',25);ylabel('Amplitude (a.u.)','fontsize',25);box
     end 
     ylim([-300,300]/100);xlim([0,0.85]);box
 
 %%========================plot waveforms mean in each cluster=================
for mm=1:1
        dataXX=datass_test{mm};
        dataXX=XX;
        [coeff{mm},score{mm}]=princomp(dataXX.');
        point=[score{mm}(:,1),score{mm}(:,2),score{mm}(:,3)];
        z=datacc{mm};
        Point=[score{mm}(:,1),score{mm}(:,2)];
        
        figure;scatterMixture(Point, z);
        set(gca,'fontsize',15);
        xlabel('Pc-1','fontsize',15);ylabel('Pc-2','fontsize',15);
       box
end
 %%===========extract the typical waveforms==============
        uniqueIdex=unique(datacc{1});
        figure(502);
        flag=0;
    for ii=1:length(uniqueIdex);
        idx=find(datacc{1}==uniqueIdex(ii));
        waveformtemplate(:,ii)= mean(datass_test{1}(:,idx),2);
        subplot(2,3,ii);hold on;plot(mean(datass_test{1}(:,idx),2)),
        set(gca,'fontsize',10);
        xlabel('Samples','fontsize',10);ylabel('Amplitude ','fontsize',10);
        title(['Cluster','-',num2str(uniqueIdex(ii))],'fontsize',10);
    end
%%==============================================================
N=length(spl.Likelihood);
Numcluster=zeros(Ncentres,N);
Uniquecluster0=zeros(1,N);
for mm=1:N
     z=spl.H_z{mm};
     Uniquecluster0(mm)=length(unique(z));
     for ii=1:Ncentres
      iidex=find(z==ii);
      Numcluster(ii,mm)=length(iidex);
     end
end
[nnn,xout] = hist(Uniquecluster0,[1:8]); nn = nnn/sum(nnn); figure(888); bar(xout,nn);xlim([3,8])
set(gca,'fontsize',25);
xlabel('Number of cluster','fontsize',25);
ylabel('Probablity','fontsize',25)

%%%=====================================================================



   