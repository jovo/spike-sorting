
%---------------------------Michigan Data p-----------------------------------------
if 1
clear all;close all;
 days={'09_24';'09_25';'09_26';'09_27';'09_28';'09_29';'09_30';'10_02';}; 
numday=8;
for tt=1:numday
filename=sprintf('Multidays_Rat004_spikes_device_1_forehalf_%s_data_80min_45sec',days{tt,1});
    load (filename)
    numgroup=length(Spikes);
    for mm=1:numgroup
         datass{tt}(:,:,mm)=Spikes{mm}(40+[-19:20],:);
         Spike{tt}(:,:,mm)=Spikes{mm};
         KeepSpike{tt}(:,:,mm)=Spikes{mm};
 
    end
    Len(tt)=size(datass{tt}(:,:,1),2);
    sptimes{tt}=Keeptime;
end
 Backpath='E:\Matlab\work\FA';
cd(Backpath);
end  
%-----------------main program-------------------------------------------------
partition=5;
fixnum=fix(Len/partition).';
restnum=(Len.'-(partition-1)*fixnum);
Configuration=[repmat(fixnum,1,5),restnum];
sumComfig=cumsum(Configuration,2);
numgroup=8;
for mm=1:partition
    if mm==1
     for tt=1:numday 
         for Ch=1:numgroup   
          tmp=datass{tt}(:,:,Ch);
%           datass_test{tt}(:,:,Ch)=tmp(:,sumComfig(tt,partition-1)+1:sumComfig(tt,partition));
          datass_test{tt}(:,:,Ch)=tmp;
%           tmp(:,sumComfig(tt,partition-1)+1:sumComfig(tt,partition))=[];
          datass_train{tt}(:,:,Ch)=tmp;
         end 
     end  
    burnin =5000; num = 5000; space = 1;Ncentres=20; K=40;
    spl{mm} =  Focused_Time_depedent_model_12(datass_train,datass_test,K,Ncentres,burnin,num,space);%
    end
end

%---------------plot part-------------------------------------
clear datass_test datass_train
   for tt=1:numday
         for Ch=1:numgroup
             clear tmp0
         tmp0= KeepSpike{tt}(:,1:end,Ch);
%          datass_test{tt}(:,:,Ch)=tmp0(:,sumComfig(tt,partition-1)+1:sumComfig(tt,partition));
         datass_testing{tt}(:,:,Ch)=tmp0;
%          tmp0(:,sumComfig(tt,partition-1)+1:sumComfig(tt,partition))=[];
         datass_train{tt}(:,:,Ch)=tmp0;
         end
   end
I=find(spl{mm}.Likelihood==max(spl{mm}.Likelihood));

% datass=spike;
figure(8800);
   for m=1:numday
       datacc{m}=spl{1}.H_z{I}{m};
       subplot(2,4,m);plot(datacc{m},'r+')
      set(gca,'fontsize',10);
       xlabel('Index of samples','fontsize',10);ylabel('Index of cluster','fontsize',10);
       title(['Day','-',num2str(m)]);
       n(m)=size(datacc{m},2);
   end
  figure(8801);plot(cell2mat(datacc),'r+');
   set(gca,'fontsize',15);
   xlabel('Index of Sample','fontsize',15);ylabel('Index of cluster','fontsize',15);
   SumN=cumsum(n);
   for m=1:numday
       Tmp0(:,1)=ones(Ncentres,1)*SumN(m);
       Tmp0(:,2)=[1:Ncentres].';
       hold on;plot(Tmp0(:,1), Tmp0(:,2),'b','linewidth',2);
   end
       
%     datass=Spike;
 for tt=1:numday
        uniqueIdex=unique(datacc{tt});
        for ch=1:numgroup
            figure(500+ch+10)
            bb=0;
            collectidx=[];
            for ii=1:length(uniqueIdex);
                idx=find(datacc{tt}==uniqueIdex(ii));
                 set(gca,'fontsize',10);
                 subplot(2,4,ii);
                    hold on;plot(datass_train{tt}(:,idx,ch)),%ylim([-0.2,0.2]);
                   xlabel('index of the feature','fontsize',10);ylabel('Amplitude ','fontsize',10)
                   title(['Cluster','-',num2str(uniqueIdex(ii))],'fontsize',10);
            end
        end
 end 
 for tt=1:numday
    for mm=1:numgroup
            dataXX=datass_train{tt}(:,:,mm);
            [coeff,score]=princomp(dataXX.');
            point=[score(:,1),score(:,2)];
            figure;plot(score(:,1),score(:,2),'o');
            z=datacc{tt};

            scatterMixture(point, z);
    end
 end
 %-----------------------------------------------------
  for tt=1:numday
      figure(10+tt)
    for mm=1:8
           dataXX=datass_train{tt}(:,:,mm);
          subplot(2,4,mm),plot(dataXX);
          set(gca,'fontsize',10);
       xlabel('Index of feature','fontsize',10);ylabel('Amplitude','fontsize',10);
       title(['Channel','-',num2str(aa(mm))]);
    end
 end
%------------------------------------------------------------- 
%--------------Calculating Likelihood-----------------------
mm=1;
L=length(spl{mm}.TotalLogLik);
for i=1:L
     [a,b]=size(spl{mm}.TotalLogLik{i}); 
for m=1:a
    for j=1:b
        Average(m,j,i)=mean(spl{1}.TotalLogLik{i}{m,j});
    end
end
end
AverageLike =reshape(Average,a*b,L);
tmp =mean(AverageLike);
meanAverageLike =mean(tmp);
stdAverageLike =std(tmp,0,2);
%%%---------------------------------------
mm=1;
L=length(spl{mm}.TotalLogLik);
for i=1:L
     [a,b,c]=size(spl{mm}.TotalLogLik{i});
     tmp=spl{mm}.TotalLogLik{i};
     tmp1=reshape(tmp,a*b,c);
     tmp2=sum(tmp1);
     idx=find(tmp2==0);
     tmp2(idx)=[];
     tmp3(i)=logsumexp(tmp2);

end
meantmp3=mean(tmp3);
stdtmp3=std(tmp3);
Loglik3=zeros(1,100);
for mm=1:1
L=length(spl{mm}.TotalLogLik);
for i=1:1:L
    [a,b,c]=size(spl{mm}.TotalLogLik{i});
    x=spl{mm}.TotalLogLik{i};
    for ch=1:8
        xx=x(ch,:,:);
        xxx=reshape(xx,b,c);
        for j=1:c
            Loglik(j)=logsumexp(xxx(:,j));
        end
            Loglik2(ch)=logsumexp( Loglik);
    end
    Loglik3(i)=logsumexp( Loglik2);
end
end
 %--------------HDP--------------------------
%  save HDP_result_8_days

 numgroup=length(datass);
I=find(spl.Likelihood==max(spl.Likelihood));
figure(80)
   for m=1:numgroup
     tablecc{m}=spl.H_z{I}(:,m).';
     datatt{m}=spl.C_z{I}{m};
     datacc{m}=tablecc{m}(datatt{m});
     subplot(2,4,m);plot(datacc{m},'r+')
   end
%    datass=Spike;
 for tt=1:numgroup
        uniqueIdex=unique(datacc{tt});
            figure(500+tt)
            for ii=1:length(uniqueIdex);
                idx=find(datacc{tt}==uniqueIdex(ii));
                subplot(2,3,ii);hold on;plot(datass{tt}(:,idx)),%ylim([-0.08,0.08]);
                xlabel('index of the feature','fontsize',10);ylabel('Amplitude ','fontsize',10)
                title(['Cluster','-',num2str(uniqueIdex(ii))],'fontsize',10)
            end
 end 
 
for mm=1:numgroup
        dataXX=datass{mm};
        [coeff{mm},score{mm}]=princomp(dataXX.');
        point=[score{mm}(:,1),score{mm}(:,2)];
        figure;plot(score{mm}(:,1),score{mm}(:,2),'o');
        z=datacc{mm};

        scatterMixture(point, z);
        set(gca,'fontsize',15);
        xlabel('Pc-1','fontsize',15);ylabel('Pc-2','fontsize',15);
       %title(['Channel','-','3'],'fontsize',15)
end
