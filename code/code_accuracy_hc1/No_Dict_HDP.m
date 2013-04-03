function spl = No_Dict_HDP(X,K,H_K,C_K,burnin,num,space)
randn('state',0); rand('state',0); 
[M] = length(X); %%% M= the number of channels; 
for m=1:M
X{m}=(X{m}-mean(X{m},2)*ones(1,size(X{m},2)));
n(m)=size(X{m},2);
end
nidx=cumsum([0 n]);
p=size(X{1},1);
K=min(K,p);
S=X;
for m=1:M 
%%%%%% local cluster%%%%
PC_z{m}=rand(n(m),C_K);
PC_z{m}=PC_z{m}./repmat(sum(PC_z{m},2),1,C_K);
for i=1:n(m)
C_z{m}(i)=randsample(C_K,1,true,PC_z{m}(i,:));
end
%%%%%%%% global cluster%%%%%%%%%
PH_z{m}=rand(C_K,H_K);
PH_z{m}=PH_z{m}./repmat(sum(PH_z{m},2),1,H_K);
for i=1:C_K
H_z(i,m)=randsample(H_K,1,true,PH_z{m}(i,:));
end
end
z=ones(K,1);
zz=logical(double(z)*double(z)');
G_v0=K;
G_w0=1e2*eye(K);
G_beta0=1;
for k=1:H_K
G_lamda{k}=eye(K);
G_mu{k}=zeros(K,1);
end

%%%%the parameters in the prior Beta of v
H_a=1e-0;
H_b=1e-0;
HV_lamda=gamrnd(H_a,1/H_b);
for k=1:H_K-1
    H_v(k)=betarnd(1,HV_lamda);
end
H_v(H_K)=1;

C_a=1e-0;
C_b=1e-1;
CV_lamda=gamrnd(C_a,1/C_b*ones(1,M));
for m=1:M
for k=1:C_K-1
    C_v(k,m)=betarnd(1,CV_lamda(m));
end
end
C_v(C_K,:)=1;


%%%%the parameters in the prior Gamma of lamda

maxit = burnin + num*space;
iter = 0;   kkk=0;acc=0;
while (iter<maxit) 
    iter = iter + 1;
                  
   tic

%%%%%%%%%%%%%%%%%%   DP Part %%%%%%%%%%%%%%%%

 for m=1:M
for k=1:C_K

    RR = chol(reshape(G_lamda{H_z(k,m)}(zz(:)),sum(z),sum(z)));
    xRinv = (S{m}(z,:) - repmat(G_mu{H_z(k,m)}(z),1,n(m)))' * RR;
    quadform = sum(xRinv.^2, 2);
    aa(:,k)=sum(log(diag(RR)))-0.5*quadform;
    PC_z{m}(:,k)=sum(log(1-C_v(1:k-1,m)+realmin)) + log(C_v(k,m)+realmin)+sum(log(diag(RR)))-0.5*quadform;

end
 PC_z{m} = exp(PC_z{m} + repmat(-max(PC_z{m},[],2),[1 C_K])) + realmin;

   PC_z{m} = PC_z{m}./repmat(sum(PC_z{m},2),[1 C_K]);
   
       for i=1:n(m)
         C_z{m}(i)=discreternd(1,PC_z{m}(i,:));      
       end
       CV_lamda(m)=gamrnd(C_a+C_K-1,1/(C_b-sum(log(1-C_v(1:C_K-1,m)+realmin))));

     for k=1:C_K-1       
            C_v(k,m)=betarnd( 1+sum(C_z{m}(:)==k),CV_lamda(m)+sum(C_z{m}(:)>k));
     end
            C_v(C_K,m)=1;
            
end

for m=1:M
   for k=1:C_K
        k_pos=find(C_z{m}(:)==k);
        kM=length(k_pos);
    if kM>0
       for jj=1:H_K

         RR = chol(reshape(G_lamda{jj}(zz(:)),sum(z),sum(z)));
         midval2=sum(log(diag(RR)));        
         midval1=sum(log(1-H_v(1:jj-1)+realmin)) + log(H_v(jj)+realmin);        
         xRinv = (S{m}(z,k_pos) - repmat(G_mu{jj}(z),1,kM))' * RR;         
         
         quadform = sum(sum(xRinv.^2, 2));
         PH_z{m}(k,jj)=midval1+midval2*kM-0.5*quadform;
       end
       PH_z{m}(k,:) = exp(PH_z{m}(k,:) -max(PH_z{m}(k,:))) + realmin;
       PH_z{m}(k,:) = PH_z{m}(k,:)./sum(PH_z{m}(k,:));
    else
       PH_z{m}(k,:)=midval1;
       PH_z{m}(k,:) = exp(PH_z{m}(k,:) -max(PH_z{m}(k,:))) + realmin;
       PH_z{m}(k,:) = PH_z{m}(k,:)./sum(PH_z{m}(k,:));       
    end
   end


       for k=1:C_K
        H_z(k,m)=discreternd(1,PH_z{m}(k,:));  
       end
     
end

  HV_lamda=gamrnd(H_a+H_K-1,1/(H_b-sum(log(1-H_v(1:H_K-1)+realmin))));

     for jj=1:H_K-1       
            H_v(jj)=betarnd( 1+sum(H_z(:)==jj),HV_lamda+sum(H_z(:)>jj));
     end
            H_v(H_K)=1;                   
            
for jj=1:H_K
    colS=[];
    for m=1:M
        h_pos=find(H_z(:,m)==jj);
        hM=length(h_pos);
        colset=[];
        if hM>0
         for i=1:hM
           colset=[colset find(C_z{m}(:)==h_pos(i))'];
         end
         colS=[colS S{m}(:,colset)];
        end        
    end
       kM=size(colS,2);
       if kM==0
        G_mu{jj}= randn(K,1);
        G_lamda{jj}= wishrnd(G_w0,G_v0);        
       else
        averS=mean(colS,2);
        G_s=cov(colS');
        G_beta=G_beta0+kM;
        G_v=G_v0+kM;

        invG_w=G_w0+G_s*kM+G_beta0*kM/G_beta*averS*(averS)';
        
        G_w=inv(invG_w);
        G_w=(G_w+G_w')/2;
        G_lamda{jj}= wishrnd(G_w,G_v);
        
        G_mumu=kM*averS/G_beta;
        Sinvsigma = chol( inv(G_lamda{jj}*G_beta))'; 
        G_mu{jj} = Sinvsigma*randn(K,1)+G_mumu;

        end
end

            %%%%%%%%%% Sampling S %%%%%%%%
                   
    % There might be numerical problems for small alpha.
    ndx = iter - burnin;
    test = mod(ndx,space);
    
     midC=[];
    for m=1:M
    midC=[midC; H_z(C_z{m}(:),m)];
    end
    uniqC=unique(midC(:));

    numC(iter)=length(uniqC);
    numZ(iter)=sum(z);
    for m=1:M
         tablecc{m}=H_z(:,m).';
         datatt{m}=C_z{m};
         datacc{m}=tablecc{m}(datatt{m});
    end
    if (ndx>0) && (test==0)
        spl.H_z{ndx} = datacc;
    end


fprintf('iter %d:  Classnum1=%d Gclassnum=%d%n',iter,length(unique(C_z{1})),numC(iter));
toc
  if mod(iter,10)==0
%         close all
%                 figure(111),
%                         zsum=sum(zw,2);
%             [vv,ord]=sort(-zsum);
%                  RowNum = fix(sqrt(sum(z)));
%             ColNum = ceil(sum(z)/RowNum);
%              for kk=1:sum(z)
%                  subplot(RowNum,ColNum,kk),plot(A(:,ord(kk)));
%                  set(gca,'units','points'); 
%              end
                for m=1:M
                     tablecc{m}=H_z(:,m).';
                     datatt{m}=C_z{m};
                     datacc{m}=tablecc{m}(datatt{m});
                end
               figure(100);
               for m=1:M
                   subplot(2,2,m);plot(datacc{m},'r+');
               end
  end

end


