function spl = DPBPFA_TIME_v6(X,K,H_K,TuningParameter1,burnin,num,space)

%%% Hierarchical Dirichelet Process and Beta Process Pseudo-SVD via Markov Chain Monte Carlo (HDPBPSVD_MCMC)
%%% written by Bo Chen, 4.19
%%%%% Please cite: Bo Chen, David Carlson and Lawrence Carin,
%%%%% 'On the Analysis of Multi-Channel Neural Spike Data', NIPS 2011.
%----------parameter setting----------------------
%G_w0=10^2*eye(K); d0 = 10^-4*ones(p,1); good result on HC-1 data
%-------------------------------------------------
% CRS20120914 original name Testing_DPBPSVD_Time_depedent_model
% renamed to DPBPFA_TIME_v6.m
randn('state',0); rand('state',0); 
[M] = length(X); %%% M= the number of channels;

for m=1:M
    n(m)=size(X{m},2);   
    X{m}=(X{m}-mean(X{m},2)*ones(1,size(X{m},2)));
%     X{m}=(X{m});
end
Len=n(1);
nidx=cumsum([0 n]);
p=size(X{1},1);
K=min(K,p);
c0 = 10^-0*ones(p,1); d0 = 10^-6*ones(p,1);% d0 = 10^-4*ones(p,1); good result
e0 = 1/K*ones(K,1); f0 = 1*(K-1)/K*ones(K,1);
g0 = 1e-0*ones(p,K); h0 = 1*10^-6*ones(p,K);
w0 = 0.01; eta0 = 0;
e1 = 1/Len*10^-1; f1 = (Len-1)/Len;
pai = betarnd(e0,f0);

[U0,H0,V0] = svd(cell2mat(X),'econ');
 A=U0(:,1:K);
 SS = H0(1:K,1:K)*V0(:,1:K)';

for m=1:M
    S{m}=SS(:,nidx(m)+1:nidx(m+1));
    RSS(:,:,m) = S{m}*S{m}'; 
%%%%%% local cluster%%%%
%  phi(:,m) = gamrnd(c0,1./(d0));
    phi(:,m)=TuningParameter1*ones(p,1);
end
PC_z=rand(Len,H_K);
PC_z=PC_z./repmat(sum(PC_z,2),1,H_K);
for i=1:Len
    H_z(i)=randsample(H_K,1,true,PC_z(i,:));
end
w = rand(K,M);
z = rand(K,1)<pai;

alpha = gamrnd(g0,1./(h0));
G_v0=K;
G_w0=eye(K);
G_beta0=1;
for i=1:M
    for k=1:H_K
%      G_lamda{k,i}=eye(K);
%      G_mu{k,i}=zeros(K,1);
     G_lamda{k,i}=wishrnd(G_w0,G_v0);
     G_mu{k,i}=randn(K,1);
    end
end

%%%%the parameters in the prior Beta of v
H_a=1e-0;
H_b=1e-1;
HV_lamda=gamrnd(H_a,1/H_b);
HV_lamda=10;
for k=1:H_K-1
    H_v(k)=betarnd(1,HV_lamda);
end
H_v(H_K)=1;


%%%%the parameters in the prior Gamma of lamda
flag=1;
maxit = burnin + num*space;
iter = 0;   kkk=0;acc=0;
while (iter<maxit) 
    iter = iter + 1;
                  
   tic

for k=1:K
      signZ=z(k);
      z(k) = 0;
      midval1=0;midval2=0;
  for m=1:M
       zw=z.*w(:,m);
       G = A'.*(ones(K,1)*phi(:,m)');  E{m} = G*A; 
       F{m} = G*X{m}; 
       midval1=midval1+w(k)^2*E{m}(k,k)*RSS(k,k,m);
       midval2=midval2+reshape(F{m}(k,:),1,size(F{m},2))*reshape(S{m}(k,:),size(S{m},2),1)-E{m}(k,:)*(zw.*RSS(:,k,m));
  end
   tmprr = log(pai(k)+eps) - log(1-pai(k)+eps)-1/2*midval1 + w(k)*midval2;
%    z(k) = binornd(1,1/(1+exp(-tmprr)));
   if signZ==0
      z(k)=binornd(1,min(1,1/exp(-tmprr)));
   else
       z(k)=binornd(1,1-min(1,exp(-tmprr))); 
   end
end
   zz=logical(double(z)*double(z)');
       
   pai = betarnd(e0 + z, f0 + 1 - z);
for m=1:M
   for k = 1:K
       w(k,m) = 0;
       tmpb =  (w0^-1+z(k)^2*E{m}(k,k)*RSS(k,k,m))^-1;
       tmpa = tmpb*(z(k)*(F{m}(k,:)*S{m}(k,:)'-E{m}(k,:)*(z.*w(:,m).*RSS(:,k,m)))+w0^-1*eta0);
       L=min(normcdf((0-tmpa)./(sqrt(tmpb)+realmin)),0.99999);
       midD=L+(1-L)*rand;
       w(k,m) = max( norminv(midD)*sqrt(tmpb)+ tmpa , 0);   
   end
end
%%%%%%%%%%%%%%%%%%   DP Part %%%%%%%%%%%%%%%%
%------------------------------------------------------------


for m=1:M
    PC_zz{m}=zeros(Len,1);
    for k=1:H_K
        RR0 = chol(reshape(G_lamda{k,m}(zz(:)),sum(z),sum(z)));
        xRinv0 = (S{m}(z,:) - repmat(G_mu{k,m}(z),1,n(m)))' * RR0;
        quadform0 = sum(xRinv0.^2, 2);
        PC_zz{m}(:,k)=sum(log(1-H_v(1:k-1))) + log(H_v(k))+sum(log(diag(RR0)))-0.5*quadform0;
      
    end
end
PC_Z=zeros(Len,H_K);
for m=1:M
    PC_Z=PC_Z+ PC_zz{m};
end
 PC_Z = exp(PC_Z + repmat(-max(PC_Z,[],2),[1 H_K])) ;

 PC_Z = PC_Z./repmat(sum(PC_Z,2),[1 H_K]);

H_z  = discreterndv2(PC_Z');
       
       HV_lamda=gamrnd(H_a+H_K-1,1/(H_b-sum(log(1-H_v(1:H_K-1)+realmin))));

     for k=1:H_K-1       
            H_v(k)=betarnd( 1+sum(H_z(:)==k)*M,HV_lamda+M*sum(H_z(:)>k));
     end
            H_v(H_K)=1; 
%---------------------------------------
    tmp0=randn(K,H_K);
  for  m=1:M         
    for k=1:H_K
           h_pos=find(H_z==k);
           colS=S{m}(:,h_pos);
           kM=size(colS,2);
           if kM==0
            G_mu{k,m}= randn(K,1);
            G_lamda{k,m}= wishrnd(G_w0,G_v0);   
%              G_mu{k,m}= zeros(K,1);
%              G_lamda{k,m}= eye(K);
           else
            averS=mean(colS,2);
            G_s=cov(colS');
            G_beta=G_beta0+kM;
            G_v=G_v0+kM;

            invG_w=G_w0+G_s*(kM-1)+G_beta0*kM/G_beta*averS*(averS)';

            G_w=inv(invG_w);
            G_w=(G_w+G_w')/2;
            G_lamda{k,m}= wishrnd(G_w,G_v);

            G_mumu=kM*averS/G_beta;
            Sinvsigma = chol(inv(G_lamda{k,m}*G_beta))'; 
%             G_mu{jj,m} = Sinvsigma*randn(K,1)+G_mumu;
             G_mu{k,m} = Sinvsigma*tmp0(:,k)+G_mumu;
            end
    end  
  end          
 %%%%%%%%%% Sampling S %%%%%%%%
 %%-------------------update weight------------------  
  temp0=randn(K,Len);
 for m=1:M
       zw= z.*w(:,m);
       zwa=repmat(zw,1,p).*A'.*repmat(phi(:,m),1,K)';
       midval=zwa*(repmat(zw,1,p).*A')';
       AX=zwa*X{m};
          for clusndx=1:H_K
            ndx=find(H_z==clusndx);
            nndx=numel(ndx);
            if nndx==0
                continue
            end
            S_sigmapart = midval+G_lamda{clusndx,m};
            S_mupart = AX(:,ndx)+...
                repmat(G_lamda{clusndx,m}*G_mu{clusndx,m},1,nndx);
            G = chol(S_sigmapart);
            S{m}(:,ndx) = G\(temp0(:,ndx)+(G')\S_mupart);
          end 
    RSS(:,:,m) = S{m}*S{m}';
 end
                  
         tmp1 = randn(p,K);  
    for k = 1:K
        if z(k)==1
            A(:,k) = 0;
            Xm =0;tmpA1=0;
            for m=1:M
            zw= z.*w(:,m);
            Xm = Xm+zw(k)*phi(:,m).*(X{m}*S{m}(k,:)' - A*(zw.*RSS(:,k,m)));
            tmpA1=tmpA1+phi(:,m)*(zw(k)^2*RSS(k,k,m));
            end
            tmpA1 = 1./(alpha(:,k) +tmpA1);
            tmpA2 = tmpA1.*Xm;
            A(:,k) = tmpA2 + tmpA1.^(0.5).*tmp1(:,k);
        else  % Draw from base
            A(:,k) = alpha(:,k).^(-0.5).*tmp1(:,k);
        end
    end  
    % There might be numerical problems for small alpha.
    alpha = gamrnd(g0+1/2,1./(h0+1/2*A.^2));
    sres=[];        rg =find(z==1);
    for m=1:M
        zw= z.*w(:,m);
        rX= A(:,rg)*(diag(zw(rg))*S{m}(rg,:));
        res = X{m} - rX;
        phi(:,m) = gamrnd(c0+0.5*n(m),1./(d0+0.5*sum(res.^2,2)));
        %----------------------impose Restrictionss-------------------
        idx= find(phi(:,m)>TuningParameter1);
        if length(idx)
            phi(idx,m)=TuningParameter1*ones(length(idx),1);
        end
%         phi(:,m) =TuningParameter1^-1*ones(p,1);%%%%%%%%%%------------
        %--------------------------------------------------------------
        sres =[sres sqrt(sum(res.^2,1))];
%         sres =[sres sqrt(sum(res.^2,1))./sqrt(sum(X{m}.^2,1))];
    end
    mse(iter)=mean(sres);
    ndx = iter - burnin;
    test = mod(ndx,space);
    

    uniqC=unique(H_z);
    ColC=[];
    for i=1:length(uniqC)
        tmp=length(find(H_z==uniqC(i)));
        if tmp>50
         ColC =[ColC,uniqC(i)];
        end
    end

    numC(iter)=length(uniqC);
    numZ(iter)=sum(z);
    C_z=[];
    if (ndx>0) && (test==0)
        spl.Likelihood(ndx)=Loglikelihood(S,G_lamda,G_mu,H_z,C_z,H_K,flag);
        spl.H_z{ndx} = H_z;
        spl.A = A;
        spl.S = S;
        spl.w = w;
        spl.z = z;
        spl.phi = phi;
        spl.alpha = alpha;
        spl.G_mu=G_mu;
        spl.G_lamda=G_lamda;
        spl.numC=numC;
        spl.pai=pai;
        spl.H_v=H_v;
        spl.HV_lamda=HV_lamda;
        spl.numCluster(ndx)=length(ColC);

    end
        spl.acc=acc;


% fprintf('iter %d: mse=%g: numDic=%g  Gclassnum=%d FirstOpenum: %d%n',iter,mse(iter),sum(z(:,1)),numC(iter),FirstOpenum);
fprintf('iter %d: mse=%g: numDic=%g %n',iter,mse(iter),sum(z(:,1)));
toc
%   if mod(iter,100)==0
% if iter>=1
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
% figure(800),plot(H_z,'r+');
%             figure(800),plot(H_z,'r+'); %CRS 20120914 turned off
%             plotting. of cluster memebership.
%    pause(0.2)
%   end

end


