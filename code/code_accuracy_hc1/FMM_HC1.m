function spl = FMM_HC1(X,K,H_K,burnin,num,space)

%%% Hierarchical Dirichelet Process and Beta Process Pseudo-SVD via Markov Chain Monte Carlo (HDPBPSVD_MCMC)
%%% written by Bo Chen, 4.19
%%%%% Please cite: Bo Chen, David Carlson and Lawrence Carin,
%%%%% 'On the Analysis of Multi-Channel Neural Spike Data', NIPS 2011.
%----------parameter setting----------------------
%-------------------------------------------------
randn('state',0); rand('state',0); 
[M] = length(X); %%% M is day number. N= the number of channels;
N=size(X{1},3);
for m=1:M 
     n(m)=size(X{m},2);
    for Ch=1:N 
        X{m}(:,:,Ch)=(X{m}(:,:,Ch)-repmat(mean(X{m}(:,:,Ch),2),1,n(m)));
    end

end
nidx=cumsum([0 n]);
p=size(X{1},1);
K=min(K,p);
c0 = 10^-0*ones(p,1); d0 = 1*10^-6*ones(p,1);% d0 = 10^-4*ones(p,1); good result
e0 = 1/K*ones(K,1); f0 = 1*(K-1)/K*ones(K,1);
g0 = 1e-0*ones(p,K); h0 = 1*10^-6*ones(p,K);
w0 = 0.01; eta0 = 0;
pai = betarnd(e0,f0);
%------------------------------------
XX=cell(1,M);
for m=1:M
    XX{m}=reshape(X{m},p,n(m)*N);
    
end
[U0,H0,V0] = svd(cell2mat(XX),'econ');
 A=U0(:,1:K);
 SS = H0(1:K,1:K)*V0(:,1:K)';

for m=1:M
    tmpS=SS(:,nidx(m)*N+1:nidx(m+1)*N);
    for Ch=1:N
    S{m}(:,:,Ch)=tmpS(:,n(m)*(Ch-1)+1:n(m)*Ch);
    tmp0=reshape(S{m}(:,:,Ch),K,n(m));
    RSS{m}(:,:,Ch) = tmp0*tmp0'; 
%%%%%% local cluster%%%%
    phi{m}(:,Ch) = gamrnd(c0,1./(d0));
    phi{m}(:,Ch)=5e5*ones(p,1);
    end
end
%---------------------------------------
for m=1:M
PC_z{m}=rand(n(m),H_K);
PC_z{m}=PC_z{m}./repmat(sum(PC_z{m},2),1,H_K);
for i=1:n(m)
H_z{m}(i)=randsample(H_K,1,true,PC_z{m}(i,:));
end
%%%%%%%% global cluster%%%%%%%%%
w{m} = rand(K,N);
end
z = rand(K,1)<pai;

alpha = gamrnd(g0,1./(h0));
G_v0=K;
G_w0=eye(K)*1e0;
G_beta0=1;
for Ch=1:N
    for k=1:H_K
%     G_lamda{k,Ch}=eye(K);
%     G_mu{k,Ch}=zeros(K,1);
    G_lamda{k,Ch}=wishrnd(G_w0,G_v0);
    G_mu{k,Ch}   =randn(K,1);
    end
end

%%%%the parameters in the prior Beta of v
%%%%the parameters in the prior Beta of v
%-------focused model added--------------

alpha0=1e-2;
alpha1 =1e-2;
% wpai=betarnd(alpha0,alpha1,[1,H_K]);
wpai=ones(1,H_K)*1e-1;
Binary=rand(M,H_K)>0;
GammaPrior_a0=1e-2;%lambda0
GammaPrior_a1=1e-2;
% Pro=betarnd(1,1,[1,H_K]);
Gamma_T=gamrnd(GammaPrior_a0,1/GammaPrior_a1,[1,H_K]);
NumberCluster=zeros(M,H_K);
%%%-----------------------------------
% for m=1:M
%     H_v(m,:)=randdir(Gamma_T.*Binary(m,:)+eps);
% end
%--------------------------------
H_a=1e-0;
H_b=1e-1;
HV_lamda=gamrnd(H_a,1/H_b);
% HV_lamda0=1;
% HV_lamda1=1;
for k=1:H_K-1
    H_v(k)=betarnd(1,HV_lamda);
%     H_v(k)=betarnd(HV_lamda0,HV_lamda1);
end
H_v =rand(H_K);
H_v(H_K)=1;
%%%%the parameters in the prior Gamma of lamda
flag=3;
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
      for Ch=1:N
       zw=z.*w{m}(:,Ch);
       G = A'.*(ones(K,1)*phi{m}(:,Ch)');  E{Ch,m} = G*A; 
       F{Ch,m} = G*X{m}(:,:,Ch); 
       midval1=midval1+w{m}(k,Ch)^2*E{Ch,m}(k,k)*RSS{m}(k,k,Ch);
       midval2=midval2+w{m}(k,Ch)*(reshape(F{Ch,m}(k,:),1,size(F{Ch,m},2))...
           *reshape(S{m}(k,:,Ch),size(S{m}(:,:,Ch),2),1)-E{Ch,m}(k,:)*(zw.*RSS{m}(:,k,Ch)));
      end
  end
   tmprr = log(pai(k)+eps) - log(1-pai(k)+eps)-1/2*midval1 + midval2;
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
    for Ch=1:N
       for k = 1:K
           w{m}(k,Ch) = 0;
           tmpb =  (w0^-1+z(k)^2*E{Ch,m}(k,k)*RSS{m}(k,k,Ch))^-1;
           tmpa = tmpb*(z(k)*(F{Ch,m}(k,:)*S{m}(k,:,Ch)'-E{Ch,m}(k,:)*(z.*w{m}(:,Ch).*RSS{m}(:,k,Ch)))+w0^-1*eta0);
           L=min(normcdf((0-tmpa)./(sqrt(tmpb)+realmin)),0.99999);
           midD=L+(1-L)*rand;
           w{m}(k,Ch) = max( norminv(midD)*sqrt(tmpb)+ tmpa , 0); 
       end
    end
end
%%%%%%%%%%%%%%%%%%   DP Part %%%%%%%%%%%%%%%%
%------------------------------------------------------------
%------------------C_v weights of atoms.--------------------
for m=1:M

    for k=1:H_K
        clear tempPC_z
        if Binary(m,k)==1
            for Ch=1:N
            RR = chol(reshape(G_lamda{k,Ch}(zz(:)),sum(z),sum(z)));
            xRinv = (S{m}(z,:,Ch) - repmat(G_mu{k,Ch}(z),1,n(m)))' * RR;%modification
            quadform = sum(xRinv.^2, 2);
            tempPC_z(:,Ch)=sum(log(diag(RR)))-0.5*quadform;
            end
        else 
            tempPC_z(:,1:N)=-10^10*ones(n(m),N);   
        end
           PC_z{m}(:,k)=sum(tempPC_z,2)+log(H_v(m,k));
    end
%      PC_z{m}=reshape(sum(tempPC_z,3),n(m),H_K);
     PC_z{m} = exp(PC_z{m} + repmat(-max(PC_z{m},[],2),[1 H_K])) + realmin;

     PC_z{m} = PC_z{m}./repmat(sum(PC_z{m},2),[1 H_K]);
     H_z{m}  = discreterndv2(PC_z{m}');
%    for i=1:n(m)
%         H_z{m}(i)=discreternd(1,PC_z{m}(i,:));      
%    end
end


%---------------------------modify----------------------------
% for m=1:M
%     clear tempPC_z
%     for k=1:H_K
%        if Binary(m,k)==1
%          for Ch=1:N
%             RR = chol(reshape(G_lamda{k,Ch}(zz(:)),sum(z),sum(z)));
%             xRinv = (S{m}(z,:,Ch) - repmat(G_mu{k,Ch}(z),1,n(m)))' * RR.';
%             quadform = sum(xRinv.^2, 2);
%             tempPC_z(:,k,Ch)= sum(log(diag(RR)))-0.5*quadform ;
% %             RR=reshape(G_lamda{k,Ch}(zz(:)),sum(z),sum(z));
% %             loglikelihood=LogNormal2(S{m}(z,:,Ch),G_mu{k,Ch}(z),RR);
% % %             Loglikelihood=LogNormal2(S{m}(:,:,Ch),G_mu{k,Ch},G_lamda{k,Ch});
% %             tempPC_z(:,k,Ch)= log(H_v(m,k))+loglikelihood;
% 
%          end
%        else
%             tempPC_z(:,k,1:N)=-10^10*ones(n(m),N);  
%            
%        end
%         tempPC_z(:,k,1:N)= tempPC_z(:,k,1:N)+ log(H_v(m,k));
%     end
%      PC_z{m}=reshape(sum(tempPC_z,3),n(m),H_K);
%      PC_z{m} = exp(PC_z{m} + repmat(-max(PC_z{m},[],2),[1 H_K])) + realmin;
% 
%      PC_z{m} = PC_z{m}./repmat(sum(PC_z{m},2),[1 H_K]);
% 
%    for i=1:n(m)
%         H_z{m}(i)=discreternd(1,PC_z{m}(i,:));      
%    end
% end

%--------------------------------------------------------------




   for m=1:M
       for k=1:H_K
           NumberCluster(m,k)=length(find(H_z{m}==k))*N; 
       end
   end
%--------------------update--Gamma_T Matrix(1,K)----------------
% [Gamma_T, rej_ac,Pro] = RejSampler_GaNeg_1(Gamma_T, GammaPrior_a0, Binary,  NumberCluster);
[Gamma_T, rej_ac,Pro] = RejSampler_GaNeg(Gamma_T, GammaPrior_a0,GammaPrior_a1,Binary,  NumberCluster);

% Pro = betarnd(a1 + sum(NumberCluster,2), b1 + sum(bsxfun(@times, Binary, Gamma_T),2));
% for k=1:H_K
%     nm=NumberCluster(:,k);
%     tmpnm=nm(Binary(:,k));
%     if isempty(tmpnm)
%         Gamma_T(k) = gamrnd(GammaPrior_a0,1/GammaPrior_a1);
%     else
%         Gamma_T(k) = Sample_r_NB( tmpnm,Gamma_T(k),Pro(Binary(:,k)),logF,GammaPrior_a0,GammaPrior_a1);
%     end
% end
 %------------------weight of binary------------
 %----------------------Binary------------------------------ 
 for m=1:M
     for k=1:H_K
         if NumberCluster(m,k)>0
             Binary(m,k)=1;
         else
             tmpratio=(log(wpai(k))-log(1-wpai(k))+Gamma_T(k)*log(1-Pro(m)));
             Binary(m,k) = binornd(1,1/(1+exp(- tmpratio)));
             Binary(m,k)=1;
         end
     end
 end
%---------------------------------------------------
%  for k=1:H_K
%      wpai(k)=betarnd(sum(Binary(:,k)),M+1-sum(Binary(:,k))+alpha0);
%  end
 for k=1:H_K
     wpai(k)=betarnd(sum(Binary(:,k))+alpha0,M+alpha1-sum(Binary(:,k)));
 end
  %---------------------------------------
%  alpha0=gamrnd(H_a+H_K-1,1/(H_b-sum(log(1-wpai(1:H_K)+realmin))));
%  alpha0=gamrnd(H_a+H_K,1/(1-sum(log(wpai(1:H_K)))));
%------------------------------weights of atoms updated---------
   for m=1:M
           gampart1 =Gamma_T.*Binary(m,:)+ NumberCluster(m,:);
           gampart2 =Pro(m);
           weight =gamrnd(gampart1,gampart2);
           H_v(m,:)=weight./sum(weight);
   end
%-----------------------------------------------------



%%%------------------------------------------------
for Ch=1:N
    for jj=1:H_K
        colS=[];
        for m=1:M
            h_pos=find(H_z{m}==jj);
            hM=length(h_pos);
            if hM>0
               colS=[colS S{m}(:,h_pos,Ch)];
            end        
        end
            kM=size(colS,2);
           if kM==0
            G_mu{jj,Ch}= randn(K,1);
            G_lamda{jj,Ch}= wishrnd(G_w0,G_v0);        
           else
            averS=mean(colS,2);
            G_s=cov(colS');
            G_beta=G_beta0+kM;
            G_v=G_v0+kM;

            invG_w=G_w0+G_s*(kM-1)+G_beta0*kM/G_beta*averS*(averS)';

            G_w=inv(invG_w);
            G_w=(G_w+G_w')/2;
            G_lamda{jj,Ch}= wishrnd(G_w,G_v);

            G_mumu=kM*averS/G_beta;
            Sinvsigma = chol( inv(G_lamda{jj,Ch}*G_beta)).'; 
            G_mu{jj,Ch} = Sinvsigma*randn(K,1)+G_mumu;

            end
    end
end

%----------------------------------------------------------------------------
%     for jj=1:H_K
%         colS=[];
%         for m=1:M
%             h_pos=find(H_z(:,m)==jj);
%             hM=length(h_pos);
%             colset=[];
%             if hM>0
%                 for Ch=1:N
%                      for i=1:hM
%                        colset=[colset find(C_z{m}(:)==h_pos(i))'];
%                      end
%                      colS=[colS S{m}(:,colset,Ch)];
%                 end
%             end        
%         end
%            kM=size(colS,2);
%            if kM==0
%             G_mu0{jj}= randn(K,1);
%             G_lamda0{jj}= wishrnd(G_w0,G_v0);        
%            else
%             averS=mean(colS,2);
%             G_s=cov(colS');
%             G_beta=G_beta0+kM;
%             G_v=G_v0+kM;
% 
%             invG_w=G_w0+G_s*kM+G_beta0*kM/G_beta*averS*(averS)';
% 
%             G_w=inv(invG_w);
%             G_w=(G_w+G_w')/2;
%             G_lamda0{jj}= wishrnd(G_w,G_v);
% 
%             G_mumu=kM*averS/G_beta;
%             Sinvsigma = chol( inv(G_lamda0{jj}*G_beta))'; 
%             G_mu0{jj} = Sinvsigma*randn(K,1)+G_mumu;
% 
%             end
%     end
%  for Ch=1:N
%      for jj=1:H_K
%         G_lamda{jj,Ch} =  G_lamda0{jj};
%         G_mu{jj,Ch}    =  G_mu0{jj};
%      end
%  end
 %%%%%%%%%% Sampling S %%%%%%%%
          

 for m=1:M
      temp0=randn(K,n(m));
     for Ch=1:N
       zw= z.*w{m}(:,Ch);
       zwa=repmat(zw,1,p).*A'.*repmat(phi{m}(:,Ch),1,K)';
       midval=zwa*(repmat(zw,1,p).*A')';
       AX=zwa*X{m}(:,:,Ch);
%     for i=1:n(m)  
%        S_sigmapart=midval+G_lamda{H_z{m}(i),Ch};
%        S_mupart=AX(:,i)+G_lamda{H_z{m}(i),Ch}*G_mu{H_z{m}(i),Ch};
%        G = chol(S_sigmapart);
% %        S{m}(:,i) = G\(randn(K,1)+(G')\S_mupart);
%        S{m}(:,i,Ch) = G\(temp0(:,i)+(G')\S_mupart);
%     end  
         for clusndx=1:H_K
            ndx=find(H_z{m}==clusndx);
            nndx=numel(ndx);
            if nndx==0
                continue
            end
            S_sigmapart = midval+G_lamda{clusndx,Ch};
            S_mupart = AX(:,ndx)+...
                repmat(G_lamda{clusndx,Ch}*G_mu{clusndx,Ch},1,nndx);
            G = chol(S_sigmapart);
            S{m}(:,ndx,Ch) = G\(temp0(:,ndx)+(G')\S_mupart);
          end 


    RSS{m}(:,:,Ch) =S{m}(:,:,Ch)*S{m}(:,:,Ch)';
     end
 end
 %--------------------------------------------------------------               
%------------------------------update Dictionary A--------------------  
         tmp1 = randn(p,K);  
    for k = 1:K
        if z(k)==1
            A(:,k) = 0;
            Xm =0;tmpA1=0;
            for m=1:M
                for Ch=1:N
                    zw= z.*w{m}(:,Ch);
                    Xm = Xm+zw(k)*phi{m}(:,Ch).*(X{m}(:,:,Ch)*S{m}(k,:,Ch)' - A*(zw.*RSS{m}(:,k,Ch)));
                    tmpA1=tmpA1+phi{m}(:,Ch)*(zw(k)^2*RSS{m}(k,k,Ch));
                end
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
        for Ch=1:N
        zw= z.*w{m}(:,Ch);
        rX= A(:,rg)*(diag(zw(rg))*S{m}(rg,:,Ch));
        res = X{m}(:,:,Ch) - rX;
        phi{m}(:,Ch) = gamrnd(c0+0.5*n(m),1./(d0+0.5*sum(res.^2,2)));
        sres =[sres sqrt(sum(res.^2,1))];
        end
    end
    mse(iter)=mean(sres);
    ndx = iter - burnin;
    test = mod(ndx,space);
    %------------------
   
    %-----------------------------------------
    
for m=1:M
    datacc{m}=H_z{m};
end
    uniqC=unique(cell2mat(datacc));

    numC(iter)=length(uniqC);
    numZ(iter)=sum(z);

    if (ndx>0) && (test==0)
%         spl.TotalLogLik{ndx}=TestdataLogLikelihood(Testdata,K,z,w,A,H_v,phi,G_mu,G_lamda);       
        spl.Likelihood(ndx)=Loglikelihood(S,G_lamda,G_mu,H_z,0,H_K,flag);
        spl.H_z{ndx}     = H_z;
        spl.datacc{ndx}  = datacc;
        spl.Binary{ndx}  = Binary;
        spl.Pro{ndx}     = Pro;
        spl.Gamma_T{ndx} = Gamma_T;
        spl.H_v{ndx}     = H_v;
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

    end
        spl.acc=acc;


% fprintf('iter %d: mse=%g: numDic=%g  Gclassnum=%d FirstOpenum: %d%n',iter,mse(iter),sum(z(:,1)),numC(iter),FirstOpenum);
fprintf('iter %d: mse=%g:  Gclassnum=%d  %n',iter,mse(iter),numC(iter));
% fprintf('iter %d: mse=%g: numDic=%g  Gclassnum=%d  %n',iter,mse(iter),sum(z(:,1)),numC(iter));
toc
%   if mod(iter,100)==0
if iter>=1
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
               figure(100);
               for m=1:M
                   plot(H_z{m},'r+');
               end
%                figure(101);imagesc(Binary);
%             figure(800),plot(H_z,'r+');
%             for m=1:M
%                 opennum(m)=length(find(Ct(m,:)));
%             end
%             figure(600);stem(opennum);
% for mm=1:M
%         dataXX=X{mm};
%         [coeff{mm},score{mm}]=princomp(dataXX.');
%         point=[score{mm}(:,1),score{mm}(:,2)];
%         figure(444);plot(score{mm}(:,1),score{mm}(:,2),'o');
%         zz=H_z;
%         scatterMixture(point, zz);
%      end


close(figure(112));
% figure(112),
showm=ceil(M/2);
pause(0.2)
  end

end

function Lsum= CRT_sum(x,r)
Lsum=0;
RND = r./(r+(0:(max(x)-1)));
for i=1:length(x);
    if x(i)>0
        Lsum = Lsum + sum(rand(1,x(i))<=RND(1:x(i)));
    end
end
