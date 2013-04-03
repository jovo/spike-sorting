function spl = No_Dict_FMM(X,K,H_K,burnin,num,space)
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
S=X;
%---------------------------------------
z=ones(K,1);   zz=logical(double(z)*double(z)');
for m=1:M
PC_z{m}=rand(n(m),H_K);
PC_z{m}=PC_z{m}./repmat(sum(PC_z{m},2),1,H_K);
for i=1:n(m)
H_z{m}(i)=randsample(H_K,1,true,PC_z{m}(i,:));
end
%%%%%%%% global cluster%%%%%%%%%
end
G_v0=K;
G_w0=eye(K);%good  result
G_beta0=50;
% G_beta0=1000;
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
alpha0=1;
alpha1 =1;
wpai=betarnd(alpha0,alpha1,[1,H_K]);
Binary=rand(M,H_K)>0;
GammaPrior_a0=0.1;%lambda0
GammaPrior_a1=1;
% Pro=betarnd(1,1,[1,H_K]);
Gamma_T=gamrnd(GammaPrior_a0,1/GammaPrior_a1,[1,H_K]);
NumberCluster=zeros(M,H_K);
%%%-----------------------------------
for m=1:M
    H_v(m,:)=randdir(Gamma_T.*Binary(m,:)+eps);
end
%--------------------------------
%%%%the parameters in the prior Gamma of lamda
flag=3;
maxit = burnin + num*space;
iter = 0;   kkk=0;acc=0;
while (iter<maxit) 
    iter = iter + 1;
                  
   tic

%%%%%%%%%%%%%%%%%%   DP Part %%%%%%%%%%%%%%%%
%------------------------------------------------------------
%------------------C_v weights of atoms.--------------------
for m=1:M

    for k=1:H_K
        clear tempPC_z
        if Binary(m,k)==1
            for Ch=1:N
            RR = chol(reshape(G_lamda{k,Ch}(zz(:)),sum(z),sum(z)));
            xRinv = (S{m}(z,:,Ch) - repmat(G_mu{k,Ch}(z),1,n(m)))' * RR.';
            quadform = sum(xRinv.^2, 2);
            tempPC_z(:,Ch)=sum(log(diag(RR)))-0.5*quadform;
            end
        else 
            tempPC_z(:,1:N)=-10^10*ones(n(m),N);   
        end
           PC_z{m}(:,k)=sum(tempPC_z,2)+log(H_v(m,k));
    end
     PC_z{m} = exp(PC_z{m} + repmat(-max(PC_z{m},[],2),[1 H_K])) + realmin;

     PC_z{m} = PC_z{m}./repmat(sum(PC_z{m},2),[1 H_K]);

%    for i=1:n(m)
%         H_z{m}(i)=discreternd(1,PC_z{m}(i,:));      
%    end
      H_z{m}  = discreterndv2(PC_z{m}');
end
   for m=1:M
       for k=1:H_K
           NumberCluster(m,k)=length(find(H_z{m}==k))*N; 
       end
   end
%--------------------update--Gamma_T Matrix(1,K)----------------
% [Gamma_T, rej_ac,Pro] = RejSampler_GaNeg_1(Gamma_T, GammaPrior_a0, Binary,  NumberCluster);
[Gamma_T, rej_ac,Pro] = RejSampler_GaNeg(Gamma_T, GammaPrior_a0,GammaPrior_a1,Binary,  NumberCluster);

 %------------------weight of binary------------
 %----------------------Binary------------------------------ 
 for m=1:M
     for k=1:H_K
         if NumberCluster(m,k)>0
             Binary(m,k)=1;
         else
             tmpratio=(log(wpai(k))-log(1-wpai(k))+Gamma_T(k)*log(1-Pro(m)));
             Binary(m,k) = binornd(1,1/(1+exp(- tmpratio)));
%              Binary(m,k)=1;
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
%----------------------------------------------------------------------------
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

    ndx = iter - burnin;
    test = mod(ndx,space);
    %------------------
   
    %-----------------------------------------
    
for m=1:M
    datacc{m}=H_z{m};
end
    uniqC=unique(cell2mat(datacc));

    numC(iter)=length(uniqC);

    if (ndx>0) && (test==0)
%         spl.TotalLogLik{ndx}=TestdataLogLikelihood(Testdata,K,z,w,A,H_v,phi,G_mu,G_lamda);       
%         spl.Likelihood(ndx)=Loglikelihood(S,G_lamda,G_mu,H_z,0,H_K,flag);
        spl.H_z{ndx}     = H_z;
        spl.datacc{ndx}  = datacc;
        spl.Binary{ndx}  = Binary;
        spl.Pro{ndx}     = Pro;
        spl.Gamma_T{ndx} = Gamma_T;
        spl.H_v{ndx}     = H_v;

    end

% fprintf('iter %d: mse=%g: numDic=%g  Gclassnum=%d FirstOpenum: %d%n',iter,mse(iter),sum(z(:,1)),numC(iter),FirstOpenum);
% fprintf('iter %d: mse=%g: numDic=%g  Gclassnum=%d rej_ac=%s %n',iter,mse(iter),sum(z(:,1)),numC(iter),rej_ac);
% fprintf('iter %d: mse=%g: numDic=%g  Gclassnum=%d  %n',iter,mse(iter),sum(z(:,1)),numC(iter));
toc
fprintf('iter %d:  Gclassnum=%d%n',iter,numC(iter));
  if mod(iter,100)==0
% if iter>=1

               figure(100);
               for m=1:M
                   subplot(1,1,m);plot(H_z{m},'r+');
               end

pause(0.2)
  end

end


