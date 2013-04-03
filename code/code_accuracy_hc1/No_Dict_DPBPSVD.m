function spl = No_Dict_DPBPSVD(X,K,H_K,burnin,num,space)

%-------------------------------------------------
randn('state',0); rand('state',0); 
[M] = length(X); %%% M= the number of channels; 
for m=1:M
      X{m}=(X{m}-mean(X{m},2)*ones(1,size(X{m},2)));
      n(m)=size(X{m},2);
end
Len=n(1);
nidx=cumsum([0 n]);
p=size(X{1},1);
K=min(K,p);
S=X;
PC_z=rand(Len,H_K);
PC_z=PC_z./repmat(sum(PC_z,2),1,H_K);
for i=1:Len
    H_z(i)=randsample(H_K,1,true,PC_z(i,:));
end
z = ones(K,1);
 zz=logical(double(z)*double(z)');
G_v0=K;
G_w0=eye(K)/10;
G_beta0=1;
for i=1:M
    for k=1:H_K
%     G_lamda{k,i}=eye(K);
%     G_mu{k,i}=zeros(K,1);
    G_lamda{k,i}=wishrnd(G_w0,G_v0);

    G_mu{k,i}=rand(K,1);
    end
end

%%%%the parameters in the prior Beta of v
H_a=1e-0;
H_b=1e-0;
HV_lamda=gamrnd(H_a,1/H_b);
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
   
% for i=1:Len
%     H_z(i)=discreternd(1,PC_Z(i,:));      
% end
       H_z  = discreterndv2(PC_Z');
       HV_lamda=gamrnd(H_a+H_K-1,1/(H_b-sum(log(1-H_v(1:H_K-1)+realmin))));

     for jj=1:H_K-1       
            H_v(jj)=betarnd( 1+sum(H_z(:)==jj)*M,HV_lamda+M*sum(H_z(:)>jj));
     end
            H_v(H_K)=1; 
%---------------------------------------
    tmp0=randn(K,H_K);
  for  m=1:M         
    for jj=1:H_K
           h_pos=find(H_z==jj);
           colS=S{m}(:,h_pos);
           kM=size(colS,2);
           if kM==0
            G_mu{jj,m}= randn(K,1);
            G_lamda{jj,m}= wishrnd(G_w0,G_v0);        
           else
            averS=mean(colS,2);
            G_s=cov(colS');
            G_beta=G_beta0+kM;
            G_v=G_v0+kM;

            invG_w=G_w0+G_s*kM+G_beta0*kM/G_beta*averS*(averS)';

            G_w=inv(invG_w);
            G_w=(G_w+G_w')/2;
            G_lamda{jj,m}= wishrnd(G_w,G_v);

            G_mumu=kM*averS/G_beta;
            Sinvsigma = chol( inv(G_lamda{jj,m}*G_beta))'; 
%             G_mu{jj,m} = Sinvsigma*randn(K,1)+G_mumu;
             G_mu{jj,m} = Sinvsigma*tmp0(:,jj)+G_mumu;
            end
    end  
  end
  %------------------------------------------------------------

 
           

 
 
    ndx = iter - burnin;
    test = mod(ndx,space);
    

    uniqC=unique(H_z);

    numC(iter)=length(uniqC);
    C_z=[];
    if (ndx>0) && (test==0)
%         spl.Likelihood(ndx)=Loglikelihood(S,G_lamda,G_mu,H_z,C_z,H_K,flag);
        spl.H_z{ndx} = H_z;

    end
        spl.acc=acc;


% fprintf('iter %d: mse=%g: numDic=%g  Gclassnum=%d FirstOpenum: %d%n',iter,mse(iter),sum(z(:,1)),numC(iter),FirstOpenum);
fprintf('iter %d:  Gclassnum=%d%n',iter,numC(iter));
toc
  if mod(iter,100)==0
% if iter>=1

            figure(800),plot(H_z,'r+');
pause(0.2)
  end

end


