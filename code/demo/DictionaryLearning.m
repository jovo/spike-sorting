function spl = DictionaryLearning(X,K,H_K,PrecisionParameter,burnin,num,space,debug)
%
%   INPUTS
%   =======================================================================
%   X       : (samples x snippet_example)
%   K       :  Number of dictionary components,  it is less than dimension 
%              of spike waveforms , i.e. K = min(K,size(X,1))
%   H_K     :  Maximal number of clusters
%   PrecisionParameter: Initialiation of phi, controls noise variance
%   burnin  :  Number of iterations from initial state to converging state. 
%   num     :  Number of iterations for collections in the converging period. 
%   space   :  Collections interval 
%   debug   :  Show information in the iterations such as runing time,
%              reconstruction error and so on when debug=1
%
%
%   IN CODE
%   ========================================================================
%   H_z : array, value for each snippet, value indicates cluster of snippet
%
%
%   OUTPUT
%   ========================================================================
%     spl.H_v = H_v;
%     spl.HV_lamda = HV_lamda;
%     spl.A = A;
%     spl.S = S;
%     spl.w = w;
%     spl.z = z;
%     spl.phi = phi;
%     spl.alpha = alpha;
%     spl.G_mu  = G_mu;
%     spl.G_lamda = G_lamda;
%     spl.numC = numC;  %# of clusters for each iteration
%     spl.pai  = pai;

randn('state',0); rand('state',0); 
n=size(X,2);
X=X/mean(max(abs(X)));
X=X-mean(X,2)*ones(1,n);
p=size(X,1);
K=min(K,p);
Len=n;
c0 = 10^-0*ones(p,1); d0 = 10^-6*ones(p,1);
e0 = 1/K*ones(K,1); f0 = 1*(K-1)/K*ones(K,1);
g0 = 1e-0*ones(p,K); h0 = 1*10^-6*ones(p,K);
w0 = 0.01; eta0 = 0; s0 = 1;
pai = betarnd(e0,f0);

[U0,H0,V0] = svd((X),'econ');
 A=U0(:,1:K);
S = H0(1:K,1:K)*V0(:,1:K)';
RSS=S*S';
phi=PrecisionParameter*ones(p,1);
PC_z=rand(Len,H_K);
PC_z=PC_z./repmat(sum(PC_z,2),1,H_K);
for i=1:Len
    H_z(i)=randsample(H_K,1,true,PC_z(i,:));
end
w = rand(K,1);
z = rand(K,1)<pai;
alpha = gamrnd(g0,1./(h0));
G_v0=K;
G_w0=eye(K)/1;
G_beta0=1;
for k=1:H_K
    G_lamda{k}=eye(K);
    G_mu{k}=zeros(K,1);
end
%%%%the parameters in the prior Beta of v
H_a=1e-0;
%%===========================
H_b=8e-1;%
%=============================
HV_lamda=gamrnd(H_a,1/H_b);
for k=1:H_K-1
    H_v(k)=betarnd(1,HV_lamda);
end
H_v(H_K)=1;
%%%%the parameters in the prior Gamma of lamda
maxit = burnin + num*space;
iter = 0;  
while (iter<maxit) 
    iter = iter + 1;
                  
   tic
%%%==================================================================
   % Notes: z that determines which dictionary components are used in the
   % model,and w determines which dictioary components are more important
   % seen Dictionary learning Section 2.1  in the reference paper
       G = A'.*(ones(K,1)*phi');  E = G*A; 
       F = G*X; 
for k=1:K
      signZ=z(k);
      z(k) = 0;
       zw=z.*w;
       midval1=w(k)^2*E(k,k)*RSS(k,k);
       midval2=F(k,:)*S(k,:)'-E(k,:)*(zw.*RSS(:,k));
       tmprr = log(pai(k)+eps) - log(1-pai(k)+eps)-1/2*midval1 + w(k)*midval2;
   if signZ==0
      z(k)=binornd(1,min(1,1/exp(-tmprr)));
   else
      z(k)=binornd(1,1-min(1,exp(-tmprr))); 
   end
end
   zz=logical(double(z)*double(z)');     
   pai = betarnd(e0 + z, f0 + 1 - z);
   for k = 1:K
       w(k) = 0;
       tmpb =  (w0^-1+z(k)^2*E(k,k)*RSS(k,k))^-1;
       tmpa = tmpb*(z(k)*(F(k,:)*S(k,:)'-E(k,:)*(z.*w.*RSS(:,k)))+w0^-1*eta0);
       w(k) = TNrnd(0,inf,tmpa,sqrt(tmpb),1); 
   end
%%%=====================================================================
%%%=============================DP Part ================================
%%%========================================================================
    %Notes: clustering operation and parameter estimations based on dirichlet process and
    % seen  Dictionary learning Section 2.1 in the reference paper
    PC_zz=zeros(Len,1);
    for k=1:H_K
        RR0 = chol(reshape(G_lamda{k}(zz(:)),sum(z),sum(z)));
        xRinv0 = (S(z,:) - repmat(G_mu{k}(z),1,n))' * RR0;
        quadform0 = sum(xRinv0.^2, 2);
        PC_zz(:,k)=sum(log(1-H_v(1:k-1))) + log(H_v(k))+sum(log(diag(RR0)))-0.5*quadform0;
      
    end
PC_Z=zeros(Len,H_K);
PC_Z= PC_zz;
PC_Z = exp(PC_Z + repmat(-max(PC_Z,[],2),[1 H_K])) ;

PC_Z = PC_Z./repmat(sum(PC_Z,2),[1 H_K]);

H_z  = discreterndv2(PC_Z');
       
HV_lamda=gamrnd(H_a+H_K-1,1/(H_b-sum(log(1-H_v(1:H_K-1)+realmin))));
     for k=1:H_K-1       
            H_v(k)=betarnd( 10^-3+sum(H_z(:)==k),HV_lamda+sum(H_z(:)>k));
     end
            H_v(H_K)=1; 
%%%============================================================================
 % updated cluster parameters including cluster means and precision
 % matrixs for each clusters
    tmp0=randn(K,H_K);        
    for k=1:H_K
           h_pos=find(H_z==k);
           colS=S(:,h_pos);
           kM=size(colS,2);
           if kM==0
            G_mu{k}= randn(K,1);
            G_lamda{k}= wishrnd(G_w0,G_v0);   
           else
            averS=mean(colS,2);
            G_s=cov(colS');
            G_beta=G_beta0+kM;
            G_v=G_v0+kM;

            invG_w=G_w0+G_s*(kM-1)+G_beta0*kM/G_beta*averS*(averS)';

            G_w=inv(invG_w);
            G_w=(G_w+G_w')/2;
            G_lamda{k}= wishrnd(G_w,G_v);

            G_mumu=kM*averS/G_beta;
            Sinvsigma = chol(inv(G_lamda{k}*G_beta))'; 
%             G_mu{k} = Sinvsigma*randn(K,1)+G_mumu;
             G_mu{k} = Sinvsigma*tmp0(:,k)+G_mumu;
            end
    end
%%%=================================================================
%%%================== DP-Part completed============================
 %%=================================================================
% Notes:signal reconstructions based on Factor analysis model, updated
% Dirtionary components ,factor scores and noise precision
% seen Dictionary learning Section 2.1  in the reference paper
 temp0=randn(K,Len);
   zw= z.*w;
   zwa=repmat(zw,1,p).*A'.*repmat(phi,1,K)';
   midval=zwa*(repmat(zw,1,p).*A')';
   AX=zwa*X;
  for clusndx=1:H_K
    ndx=find(H_z==clusndx);
    nndx=numel(ndx);
    if nndx==0
        continue
    end
    S_sigmapart = midval+G_lamda{clusndx};
    S_mupart = AX(:,ndx)+...
        repmat(G_lamda{clusndx}*G_mu{clusndx},1,nndx);
    G = chol(S_sigmapart);
    S(:,ndx) = G\(temp0(:,ndx)+(G')\S_mupart);
  end 
    RSS = S*S';   
               
         tmp1 = randn(p,K);  
    for k = 1:K
        if z(k)==1
            A(:,k) = 0;
            zw= z.*w;
            Xm = zw(k)*phi.*(X*S(k,:)' - A*(zw.*RSS(:,k)));
            tmpA1=phi*(zw(k)^2*RSS(k,k));
            tmpA3 = 1./(alpha(:,k) +tmpA1);
            tmpA2 = tmpA3.*Xm;
            A(:,k) = tmpA2 + tmpA3.^(0.5).*tmp1(:,k);
        else  % Draw from base
            A(:,k) = alpha(:,k).^(-0.5).*tmp1(:,k);
        end
    end  
    alpha = gamrnd(g0+1/2,1./(h0+1/2*A.^2));
    sres=[];        rg =find(z==1);
    rg =find(z==1); res = X - A(:,rg)*(diag(zw(rg))*S(rg,:));

    phi = gamrnd(c0+0.5*n,1./(d0+0.5*sum(res.^2,2))); 
    %%=================================
    %Log Stuff
    
    ndx = iter - burnin;
    test = mod(ndx,space);
    mse(iter)=mean(sqrt(sum((X - A*diag(z.*w)*S).^2,1)));
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
    
    
   
    
    %Print Things
    %----------------------------------------------------------------------
    if mod(iter,50) == 0,
        if (debug)
            % fprintf('iter %d: mse=%g: numDic=%g  Gclassnum=%d FirstOpenum: %d%n',iter,mse(iter),sum(z(:,1)),numC(iter),FirstOpenum);
            fprintf('iter %d: mse=%g: numDic=%g  Gclassnum=%d%n',iter,mse(iter),sum(z(:,1)),numC(iter));
            toc
        else
            fprintf('.')
        end
    end
    if (ndx>0) && (test==0)
%         spl.Likelihood(ndx)=Loglikelihood(S,G_lamda,G_mu,H_z,[],H_K,1);
        spl.H_z{ndx} = H_z;
        spl.A{ndx} = A;
        spl.S{ndx} = S;
        spl.w{ndx} = w;
        spl.z{ndx} = z;
        spl.phi{ndx} = phi;
        spl.alpha{ndx} = alpha;
        spl.G_mu{ndx}=G_mu;
        spl.G_lamda{ndx}=G_lamda;
        spl.numC{ndx}=numC;
        spl.pai{ndx}=pai;
        spl.H_v{ndx}=H_v;
        spl.HV_lamda{ndx}=HV_lamda;
        spl.numCluster(ndx)=length(ColC);
    end
  if mod(iter,100)==0

            figure(800),plot(H_z,'r+'); %CRS 20120914 turned off

  end
end


function out = TNrnd(a,b,mu,sigma,sampleSize)
PHIl = normcdf((a-mu)/sigma);                                                 
PHIr = normcdf((b-mu)/sigma);                                                                                               
out = mu + sigma*( sqrt(2)*erfinv(2*(PHIl+(PHIr-PHIl)*rand(sampleSize))-1) ); 
return

