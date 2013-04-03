function spl = TSPLIT_DPBPSVD_Time_depedent_model(X,B,K,H_K,burnin,num,space)

%%% Hierarchical Dirichelet Process and Beta Process Pseudo-SVD via Markov Chain Monte Carlo (HDPBPSVD_MCMC)
%%% written by Bo Chen, 4.19
%%%%% Please cite: Bo Chen, David Carlson and Lawrence Carin,
%%%%% 'On the Analysis of Multi-Channel Neural Spike Data', NIPS 2011.
%----------parameter setting----------------------
%G_w0=10^2*eye(K); d0 = 10^-4*ones(p,1); good result on HC-1 data
%-------------------------------------------------
randn('state',0); rand('state',0); 
XX{1}=X;
clear X
X=XX{1};
clear XX
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
c0 = 10^-0*ones(p,1); d0 = 1*10^-4*ones(p,1);% d0 = 10^-4*ones(p,1); good result
e0 = 1/K*ones(K,1); f0 = 1*(K-1)/K*ones(K,1);
g0 = 1e-0*ones(p,K); h0 = 1*10^-5*ones(p,K);
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
 phi(:,m) = gamrnd(c0,1./(d0));
%  phi(:,m) =10^4*ones(p,1);
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
G_w0=eye(K)*1;%good  result
G_beta0=1;
for i=1:M
    for k=1:H_K
%     G_lamda{k,i}=eye(K);
%     G_mu{k,i}=zeros(K,1);
    G_lamda{k,i}=wishrnd(G_w0,G_v0);
    G_mu{k,i}=randn(K,1);
    end
end

%%%%the parameters in the prior Beta of v
H_a=1e-0;
H_b=1e-1;
HV_lamda=gamrnd(H_a,1/H_b);
for k=1:H_K-1
    H_v(k)=betarnd(1,HV_lamda);
end
H_v(H_K)=1;
%%%%-----------------------------------------
%%%%------------------------------------------

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
       for i=1:n(m)
           E=(B(:,i).*A(:,k).*phi(:,m));F=(E'*(B(:,i).*A(:,k)));
%            xtmp=X{m}(:,i)-sum(repmat(B(:,i),1,K).*A.*(repmat((zw.*S{m}(:,i)).',p,1)),2);
           xtmp=X{m}(:,i)-repmat(B(:,i),1,K).*A*(zw.*S{m}(:,i));
           G=E'*xtmp;
           midval1= midval1+F*w(k,m).^2*S{m}(k,i).^2;
           midval2= midval2+G*w(k,m)*S{m}(k,i);
       end
  end
   tmprr = log(pai(k)+eps) - log(1-pai(k)+eps)-1/2*midval1 + midval2;
%    z(k) = binornd(1,1/(1+exp(-tmprr)));
   if signZ==0
      z(k)=binornd(1,min(1,1/exp(-tmprr)));
   else
      z(k)=binornd(1,1-min(1,exp(-tmprr))); 
   end
%      z(k)=1;
end
   zz=logical(double(z)*double(z)');
       
   pai = betarnd(e0 + z, f0 + 1 - z);
   
 for m=1:M
%      zw=z.*w(:,m);
   for k = 1:K
%        w(k,m) = 0;tmpb0=0;  tmpa0=0;
%        for i=1:n(m)
%            E=(B(:,i).*A(:,k).*phi(:,m));F=(E'*(B(:,i).*A(:,k)));
%            xtmp=X{m}(:,i)-repmat(B(:,i),1,K).*A*(zw.*S{m}(:,i));
%            G=E'*xtmp;
%            tmpb0 =tmpb0+  (z(k)^2*F*S{m}(k,i).^2);
%            tmpa0 =tmpa0+ G*z(k)*S{m}(k,i);
%        end
%        tmpb=(tmpb0+w0^-1)^(-1);
%        tmpa=tmpb*(tmpa0+w0^-1*eta0);
%        L=min(normcdf((0-tmpa)./(sqrt(tmpb)+realmin)),0.99999);
%        midD=L+(1-L)*rand;
%        w(k,m) = max( norminv(midD)*sqrt(tmpb)+ tmpa , 0);
         w(k,m)=1;
   end

end  
% for m=1:M
%    for k = 1:K
%        w(k,m) = 0;
%        tmpb =  (w0^-1+z(k)^2*E{m}(k,k)*RSS(k,k,m))^-1;
%        tmpa = tmpb*(z(k)*(F{m}(k,:)*S{m}(k,:)'-E{m}(k,:)*(z.*w(:,m).*RSS(:,k,m)))+w0^-1*eta0);
%        L=min(normcdf((0-tmpa)./(sqrt(tmpb)+realmin)),0.99999);
%        midD=L+(1-L)*rand;
%        w(k,m) = max( norminv(midD)*sqrt(tmpb)+ tmpa , 0);   
%    end
% end




%%%%%%%%%%%%%%%%%%   DP Part %%%%%%%%%%%%%%%%
%------------------------------------------------------------
if 1

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
 end
 %%-------------------update weight------------------  
%   temp0=randn(K,Len);
%  for m=1:M
%        zw= z.*w(:,m);
%        zwa=repmat(zw,1,p).*A'.*repmat(phi(:,m),1,K)';
%        midval=zwa*(repmat(zw,1,p).*A')';
%        AX=zwa*X{m};
%           for clusndx=1:H_K
%             ndx=find(H_z==clusndx);
%             nndx=numel(ndx);
%             if nndx==0
%                 continue
%             end
%             S_sigmapart = midval+G_lamda{clusndx,m};
%             S_mupart = AX(:,ndx)+...
%                 repmat(G_lamda{clusndx,m}*G_mu{clusndx,m},1,nndx);
%             G = chol(S_sigmapart);
%             S{m}(:,ndx) = G\(temp0(:,ndx)+(G')\S_mupart);
%           end 
%     RSS(:,:,m) = S{m}*S{m}';
% end
 for m=1:M
       zw= z.*w(:,m);
    for i=1:n(m) 
       zwa=repmat(zw,1,p).*(repmat(B(:,i),1,K).*A)'.*repmat(phi(:,m),1,K)';
       midval=zwa*(repmat(zw,1,p).*(repmat(B(:,i),1,K).*A)')';
       AX=zwa*X{m}(:,i);
       S_sigmapart=midval+G_lamda{H_z(i),m};
       S_mupart=AX+G_lamda{H_z(i),m}*G_mu{H_z(i),m};
%        S_sigmapart=midval+eye(K);
%        S_mupart=AX;
       G = chol(S_sigmapart);
       S{m}(:,i) = G\(randn(K,1)+(G')\S_mupart);
    end 
      RSS(:,:,m) = S{m}*S{m}';
 end
%---------------------------------------------
    tmp1 = randn(p,K);  
    for l = 1:p
                Xm =0;tmpA1=0;
            for m=1:M
                zw= z.*w(:,m);
                tmpX = repmat(zw,1,size(S{m},2)).*S{m}.*repmat(B(l,:),K,1);
                tmpA1 = tmpA1+tmpX*phi(l,m)*tmpX';
                Xm    = Xm+tmpX*phi(l,m)*X{m}(l,:)';
            end
                tmpA2 = diag(alpha(l,:)) +tmpA1;   
                tmpA3 = tmpA2\Xm;
                A(l,:)= tmpA3+ chol(tmpA2)\tmp1(l,:).';              
    end 



%----------------------------------------------
%     tmp1 = randn(p,K);  
%     for k = 1:K
%         if z(k)==1
%             A(:,k) = 0;
%             Xm =0;tmpA1=0;
%             for m=1:M
%                 zw= z.*w(:,m);
%                 for i=1:n(m)
%                     tmpX=X{m}(:,i)-(repmat(B(:,i),1,K).*A)*diag(zw)*S{m}(:,i);
%                     Xm = Xm+zw(k)*S{m}(k,i)*diag(B(:,i).*phi(:,m))*tmpX;
%                     tmpA1=tmpA1+(B(:,i).*phi(:,m).*B(:,i))*zw(k)^2*S{m}(k,i)^2;
%                 end
%             end
%             tmpA1 = 1./(alpha(:,k) +tmpA1);
%             tmpA2 = tmpA1.*Xm;
%             A(:,k) = tmpA2 + tmpA1.^(0.5).*tmp1(:,k);
%         else  % Draw from base
%             A(:,k) = alpha(:,k).^(-0.5).*tmp1(:,k);
%         end
%     end 
%---------------------------------------------
                  
%          tmp1 = randn(p,K);  
%     for k = 1:K
%         if z(k)==1
%             A(:,k) = 0;
%             Xm =0;tmpA1=0;
%             for m=1:M
%             zw= z.*w(:,m);
%             Xm = Xm+zw(k)*phi(:,m).*(X{m}*S{m}(k,:)' - A*(zw.*RSS(:,k,m)));
%             tmpA1=tmpA1+phi(:,m)*(zw(k)^2*RSS(k,k,m));
%             end
%             tmpA1 = 1./(alpha(:,k) +tmpA1);
%             tmpA2 = tmpA1.*Xm;
%             A(:,k) = tmpA2 + tmpA1.^(0.5).*tmp1(:,k);
%         else  % Draw from base
%             A(:,k) = alpha(:,k).^(-0.5).*tmp1(:,k);
%         end
%     end  
    % There might be numerical problems for small alpha.
    alpha = gamrnd(g0+1/2,1./(h0+1/2*A.^2));
    sres=[];        rg =find(z==1);
    for m=1:M
        zw= z.*w(:,m);
        rX= A(:,rg)*(diag(zw(rg))*S{m}(rg,:)).*B;
        res = X{m} - rX;
        phi(:,m) = gamrnd(c0+0.5*n(m),1./(d0+0.5*sum(res.^2,2)));
        sres =[sres sqrt(sum(res.^2,1))];
%         sres =[sres sqrt(sum(res.^2,1))./sqrt(sum(X{m}.^2,1))];
    end
    mse(iter)=mean(sres);
    ndx = iter - burnin;
    test = mod(ndx,space);

    uniqC=unique(H_z);

    numC(iter)=length(uniqC);
    numZ(iter)=sum(z);
    C_z=[];
    if (ndx>0) && (test==0)
        spl.Likelihood(ndx)=Loglikelihood(S,G_lamda,G_mu,H_z,C_z,H_K,flag);
        spl.H_z{ndx} = H_z;
        spl.A{ndx} = A;
        spl.S{ndx} = S;
        spl.w{ndx} = w;
        spl.z{ndx} = z;
%         spl.phi = phi;
%         spl.alpha = alpha;
%         spl.G_mu=G_mu;
%         spl.G_lamda=G_lamda;
%         spl.numC=numC;
%         spl.pai=pai;
%         spl.H_v=H_v;
%         spl.HV_lamda=HV_lamda;

    end
        spl.acc=acc;


% fprintf('iter %d: mse=%g: numDic=%g  Gclassnum=%d FirstOpenum: %d%n',iter,mse(iter),sum(z(:,1)),numC(iter),FirstOpenum);
fprintf('iter %d: mse=%g: numDic=%g  Gclassnum=%d%n',iter,mse(iter),sum(z(:,1)),numC(iter));
% fprintf('iter %d: mse=%g: numDic=%g  %n',iter,mse(iter),sum(z(:,1)));
toc
  if mod(iter,10)==0
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

            figure(800),plot(H_z,'r+');
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


