function spl = HDP_DL(X,K,H_K,C_K,burnin,num,space)


randn('state',0); rand('state',0); 

[M] = length(X); %%% M= the number of channels; 
for m=1:M
X{m}=(X{m}-mean(X{m},2)*ones(1,size(X{m},2)));
n(m)=size(X{m},2);
end
nidx=cumsum([0 n]);
p=size(X{1},1);
K=min(K,p);
c0 = 10^-1*ones(p,1); d0 = 10^-6*ones(p,1);
e0 = 1/K*ones(K,1); f0 = 1*(K-1)/K*ones(K,1);
g0 = 1e-0*ones(p,K); h0 = 10^-6*ones(p,K);
 w0 = 0.01; eta0 = 0;
pai = betarnd(e0,f0);

[U0,H0,V0] = svd(cell2mat(X),'econ');
 A=U0(:,1:K);
 SS = H0(1:K,1:K)*V0(:,1:K)';

for m=1:M
    S{m}=SS(:,nidx(m)+1:nidx(m+1));
    RSS(:,:,m) = S{m}*S{m}'; 
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
 phi(:,m) = gamrnd(c0,1./(d0));
end

w = rand(K,M);
z = rand(K,1)<pai;

alpha = gamrnd(g0,1./(h0));
G_v0=K;
G_w0=1*eye(K);
G_beta0=1;
for k=1:H_K
G_lamda{k}=eye(K);
G_mu{k}=zeros(K,1);
end

%%%%the parameters in the prior Beta of v
H_a=1e-0;
H_b=1e-1;
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

for k=1:K
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
   z(k) = binornd(1,1/(1+exp(-tmprr)));
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
                   
for m=1:M
       zw= z.*w(:,m);
       zwa=repmat(zw,1,p).*A'.*repmat(phi(:,m),1,K)';
       midval=zwa*(repmat(zw,1,p).*A')';
       AX=zwa*X{m};
    for i=1:n(m)  
       S_sigmapart=midval+G_lamda{H_z(C_z{m}(i),m)};
       S_mupart=AX(:,i)+G_lamda{H_z(C_z{m}(i),m)}*G_mu{H_z(C_z{m}(i),m)};
       G = chol(S_sigmapart);
       S{m}(:,i) = G\(randn(K,1)+(G')\S_mupart);
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
        sres =[sres sqrt(sum(res.^2,1))];
    end
    mse(iter)=mean(sres);
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

        spl.A = A;
        spl.S = S;
        spl.w = w;
        spl.z = z;
        spl.phi = phi;
        spl.alpha = alpha;
        spl.C_z = C_z;
        spl.G_mu=G_mu;
        spl.G_lamda=G_lamda;
        spl.numC=numC;
        spl.pai=pai;
        spl.C_v=C_v;
        spl.H_v=H_v;
        spl.CV_lamda=CV_lamda;
        spl.HV_lamda=HV_lamda;
        spl.PC_z=PC_z;
        spl.PH_z=PH_z;
        spl.numZ=numZ;
        spl.H_z{ndx} = datacc;
    end
        spl.acc=acc;


fprintf('iter %d: mse=%g: numDic=%g Classnum1=%d Gclassnum=%d%n',iter,mse(iter),sum(z(:,1)),length(unique(C_z{1})),numC(iter));
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

% close(figure(112));
% figure(112),
showm=ceil(M/2);
pause(0.2)
  end

end


