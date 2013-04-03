function Loglikelihood = Loglikelihood_v3(X,nu,mu,H_z,H_K)
%X data with cell tpye; nu: covariance matrix, mu mean vector, H_z
%indicator of dish to table, C_z local indicator of table to custome
datacc=H_z;
      for mm=1:H_K
         idx=find(datacc==mm);
         KM=length(idx);
         RR=chol(nu{mm});
         xRinv=(X(:,idx)-repmat(mu{mm},1,KM)).'*RR.';
         midval2=sum(log(diag(RR)));  
         quadform = sum(sum(xRinv.^2, 2));
         PH_z(mm)=midval2*KM-0.5*quadform; 
      end
   Loglikelihood=(sum(PH_z));
