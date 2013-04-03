function Loglikelihood = Loglikelihood(X,nu,mu,H_z,C_z,H_K, flag)
%X data with cell tpye; nu: covariance matrix, mu mean vector, H_z
%indicator of dish to table, C_z local indicator of table to customer

if flag==0
num=length(X);
  for jj=1:num
         datacc{jj}=H_z(C_z{jj},jj);
      for mm=1:H_K
         idx=find(datacc{jj}==mm);
         KM=length(idx);
         RR=chol(nu{mm});
         xRinv=(X{jj}(:,idx)-repmat(mu{mm},1,KM)).'*RR;
         midval2=sum(log(diag(RR)));  
         quadform = sum(sum(xRinv.^2, 2));
         PH_z(mm,jj)=midval2*KM-0.5*quadform; 
     end
  end
   Loglikelihood=sum(sum(PH_z));
elseif flag==1
    num=length(X);
  for jj=1:num
         datacc{jj}=H_z;
      for mm=1:H_K
         idx=find(datacc{jj}==mm);
         KM=length(idx);
         RR=chol(nu{mm,jj});
         xRinv=(X{jj}(:,idx)-repmat(mu{mm,jj},1,KM)).'*RR;
         midval2=sum(log(diag(RR)));  
         quadform = sum(sum(xRinv.^2, 2));
         PH_z(mm,jj)=midval2*KM-0.5*quadform; 
     end
  end
   Loglikelihood=sum(sum(PH_z));
elseif flag==2
    N=size(nu,2);
    num=length(X);
    for n=1:N
      for jj=1:num
             datacc{jj}=H_z(C_z{jj},jj);
          for mm=1:H_K
             idx=find(datacc{jj}==mm);
             KM=length(idx);
             RR=chol(nu{mm,n});
             xRinv=(X{jj}(:,idx)-repmat(mu{mm,n},1,KM)).'*RR;
             midval2=sum(log(diag(RR)));  
             quadform = sum(sum(xRinv.^2, 2));
             PH_z(mm,jj,n)=midval2*KM-0.5*quadform; 
         end
      end
    end
   Loglikelihood=sum(sum(sum(PH_z))); 
   elseif flag==3
    N=size(nu,2);
    num=length(X);
    for n=1:N
      for jj=1:num
             datacc{jj}=H_z{jj};
          for mm=1:H_K
             idx=find(datacc{jj}==mm);
             KM=length(idx);
             RR=chol(nu{mm,n});
             xRinv=(X{jj}(:,idx)-repmat(mu{mm,n},1,KM)).'*RR;
             midval2=sum(log(diag(RR)));  
             quadform = sum(sum(xRinv.^2, 2));
             PH_z(mm,jj,n)=midval2*KM-0.5*quadform; 
         end
      end
    end
   Loglikelihood=sum(sum(sum(PH_z)));  
end












% num=length(X);
%   for jj=1:num
%          tmp{jj}=H_z(:,jj).';
%          datacc{jj}=tmp{jj}(C_z{jj});
%       for mm=1:H_K
%          idx=find(datacc{jj}==mm);
%          KM=length(idx);
%          RR=chol(nu{mm});
%          xRinv=(X{jj}(:,idx)-repmat(mu{mm},1,KM)).'*RR;
%          midval2=sum(log(diag(RR)));  
%          quadform = sum(sum(xRinv.^2, 2));
%          PH_z(mm,jj)=midval2*KM-0.5*quadform; 
%      end
%   end
%    Loglikelihood=sum(sum(PH_z));


