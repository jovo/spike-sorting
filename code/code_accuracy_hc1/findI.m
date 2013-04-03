function [minnum,I]=findI(spl,label,flag)      
 N=length(spl.H_z);
 Uz=[];   
for mm=1:N
    if flag==0||flag==3
    z=spl.H_z{mm};
    elseif flag==1||flag==4
    z=spl.H_z{1,mm}{1,1};
    elseif flag>4
      z=spl.H_z{mm};  
    end
    uniquez=unique(z);
    Uz=[uniquez,Uz];
end
Uniquez=unique(Uz);
for ii=1:length(Uniquez)
    for mm=1:N
           if flag==0||flag==3
            z=spl.H_z{mm};
            elseif flag==1||flag==4
            z=spl.H_z{1,mm}{1,1};       
            elseif flag>4
            z=spl.H_z{mm};  
           end
           iidex=find(z==Uniquez(ii));
           pp=zeros(2491,1);
           pp(iidex)=1;
           aa(ii,mm)=length(find(pp-label));
    end
end
      [rowidx,colidx]=find(min(min(aa))==aa);
      I=colidx;
      minnum=aa(rowidx,:);  
     
     
     
     
     

