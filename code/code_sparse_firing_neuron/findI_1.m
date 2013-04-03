function I=findI(spl,label)
 N=length(spl.Likelihood);
 Uz=[];
for mm=1:N
    z=spl.H_z{mm};
    uniquez=unique(z);
    Uz=[uniquez,Uz];
end
Uniquez=unique(Uz);
for ii=1:length(Uniquez)
    for mm=1:N
         z=spl.H_z{mm};
          iidex=find(z==Uniquez(ii));
            pp=zeros(2491,1);
            pp(iidex)=1;
            aa(ii,mm)=length(find(pp-label));
    end
end
      [rowidx,colidx]=find(min(min(aa))==aa);
      I=colidx;
