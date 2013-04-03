function [Y,Bmatrix]= alignment_function(spike,Minposition)
% alignment at the point of the minimal value 
% input of spike is a cell
X=spike;
[p,N]=size(X);
tmpx=min(X);
Y=zeros(p,N);
Bmatrix=zeros(p,N);
for i=1:N
    [J]=find(X(:,i)==tmpx(i));
    if length(J)>1;
        J(2:end)=[];
    end
    if J<Minposition
       Delta=Minposition-J;
       Y(1:Delta,i)      = 0;
       Y(Delta+1:p,i)    = X(1:p-Delta,i);
       Bmatrix(Delta+1:p,i)  = 1;
    elseif J==Minposition
        Y(:,i)=X(:,i);
        Bmatrix(1:p,i)= 1; 
    elseif J>Minposition
       Delta=J-Minposition; 
       Y(1:p-Delta,i)      = X(Delta+1:p,i);
       Y(p-Delta+1:p,i)    = 0;
       Bmatrix(1:p-Delta,i)= 1;  
    end
end
