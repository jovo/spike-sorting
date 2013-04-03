function Xaligned=alignSpikes(X)
% function Xaligned=alignSpikes(X)
%
% This function aligns spikes by their minimum point. This minimum is
% shifted to the most common minimum point.
%
% It has one return, which is the aligned versions of the spikes.  Unknown
% values are listed as NaNs.
%
% 10/4/12
% David Carlson
% david.carlson (at) duke.edu
[P,N]=size(X);
if N<P
    X=X';
    [P,N]=size(X);
end
[minval,minloc]=min(X);

loc=mode(minloc);
Xaligned=zeros(P,N);
for n=1:N
    if minloc(n)<=loc;
        adj=loc-minloc(n);
        Xaligned(adj+1:end,n)=X(1:end-adj,n);
        Xaligned(1:adj-1,n)=nan;
    else
        adj=minloc(n)-loc;
        Xaligned(1:end-adj,n)=X(adj+1:end,n);
        Xaligned(end-adj+1:end,n)=nan;
    end
end