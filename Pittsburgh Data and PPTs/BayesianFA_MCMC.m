function [Xreconstructed, Xhat]=BayesianFA_MCMC(X,K);
% function [Xreconstructed, Xhat]=BayesianFA_MCMC(X,K);
% 
% Given an input of data X and the number of factors K you can determine
% the values of the missing data as well as the noiseless estimate of the
% input.  This code considers that anything that has an input of nan to be
% missing data. K defaults to 5 without an input.
%
% The method returns two results: Xreconstructed, which is the original
% data with only the missing data values replaced.  Xhat replaces all data
% with the minimum mean square error estimator.
%
% 10/4/12
% David Carlson
% david.carlson (at) duke.edu
%%
maxiter=300;
burnin=50;
collect=5;
a0=1e-2;
b0=1e-2;
c0=1e-2;
d0=1e-2;
if nargin<2
    K=5;
end
[P,N]=size(X);
if (N<P)
    X=X';
    [P,N]=size(X);
end
D=randn(P,K);
S=randn(K,N);
gamS=gamrnd(a0,1./b0,K,1);
B=~isnan(X);
Xo=X;
Xo(isnan(Xo))=0;
Bp=sum(B,2);
Bn=sum(B,1);
gamX=gamrnd(c0,1./d0,P,1);
%%
c=0;
Xreconstructed=zeros(size(X));
Xhat=zeros(size(X));
%%
for iter=1:maxiter
    % update D
    for p=1:P
        stmp=S(:,B(p,:));
        xtmp=Xo(p,B(p,:));
        precD=eye(K)+gamX(p)*stmp*stmp';
        muuD=precD\(gamX(p)*stmp*xtmp');
        D(p,:)=muuD'+(chol(precD)\randn(K,1))';
        gamX(p)=gamrnd(c0+Bp(p)/2,1./(d0+.5*sum((xtmp-D(p,:)*stmp).^2)));
    end
    % update S
    for n=1:N
        Dtmp=D(B(:,n),:);
        xtmp=Xo(B(:,n),n);
        precS=diag(gamS)+Dtmp'*diag(gamX(B(:,n)))*Dtmp;
        muuS=precS\(Dtmp'*diag(gamX(B(:,n)))*xtmp);
        S(:,n)=muuS+chol(precS)\randn(K,1);
    end
    % update gamS
    S2=sum(S.^2,2);
    gamS=gamrnd(a0+N/2,1./(b0+.5*S2));
    if iter>burnin
        if mod(iter,collect)==0
            c=c+1;
            Xreconstructed=Xreconstructed+D*S;
        end
    end
    %
    if mod(iter,50)==0
        fprintf('Iteration %d complete.\n',iter);
    end
end
Xhat=Xreconstructed./c;
Xreconstructed=Xhat;
Xreconstructed(B)=X(B);
