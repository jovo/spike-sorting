function z = discreterndv2(p)
% Takes a m x n matrix and draws one discrete value for each
% column.  Returns draws from a discrete distribution in a column form.
% DEC 8/4/12
[m n]=size(p);

r = rand(1,n);   % rand numbers are in the open interval (0,1)
p = cumsum(p);
pn=p./repmat(p(m,:),m,1);
z = sum(repmat(r,m,1) > pn)+1;z=z(:).';