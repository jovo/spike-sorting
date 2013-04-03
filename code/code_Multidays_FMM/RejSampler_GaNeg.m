function [Lambda, rej_ac,p] = RejSampler_GaNeg(Lambda, gamma1, gamma2, B, Stats)

%Lambda is the parameter of interest, rej_ac is the acceptance rate

%gamma1 is the hyperparameter
%B is the binary matrix, 
%Stats is the observation-cluster count matrix

%p is the parameter sampled from belowing beta distribution
p = betarnd(1 + sum(Stats,2), 1 + sum(bsxfun(@times, B, Lambda),2));
% p = betarnd(1 + Stats, 1 + sum(bsxfun(@times, B, Lambda),2));


K = length(Lambda);
Stats = B.*Stats;

Iter_gr = 200;
Iter_rj = 500;
Grid = 20;
Threshold = 1e-5;
BS = 20;

br = (Grid-1e-15)*ones(1,K);
bl = realmin*ones(1,K);
b = 0.5*ones(BS,K);
LL = -inf*ones(BS,K);
oldfbmax = ones(1,K);
m = ones(1,K);
fmax = -inf*ones(1,K);

Count_gr = 0;
ml = true(1,K);
indexesm = 0:BS:(K-1)*BS;

beta = gamma2 - sum(bsxfun(@times, B, log(1-p)), 1);

% Comp_in = repmat(Stats > 1, 1, BS);

%first do the grid [0 Right] based search
while nnz(ml) > 0 && Count_gr < Iter_gr
    
    oldfbmax(ml) = fmax(ml);
    
    Count_gr = Count_gr + 1;   
    
    b(:,ml) = ndlinspace(bl(ml),br(ml),BS);
    
    b(b==0) = 1e-10;
%     b(b==Grid) = Grid;
    bt = b(:,ml)';
    Stats_temp = repmat(Stats(:,ml), 1, BS);
    Comp = bsxfun(@minus, gammaln(bsxfun(@plus, Stats_temp, bt(:)')), gammaln(bt(:)'));
    
    LL(:,ml)=bsxfun(@times,gamma1-1,log(b(:,ml)))-bsxfun(@times,beta(ml),b(:,ml)) ...
        + reshape(sum(Comp, 1), nnz(ml), BS)';
    
    [fmax,m] = max(LL);
    
    newindexes = indexesm + m;
    Left = newindexes;
    Left(mod(newindexes-1,BS)~=0)=newindexes(mod(newindexes-1,BS)~=0)-1;
    bl = b(Left);
    Right = newindexes;
    Right(mod(newindexes,BS)~=0)=newindexes(mod(newindexes,BS)~=0)+1;
    br = b(Right);
    
    ml = abs(oldfbmax-fmax) > Threshold;
end


if Count_gr == Iter_gr
     disp(['binary search failed for ',num2str(nnz(ml)),' variables']);
end

if nnz(isnan(b(newindexes)))
    disp('NaN elements returned in the binary search');
end


newindexes = indexesm + m;
Phi_Candidate = b(newindexes);
Rj = Phi_Candidate > Grid - 1e-5;
% Lambda(~Rj) = Phi_Candidate(~Rj);
lmax_new = Phi_Candidate;
%for the rest(Rj), perform the rejection sampling:

lmax = lmax_new;
Count_rj = 0;

%Newton's method to find the global maximum value
step = rand;
while nnz(Rj)>0 && Count_rj < Iter_rj
    
    Count_rj = Count_rj + 1;
    
    Comp_1 = bsxfun(@minus, psi(bsxfun(@plus, lmax(Rj), Stats(:,Rj))), psi(lmax(Rj)));
    fp = (gamma1-1)./lmax(Rj) - beta(Rj) + sum(Comp_1,1);
    Comp_2 = bsxfun(@minus, psi(1,bsxfun(@plus, lmax(Rj),Stats(:,Rj))),psi(1,lmax(Rj)));
    fpp = -(gamma1-1)./(lmax(Rj).^2) + sum(Comp_2,1);
    
    lmax_new(Rj) = lmax(Rj) - step.*fp./fpp;
    if nnz(isnan(lmax_new))>0 || (max(abs(lmax_new)) == inf)
        disp('lmax_new has NaN value or becomes inf');
    end
    
    lmax(Rj) = lmax_new(Rj);
    
    lmax(lmax<=0) = 10^(-5*rand);
    lmax(lmax<=1e-10) = 1e-10; 
    Rj(Rj) = abs(fp) > Threshold;
    
    step = 0.5*rand(1,nnz(Rj));
end

% Lambda = lmax;
Add = Stats > 0;
Linear = sum(bsxfun(@minus, psi(bsxfun(@plus, lmax, Stats)), psi(lmax)).*B,1);
gamma11 = gamma1 + sum(Add,1);
gamma12 = beta - Linear + sum(Add,1)./lmax;

Orig_S = sum(bsxfun(@minus,gammaln(bsxfun(@plus, lmax, Stats)),gammaln(lmax)).*B,1)...
    - sum(Add,1).*log(lmax);

if min(gamma12)<0
    disp([num2str(nnz(gamma12<0)),' gamma1s had scale parameter less than 0']);
end

L_P1 = gamrnd(gamma11,1./gamma12);

in = Orig_S - lmax.*Linear + sum(Add,1);

Orig = sum(bsxfun(@minus,gammaln(bsxfun(@plus, L_P1, Stats)),gammaln(L_P1)).*B,1) ...
    - sum(Add,1).*log(L_P1);

r_er = nnz(L_P1.*Linear - L_P1.*sum(Add,1)./lmax + in < Orig - 1e-4);
if r_er > 0
    disp(['rejection sampling error with ', num2str(r_er), 'variables']);
end

ran = log(rand(1,K));
u = ran + L_P1.*Linear - L_P1.*sum(Add,1)./lmax + in < Orig;

rej_ac = nnz(u)/K;

Lambda(u) = L_P1(u);

end