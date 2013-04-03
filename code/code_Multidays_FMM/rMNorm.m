function rnorm = rMNorm(m,V,T)
%%% Minhua Chen, minhua.chen@duke.edu

%%%%% Please cite: Bo Chen, Minhua Chen, John Paisley, Aimee Zaas, Christopher Woods,
%%%%% Geoffrey S Ginsburg, Alfred Hero III, Joseph Lucas, David Dunson and Lawrence Carin,
%%%%% "Bayesian Inference of the Number of Factors in Gene-Expression Analysis: Application
%%%%% to Human Virus Challenge Studies", BMC Bioinformatics 2010, 11:552.
p=length(m); 
rnorm = repmat(reshape(m,p,1),1,T) +chol(V)'*randn(p,T);



