%%running spike sorting code in the dictionary learning method.
%%%%% For the detailed algorithm description, Please refer to the references:
%%%%% 'On the Analysis of Multi-Channel Neural Spike Data', NIPS 2011.
%%%%% 'Sorting Electrophysiological Data via Dictionary Learning & Mixture
%%%%% Modeling'.
%%=================================================================================
%   X       : (samples x snippet_example) [p,n]=size(X),p is dimension of
%              spike, n is number of spikes
%   K       :  Number of dictionary components,  it is less than dimension 
%              of spike waveforms , i.e. K = min(K,size(X,1))
%   Ncentres     :  Maximal number of clusters
%   TuningParameter1: Initialiation of phi, controls noise variance
%   burnin  :  Number of iterations from initial state to converging state. 
%   num     :  Number of iterations for collections in the converging period. 
%   space   :  Collections interval 
%   debug   :  Show information in the iterations such as runing time,
%              reconstruction error and so on when debug=1
%%==================================================================
%% For example
clear all;close all
load data %simulation dataset
X=Spike;burnin =20000; num = 20000; space = 1;Ncentres=20; K=32; debug=1;PrecisionParameter=10^8;
spl = DictionaryLearning(X,K,Ncentres,PrecisionParameter,burnin,num,space,debug);
%%=========================================================================