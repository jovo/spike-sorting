the matlabe script is spike_sorting file.

 INPUTS
%   =======================================================================
%   X       : (samples x snippet_example)
%   K       :  Number of dictionary components,  it is less than dimension 
%              of spike waveforms , i.e. K = min(K,size(X,1))
%   H_K     :  Maximal number of clusters
%   PrecisionParameter: Initialiation of phi, controls noise variance
%   burnin  :  Number of iterations from initial state to converging state. 
%   num     :  Number of iterations for collections in the converging period. 
%   space   :  Collections interval 
%   debug   :  Show information in the iterations such as runing time,
%              reconstruction error and so on when debug=1
%
%
%
%   OUTPUT
%   ========================================================================
%     spl.H_z       =  H_z ;       Value for each spikes, value indicates cluster of spikes
%     spl.H_v       =  H_v;        Weight for each clusters, vector with length H_K,H_K is an assumed cluster number.
%     spl.HV_lamda  =  HV_lamda;   Parameter for stick breaking  H_v ~GEM(HV_lamda)
%     spl.A         =  A;          Facorloading (Dictionary) (p*K)        
%     spl.S         =  S;          Factor score (K*n),n is number of snippets (spikes)
%     spl.w         =  w;          Weight (diagnal matrix with elements K)  determining which dictioanry elements are important; 
%                                  that the values are large represents the correspding elements are more important.        
%     spl.z         =  z;          Binary diagnal matrix with elements K determining which dictioanry elements are used;
%                                  z(k)=1 denotes k-th dictionary elements is used, otherwise          
%     spl.phi       =  phi;        noise Precision of the vector with p elements
%     spl.alpha     =  alpha;      Parameters (vector with p elements)for dictionary elements A(:,k)~Gaussian(0,alpha);
%     spl.G_mu      =  G_mu;       Parameters of cluster means, a matrix with size of (K*H_K)
%     spl.G_lamda   =  G_lamda;    Parameters of cluster precison, a matrix with size of (K*H_K)
%     spl.numC      =  numC;       Number of clusters for each iteration
%     spl.pai       =  pai;        Parameters(vector with K elements) for binary dignal matrix, z(k)~beta(pai(k));
