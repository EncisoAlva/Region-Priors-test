%% EXAMPLE INFORMATION
%
% Synthetic data was created using one dipole on the cortex, with 
% orientation normal to the cortex surface and random magnitude. 
%
% Head model was created from sample MRI ICBM 152. Distributed dipoles were
% located at the cortex surface, and forward model was computed using
% OpenMEEG. 
%
% Dipole for synthetic data is located at the following SCS coordinates
% 69 37 85
% Artificial data was created using SimMEEG

%% DEBUG PARAMETERS
show_figs = true;

%% DATA LOAD AND PRE-PROCESS
pathstr = pwd;
load(fullfile(pathstr,".\data_example\synth_Y.mat"))
load(fullfile(pathstr,".\data_example\synth_G.mat"))
load(fullfile(pathstr,".\data_example\synth_S.mat"))

Gog = forward_model_icbm152.Gain;
Yog = synthetics_data_01.F;

Yog([19,20],:) = [];
Gog([19,20],:) = [];

K = size( atlas_regions,2 ) +1;
N = size(Gog,2)/3;

%% PRE PROCESSING

% Whitening of Y
SG = cov(Yog');
if show_figs
  figure()
  image(SG);
  colorbar
  title("Covariance matrix for $Y$",'interpreter','latex')
  figure()
  histogram(SG(:))
  title("Covariance matrix for $Y$",'interpreter','latex')
  ylabel("Frequency",'interpreter','latex')
end
Y = sqrtm(inv(SG))*Yog;
G = sqrtm(inv(SG))*Gog;

% remove later
G = G(:,(3*(1:N)-2));

% TODO: a better way to select active regions
%S  = false(K,1);
%S(55) = true;
S  = true(K,1);

Nk = zeros(K,1);
Rk = [];
unassigned = true(N,1);
for k = 1:(K-1)
  Rk{k} = atlas_regions(k).Vertices;
  Nk(k) = size(Rk{k},2);
  unassigned(Rk{k}) = false;
end
idx = 1:N;
Rk{K} = idx(unassigned);
Nk(K) = size(Rk{K},2);

% other matrices that are constant
L = zeros(N,K);
for k = 1:K
  L(Rk{k},k) = 1;
end

Nk_inv = inv(L'*L);
I_A = eye(N) - L*diag(S*1)* Nk_inv * L';
GG  = G'*G;
GY  = G'*Y;

Niter = 10;
% main cycle
ERR = zeros(Niter,1);
collect_gamma = zeros(K,Niter);
gamma = ones(K,1);

iter = 1;
for iter = 1:Niter
  GAM2 = diag(L*(gamma.^2));
  J = ( GG + I_A*GAM2 ) \ GY;
  if show_figs
    figure()
    plot(J(5942,:)) % arbitrary dipole in region where the actual source is located
  end
  %V = L'*J;
  %plot(V(1,:))
  Q = Nk_inv * L' * mean((I_A*J).^2,2);
  %gamma = min(Q.^(-1),1/tol);
  gamma = 1./Q;
  collect_gamma(:,iter) = gamma;
  ERR(iter) = norm(G*J-Y,'fro');
  iter = iter +1;
end