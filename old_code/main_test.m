<<<<<<< HEAD:old_code/main_test.m
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
T = size(Yog,2);

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
collect_sigma = zeros(1,Niter);
gamma2 = ones(K,1);
sigma2 = 1;

%iter = 1;
for iter = 1:Niter
  GAM2 = diag(L*gamma2);
  J = ( GG + I_A*GAM2 ) \ GY;
  if show_figs
    figure()
    plot(J(5942,:)) % arbitrary dipole in region where the actual source is located
  end
  sigma2 = 1/( ( norm(G*J-Y,'fro')^2 )/(N*T) );
  IA_J = I_A*J;
  for k = 1:K
    gamma2(k) = 1/( ( norm(IA_J(Rk{k},:),'fro')^2 )/(Nk(k)*T) );
  end
  gamma2 = gamma2/sigma2;
  collect_gamma(:,iter) = gamma2/sigma2;
  collect_sigma(iter) = sigma2;
  ERR(iter) = norm(G*J-Y,'fro');
  %iter = iter +1;
end


figure()
plot(log(ERR))
%title("Covariance matrix for $Y$",'interpreter','latex')
xlabel("Iteration",'interpreter','latex')
ylabel("$log \left\Vert G \hat{J} - Y\right\Vert_F$",'interpreter','latex')

figure()
plot(log(collect_gamma(S,:)'))
xlabel("Iteration",'interpreter','latex')
ylabel("$log \gamma_k^2$",'interpreter','latex')
=======
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
T = size(Yog,2);

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
collect_sigma = zeros(1,Niter);
gamma2 = ones(K,1);
sigma2 = 1;

K  = zeeros(M,N);
%iter = 1;
for iter = 1:Niter
  GAM2 = diag(L*gamma2);
  %J = ( GG + I_A*GAM2 ) \ GY;
  [K,~] = 
  if show_figs
    figure()
    plot(J(5942,:)) % arbitrary dipole in region where the actual source is located
  end
  sigma2 = 1/( ( norm(G*J-Y,'fro')^2 )/(N*T) );
  IA_J = I_A*J;
  for k = 1:K
    gamma2(k) = 1/( ( norm(IA_J(Rk{k},:),'fro')^2 )/(Nk(k)*T) );
  end
  gamma2 = gamma2/sigma2;
  collect_gamma(:,iter) = gamma2/sigma2;
  collect_sigma(iter) = sigma2;
  ERR(iter) = norm(G*J-Y,'fro');
  %iter = iter +1;
end


figure()
plot(log(ERR))
%title("Covariance matrix for $Y$",'interpreter','latex')
xlabel("Iteration",'interpreter','latex')
ylabel("$log \left\Vert G \hat{J} - Y\right\Vert_F$",'interpreter','latex')

figure()
plot(log(collect_gamma(S,:)'))
xlabel("Iteration",'interpreter','latex')
ylabel("$log \gamma_k^2$",'interpreter','latex')


n= 3;
I_A = eye(n) - ones(n,n)/n; 
>>>>>>> f9a985449b8c25072bd09239668c77773714c693:main_test.m
