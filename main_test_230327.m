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

time = synthetics_data_01.Time;

Gog  = forward_model_icbm152.Gain;
Yog  = synthetics_data_01.F;

Yog([19,20],:) = [];
Gog([19,20],:) = [];

K = size( atlas_regions,2 ) +1;
N = size(Gog,2);
T = size(Yog,2);
M = size(Yog,1);

meta = [];
meta.K = K;
meta.N = N;
meta.T = T;
meta.M = M;

params = [];
params.MaxIter = 250;
params.Method  = 'ConjugateGradient';
%params.Method  = 'SteepestDescent';
params.PlotError = show_figs;

%% PRE PROCESSING

% Whitening of Y
SG = cov(Yog');
% if show_figs
%   figure()
%   image(SG);
%   colorbar
%   title("Covariance matrix for $Y$",'interpreter','latex')
%   figure()
%   histogram(SG(:))
%   title("Covariance matrix for $Y$",'interpreter','latex')
%   ylabel("Frequency",'interpreter','latex')
% end
%Y = sqrtm(inv(SG))*Yog;
%G = sqrtm(inv(SG))*Gog;
Y = Yog; % pre-whitening is no longer required by the model
G = Gog;

% remove later
%G = G(:,(3*(1:N)-2));

% TODO: a better way to select active regions
%S  = false(K,1);
%S(55) = true;
S  = true(K,1);

Nk = zeros(K,1);
R = [];
unassigned = true(N,1);
for k = 1:(K-1)
  R{k} = atlas_regions(k).Vertices;
  Nk(k) = size(R{k},2);
  unassigned(R{k}) = false;
end
idx = 1:N;
R{K} = idx(unassigned);
Nk(K) = size(R{K},2);

% other matrices that are constant
L = zeros(N,K);
for k = 1:K
  L(R{k},k) = 1;
end

Nk_inv = inv(L'*L);
%I_A = eye(N) - L*diag(S*1)* Nk_inv * L';
%GG  = G'*G;
%GY  = G'*Y;

%%
Niter = 5;
% main cycle
ERR = zeros(Niter,1);
collect_gamma = zeros(K,Niter+1);
collect_sigma = zeros(1,Niter);
%sigma2 = 1;

SIGMAi = inv(cov(Y'));
reg = mean(diag(SIGMAi));
%SIGMAi = SIGMAi/reg;
if show_figs
  PLotCov(SIGMAi);
end
%gamma2 = ones(K,1);
gamma2 = zeros(K,1);
%for k = 1:K
%  gamma2(k) = mean( diag( GG(R{k},R{k}) ) );
%end

collect_gamma(:,1) = gamma2;
collect_sigma(1) = reg;

W = zeros(N,M);

%%
iter = 1;
for iter = 1:Niter
  [W,~] = InversionKernel(G,SIGMAi,gamma2,R,S,W, meta,params);
  if iter==1
    % extra computing time if the initial guess was bad
    [W,~] = InversionKernel(G,SIGMAi,gamma2,R,W, meta,params);
  end
  J = W*Y;
  if show_figs
    figure()
    plot(time,J(5942,:)) % arbitrary dipole in region where the actual source is located
    xlabel('Time [s]')
    ylabel('Current density (J) []')
  end

  %if show_figs
  %  figure()
  %  imagesc(G*W);
  %  colorbar
  %  title("$G W \approx I_M$",'interpreter','latex')
  %end

  Q = zeros(M,M);
  for t = 1:T
    Q = Q + (G*J(:,t)-Y(:,t))*(G*J(:,t)-Y(:,t))';
  end
  Q = Q/T;
  %SIGMAi = inv(Q);

  if show_figs
    PLotCov(Q);
  end
  
  for k = 1:K
    gamma2(k) = 1/( ( norm(J(R{k},:)-mean(J(R{k},:)),'fro')^2 )/(Nk(k)*T) ) ;
  end
  
  ERR(iter) = norm(G*J-Y,'fro')/(N*T);
  reg = abs(mean(diag(SIGMAi)));
  
  collect_gamma(:,iter+1) = gamma2;
  collect_sigma(iter+1) = reg;
  
  iter = iter +1;
end

%plot()

%%
figure()
plot(log(ERR))
%title("Covariance matrix for $Y$",'interpreter','latex')
xlabel("Iteration",'interpreter','latex')
ylabel("$log \left\Vert G \hat{J} - Y\right\Vert_F$",'interpreter','latex')

figure()
plot(log(collect_gamma(S,:)'))
xlabel("Iteration",'interpreter','latex')
ylabel("$log \gamma_k^2$",'interpreter','latex')
