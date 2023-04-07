function varargout = process_regionpriors( varargin )
% PROCESS_REGION_PRIORS:
% [This function exists for technical purposes]
%
% @========================================================================
% TODO: Fill descritpion
% ========================================================================@
%
% Author: Julio Cesar Enciso-Alva, 2023
%         (juliocesar.encisoalva@mavs.uta.edu)
%
eval(macro_method);
end

%% ===== GET DESCRIPTION =====
function sProcess = GetDescription()
  % Description the process
  sProcess.Comment     = 'Source Estimation w/Region Activation Priors';
  sProcess.Category    = 'Custom';
  sProcess.SubGroup    = 'Sources';
  sProcess.Index       = 1000;
  sProcess.FileTag     = '';
  sProcess.Description = 'github.com/EncisoAlva/region_priors';
  % Definition of the input accepted by this process
  sProcess.InputTypes  = {'data', 'raw'};
  sProcess.OutputTypes = {'data', 'raw'};
  sProcess.nInputs     = 1;
  %sProcess.nMinFiles   = 1;
  %
  % Debug options
  sProcess.options.Debug.Comment    = 'Enable debug options';
  sProcess.options.Debug.Type       = 'checkbox';
  sProcess.options.Debug.Value      = 0;                 % Selected or not by default
  sProcess.options.Debug.Controller = 'Debug';
  %sProcess.options.Debug.Hidden  = 0;
  %
  sProcess.options.DebugFigs.Comment = 'Show debug figures';
  sProcess.options.DebugFigs.Type    = 'checkbox';
  sProcess.options.DebugFigs.Value   = 0;                 % Selected or not by default
  sProcess.options.DebugFigs.Class   = 'Debug';
  %sProcess.options.DebugFigs.Hidden  = 0;
  %
  sProcess.options.MaxIterGrad.Comment = 'Max iterations (Kernel): ';
  sProcess.options.MaxIterGrad.Type    = 'value';
  sProcess.options.MaxIterGrad.Value   = {350, '', 0};   % {Default value, units, precision}
  sProcess.options.MaxIterGrad.Class   = 'Debug';
  %sProcess.options.MaxIterGrad.Hidden  = 0;
  %
  sProcess.options.MethodGrad.Comment = 'Method (Kernel): ';
  sProcess.options.MethodGrad.Type    = 'combobox_label';
  sProcess.options.MethodGrad.Value   = {'ConjugateGradient',...
    {'ConjugateGradient','SteepestDescent';...
    'ConjugateGradient','SteepestDescent'}};
  sProcess.options.MethodGrad.Class   = 'Debug';
  %sProcess.options.MethodGrad.Hidden  = 0;
  %
  % Process options
  sProcess.options.Prewhiten.Comment = 'Prewhiten';
  sProcess.options.Prewhiten.Type    = 'checkbox';
  sProcess.options.Prewhiten.Value   = 0;                 % Selected or not by default
  sProcess.options.DebugFigs.Class   = 'Pre';
  %
  sProcess.options.FullRes.Comment = 'Compute full estimate (not only kernel)';
  sProcess.options.FullRes.Type    = 'checkbox';
  sProcess.options.FullRes.Value   = 1;                 % Selected or not by default
  %
  sProcess.options.MaxIter.Comment = 'Max iterations: ';
  sProcess.options.MaxIter.Type    = 'value';
  sProcess.options.MaxIter.Value   = {5, '', 0};   % {Default value, units, precision}
  %
  % Definition of the options
  % Options: Scouts-based
  %sProcess.options.test1.Type  = {'scout_confirm'};
  %sProcess.options.test1.Value = [];
  %sProcess.options.test2.Type = {'scout'};
  % Option: Atlas
  sProcess.options.AtlasRegions.Comment = 'Select atlas (currently not working):';
  sProcess.options.AtlasRegions.Type    = 'atlas';
  sProcess.options.AtlasRegions.Value   = [];    
  %
  % Option: Inverse method
  sProcess.options.method.Comment = 'Weight estimation (gamma):';
  sProcess.options.method.Type    = 'combobox_label';
  sProcess.options.method.Value   = {'mle', {'Max Likelihood', '(future work)'; ...
                                             'mle',            'mgcv'}};
  % Option: Sensors selection
  sProcess.options.sensortype.Comment = 'Sensor type:';
  sProcess.options.sensortype.Type    = 'combobox_label';
  %sProcess.options.sensortype.Value   = {'EEG', {'MEG', 'MEG GRAD', 'MEG MAG', 'EEG', 'SEEG', 'ECOG'; ...
  %                                               'MEG', 'MEG GRAD', 'MEG MAG', 'EEG', 'SEEG', 'ECOG'}};
  sProcess.options.sensortype.Value   = {'EEG', {'EEG', 'SEEG', 'ECOG'; ...
                                                 'EEG', 'SEEG', 'ECOG'}};
end

%% ===== FORMAT COMMENT =====
function Comment = FormatComment(sProcess)
    %if isempty(sProcess.options.sensortype.Value)
    %    Comment = 'No sensor selected';
    %else
    %    %strValue = sprintf('%1.0fHz ', sProcess.options.freqlist.Value{1});
    %    Comment = ['sensors: ' , sProcess.options.sensortype.Value];
    %end
    Comment = sProcess.Comment;
end

function OutputFiles = Run(sProcess, sInputs)
  % Initialize returned list of files
  OutputFiles = {};

  % ===== GET OPTIONS =====
  % General
  params = [];
  params.MaxIter      = sProcess.options.MaxIter.Value{1};
  params.PlotError    = sProcess.options.DebugFigs.Value;
  params.PreWhiten    = sProcess.options.Prewhiten.Value;
  params.FullResult   = sProcess.options.FullRes.Value;
  params.DebugFigures = sProcess.options.DebugFigs.Value;
  %
  params_gradient = [];
  params_gradient.MaxIter   = sProcess.options.MaxIterGrad.Value{1};
  params_gradient.Method    = sProcess.options.MethodGrad.Value{1};
  params_gradient.PlotError = sProcess.options.DebugFigs.Value;
  % Inverse options
  Method   = sProcess.options.method.Value{1};
  Modality = sProcess.options.sensortype.Value{1};
  % Get unique channel files 
  AllChannelFiles = unique({sInputs.ChannelFile});
  % Progress bar
  bst_progress('start', 'ft_sourceanalysis', 'Loading input files...', 0, 2*length(sInputs));

  % ===== LOOP ON FOLDERS =====
  for iChanFile = 1:length(AllChannelFiles)
    bst_progress('text', 'Loading input files...');
    % Get the study
    %[sStudyChan, iStudyChan] = bst_get('ChannelFile', AllChannelFiles{iChanFile});
    [sStudyChan, ~] = bst_get('ChannelFile', AllChannelFiles{iChanFile});
    % Error if there is no head model available
    if isempty(sStudyChan.iHeadModel)
      bst_report('Error', sProcess, [], ['No head model available in folder: ' bst_fileparts(sStudyChan.FileName)]);
      continue;
    elseif isempty(sStudyChan.NoiseCov) || isempty(sStudyChan.NoiseCov(1).FileName)
      bst_report('Error', sProcess, [], ['No noise covariance matrix available in folder: ' bst_fileparts(sStudyChan.FileName)]);
      continue;
    end
    % Load channel file
    ChannelMat = in_bst_channel(AllChannelFiles{iChanFile});
    % Get selected sensors
    iChannels = channel_find(ChannelMat.Channel, Modality);
    if isempty(iChannels)
      bst_report('Error', sProcess, sInputs, ['Channels "' Modality '" not found in channel file.']);
      return;
    end
    % Load head model
    HeadModelFile = sStudyChan.HeadModel(sStudyChan.iHeadModel).FileName;
    HeadModelMat  = in_bst_headmodel(HeadModelFile);
    % Load data covariance matrix
    NoiseCovFile = sStudyChan.NoiseCov(1).FileName;
    NoiseCovMat  = load(file_fullpath(NoiseCovFile));

    % ===== LOOP ON DATA FILES =====
    % Get data files for this channel file
    iChanInputs = find(ismember({sInputs.ChannelFile}, AllChannelFiles{iChanFile}));
    % Loop on data files
    for iInput = 1:length(iChanInputs)
      % === LOAD DATA ===
      % Load data
      DataFile = sInputs(iChanInputs(iInput)).FileName;
      DataMat  = in_bst_data(DataFile);
      iStudyData = sInputs(iChanInputs(iInput)).iStudy;
      % Remove bad channels
      iBadChan = find(DataMat.ChannelFlag == -1);
      iChannelsData = setdiff(iChannels, iBadChan);
      % Error: All channels tagged as bad
      if isempty(iChannelsData)
        bst_report('Error', sProcess, sInputs, 'All the selected channels are tagged as bad.');
        return;
      end

      % ===== LOAD SURFACE ATLAS INFO =====
      % Load the surface filename from results file
      %ResultsMat_atlas = in_bst_results(DataFile);
      %ResultsMat_atlas = in_bst_results(DataFile,0,{'SurfaceFile','Atlas'});
      ResultsMat_atlas = in_bst_data(DataFile);
      % Error: cannot process results from volume grids
      if ismember(HeadModelMat.HeadModelType, {'volume', 'mixed'})
        bst_report('Error', sProcess, sInput, 'Atlases are not supported yet for volumic grids.');
        return;
      elseif isempty(HeadModelMat.SurfaceFile)
        bst_report('Error', sProcess, sInput, 'Surface file is not defined.');
        return;
      elseif isfield(ResultsMat_atlas, 'Atlas') && ~isempty(ResultsMat_atlas.Atlas)
        bst_report('Error', sProcess, sInput, 'File is already based on an atlas.');
        return;
      end
      % Load surface
      SurfaceMat = in_tess_bst(HeadModelMat.SurfaceFile);
      if isempty(SurfaceMat.Atlas) 
        bst_report('Error', sProcess, sInput, 'No atlases available in the current surface.');
        return;
      end
      % Forbid this process on mixed head models
      %if (ResultsMat_atlas.nComponents == 0)
      %  bst_report('Error', sProcess, sInput, 'Cannot run this process on mixed source models.');
      %  return;
      %end
      % Get the atlas to use
      iAtlas = [];
      if ~isempty(sProcess.options.AtlasRegions.Value)
        iAtlas = find(strcmpi({SurfaceMat.Atlas.Name}, sProcess.options.AtlasRegions.Value));
        if isempty(iAtlas)
          bst_report('Warning', sProcess, sInput, ['Atlas not found: "' sProcess.options.AtlasRegions.Value '"']);
        end
      end
      if isempty(iAtlas)
        iAtlas = SurfaceMat.iAtlas;
      end
      if isempty(iAtlas)
        iAtlas = 1;
      end
      % Check atlas 
      if isempty(SurfaceMat.Atlas(iAtlas).Scouts)
        bst_report('Error', sProcess, sInput, 'No available scouts in the selected atlas.');
        return;
      end
      bst_report('Info', sProcess, sInputs, ['Using atlas: "' SurfaceMat.Atlas(iAtlas).Name '"']);
      % Get all the scouts in current atlas
      sScouts = SurfaceMat.Atlas(iAtlas).Scouts;

      %% === ESTIMATION OF PARAMETERS ===
      % replace later: using fieldtrip functions to extract leadfield
      % matrix with appropriate filtering of bad channels
      bst_progress('text', 'Estimating inversion kernel...');
      [InvKernel, Estim, EstimAbs, debug] = Compute(...
        HeadModelMat.Gain(iChannelsData,:), ...
        DataMat.F(iChannelsData,:), ...
        NoiseCovMat.NoiseCov(iChannelsData,iChannelsData), ... % noise covariance
        sScouts, ...
        params, params_gradient, ...
        DataMat.Time);

      % === CREATE OUTPUT STRUCTURE ===
      bst_progress('text', 'Saving source file...');
      bst_progress('inc', 1);
      % Create structure
      ResultsMat = db_template('resultsmat');
      ResultsMat.ImagingKernel = InvKernel;
      ResultsMat.ImageGridAmp  = Estim;
      ResultsMat.nComponents   = 3;
      ResultsMat.Comment       = ['Source Estimate w/Region Priors; weighted via ', Method];
      ResultsMat.Function      = Method;
      ResultsMat.Time          = DataMat.Time;
      ResultsMat.DataFile      = DataFile;
      ResultsMat.HeadModelFile = HeadModelFile;
      ResultsMat.HeadModelType = HeadModelMat.HeadModelType;
      ResultsMat.ChannelFlag   = DataMat.ChannelFlag;
      ResultsMat.GoodChannel   = iChannelsData;
      ResultsMat.SurfaceFile   = HeadModelMat.SurfaceFile;
      ResultsMat.nAvg          = DataMat.nAvg;
      ResultsMat.Leff          = DataMat.Leff;
      ResultsMat.params        = params;
      ResultsMat.params_grad   = params_gradient;
      ResultsMat.DebugData     = debug;
      switch lower(ResultsMat.HeadModelType)
        case 'volume'
          ResultsMat.GridLoc    = HeadModelMat.GridLoc;
          % ResultsMat.GridOrient = [];
        case 'surface'
          ResultsMat.GridLoc    = [];
          % ResultsMat.GridOrient = [];
        case 'mixed'
          ResultsMat.GridLoc    = HeadModelMat.GridLoc;
          ResultsMat.GridOrient = HeadModelMat.GridOrient;
      end
      ResultsMat = bst_history('add', ResultsMat, 'compute', ...
        ['Source Estimate w/Region Priors; weighted via ' Method ' ' Modality]);
        
      % === SAVE OUTPUT FILE ===
      % Output filename
      OutputDir = bst_fileparts(file_fullpath(DataFile));
      ResultFile = bst_process('GetNewFilename', OutputDir, ['results_', Method, '_', Modality, ]);
      % Save new file structure
      bst_save(ResultFile, ResultsMat, 'v6');

      % ===== REGISTER NEW FILE =====
     bst_progress('inc', 1);
      % Create new results structure
      newResult = db_template('results');
      newResult.Comment       = ResultsMat.Comment;
      newResult.FileName      = file_short(ResultFile);
      newResult.DataFile      = DataFile;
      newResult.isLink        = 0;
      newResult.HeadModelType = ResultsMat.HeadModelType;
      % Get output study
      sStudyData = bst_get('Study', iStudyData);
      % Add new entry to the database
      iResult = length(sStudyData.Result) + 1;
      sStudyData.Result(iResult) = newResult;
      % Update Brainstorm database
      bst_set('Study', iStudyData, sStudyData);
      % Store output filename
      OutputFiles{end+1} = newResult.FileName;
      % Expand data node
      panel_protocols('SelectNode', [], newResult.FileName);
    end
  end
  % Save database
  db_save();
  % Hide progress bar
  bst_progress('stop');
end

%%
% USAGE: x = process_notch('Compute', x, sfreq, FreqList)
function [kernel, estim, estim_abs, debug] = Compute(G, Y, COV, atlas_regions, ...
  params, params_gradient, time)
% TODO: Add documentation
%
%-------------------------------------------------------------------------
% INPUT
%
%        G  Leadfiel matrix, MxN
%        Y  Sensors data, MxT
%      COV  Covariance matrix of Y|J, MxM
%     time  Vector wit time, 1xT
%
%-------------------------------------------------------------------------
% INPUT (OPTIONAL)
%   params  Additional parameters, like error tolerance and max iterations
% params_gradient Additional parameters for gradient descent
%
%-------------------------------------------------------------------------
% OUTPUT
%
%   kernel  (W) Inversion kernel such that J=W*Y, NxM
%    estim  (J) Estimation of current density, NxT
%    debug  Errors and gamma at each iteration
%
%-------------------------------------------------------------------------
% Author: Julio Cesar Enciso-Alva, 2023
%         (juliocesar.encisoalva@mavs.uta.edu)
%
  debug = [];
  % === METADATA ===
  meta = [];
  meta.K = size( atlas_regions,2 ) +1;
  meta.N = size(G,2);
  meta.T = size(Y,2);
  meta.M = size(Y,1);

  % === PRE PROCESSING ===
  % Prewhitening
  iCOV = inv(COV);
  if params.PreWhiten
    Y = sqrtm(iCOV)*Y;
    G = sqrtm(iCOV)*G;
    iCOV = eye(meta.M);
  end

  % Region activations
  S  = true(meta.K,1);
  n = zeros(meta.K,1);
  R = [];
  unassigned = true(meta.N,1);
  for k = 1:(meta.K-1)
    R{k} = atlas_regions(k).Vertices;
    n(k) = size(R{k},2);
    unassigned(R{k}) = false;
  end
  if any(unassigned)
    idx = 1:meta.N;
    R{meta.K} = idx(unassigned);
    n(meta.K) = size(R{meta.K},2);
  else
    meta.K = meta.K-1;
    n(meta.K) = [];
  end
  
  % === INITIALIZE ===
  debug.error = zeros(1,      params.MaxIter);
  debug.gamma = zeros(meta.K, params.MaxIter+1);
  debug.sigma = zeros(1,      params.MaxIter+1);

  gamma2 = zeros(meta.K,1);
  W = zeros(meta.N, meta.M);

  % === MAIN CYCLE ===
  debug.gamma(:,1) = gamma2;
  debug.sigma(1)   = mean(diag(iCOV));
  for iter = 1:params.MaxIter
    % kernel is computed using another function
    [W,~] = InversionKernel(G,iCOV,gamma2,R,S,W, meta,params_gradient);
    if iter==1
      % extra computing time if the initial guess was bad
      [W,~] = InversionKernel(G,iCOV,gamma2,R,S,W, meta,params_gradient);
    end
    % if the sources are actually requested
    if params.FullResult
      % todo
    end
    J = W*Y;
    if params.DebugFigures
      figure()
      plot(time,J(5942,:)) % arbitrary dipole in region where the actual source is located
      xlabel('Time [s]')
      ylabel('Current density (J) []')
    end
    debug.error(:,iter) = norm(G*J-Y,'fro')/(meta.N*meta.T);
    % covariance of residuals
    Q = zeros(meta.M,meta.M);
    for t = 1:meta.T
      Q = Q + (G*J(:,t)-Y(:,t))*(G*J(:,t)-Y(:,t))';
    end
    Q = Q/meta.T;
    debug.sigma(iter+1) = mean(diag(iCOV));
    if params.DebugFigures
      PLotCov(Q);
    end
    
    % update of gammas  
    for k = 1:meta.K
      gamma2(k) = 1/( ( norm(J(R{k},:)-mean(J(R{k},:)),'fro')^2 )/(n(k)*meta.T) ) ;
    end
    debug.gamma(:,iter+1) = gamma2;
  end
  kernel = W;
  estim  = W*Y;
  % Get magnitude at each dipole, for visualization
  estim_abs = zeros(meta.N/3,meta.T);
  for i = 1:meta.N/3
    estim_abs(i,:) = vecnorm( estim((3*(i-1)+(1:3)),:), 2,1 );
  end
  if params.DebugFigures
    figure()
    plot(log(debug.error))
    %title("Covariance matrix for $Y$",'interpreter','latex')
    xlabel("Iteration",'interpreter','latex')
    ylabel("$log \left\Vert G \hat{J} - Y\right\Vert_F$",'interpreter','latex')
    %
    figure()
    plot(log(debug.gamma(S,:)'))
    xlabel("Iteration",'interpreter','latex')
    ylabel("$log \gamma_k^2$",'interpreter','latex')
  end
end

%% ===== ADDITIONAL FUNCTIONS =====

function [W, debug_info] = ...
  InversionKernel(G,iCOV,gamma2,R,S,W0, meta,params)
% This algortihm construct the following matrix
%       W = [ G'*iCOV*G + (I-A)*GAMMA2 ]^-1 * G'*iCOV
% which is referred as the inversion kernel for the problem. This matrix
% arise from the MAP source estimator, which is linear 
%       J_MAP = W * Y
% This matrix W is computed by solvng the linear system
%       [ G'*iCOV*G + (I-A)*GAMMA2 ] * W = G'*iCOV
% which is solved column-wise, with some code optimizations.
%
% Two method is implemented: steepest gradient descent, conjugate gradient.
%
%-------------------------------------------------------------------------
% INPUT
%
%        G  Leadfiel matrix, MxN
%     iCOV  Inverse of covariance matrix of Y|J, MxM
%   gamma2  Regional weights, Kx1
%       Rk  Region indicators, Nx1
%       W0  Initial estimation of the kernel, NxM
%     meta  Metadata of matrices: N, M, K
%
%-------------------------------------------------------------------------
% INPUT (OPTIONAL)
%   params  Additionalparameters, like error tolerance and max iterations
%  MaxIter  Max number of iterations
%   Method  SteepestDescent, ConjugateGradient
%
%-------------------------------------------------------------------------
% OUTPUT
%
%        W  Inversion kernel, NxM
%
%-------------------------------------------------------------------------
% INPUT (OPTIONAL)
%      ERR  Errors at each iteration
%
%-------------------------------------------------------------------------
% Author: Julio Cesar Enciso-Alva, 2023
%         (juliocesar.encisoalva@mavs.uta.edu)
%
W = W0;
debug_info = [];
debug_info.ERR = zeros(meta.M, params.MaxIter);
GS = G'*iCOV;
switch params.Method
case 'SteepestDescent'
  for i = 1:meta.M
    Gi = GS(:,i);
    Wi = W(:,i);
    for iter = 1:params.MaxIter
      % p = b-A*x
      p  = zeros(meta.N,1);
      for k = 1:meta.K
        if S(k)
          p(R{k}) = -(Wi(R{k}) - mean(Wi(R{k})) )*gamma2(k);
        else
          p(R{k}) = -(Wi(R{k}) )*gamma2(k);
        end
      end
      p = p + Gi - (G'*(iCOV*(G*Wi)));
      % alpha = p'*p / p'*A*p
      Ap = zeros(meta.N,1);
      for k = 1:meta.K
        if S(k)
          Ap(R{k}) = (p(R{k}) - mean(p(R{k})) )*gamma2(k);
        else
          Ap(R{k}) = (p(R{k}) )*gamma2(k);
        end
      end
      Ap = Ap + (G'*(iCOV*(G*p)));
      alpha = (p'*p) / (p'*Ap);
      % x_new = x + alpha*p
      Wi = Wi + alpha*p;
      debug_info.ERR(i,iter) = norm(p);
    end
    W(:,i) = Wi;
  end
case 'ConjugateGradient'
  for i = 1:meta.M
    % b <- Gi,   x <- Wi
    Gi = GS(:,i);
    Wi = W(:,i);
    bestW = Wi;
    bestE = Inf;
    % p0 = b-A*x0,   r0 = p0
    p  = zeros(meta.N,1);
    for k = 1:meta.K
      if S(k)
        p(R{k}) = -(Wi(R{k}) - mean(Wi(R{k})) )*gamma2(k);
      else
        p(R{k}) = -(Wi(R{k}) )*gamma2(k);
      end
    end
    p = p + Gi - (G'*(iCOV*(G*Wi)));
    r = p;
    for iter = 1:params.MaxIter
      % alpha = p'*r / p'*A*p
      Ap = zeros(meta.N,1);
      for k = 1:meta.K
        if S(k)
          Ap(R{k}) = (p(R{k}) - mean(p(R{k})) )*gamma2(k);
        else
          Ap(R{k}) = (p(R{k}) )*gamma2(k);
        end
      end
      Ap = Ap + (G'*(iCOV*(G*p)));
      alpha = (p'*r) / (p'*Ap);
      % x_new = x + alpha*p,   r_new = r - alpha*A*p
      Wi = Wi + alpha*p;
      r  = r  - alpha*Ap;
      % beta = (A*p)'*r_new / p'*A*p
      beta = -( Ap'*r )/( p'*Ap );
      % p_new = r_new + beta*p
      p  = r + beta*p;
      er = norm(r);
      debug_info.ERR(i,iter) = er;
      if er < bestE
        bestE = er;
        bestW = Wi;
      end
    end
    W(:,i) = bestW;
  end
end
if params.PlotError
  PlotWithErrorshade(1:params.MaxIter,log(debug_info.ERR))
  xlabel('Iteration')
  ylabel('log( error )')
end
end

%% generic function
function PlotWithErrorshade(idx, DATA, yl)
%
% Plot of mean with variance as a shaded region.
%
% Will plot mean( X(t) ) vs t as solid line, and SD( X(t) ) vs t as shaded
% region. It is created on a new figure().
%
%---------------------------
% Required Inputs:
%---------------------------
%   idx   vector of t,    should be 1xN
%     X   vector of X(t), should be RxN
%
%---------------------------
% Optional Inputs:
%---------------------------
%    yl   limits at y-axis
%
%---------------------------
% Output
%---------------------------
%    []   Only plot, no value
%
MEAN  = mean(DATA,1);
STD   = mean((DATA-MEAN).^2,1).^(1/2);
figure()
% fill( X,Y,C ), fills points (x,y) counterclockwise
hold on
fill([idx,fliplr(idx)], [MEAN-STD,fliplr(MEAN+STD)], ...
  [.7,.7,.7], 'EdgeColor','none' )
plot(idx,MEAN,'k')
if nargin == 3
  ylim(yl)
end
end

%% another generic function
function PLotCov( C )
% This function grpahs theinverse correlation matrix C with custom 
% formatting fitting my personal tastes.
figure()
t = tiledlayout(1,2);
t.Padding = 'compact';
t.TileSpacing = 'compact';
nexttile
imagesc(C);
colorbar
title("Inverse of Covariance matrix for $Y$, $\Sigma^{-1}$",'interpreter','latex')
nexttile
histogram(C(:))
title("Histgram of $\Sigma^{-1}$ entries",'interpreter','latex')
ylabel("Frequency",'interpreter','latex')
end