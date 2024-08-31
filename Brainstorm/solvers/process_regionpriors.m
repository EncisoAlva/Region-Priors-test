function varargout = process_RegionPriors( varargin )
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
  sProcess.Comment     = 'Binary Region Priors';
  sProcess.Category    = 'Custom';
  sProcess.SubGroup    = 'Sources';
  sProcess.Index       = 1000;
  sProcess.FileTag     = '';
  sProcess.Description = 'https://github.com/EncisoAlva/Region-Priors';
  % Definition of the input accepted by this process
  sProcess.InputTypes  = {'data', 'raw'};
  sProcess.OutputTypes = {'data', 'raw'};
  sProcess.nInputs     = 1;
  %sProcess.nMinFiles   = 1;
  %
  % Debug options
  %sProcess.options.Debug.Comment    = 'Enable debug options';
  %sProcess.options.Debug.Type       = 'checkbox';
  %sProcess.options.Debug.Value      = 0; % Selected or not by default
  %sProcess.options.Debug.Controller = 'Debug';
  %sProcess.options.Debug.Hidden  = 0;
  %
  sProcess.options.DebugFigs.Comment = 'Show debug figures';
  sProcess.options.DebugFigs.Type    = 'checkbox';
  sProcess.options.DebugFigs.Value   = 0;                 % Selected or not by default
  sProcess.options.DebugFigs.Class   = 'Debug';
  sProcess.options.DebugFigs.Hidden  = 0;
  %
  %sProcess.options.MaxIterGrad.Comment = 'Max iterations (Kernel): ';
  %sProcess.options.MaxIterGrad.Type    = 'value';
  %sProcess.options.MaxIterGrad.Value   = {100, '', 0};   % {Default value, units, precision}
  %sProcess.options.MaxIterGrad.Class   = 'Debug';
  %sProcess.options.MaxIterGrad.Hidden  = 0;
  %
  %sProcess.options.MethodGrad.Comment = 'Method (Kernel): ';
  %sProcess.options.MethodGrad.Type    = 'combobox_label';
  %sProcess.options.MethodGrad.Value   = {'ConjugateGradient',...
  %  {'ConjugateGradient','SteepestDescent';...
  %  'ConjugateGradient','SteepestDescent'}};
  %sProcess.options.MethodGrad.Class   = 'Debug';
  %sProcess.options.MethodGrad.Hidden  = 0;
  %
  %sProcess.options.scouts.Comment = 'Select active regions: (Default is None)';
  %sProcess.options.scouts.Type    = 'scout';
  %sProcess.options.scouts.Value   = {};
  %
  % Process options
  sProcess.options.Prewhiten.Comment = 'Prewhiten';
  sProcess.options.Prewhiten.Type    = 'checkbox';
  sProcess.options.Prewhiten.Value   = 0;                 % Selected or not by default
  sProcess.options.DebugFigs.Class   = 'Pre';
  %
  %sProcess.options.MaxIter.Comment = 'Max iterations: ';
  %sProcess.options.MaxIter.Type    = 'value';
  %sProcess.options.MaxIter.Value   = {5, '', 0};   % {Default value, units, precision}
  %
  % Option: Atlas
  %sProcess.options.AtlasRegions.Comment = 'Select atlas (currently not working):';
  %sProcess.options.AtlasRegions.Type    = 'atlas';
  %sProcess.options.AtlasRegions.Value   = [];    
  %
  % Option: Parameter Tuning
  sProcess.options.tuning.Comment = 'Parameter Tuning via';
  sProcess.options.tuning.Type    = 'combobox_label';
  sProcess.options.tuning.Value   = {'gcv', {...
    'Gen. Cross-Validation', 'gcv' ;...
    'L-curve', 'Lcurv' ;...
    'U-curve', 'Ucurv' ;...
    'CRESO', 'CRESO' ;...
    'Median eigenv.', 'median'; ...
    }'};
  % Option: Sensors selection
  sProcess.options.sensortype.Comment = 'Sensor type:';
  sProcess.options.sensortype.Type    = 'combobox_label';
  %sProcess.options.sensortype.Value   = {'EEG', {'MEG', 'MEG GRAD', 'MEG MAG', 'EEG', 'SEEG', 'ECOG'; ...
  %                                               'MEG', 'MEG GRAD', 'MEG MAG', 'EEG', 'SEEG', 'ECOG'}};
  sProcess.options.sensortype.Value   = {'EEG', {'EEG', 'SEEG', 'ECOG'; ...
                                                 'EEG', 'SEEG', 'ECOG'}};
  %
  sProcess.options.AbsVal.Comment = 'Compute absolute values only';
  sProcess.options.AbsVal.Type    = 'checkbox';
  sProcess.options.AbsVal.Value   = 0;                 % Selected or not by default
  %sProcess.options.DebugFigs.Hidden  = 0;
  %
  sProcess.options.FullKernel.Comment = 'Full Results?';
  sProcess.options.FullKernel.Type    = 'combobox_label';
  %sProcess.options.sensortype.Value   = {'EEG', {'MEG', 'MEG GRAD', 'MEG MAG', 'EEG', 'SEEG', 'ECOG'; ...
  %                                               'MEG', 'MEG GRAD', 'MEG MAG', 'EEG', 'SEEG', 'ECOG'}};
  sProcess.options.FullKernel.Value   = {'kernel', {'Full Result (NxT)', 'Only Kernel (NxM)', 'Both'; ...
                                                 'full', 'kernel','both'}};
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
  params.PreWhiten   = sProcess.options.Prewhiten.Value;
  params.AbsRequired = sProcess.options.AbsVal.Value;
  params.ResFormat   = sProcess.options.FullKernel.Value{1};
  %
  params.Tuner       = sProcess.options.tuning.Value{1};
  %params.MaxIter     = sProcess.options.MaxIter.Value{1};
  %params.PlotError    = sProcess.options.DebugFigs.Value;
  params.DebugFigures = sProcess.options.DebugFigs.Value;
  %
  %params_gradient = [];
  %params_gradient.MaxIter   = sProcess.options.MaxIterGrad.Value{1};
  %params_gradient.Tol       = 10^(-10);
  %params_gradient.Method    = sProcess.options.MethodGrad.Value{1};
  %params_gradient.PlotError = sProcess.options.DebugFigs.Value;
  % Inverse options
  Modality = sProcess.options.sensortype.Value{1};
  %
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
    % model-derived parameters
    params.SourceType  = HeadModelMat.HeadModelType;

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

      % Forbid this process on mixed head models
      if ismember(HeadModelMat.HeadModelType, {'mixed'})
        bst_report('Error', sProcess, sInputs, 'Mixed sources are not supported yet.');
        return;
      elseif isempty(HeadModelMat.SurfaceFile)
        bst_report('Error', sProcess, sInputs, 'Surface file is not defined.');
        return;
      end
      params.SourceType = HeadModelMat.HeadModelType;

      %% === ESTIMATION OF PARAMETERS ===
      % replace later: using fieldtrip functions to extract leadfield
      % matrix with appropriate filtering of bad channels
      bst_progress('text', 'Computing inversion kernel...');
      [InvKernel, Estim, ~] = Compute( ...
        HeadModelMat.Gain(iChannelsData,:), ... % G: leadfield
        DataMat.F(iChannelsData,:), ...         % Y: EEG data
        NoiseCovMat.NoiseCov(iChannelsData,iChannelsData), ... % COV: covariance matrix for Y
        params ); % parameters of many kinds

      % === CREATE OUTPUT STRUCTURE ===
      bst_progress('text', 'Saving source file...');
      bst_progress('inc', 1);
      % Create structure
      ResultsMat = db_template('resultsmat');
      ResultsMat.ImagingKernel = InvKernel;
      ResultsMat.ImageGridAmp  = Estim;
      if params.AbsRequired
        if strcmp(params_gradient.ResFormat,'kernel')
          ResultsMat.nComponents   = 3;
          disp('Request for asolute value is safely ignored.')
        else
          ResultsMat.nComponents   = 1;
        end
      else
        ResultsMat.nComponents   = 3;
      end
      ResultsMat.Comment       = ['Source Estimate via MNE; tuning via ', params.Tuner];
      ResultsMat.Function      = params.Tuner;
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
      %ResultsMat.params_grad   = params_gradient;
      %ResultsMat.DebugData     = debug;
      switch lower(ResultsMat.HeadModelType)
        case 'volume'
          ResultsMat.GridLoc    = HeadModelMat.GridLoc;
          % ResultsMat.GridOrient = [];
        case 'surface'
          ResultsMat.GridLoc    = HeadModelMat.GridLoc;
          ResultsMat.GridOrient = HeadModelMat.GridOrient;
        case 'mixed'
          ResultsMat.GridLoc    = HeadModelMat.GridLoc;
          ResultsMat.GridOrient = HeadModelMat.GridOrient;
      end
      ResultsMat = bst_history('add', ResultsMat, 'compute', ...
        ['WMNE Source Estimate; tuned via ' params.Tuner ' ' Modality]);
        
      % === SAVE OUTPUT FILE ===
      % Output filename
      OutputDir = bst_fileparts(file_fullpath(DataFile));
      ResultFile = bst_process('GetNewFilename', OutputDir, ['results_', params.Tuner, '_', Modality, ]);
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
function [kernel, estim, debug] = Compute(G, Y, COV, ...
  params)
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
%
%   params  Additional parameters, like error tolerance and max iterations
% params_gradient Additional parameters for gradient descent
%
%-------------------------------------------------------------------------
% OUTPUT
%
%   kernel  (W) Inversion kernel such that J=W*Y, NxM
%    estim  (J) Estimation of current density, NxT
%    debug  [used on other functions]
%
%-------------------------------------------------------------------------
% Author: Julio Cesar Enciso-Alva, 2024
%         (juliocesar.encisoalva@mavs.uta.edu)
%
  debug = [];
  % === METADATA ===
  meta = [];
  meta.T = size(Y,2);
  meta.M = size(Y,1);
  meta.N = size(G,2);
  %switch params.SourceType
  %  case 'volume'
  %    meta.n = size(G,2)/3;
  %  case 'surface'
  %    meta.n = size(G,2);
  %end
  meta.R = min(meta.M, meta.N);
  %
  meta.G = G;
  meta.Y = Y;
  meta.COV = COV;

  % === PRE PROCESSING ===
  % Prewhitening
  meta.iCOV = pinv(meta.COV);
  if params.PreWhiten
    iCOV_half = sqrtm(meta.iCOV);
    meta.Y = iCOV_half*meta.Y;
    meta.G = iCOV_half*meta.G;
    meta.iCOV = eye(meta.M);
  end

  % Column normalization
  meta.ColNorm = vecnorm(meta.G,2,1);
  meta.G_ = meta.G;
  for q = 1:size(meta.G,2)
    meta.G_(:,q) = meta.G(:,q)/meta.ColNorm(q);
  end

  % SVD decomposition of leadfield matrix
  [U,S,V] = svd(meta.G_, "econ", "vector");
  meta.U = U;
  meta.S = S;
  meta.V = V;

  % provisional values for hot start: p1 is empty
  meta.P  = zeros(meta.N, 1);
  meta.g0 = 1;
  meta.g1 = 1;
  meta.P1 = [];
  switch info.SourceType
    case 'surface'
      meta.nu = meta.N;
    case 'volume'
      meta.nu = meta.N*3;
  end

  % hot start: the WMNE solution
  oldJ = RegionMNEsol( meta, median(meta.S)^2, meta.Y);
  switch params.SourceType
    case 'surface'
      oldJnorm = vecnorm( oldJ.^2, 2, 2);
    case 'volume'
      oldJnorm = vecnorm( dip_norm(oldJ).^2, 2, 2);
  end

  %initial partition of P0
  g0 = min( oldJnorm(:) );
  g1 = max( oldJnorm(:) );
  N0 = max( sum( oldJnorm < (g0+g1)/2 ), meta.N/2 );
  lambda0 = median(meta.S)^2*( g1 );

  % loop
  counter = 0; err = Inf;
  idxs = 1:meta.N;
  while (counter < 20) && ( err>10^-5 )
    counter = counter + 1;
    for iter2 = 1:5
      % identification of regions based on a threshold
      thr = ( N0*g0 + (meta.N-N0)*g1 ) / meta.N;
      if (thr < min(oldJnorm)) || (thr > max(oldJnorm))
        thr = ( min(oldJnorm) + max(oldJnorm) )/2;
      end
      R0  = ( oldJnorm <= thr );
      R1  = ( oldJnorm >  thr );
      N0  = sum(R0);
      %
      if params.DebugFigures
        figure()
        tiledlayout(2,2)
        nexttile([1,2])
        histogram(oldJnorm)
        xline(thr, '-', {'Threshold'})
        %
        nexttile
        histogram(oldJnorm(R0))
        nexttile
        histogram(oldJnorm(R1))
      end
      % copmutation of parameters from each region
      g0  = mean( oldJnorm(R0) );
      g1  = mean( oldJnorm(R1) );
    end
    meta.g0 = g0;
    meta.g1 = g1;
    meta.P1 = idxs(R1);

    % P-regions matrix
    switch info.SourceType
      case 'surface'
        P = zeros(meta.N, 1);
        P( R1 ) = 1;
        nu = meta.N;
      case 'volume'
        P = zeros(meta.N*3, 1);
        P( (kron(R1, [1,1,1]'))>0 ) = 1;
        nu = meta.N*3;
    end
    meta.P  = P;
    meta.nu = nu;
    
    % regional-weighted matrix GA
    GR = mean( meta.G_(:,P>0), 2 );
    meta.GA = sqrt(meta.g0)*G_;
    meta.GA(meta.P1) = meta.GA(meta.P1) + (sqrt(meta.g0+meta.g1)-sqrt(meta.g0))*GR;
    [U,S,V] = svd(meta.GA, "econ", "vector");
    meta.U = U;
    meta.S = S;
    meta.V = V;

    % parameter tuning
    InvParams = RegionsMNE_tune(params, meta, lambda0);
    lambda0   = InvParams.best_lambda;

    % new solution
    newJ = RegionMNEsol( meta, InvParams.best_lambda, meta.Y);
    %newJ = Hs * meta.Leadfield' * ...
    %  pinv( meta.Leadfield*Hs*meta.Leadfield' + best_lambda*eye(pars.M) ) * result.data.Y;
    err = norm( newJ-oldJ, 'fro' );

    % update for next iteration, if any
    oldJ = newJ;
    switch params.SourceType
      case 'surface'
        oldJnorm = vecnorm( oldJ.^2, 2, 2);
      case 'volume'
        oldJnorm = vecnorm( dip_norm(oldJ).^2, 2, 2);
    end
    maxJ = max(oldJnorm(:));

    % print partial results
    fprintf("Iter %2d. Err = %3.3d. (0.00): %5d dips. (0.05) : %4d dips.  (0.50) : %3d dips.\n", ...
      counter, err, sum(oldJnorm~=0), sum(oldJnorm>0.05*maxJ), sum(oldJnorm>0.5*maxJ))
  end

  % X ~ chi(1)
  %   E[   X ] = 1
  % Var(   X ) = 2
  %   E[ k*X ] = k
  % Var( k*X ) = 2*k^2

  % === INITIALIZE ===
  %debug.error = zeros(1,      params.MaxIter);

  % === PARAMETER TUNING ===
  % already tuned at the last iteration
  %InvParams = RegionsMNE_tune(params, meta, lambda0);
  
  % === MAIN CYCLE ===
  [kernel, estim ] = RegionMNE(params, InvParams, meta);
  %RegionMNEsol( meta, InvParams.best_lambda, P, g0, g1, meta.Y);

  % Get magnitude at each dipole, for visualization 
  if params.AbsRequired
    switch params.ResFormat
      case 'kernel'
        disp('Absolute values are not computed by this function.')
      case {'full', 'both'}
        switch params.SourceType
          case 'surface'
            estim_abs = abs( estim );
          case 'volume'
            estim_abs = dip_norm( estim );
        end
        estim = estim_abs;
    end
  end
end

%% ===== ADDITIONAL FUNCTIONS =====

function [kernel, J] = RegionMNE(params, InvParams, meta)
% Weighten Minimum-Norm Estimator (wMNE), follows the basic Tikonov
% regularized estimation
%   J^ = argmin_J || G*J-Y ||^2_F + alpha || W*J ||^2_F
% with W the weight induced by column-normalization of G and ||*||_F is the
% Frobenius norm.
%
% The same is achieved by column-normalizing G and unit weight for J.
%
% The parameter alpha is found using Generalized Cross Validation.

% intialize
switch params.ResFormat
  case {'full', 'both'}
    J = RegionMNEsol( meta, InvParams.alpha, meta.Y);
  otherwise
    J = [];
end
switch params.ResFormat
  case {'kernel', 'both'}
    kernel = RegionMNEsol( meta, InvParams.alpha, eye(meta.m));
  otherwise
    kernel = [];
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function JB = RegionMNEsol( meta, alpha, YY)
% short for estimator
JB = zeros( meta.n, size(YY,2) ); %  m*t or m*n, usable for both
for i = 1:meta.r
  JB = JB + ( meta.S(i)/( meta.S(i)^2 + alpha ) ) * ...
    reshape( meta.V(:,i), meta.n, 1 ) * ( meta.U(:,i)' * YY );
end
JR = mean( JB(meta.P1,:), 2 );
J  = meta.g0*JB;
if ~empty(meta.P1)
  J(meta.P1) = J(meta.P1) + meta.g1*JR;
end

for q = 1:meta.n
  J(q,:) = J(q,:) / meta.ColNorm(q);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function InvParams = WMNE_tune(params, meta, lambda0)
% Weighten Minimum-Norm Estimator (wMNE), follows the basic Tikonov
% regularized estimation
%   J^ = argmin_J || G*J-Y ||^2_F + alpha || W*J ||^2_F
% with W the weight induced by column-normalization of G and ||*||_F is the
% Frobenius norm.
%

% init
InvParams = [];

switch params.Tuner
  case 'median'
    % median eigenvalue
    InvParams.alpha = lambda0;
  otherwise
    % common cyle for the other algorithms
    % starting at median eigenvalue
    best_alpha = lambda0;
    scale  = 10;
    for iter = 1:4
      % try values for alpha, compute quanity, get the min
      alphas = best_alpha * (2.^( (-scale):(scale/10):scale ));
      switch params.Tuner
        case 'gcv'
          % Generalized Cross-Validation
          Gs = zeros( size(alphas) );
          for q = 1:length(alphas)
            alpha = alphas(q);
            Gs(q) = metricGCV( meta, alpha );
          end
          [~, idx]   = min(Gs);
          best_alpha = alphas(idx);
        case 'Lcurv'
          % L-curve criterion
          Ns = zeros( size(alphas) );
          Rs = zeros( size(alphas) );
          if iter==1
            maxN = -Inf; maxR = -Inf;
          end
          for q = 1:length(alphas)
            alpha = alphas(q);
            [Ns(q), Rs(q)] = metricLcurve( meta, alpha);
          end
          % scaling, mimicking how I do it manually
          maxN = max(maxN, max(abs(Ns)));
          maxR = max(maxR, max(abs(Rs)));
          % finding the 'elbow' the way I do it manually
          [~,idx] = min( abs(Ns)/maxN +abs(Rs)/maxR );
          best_alpha = alphas(idx);
        case 'Ucurv'
          % U-curve criterion
          Us = zeros( size(alphas) );
          for q = 1:length(alphas)
            alpha = alphas(q);
            Us(q) = metricUcurve( meta, alpha );
          end
          [~, idx]   = min(Us);
          best_alpha = alphas(idx);
        case 'CRESO'
          % Composite Residual and Smoothing Operator
          Cs = zeros( size(alphas) );
          for q = 1:length(alphas)
            alpha = alphas(q);
            Cs(q) = metricCRESO( meta, alpha );
          end
          % numerical derivative
          dCs = zeros( size(alphas) );
          dCs(1)   = ( Cs(2)   - Cs(1)     )/(alphas(2)   - alphas(1));
          dCs(end) = ( Cs(end) - Cs(end-1) )/(alphas(end) - alphas(end-1));
          for q = 2:(length(alphas)-1)
            dCs(q) = ( Cs(q+1) - Cs(q-1) )/(alphas(q+1) - alphas(q-1));
          end
          % first root of the derivative
          idx = find( dCs>0, 1, 'last' );
          best_alpha = alphas(idx);
        otherwise
          % will default to medain eigenvalue
          break
      end
      % if not on the border, reduce scale; else, increase it
      if (1<idx) && (idx<length(alphas))
        scale = scale/10;
      else
        scale = scale*10;
      end
    end
    InvParams.alpha = max(best_alpha, 0.00001);
end

% print the results nicely
switch params.Tuner
  case 'median'
    fprintf("Parameter tuning using median eigenvalue.\n Optimal lambda: %d\n\n",...
      InvParams.alpha)
  case 'gcv'
    fprintf("Parameter tuning via Generalized Cross-Validation.\n Optimal lambda: %d\n\n",...
      InvParams.alpha)
  case 'Lcurv'
    fprintf("Parameter tuning using L-curve criterion.\n Optimal lambda: %d\n\n",...
      InvParams.alpha)
  case 'Ucurv'
    fprintf("Parameter tuning using U-curve criterion.\n Optimal lambda: %d\n\n",...
      InvParams.alpha)
  case 'CRESO'
    fprintf("Parameter tuning using Composite Residual and Smoothing Operator (CRESO) criterion.\n Optimal lambda: %d\n\n",...
      InvParams.alpha)
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function G = metricGCV( meta, alpha)
% Generalized Cross-Validation given the SVD decomposition

% solution
J = RegionMNEsol( meta, alpha, meta.Y);

% residual
R = norm( meta.G*J - meta.Y, 'fro' )^2;

% trace
tra = 0;
for i = 1:meta.r
  tra = tra + (( meta.S(i)^2 )/( meta.S(i)^2 + alpha ));
end

% GCV quantity
G = R /( (meta.m - tra)^2 );
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function C = metricCRESO( meta, alpha)
% CRESO: Composite Residual and Smoothing Operator

% solution
J = RegionMNEsol( meta, alpha, meta.Y);

% norm
N = norm( J, 'fro' )^2;

% residual
R = norm( meta.G*J - meta.Y, 'fro' )^2;

% CRESO quantity
C = -R + alpha*N;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [N, R] = metricLcurve( meta, alpha)
% L-Curve Criterion

% solution
J = RegionMNEsol( meta, alpha, meta.Y);

% norm
N = norm( J, 'fro' )^2;

% residual
R = norm( meta.G*J - meta.Y, 'fro' )^2;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function U = metricUcurve( meta, alpha)
% U-curve criterion

% solution
J = RegionMNEsol( meta, alpha, meta.Y);

% norm
N = norm( J, 'fro' )^2;

% residual
R = norm( meta.G*J - meta.Y, 'fro' )^2;

% geometric mean of resitual and solution norms
U = 1/R + 1/N;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function normJ = dip_norm( J )
% In the case of unconstrained dpoles, the magnitude of J needs to be
% extracted. This happens very often so it is made into a function.
nDips = size(J,1)/3;
normJ = zeros(nDips, size(J,2));
for ii = 1:nDips
  normJ(ii,:) = vecnorm( J(3*(ii-1)+[1,2,3],:), 2, 1 );
end

end