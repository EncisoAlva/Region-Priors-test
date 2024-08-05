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
  sProcess.options.MaxIterGrad.Value   = {100, '', 0};   % {Default value, units, precision}
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
  params.MaxIter      = sProcess.options.MaxIter.Value{1};
  params.PlotError    = sProcess.options.DebugFigs.Value;
  params.PreWhiten    = sProcess.options.Prewhiten.Value;
  params.DebugFigures = sProcess.options.DebugFigs.Value;
  params.AbsRequired = sProcess.options.AbsVal.Value;
  params.ResFormat   = sProcess.options.FullKernel.Value{1};
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
        bst_report('Error', sProcess, sInputs, 'Atlases are not supported yet for volumic grids.');
        return;
      elseif isempty(HeadModelMat.SurfaceFile)
        bst_report('Error', sProcess, sInputs, 'Surface file is not defined.');
        return;
      elseif isfield(ResultsMat_atlas, 'Atlas') && ~isempty(ResultsMat_atlas.Atlas)
        bst_report('Error', sProcess, sInputs, 'File is already based on an atlas.');
        return;
      end
      % Load surface
      SurfaceMat = in_tess_bst(HeadModelMat.SurfaceFile);
      if isempty(SurfaceMat.Atlas) 
        bst_report('Error', sProcess, sInputs, 'No atlases available in the current surface.');
        return;
      end
      % Forbid this process on mixed head models
      %if (ResultsMat_atlas.nComponents == 0)
      %  bst_report('Error', sProcess, sInputs, 'Cannot run this process on mixed source models.');
      %  return;
      %end
      % Get the atlas to use
      iAtlas = [];
      if ~isempty(sProcess.options.AtlasRegions.Value)
        iAtlas = find(strcmpi({SurfaceMat.Atlas.Name}, sProcess.options.AtlasRegions.Value));
        if isempty(iAtlas)
          bst_report('Warning', sProcess, sInputs, ['Atlas not found: "' sProcess.options.AtlasRegions.Value '"']);
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
        bst_report('Error', sProcess, sInputs, 'No available scouts in the selected atlas.');
        return;
      end
      bst_report('Info', sProcess, sInputs, ['Using atlas: "' SurfaceMat.Atlas(iAtlas).Name '"']);
      % Get all the scouts in current atlas
      sScouts = SurfaceMat.Atlas(iAtlas).Scouts;

      %% === ESTIMATION OF PARAMETERS ===
      % replace later: using fieldtrip functions to extract leadfield
      % matrix with appropriate filtering of bad channels
      bst_progress('text', 'Estimating inversion kernel...');
      [InvKernel, Estim, debug] = Compute(...
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
      if params.AbsRequired
        if strcmp(params_gradient.ResFormat,'kernel')
          ResultsMat.nComponents   = 3;
          disp('Request for asolute value is being ignored.')
        else
          ResultsMat.nComponents   = 1;
        end
      else
        ResultsMat.nComponents   = 3;
      end
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
function [kernel, estim, debug] = Compute(G, Y, COV, atlas_regions, ...
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
  S = false(meta.K,1);
  n = zeros(meta.K,1);
  R = cell( 3*(meta.K-1),1 );
  unassigned = true(meta.N,1);
  for k = 1:(meta.K-1)
    for nu = 1:3
      R{3*k-3+nu} = atlas_regions(k).Vertices*3-3+nu;
      unassigned(R{3*k-3+nu}) = false;
    end
    %R{k} = atlas_regions(k).Vertices;
    n(k) = size(R{3*k-1},2);
  end
  if any(unassigned)
    idx = 1:(meta.N);
    idx = unique(ceil(idx(unassigned)/3-1));
    for nu = 1:3
      R{3*meta.K-3+nu} = idx*3-3+nu;
    end
    %R{meta.K} = idx(unassigned);
    n(meta.K) = size(R{meta.K*3-1},2);
  else
    meta.K = meta.K-1;
    n(meta.K) = [];
  end
  if false
    figure()
    hold on
    for k = 1:meta.K
      idx = unique(ceil(R{k*3-1}/3));
      if k ~= meta.K
        %q = k*ones(n(k),1);
        q = 'blue';
      else
        q = 'red';
      end
      scatter3( forward_model_icbm152.GridLoc(idx,1),...
        forward_model_icbm152.GridLoc(idx,2),...
        forward_model_icbm152.GridLoc(idx,3),100,q,'.')
    end
  end
  
  % === INITIALIZE ===
  debug.error = zeros(1,      params.MaxIter);
  debug.gamma = zeros(meta.K, params.MaxIter+1);
  debug.sigma = zeros(1,      params.MaxIter+1);

  gamma2 = zeros(meta.K,1);
  Gnorm  = vecnorm(G,2,1);
  for k = 1:meta.K
    gamma2(k) = median( [Gnorm(R{3*k-3+1}),Gnorm(R{3*k-3+2}),Gnorm(R{3*k-3+3})] )^2;
  end

  % === MAIN CYCLE ===
  debug.gamma(:,1) = gamma2;
  debug.sigma(1)   = mean(diag(iCOV));
  switch params.ResFormat
    case {'kernel','both'}
      W  = zeros(meta.N, meta.M);
      Id = eye(meta.M);
      for iter = 1:params.MaxIter
        % kernel is computed using another function
        if iter==1
          % extra computing time if the initial guess was bad
          params_gradient.MaxIter = params_gradient.MaxIter*10;
          [W,~] = GradDescent(G,iCOV,gamma2,R,S,Id,W, ...
            meta,params_gradient);
          params_gradient.MaxIter = params_gradient.MaxIter/10;
        end
        [W,~] = GradDescent(G,iCOV,gamma2,R,S,Id,W, ...
          meta,params_gradient);
        if params.DebugFigures
          figure()
          random_line = W(5942,:)*Y;
          plot(time,random_line) % arbitrary dipole in region where the actual source is located
          xlabel('Time [s]')
          ylabel('Current density (J) []')
        end
        err = 0;
        for t = 1:meta.T
          err = err + norm(G*W*Y-Y,'fro');
        end
        debug.error(:,iter) = err/(meta.N*meta.T);
        % covariance of residuals
        Q = zeros(meta.M,meta.M);
        for t = 1:meta.T
          Q = Q + (G*W*Y(:,t)-Y(:,t))*(G*W*Y(:,t)-Y(:,t))';
        end
        Q = Q/meta.T;
        debug.sigma(iter+1) = mean(diag(Q));
        if params.DebugFigures
          PLotCov(Q);
        end
        % update of gammas  
        for k = 1:meta.K
          nrm = 0;
          for nu = 1:3
            for t = 1:meta.T
              nrm = nrm + norm(W(R{3*k-3+nu},:)*Y(:,t)-mean(W(R{3*k-3+nu},:)*Y(:,t)),'fro')^2;
            end
          end
          gamma2(k) = 1/( ( nrm )/( 3*n(k)*meta.T ) ) ;
          %gamma2(k) = 1/( ( norm(J(R{k},:)-mean(J(R{k},:)),'fro')^2 )/(n(k)*meta.T) ) ;
        end
        debug.gamma(:,iter+1) = gamma2;
      end
      switch params.ResFormat
        case 'kernel'
          kernel = W;
          estim  = [];
        case 'both'
          kernel = W;
          estim  = W*Y;
      end
    case 'full'
      J = zeros(meta.N, meta.T);
      for iter = 1:params.MaxIter
        % full result is computed using another function
        if iter==1
          % extra computing time if the initial guess was bad
          params_gradient.MaxIter = params_gradient.MaxIter*10;
          [J,~] = GradDescent(G,iCOV,gamma2,R,S,Y,J, ...
            meta,params_gradient);
          params_gradient.MaxIter = params_gradient.MaxIter/10;
        end
        [J,~] = GradDescent(G,iCOV,gamma2,R,S,Y,J, ...
          meta,params_gradient);
        if params.DebugFigures
          figure()
          random_line = J(5942,:);
          plot(time,random_line) % arbitrary dipole in region where the actual source is located
          xlabel('Time [s]')
          ylabel('Current density (J) []')
        end
        err = 0;
        for t = 1:meta.T
          err = err + norm(G*J-Y,'fro');
        end
        debug.error(:,iter) = err/(meta.N*meta.T);
        % covariance of residuals
        Q = zeros(meta.M,meta.M);
        for t = 1:meta.T
          Q = Q + (G*J(:,t)-Y(:,t))*(G*J(:,t)-Y(:,t))';
        end
        Q = Q/meta.T;
        debug.sigma(iter+1) = mean(diag(Q));
        if params.DebugFigures
          PLotCov(Q);
        end
        % update of gammas  
        for k = 1:meta.K
          nrm = 0;
          for nu = 1:3
            nrm = nrm + norm(J(R{3*k-3+nu},:)-mean(J(R{3*k-3+nu},:)),'fro')^2;
          end
          gamma2(k) = 1/( ( nrm )/( 3*n(k)*meta.T ) ) ;
        end
        debug.gamma(:,iter+1) = gamma2;
      end
      kernel = [];
      estim  = J;
  end
  % Get magnitude at each dipole, for visualization 
  if params.AbsRequired
    switch params.ResFormat
      case 'kernel'
        disp('Absolute values are not computed by this function.')
      case {'full', 'both'}
        estim_abs = zeros(meta.N/3,meta.T);
        for i = 1:meta.N/3
          estim_abs(i,:) = vecnorm( estim((3*(i-1)+(1:3)),:), 2,1 );
        end
      estim = estim_abs;
    end
  end
  if params.DebugFigures
    figure()
    plot(log(debug.error))
    %title("Covariance matrix for $Y$",'interpreter','latex')
    xlabel("Iteration",'interpreter','latex')
    ylabel("$log \left\Vert G \hat{J} - Y\right\Vert_F$",'interpreter','latex')
    %
    figure()
    plot(log(debug.gamma'))
    xlabel("Iteration",'interpreter','latex')
    ylabel("$log \gamma_k^2$",'interpreter','latex')
  end
end

%% ===== ADDITIONAL FUNCTIONS =====

function [W, debug_info] = ...
  GradDescent(G,iCOV,gamma2,R,S,B,W0, meta,params)
% This algorithm solves the following system for W
%       [ G'*iCOV*G + (I-A)*GAMMA2 ] * W = iCOV*G'*B
% via Gradient Descent. This implementation is optimized for the particular
% matrices that occur in the context of the probelm of interest.
%
% Two method implemented: steepest gradient descent, conjugate gradient.
%
%-------------------------------------------------------------------------
% INPUT
%
%        G  Leadfiel matrix, MxN
%     iCOV  Inverse of covariance matrix of Y|J, MxM
%   gamma2  Regional weights, Kx1
%       Rk  Region indicators, Nx1
%        B  (kernel) idetity, M*M (full) Y, MxT
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
%        W  (kernel) Inversion kernel, NxM (full) J, N*T
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
switch params.Method
case 'SteepestDescent'
  for i = 1:size(W0,2)
    Bi = G'*iCOV* B(:,i);
    Wi = W0(:,i);
    for iter = 1:params.MaxIter
      % p = b-A*x
      p  = zeros(meta.N,1);
      for k = 1:meta.K
        if S(k)
          for nu = 1:3
            p(R{3*k-3+nu}) = -(Wi(R{3*k-3+nu}) - mean(Wi(R{3*k-3+nu})) )*gamma2(k);
          end
        else
          for nu = 1:3
            p(R{3*k-3+nu}) = -(Wi(R{3*k-3+nu}) )*gamma2(k);
          end
        end
      end
      p = p + Bi - (G'*(iCOV*(G*Wi)));
      % alpha = p'*p / p'*A*p
      Ap = zeros(meta.N,1);
      for k = 1:meta.K
        if S(k)
          for nu = 1:3
            Ap(R{3*k-3+nu}) = (p(R{3*k-3+nu}) - mean(p(R{3*k-3+nu})) )*gamma2(k);
          end
        else
          for nu=1:3
            Ap(R{3*k-3+nu}) = (p(R{3*k-3+nu}) )*gamma2(k);
          end
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
  for i = 1:size(W0,2)
    % b <- Bi,   x <- Wi
    Bi = G'*iCOV* B(:,i);
    Wi = W0(:,i);
    bestW = Wi;
    bestE = Inf;
    % p0 = b-A*x0,   r0 = p0
    p  = zeros(meta.N,1);
    for k = 1:meta.K
      if S(k)
        for nu = 1:3
          p(R{3*k-3+nu}) = -(Wi(R{3*k-3+nu}) - mean(Wi(R{3*k-3+nu})) )*gamma2(k);
        end
      else
        for nu = 1:3
          p(R{3*k-3+nu}) = -(Wi(R{3*k-3+nu}) )*gamma2(k);
        end
      end
    end
    p = p + Bi - (G'*(iCOV*(G*Wi)));
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
      alpha = (r'*r) / (p'*Ap);
      % x_new = x + alpha*p,   r_new = r - alpha*A*p
      r_ = r; %(non-updated r)
      Wi = Wi + alpha* p;
      r  = r  - alpha*Ap;
      % beta = (A*p)'*r_new / p'*A*p
      %beta = -( Ap'*r )/( p'*Ap );
      % beta = r_new'*r_new / r_old'*r_old
      beta = (r'*r) / ( r_'*r_);
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