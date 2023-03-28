function [W, debug_info] = ...
  InversionKernel(G,SIGMAi,gamma2,R,S,W0, meta,params)
% This algortihm construct the following matrix
%       W = [ G'*SIGMAi*G + (I-A)*GAMMA2 ]^-1 * G'*SIGMAi
% which is referred as the inversion kernel for the problem. This matrix
% arise from the MAP source estimator, which is linear 
%       J_MAP = W * Y
% This matrix W is computed by solvng the linear system
%       [ G'*SIGMAi*G + (I-A)*GAMMA2 ] * W = G'*SIGMAi
% which is solved column-wise, with some code optimizations.
%
% One method is implemented: steepest gradient descent.
%
%-------------------------------------------------------------------------
% INPUT
%
%        G  Leadfiel matrix, MxN
%   SIGMAi  Inverse of covariance matrix of Y|J, MxM
%   gamma2  Regional weights, Kx1
%       Rk  Region indicators, Nx1
%       W0  Initial estimation of the kernel, NxM
%     meta  Metadata of matrices: N, M, K
%
%-------------------------------------------------------------------------
% INPUT (OPTIONAL)
%   params  Additionalparameters, like error tolerance and max iterations
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
% Written by Julio Cesar Enciso-Alva (juliocesar.encisoalva@mavs.uta.edu)
%
W = zeros(meta.N, meta.M);
debug_info = [];
debug_info.ERR = zeros(meta.M,params.MaxIter);
GS = G'*SIGMAi;
if strcmp(params.Method,'SteepestDescent')
  for i = 1:meta.M
    Gi = GS(:,i);
    Wi = W0(:,i);
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
      p = p + Gi - (G'*(SIGMAi*(G*Wi)));
      % alpha = p'*p / p'*A*p
      Ap = zeros(meta.N,1);
      for k = 1:meta.K
        if S(k)
          Ap(R{k}) = (p(R{k}) - mean(p(R{k})) )*gamma2(k);
        else
          Ap(R{k}) = (p(R{k}) )*gamma2(k);
        end
      end
      Ap = Ap + (G'*(SIGMAi*(G*p)));
      alpha = (p'*p) / (p'*Ap);
      % x_new = x + alpha*p
      Wi = Wi + alpha*p;
      debug_info.ERR(i,iter) = norm(p);
    end
    W(:,i) = Wi;
  end
elseif strcmp(params.Method,'ConjugateGradient')
  for i = 1:meta.M
    % b <- Gi,   x <- Wi
    Gi = GS(:,i);
    Wi = W0(:,i);
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
    p = p + Gi - (G'*(SIGMAi*(G*Wi)));
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
      Ap = Ap + (G'*(SIGMAi*(G*p)));
      alpha = (p'*r) / (p'*Ap);
      % x_new = x + alpha*p,   r_new = r - alpha*A*p
      Wi = Wi + alpha*p;
      r  = r  - alpha*Ap;
      % beta = (A*p)'*r_new / p'*A*p
      beta = -( Ap'*r )/( p'*Ap );
      % p_new = r_new + beta*p
      p = r + beta*p;
      debug_info.ERR(i,iter) = norm(r);
      if debug_info.ERR(i,iter) < bestE
        bestE = debug_info.ERR(i,iter);
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