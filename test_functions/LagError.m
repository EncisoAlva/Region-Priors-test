function metric = LagError( meta, result, solution )

% time averaging
Javg = movmean( solution.J, 3 );

% identify time with the maximum peak
[~,IDX]    = max( Javg, [], "all" );
[~, peakT] = ind2sub( size(solution.J) ,IDX);

%eval.LocErr = norm( meta.Gridloc(peakIDX,:) - result.meta.TrueLocs );
metric = abs(peakT - 8);

end