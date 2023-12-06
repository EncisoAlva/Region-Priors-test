function metric = LocalizationError( meta, result, solution )

% time averaging
Javg = movmean( solution.J, 3 );

% identify time with the maximum peak
[~,IDX]    = max( Javg, [], "all" );
[peakIDX, ~] = ind2sub( size(solution.J) ,IDX);

metric = norm( meta.Gridloc(peakIDX,:) - result.meta.TrueLocs );

end