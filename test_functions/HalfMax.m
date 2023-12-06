function metric = HalfMax( meta, result, solution )

% time averaging
Javg = movmean( solution.J, 3 );

% identify time with the maximum peak
[Jmax,IDX]    = max( Javg, [], "all" );
[~,peakTIME] = ind2sub( size(solution.J) ,IDX);

metric = sum(meta.GridVolume( (Javg(:,peakTIME)/Jmax) > 0.5 ));

end