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