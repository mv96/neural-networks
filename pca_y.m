function pca_y=pca_y(X,c)
N = size(X, 1);
m = mean(X); % each row is a X sample
X_m = X - repmat(m, N, 1);
covar = X_m'*X_m/N; % or N-1 for unbiased estimate
[U,S,V] = svd(covar);
reduced_X = X_m*V(:,1:2); % reduce to 2 components