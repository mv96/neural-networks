function  z= pca(X)
% [pc,sv,n_sv]  = pca(x)
%
% Input:
%   x - Data stored column-vise .
%% Useful values
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);
% Output:
% pc - Principal components (eigenvectors of the covariance matrix).
%  sv     - Singular values.
%  n_sv - Normalized singular values.

Sigma = 1.0/m .* X' * X;

[U, S, V] = svd(Sigma);  
energy_distribution=sum(S,2);
energy_total=sum(energy_distribution);
energy_threshold=90;
e_sum=(energy_total*energy_threshold)/100 ;
i=0; e=0;
while(e<=e_sum)
i=i+1;
e=e+energy_distribution(i);
k=i;
end
fprintf('k= %f \n',k); 
energy_total=sum(sum(S,1));
disp(S);
U_reduce =U(:,1:k);
z= X*U_reduce;

end