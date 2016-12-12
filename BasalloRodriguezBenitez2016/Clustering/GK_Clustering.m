function [U, C, F, Labs, Dist, iter] = GK_Clustering(X, k, q, tol)

% GK_Clustering
% Gustafson-Kessel clustering algorithm 

% DESCRIPTION
% This function performa partition of a data set X according to the fuzzy
% Gustafson-Kessel clustering algorithm.

% INPUTS
% X: An N by M data set of N data points at M features.
% k: Number of clusters
% q: Fuzziness parameter
% tol: Minimum deviation of the objective value between sucessive
% iterations of the algorithm (stopping criterion)

% OUTPUTS
% C: An k by M matrix of cluster centroids or prototypes
% Labs: An N by 1 vector containing the labels of the datapoints
% Dist: An N by k matrix containing the distances between datapoints and
% cluster centroids.
% U: An N by k matrix containing the membership values of datapoints to
% each cluster.
% iter: Number of iterations required until reach at least "tol" deviation
% in objective function.

[N, l] = size(X);

rand('state',0)
U = rand(N,k);
U = U./repmat(sum(U,2),1,k);% Select the initial partition matrix
F = repmat(eye(l),1,1,k);   % Select the initial covariance matrix

lambda = 0.001;
beta = 1e15;
F0 = eye(l)*det(cov(X)).^(1/l);

C = zeros(k,l);
Dist = zeros(N, k);
for i = 1:k
    % Update the centers matrix
    C(i,:) = (U(:,i).^q)'*X / sum(U(:,i).^q);
    
    % Compute the initial distance matrix
    Xc = X - ones(N,1)*C(i,:);
    [V, D] = eig(F(:,:,i)); D = diag(D);
    Dist(:,i) = ( (Xc*V).^2 )*D;
end

% Compute the initial error
Uant=zeros(size(U));

Errors = [];
iter = 0;
while max(max(U-Uant))>tol && iter <= 300
    Uant = U;
    
    % Update the memberships
    U = ( Dist.^(-1/(q-1)) )./repmat( sum( Dist.^(-1/(q-1)), 2 ), 1, k );
    U(isnan(U)) = 1;
    
    for i = 1:k
        
        % Update the centrs
        C(i,:) = ( U(:,i).^q )'*X / sum(U(:,i).^q);
        
        % Update the covariance matrices
        F(:,:,i) = (ones(l,1)*U(:,i)') .* (X - ones(N,1)*C(i,:))' * (X - ones(N,1)*C(i,:));
        F(:,:,i) = (1-lambda)*(F(:,:,i)/sum(U(:,i))) + lambda*F0;
        [ev, ei] = eig(F(:,:,i)); eimax = max(diag(ei));
        ei(beta*ei < eimax) = eimax/beta;
        F(:,:,i) = ev*diag(diag(ei))*inv(ev);
        
        % Compute the distances
        Xc = X - ones(N,1)*C(i,:);
        ei = diag(ei);
        [~,IX] = sort(ei);
        ei = ei(IX); ev = ev(:,IX); ei = ei(end)./ei;
        Dist(:,i) = ( (Xc*ev).^2 )*ei + 1e-100;
        
    end

    % Compute the performance measure
    J = sum(sum((U.^q).*Dist));
    Errors = cat(1, Errors, J);
    
    iter = iter + 1;
end

% Update the centroids and covariance matrices
for i = 1:k
    C(i,:) = (U(:,i).^q)'*X / sum(U(:,i).^q);
    F(:,:,i) = (ones(l,1)*U(:,i)') .* (X - ones(N,1)*C(i,:))' * (X - ones(N,1)*C(i,:));
    F(:,:,i) = F(:,:,i)/sum(U(:,i));
end

% Obtain the hard partition
[~,Labs] = max(U,[],2);

end