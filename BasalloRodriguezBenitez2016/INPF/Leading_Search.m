function [ypred, L_ind] = Leading_Search(X, y, L, M)

% Leading_Indicator_Machine
% Selection of a set of leading indicators from a time-series data set, and
% developing of forecasts based on leading indicators.

% X:    An N by T matrix containing N time series of Length T, i.e. 
%       the product group dataset.
% y:    A 1 by T vector containing the current time series, available to 
%       forecast.
% L:    Number of lags
% M:    Number of leading indicators to be selected.

% OUTPUTS
% ypred:    A 1 by L vector containig L predictions ahead of time series "y".
% L_ind:    An M by T matrix containing the set of M leading indicators.

[N, T] = size(X);
L = min(T - 3, L);      % Deben reservarse al menos 3 datos para obtener resultados confiables de regresión

M = min(N,M);

% Obtain the correlation coefficient for each product in X
Rho = zeros(N, 1);
for i = 1:N
    Rho(i, :) = Corr_Coef( X(i, :), y, L );
end

% Select the leading indicators
[~, IX] = sort(Rho, 'descend');
X = X(IX, :);
L_ind = X(1:M, :);

% Obtain the regression coefficients
Coefs = zeros(M, 2);
for i = 1:M
    b = regress(y(L + 1:end)', [ones(length(L_ind(i, 1:T - L)'), 1) L_ind(i, 1:T - L)']);
    Coefs(i, :) = b';
end

% Generate time series data
ypred = zeros(M, L);
for i = 1:M
    ypred(i, :) = Coefs(i, 1) + Coefs(i, 2)*L_ind(i, T - L + 1:T);
end

ypred(ypred<0) = 0;

end