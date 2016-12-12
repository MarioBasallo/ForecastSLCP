function [xbar, RMSE, Time] = AFA_Forecast(Train_set, Test_set, TauSigmaRatio)

% AFA_Forecast
% Adaptive Forecasting Algorithm forecast for Short Life Cycel Products 
% (SLCPs).

% DESCRIPTION
% This function perform forecasts for SLCPs demand using the adaptive 
% forecasting algorithm (AFA) method.

% INPUTS
% Train_set:    An N by T matrix containing a set of N time series of
%               length T, the training set.
% Test_set:     An M by T matrix containing a set if M time series of 
%               length T required to forecast, the test set.
% TauSigmaRatio: AFA Parameter that relates the uncertainty in market
%               potential and the variability of random fluctuations.

% OUTPUTS
% xbar:         An M by T matrix containing the forecasts of the time
%               series in Test_set.
% RMSE:         An M by 1 vector containing the results for root mean
%               square error of the forecasts. For the computation of RMSE
%               we consider that the forecasts start at period 2.
% Time:         Processing time

tic

[N, ~] = size(Train_set);
[M, ~] = size(Test_set);

T1 = 0;
for i = 1:N
    T1 = max(length(Train_set{i}), T1);
end
T2 = 0;
for i = 1:M
    T2 = max(length(Test_set{i}), T2);
end
T = max(T1, T2);

% Prior probability distribution
Pr = (1/N)*ones(N, 1);

% Select L appropriate pairs of shape parameters (p, q)
st = zeros(N, T); St = zeros(N, T);
m = zeros(N,1);
for i = 1:N
    [~, ~, st(i, :), St(i, :)] = NLLSEstimation(Train_set{i}, T, 0);
    m(i) = sum(Train_set{i});
end

% Determine the prior mean
Mu0pq = mean( m );

% Determine the prior variances
SSq = var( m );
Sigma2 = SSq/( 1 + TauSigmaRatio^2 );     % Prior variance of demand, fluctuations
Tao2 = Sigma2*( TauSigmaRatio^2 );        % Prior variance of total demand

% Obtain forecast at period 1
xbar = cell(M,1);    RMSE = zeros(M, 1);
for i = 1:M
    
    xf = zeros(1,length(Test_set{i}));
    xf(1) = inf;
    
    for t = 2:length(Test_set{i})
        
        % Get the posterior distribution of m, update parameters
        vars = St(:, t - 1).*Tao2 + Sigma2;
        y = Test_set{i};
        Mutpq = Mu0pq*Sigma2./vars + repmat( sum( y(1:t - 1), 2 ), N, 1 )*Tao2./vars;
        
        % Get the posterior joint distribution of p and q
        out_factor = 1./( sqrt( ( ( Sigma2^(t - 2) )*vars ).*prod(st(:, 1:t - 1), 2) ) );
        
        exp1 = -0.5*( (Mu0pq^2)/Tao2 + (1/Sigma2)*sum( repmat(y(1:t - 1).^2, N, 1)./st(:, 1:t - 1), 2 ) );
        exp2 = 0.5*( Mutpq.*( sum(y(1:t - 1), 2)/Sigma2 + Mu0pq/Tao2 ) );
        
        wtpq = out_factor.*exp( exp1 + exp2 );
        if any(wtpq) == 1 && (wtpq'*Pr) ~= 0
            Prt = wtpq.*Pr/(wtpq'*Pr);
        else
            Prt = (1/N)*ones(N, 1);
        end
        
        % Get the forecast
        xf(t) = sum( st(:, t).*Mutpq.*Prt );
    end
    
    xbar{i} = xf;
    RMSE(i) = sqrt( sum( (xf(2:end) - y(2:end)).^2 )/(length(y) - 1) );
    RMSE(i) = RMSE(i)/mean(y(2:end));
end

% Plot of results
if M == 1
    plot(xbar{1},'-+r'); hold on; plot(Test_set{1}, 'k');
    ylabel('x_t')
    xlabel('t')
    legend('xbar', 'x');
end

Time = toc;

end