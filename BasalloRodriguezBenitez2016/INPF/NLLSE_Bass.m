function [VAR, p, q, st, St, Error, Fit] = NLLSE_Bass(x, Lag, graph)

% NLLSE_Bass
% Nonlinear least squares estimation for the Bass diffusion model.

% DESCRIPTION
% This function perform a nonlinear least squares estimation (NLLSE) for
% parameters p, q of the Bass diffusion model used in INPF forecasting 
% method.

% INPUTS
% x:    A 1 by T vector containing the time series for fitting.
% graph:A 0-1 value. If 1, then a plot is generated with the
%       results.
% Lag:  Number of periods ahead to forecast

% OUTPUT
% VAR:  Variance of the prediction "Lag" periods ahead.
% p and q: Shape parameters of the Bass diffusion model.
% st:   Expected fraction of demand at the different periods.
% St:   Cumilative expected fraction of demand.
% Error:Fitting error.
% Fit:  Forecast "Lag" periods ahead.

% EXAMPLE
%%% From this example you must load RD1_dataset.mat file.
% load RD1_dataset
% r = unidrnd(size(Train_set,1),1,1);
% [VAR, p, q, st, St, Error, Fit] = NLLSE_Bass(Train_set(r,:), 1, 1);

l = length(x);
X = cumsum(x);
t = 1:l;


val = [0.5, 0.5, sum(x)];
options = optimset('Largescale','off','Display','off','Algorithm','interior-point');
[param1, Error, ~, ~, ~, ~, H]  = fmincon( @Squared_error, val, [], [], [], [], [1e-6; 1e-6; 1e-6], [1; 1; sum(x)*100], [], options );
m = param1(3); p = param1(1); q = param1(2);

% Covariance matrix of parameters
Cov = ( 2*Error/(l - length(param1)) )*pinv(H);
% Gradient
Grad = [m*q*( 1 + exp((p + q)*(l + q))*(-1 + (p + q)*Lag) )/( (p + q*exp((p + q)*Lag))^2 ), ...
    (-m*p + exp((p + q)*Lag)*m*(p + q*(p + q)*Lag))/( (p + exp((p + q)*Lag)*q)^2 ), ...
    ((-1 + exp((p + q)*Lag))*q)/(p + exp((p + q)*Lag)*q)];
% Variance estimation
VarM = Cov(1,1)*Grad(1)^2 + Cov(2,2)*Grad(2)^2 + Cov(3,3)*Grad(3)^2 +2* Cov(1,2)*Grad(1)*Grad(2) + ...
    2*Cov(1,3)*Grad(1)*Grad(3) + 2*Cov(2,3)*Grad(2)*Grad(3);
VarE = Error/(l - 1);
VAR = VarM + VarE;

% Get the results
Fit = m*( ( 1 - exp( -(p + q)*t ) )./( 1 + (q/p)*exp( -(p + q)*t ) ) - ...
    ( 1 - exp( -(p + q)*(t - 1) ) )./( 1 + (q/p)*exp( -(p + q)*(t - 1) ) ) );

st = ( 1 - exp( -(p + q)*t ) )./( 1 + (q/p)*exp( -(p + q)*t ) ) - ...
    ( 1 - exp( -(p + q)*(t - 1) ) )./( 1 + (q/p)*exp( -(p + q)*(t - 1) ) );

St = ( 1 - exp( -(p + q)*t ) )./( 1 + (q/p)*exp( -(p + q)*t ) );

if graph == 1
    plot(x, 'k'); hold on; plot(Fit, 'g');
end

Fit = m*( ( 1 - exp( -(p + q)*Lag ) )./( 1 + (q/p)*exp( -(p + q)*Lag ) ) - ...
    ( 1 - exp( -(p + q)*(Lag - 1) ) )./( 1 + (q/p)*exp( -(p + q)*(Lag - 1) ) ) );

    % Objective function
    function E = Squared_error(param)
        E = sum( ( X - param(3)*( 1 - exp( -(param(1) + param(2))*t ) )./...
            ( 1 + (param(2)/param(1))*exp( -(param(1) + param(2))*t ) ) ).^2 );
    end

end