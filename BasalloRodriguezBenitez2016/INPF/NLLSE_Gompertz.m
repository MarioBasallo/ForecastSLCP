function [VAR, m, a, b, Error, Fit] = NLLSE_Gompertz(x, Lag, graph)

% NLLSE_Gompertz
% Nonlinear least squares estimation for the Gompertz diffusion model.

% DESCRIPTION
% This function perform a nonlinear least squares estimation (NLLSE) for
% parameters m, a, and b of the Gompertz diffusion model used in INPF 
% forecasting method.

% INPUTS
% x:    A 1 by T vector containing the time series for fitting.
% graph:A 0-1 value. If 1, then a plot is generated with the
%       results.
% Lag:  Number of period ahead to forecast

% OUTPUT
% VAR:  Variance of the prediction "Lag" periods ahead.
% a, and b: Shape parameters of the Bass diffusion model.
% Error:Fitting error.
% Fit:  Forecast "Lag" periods ahead.

% EXAMPLE
%%% From this example you must load RD1_dataset.mat file.
% load RD1_dataset
% r = unidrnd(size(Train_set,1),1,1);
% [VAR, m, a, b, Error, Fit] = NLLSE_Gompertz(Train_set(r,:), 1, 1);

l = length(x);
X = cumsum(x);
t = 1:l;

val = [sum(x); 0.5; 0.5];
options = optimset('Largescale','off','Display','off','Algorithm','interior-point');
[param1, Error, ~, ~, ~, ~, H]  = fmincon( @Squared_error, val, [], [], [], [], [1e-6; 1e-6; 1e-6], [100*sum(x); 100; 100], [], options );
m = param1(1); a = param1(2); b = param1(3);

% Covariance matrix of parameters
Cov = ( 2*Error/(l - length(param1)) )*pinv(H);
% Gradient
Grad = [exp(-a*exp(-b*(Lag))), -m*exp(-a*exp(-b*(Lag)) - b*(Lag)), ...
    a*m*(Lag)*exp(-a*exp(-b*(Lag)) - b*(Lag))];
% Variance estimation
VarM = Cov(1,1)*Grad(1)^2 + Cov(2,2)*Grad(2)^2 + Cov(3,3)*Grad(3)^2 +2* Cov(1,2)*Grad(1)*Grad(2) + ...
    2*Cov(1,3)*Grad(1)*Grad(3) + 2*Cov(2,3)*Grad(2)*Grad(3);
VarE = Error/(l - 1);
VAR = VarM + VarE;

% Get the results
Fit = m*exp(-a*exp(-b*t)) - m*exp(-a*exp(-b*(t - 1)));
if graph == 1
    plot(x, 'k'); hold on; plot(Fit, 'g');
end

Fit = m*exp(-a*exp(-b*(Lag))) - m*exp(-a*exp(-b*(Lag - 1)));

    % Objective function
    function E = Squared_error(param)
        E = sum( ( X - param(1)*exp( -param(2)*exp(-param(3)*t) ) ).^2 );
    end

end