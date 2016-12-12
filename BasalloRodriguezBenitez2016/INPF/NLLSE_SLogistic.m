function [VAR, m, a, b, Error, Fit] = NLLSE_SLogistic(x, Lag, graph)

% NLLSE_SLogistic
% Nonlinear least squares estimation for the logistic diffusion model.

% DESCRIPTION
% This function perform a nonlinear least squares estimation (NLLSE) for
% parameters p, q of the simple logistic diffusion model used in INPF
% forecasting method.

% INPUTS
% x:    An 1 by T vector containing the time series for fitting.
% graph:A 0-1 value. If 1, then a plot is generated with the
%       results.
% Lag:  Number of periods ahead to forecast

% OUTPUT
% VAR:  Variance of the prediction "Lag" periods ahead.
% m, a, and b: Shape parameters of the simple logistic diffusion model.
% Error:Fitting error.
% Fit:  Forecast "Lag" periods ahead.

% EXAMPLE
%%% From this example you must load the RD1_dataset.mat file.
% load RD1_dataset
% r = unidrnd(size(Train_set,1),1,1);
% [VAR, m, a, b, Error, Fit] = NLLSE_SLogistic(Train_set(r,:), 1, 1);

l = length(x);
X = cumsum(x);
t = 1:l;


val = [sum(x); 0.5; 0.5];
options = optimset('Largescale','off','Display','off','Algorithm','interior-point');
[param1, Error, ~, ~, ~, ~, H]  = fmincon( @Squared_error, val, [], [], [], [], [1e-6; 1e-6; 1e-6], [sum(x)*100; 100; 100], [], options );
m = param1(1); a = param1(2); b = param1(3);


% Covariance matrix of parameters
Cov = ( 2*Error/(l - length(param1)) )*pinv(H);
% Gradient
Grad = [1/(1 + a*exp( -b*(Lag) )), -m*exp( -b*(Lag) )/( (1 + a*exp( -b*(Lag)) )^2 ), ...
    a*m*(Lag)*exp( -b*(Lag) )/( (1 + a*exp( -b*(Lag)))^2 )];
% Variance estimation
VarM = Cov(1,1)*Grad(1)^2 + Cov(2,2)*Grad(2)^2 + Cov(3,3)*Grad(3)^2 +2* Cov(1,2)*Grad(1)*Grad(2) + ...
    2*Cov(1,3)*Grad(1)*Grad(3) + 2*Cov(2,3)*Grad(2)*Grad(3);
VarE = Error/(l-1);
VAR = VarM + VarE;
% VAR = VarE;


% Plot the results
Fit = m./(1 + a*exp(-b*t)) - m./(1 + a*exp( -b*(t - 1) ));
if graph == 1
    plot(x, 'k'); hold on; plot(Fit, 'g');
end

Fit = m/(1 + a*exp(-b*(Lag))) - m/(1 + a*exp( -b*(Lag - 1) ));


    % Objective function
    function E = Squared_error(param)
        E = sum( ( X - param(1)./(1 + param(2)*exp(- param(3)*t)) ).^2 );
    end

end