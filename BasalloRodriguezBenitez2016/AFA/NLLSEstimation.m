function [p, q, st, St, Error, Fit] = NLLSEstimation(x, T, graph)

% NLLSEstimation
% Nonlinear least squares estimation for parameters of Bass diffusion
% model.

% DESCRIPTION
% This function perform a nonlinear least squares estimation (NLLSE) for
% parameters p and q of the Bass diffusion model used in AFA forecasting
% method.

% INPUTS
% x:    A 1 by T vector containing the time series for fitting.
% graph:A 0-1 value. If 1, then a plot is generated with the
%       results.

% OUTPUTS
% p and q: Shape parameters of the Bass diffusion model.
% st:   Expected fraction of demand at the different periods.
% St:   Cumilative expected fraction of demand.
% Error:Fitting error.
% Fit:  An 1 by T vector containing the fitting results.

% EXAMPLE
%%% From this example you require to load the RD1_dataset.mat file.
% load RD1_dataset
% r = unidrnd(size(Train_set,1),1,1);
% [p, q, st, St, Error, Fit] = NLLSEstimation(Train_set(r,:), 1)

X = cumsum(x);
m = sum(x);
t = 1:length(x);
T = 1:T;


val = [rand(); rand()];
options = optimset('Largescale','off','Display','off','Algorithm','interior-point');
[param1, Error]  = fmincon( @Squared_error, val, [], [], [], [], [0; 0], [1; 1], [], options );
p = param1(1); q = param1(2);

% Get the results
Fit = m*( ( 1 - exp( -(p + q)*T ) )./( 1 + (q/p)*exp( -(p + q)*T ) ) - ...
    ( 1 - exp( -(p + q)*(T - 1) ) )./( 1 + (q/p)*exp( -(p + q)*(T - 1) ) ) );

st = ( 1 - exp( -(p + q)*T ) )./( 1 + (q/p)*exp( -(p + q)*T ) ) - ...
    ( 1 - exp( -(p + q)*(T - 1) ) )./( 1 + (q/p)*exp( -(p + q)*(T - 1) ) );

St = ( 1 - exp( -(p + q)*T ) )./( 1 + (q/p)*exp( -(p + q)*T ) );

    % Objective function
    function E = Squared_error(param)
        E = sum( ( X - m*( 1 - exp( -(param(1) + param(2))*t ) )./...
            ( 1 + (param(2)/param(1))*exp( -(param(1) + param(2))*t ) ) ).^2 );
    end

if graph == 1
    plot(x, 'k'); hold on; plot(Fit, 'g');
end

end