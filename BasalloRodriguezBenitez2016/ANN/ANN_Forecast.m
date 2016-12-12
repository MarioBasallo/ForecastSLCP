function [xbar, RMSE, Time] = ANN_Forecast(Train_set, Test_set, p, ANN_Parameters)

% ANN_Forecast
% Artificial Neural Network forecast for Short Life Cycle Products.

% DESCRIPTION
% This function performs forecasts for SLCPs demand using a multilayer
% feed-forward neural network.

% INPUTS
% Train_set:    An N by T matrix containing a set of N time series of
%               length T, i.e. the training set.
% Test_set:     An M by T matrix containing a set if M time series of 
%               length T, the test set.
% p:            Number of delays considered in regression.
% ANN_Parameters: A vector containing the number of neurons in each hidden
%               layer, up to two hidden layers are considered.

% OUTPUTS
% xbar:         An M by T matrix containing the forecasts of the time
%               series in Test_set. The forecast of first period is an
%               average of the time series in Train_set.
% RMSE:         An M by 1 vector containing the results for root mean
%               square error of the forecasts. The RMSE consider only the
%               forecasts generated from period 2.
% Time:         Processing time of the method.


tic

N =  size(Train_set, 1);
M = size(Test_set, 1);

T1 = 0;
for i = 1:N
    T1 = max(length(Train_set{i}), T1);
end
T2 = 0;
for i = 1:M
    T2 = max(length(Test_set{i}), T2);
end
T = min(T1, T2);

% Training machines
ANNs = cell(1, T);
for t = 2:T
    
    inputs = [];
    output = [];
    for i = 1:N
        x = Train_set{i};
        if length(Train_set{i}) >= t
            inputs = cat(1, inputs, x(t - 1:-1:max(1, t - p)));
            output = cat(1, output, x(:, t));
        elseif length(x) >= p + 1
            inputs = cat(1, inputs, x(end - 1:-1:max(1, end - p)));
            output = cat(1, output, x(:, end));
        end
    end
    
    % Outlier detection
    XX = [inputs, output];
    outlier_idx = sqrt( sum((XX - repmat(median(XX), size(XX,1),1)).^2,2) ) > 3*max(std(XX));
    inputs(outlier_idx,:) = [];
    output(outlier_idx,:) = [];
    
    Neurons = ANN_Parameters(find(ANN_Parameters));
    net = feedforwardnet(Neurons, 'trainbr'); % Training with Lebenverg-Marquardt with Bayesian Regularization
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    nntraintool('close');
    net.trainParam.epochs = 300;    % Number of runs
    net.trainParam.lr = 0.01;       % Learning rate
    net = train(net, inputs', output');
    
    ANNs{t} = net;
    
end

% Generation of forecasts
xbar = cell(M,1); 
RMSE = zeros(M,1);
for i = 1:M
    
    xf = zeros(1,length(Test_set{i}));
    xf(1) = inf;
    x = Test_set{i};
    for t = 2:min(length(Test_set{i}), T)
        
        Ri = x(t - 1:-1:max(1, t - p));
        xf(t) = sim(ANNs{t}, Ri');
        xf(t) = max(0,xf(t));
        
    end
    
    xbar{i} = xf;
    RMSE(i) = sqrt( sum( (xf(2:end) - x(2:end)).^2 )/(length(x) - 1) );
    RMSE(i) = RMSE(i)/mean(x(2:end));
    
end

% Plot of results
if M == 1
    plot(xbar{1},'r'); hold on; plot(Test_set{1}, 'k');
    ylabel('x_t')
    xlabel('t')
    legend('xbar', 'x');
end

Time = toc;

end