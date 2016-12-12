function [xbar, RMSE, Time] = MLR_Forecast(Train_set, Test_set, p)

% MLR_Forecast
% Multiple linear regression forecast for Short Life Cycel Products (SLCP)
% demand.

% DESCRIPTION
% This function perform forecasts for SLCPs demand using multiple linear
% regression (MLR) method.

% INPUTS
% Train_set:    An N by T matrix containing a set of N time series of
%               length T, the training set, used to obtain multiple linear 
%               regression coefficients.
% Test_set:     An M by T matrix containing a set if M time series of 
%               length T required to forecast.
% p:            Number of delays considered in regression.

% OUTPUTS
% xbar:         An M by T matrix containing the forecasts of the time
%               series in Test_set. The forecast of first period is an
%               average of the time series in Train_set.
% RMSE:         An M by 1 vector containing the results for root mean
%               square error of the forecasts. The RMSE consider only the
%               forecasts generated from period 2.
% Time:         Processing time


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

tic

% Estimation of regression coefficients
Beta = cell(T,1);
for t = 2:T
    
    inputs = [];
    output = [];
    for i = 1:N
        x = Train_set{i};
        if length(Train_set{i}) >= t
            inputs = cat(1, inputs, [1, x(t - 1:-1:max(1, t - p))]);
            output = cat(1, output, x(:, t));
        elseif length(x) >= p + 1
            inputs = cat(1, inputs, [1, x(end - 1:-1:max(1, end - p))]);
            output = cat(1, output, x(:, end));
        end
    end
    
    % Outlier detection
    XX = [inputs(:,2:end), output];
    outlier_idx = sqrt( sum((XX - repmat(median(XX), size(XX,1),1)).^2,2) ) > 3*max(std(XX));
    inputs(outlier_idx,:) = [];
    output(outlier_idx,:) = [];
    
    Beta{t} = regress(output, inputs);
    
end
% plot(Multic); figure;

% Development of forecasts
xbar = cell(M,1); 
RMSE = zeros(M,1);
for i = 1:M
    
    xf = zeros(1,length(Test_set{i}));
    xf(1) = inf;
    x = Test_set{i};
    for t = 2:length(Test_set{i})
        
        if t <= T1
            Ri = [1, x(t - 1:-1:max(1, t - p))];
            xf(t) = Ri*Beta{t};
            xf(t) = max(0,xf(t));
        elseif p < T1
            Ri = [1, x(t - 1:-1:max(1, t - p))];
            xf(t) = Ri*Beta{end};
            xf(t) = max(0,xf(t));
        end
        
    end
    xbar{i} = xf;
    RMSE(i) = sqrt( sum( (xf(2:end) - x(2:end)).^2 )/(length(x) - 1) );
    RMSE(i) = RMSE(i)/mean(x(2:end));
    
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