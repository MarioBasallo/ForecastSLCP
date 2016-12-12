function [xbar, RMSE, Time] = SVR_Forecast(Train_set, Test_set, p, SVR_Parameters)

% IMPORTANT
% This function requires the LIBSVM toolbox to be instaled in your
% computer. See Chang, C.-C., & Lin, C.-J. (2011). LIBSVM: a library for
% support vector machines. ACM Transactions on Inteligent Systems and
% Technology, (pp. 1-27).

% SVR_Forecast
% Epsilon Support Vector regression forecast for Short Life Cycel Products 
% (SLCP).

% DESCRIPTION
% This function perform forecasts for SLCPs demand using the Epsilon Support 
% Vector Regression method.

% INPUTS
% Train_set:    An N by T matrix containing a set of N time series of
%               length T, the training set.
% Test_set:     An M by T matrix containing a set if M time series of 
%               length T, the test set.
% p:            Number of delays considered in regression.
% SVR_Parameters: A 1 by 3 vector containing the base-2 logarithm of the
%               epsilon-SVR parameters, in the following order: epsilon 
%               (width of the tube), constant penalty C, and gamma 
%               (parameter of Gaussian kernel).

% OUTPUTS
% xbar:         An M by T matrix containing the forecasts of the time
%               series in Test_set. The forecast of first period is an
%               average of the time series in Train_set.
% RMSE:         An M by 1 vector containing the results for root mean
%               square error of the forecasts. The RMSE consider only the
%               forecasts generated after period 2.
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

% Defining parameters of the SVR
epsilon = 2^SVR_Parameters(1);   % epsilon-SVR parameters, Gaussian Kernel
penalty = 2^SVR_Parameters(2);
gamma = 2^SVR_Parameters(3);

% Training machines
SVMs = cell(1, T);
param = [ '-q -s 3 -t 2 -c ', num2str(penalty), ' -p ', num2str(epsilon), ' -g ', num2str(gamma) ];
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
    
    SVMs{t} = svmtrain( output, inputs, param );
    
end

% Development of forecasts
xbar = cell(M,1); 
RMSE = zeros(M,1);
for i = 1:M
    
    xf = zeros(1,length(Test_set{i}));
    xf(1) = inf;
    x = Test_set{i};
    for t = 2:min(T, length(Test_set{i}) )
        
        Ri = x(t - 1:-1:max(1, t - p));
        xf(t) = svmpredict( rand( size(Ri,1), 1 ), Ri, SVMs{t}, '-q' );
        xf(t) = max(0,xf(t));
        
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