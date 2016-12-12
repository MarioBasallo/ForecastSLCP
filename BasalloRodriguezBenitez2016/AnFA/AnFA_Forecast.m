function [xbar, RMSE, Time] = AnFA_Forecast(Train_set, Test_set)

% AnFA_Forecast
% Analogous forecasting algorithm for Short Life Cycle Products (SLCP).

% DESCRIPTION
% This function performs forecasts for SLCPs demand using the analogous
% forecasting algorithm (AnFA).

% INPUTS
% Train_set:    An N by T matrix containing a set of N time series of
%               length T, i.e. the training set.
% Test_set:     An M by T matrix containing a set if M time series of 
%               length T, i.e. the test set.

% OUTPUTS
% xbar:         An M by T matrix containing the forecasts of the time
%               series in Test_set. The forecast of first period is an
%               average of the time series in Train_set.
% RMSE:         An M by 1 vector containing the results for the root mean
%               square error of the forecasts.
% Time:         Processing time of the method.

tic

[N, ~] = size(Train_set);
[M, ~] = size(Test_set);

lengths = zeros(N,1);
for i = 1:N
    lengths(i) = length(Train_set{i});
end

xbar = cell(M,1); 
RMSE = zeros(M,1);
for i = 1:M     % For each real time time series
    
    xf = zeros(1,length(Test_set{i}));
    xf(1) = inf;
    y = Test_set{i};
    
    for t = 2:length(Test_set{i})     % For each period of the real time time series
        
        if t == 2
            Dist = zeros(N,1);
            x = zeros(N,1);
            for j = 1:N
                xx = Train_set{j};
                x(j,:) = xx(t);
                Dist(j, 1) = sqrt( (xx(1) - y(1)).^2 );
            end
            xf(t) = mean(x(Dist <= min(Dist)));
        else
            
            Nl = length(find(lengths >= t));
            Trainl = Train_set(lengths >= t);
            
            % Compute the distances and select the 20 most similar time
            % series
            Dist = zeros(Nl,1);
            for j = 1:Nl
                x = Trainl{j};
                [~, Dist(j, 1)] = Analogous_dist( y(1:t - 1), x(1:t - 1) );
            end
            [~, IX] = sort(Dist, 'ascend'); IX = IX(1:min(20, Nl));
            Trainl = Trainl(IX); Nl = min(20,Nl);
            
            %  Calibrating of time series
            V = cell(Nl,1);
            for j = 1:Nl
                x = Trainl{j};
                V{j} = Calibrating( y(1:t - 1), x, 0 );
            end
            
            % Adjust the length of the time series
            Dist = zeros(Nl,1);
            f = zeros(Nl,1);
            E = cell(Nl,1);
            for j = 1:size(V, 1)
                [E{j}, Dist(j,:)] = Adjust_length(y(1:t - 1), V{j}, 0);
                e = E{j};
                f(j) = e(t);
            end
            
            % Perform forecasts
            xf(t) = mean( f(Dist <= min(Dist)) );
            
        end
        
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