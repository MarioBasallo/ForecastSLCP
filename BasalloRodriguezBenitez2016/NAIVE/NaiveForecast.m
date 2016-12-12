function [xbar, RMSE, Time] = NaiveForecast(Test_set, p)

tic

M = size(Test_set, 1);

% Development of forecasts
xbar = cell(M,1); 
RMSE = zeros(M,1);
for i = 1:M
    
    xf = zeros(1,length(Test_set{i}));
    xf(1) = inf;
    x = Test_set{i};
    for t = 2:length(Test_set{i})
        
        xf(t) = mean(x(t - 1:-1:max(1, t - p)));
        
    end
    xbar{i} = xf;
    RMSE(i) = sqrt( sum( (xf(2:end) - x(2:end)).^2 )/(length(x) - 1) );
    RMSE(i) = RMSE(i)/mean(x(2:end));
    
end

% Plot of results
if M == 1
    plot(xbar{1},'-or'); hold on; plot(Test_set{1}, 'k');
    ylabel('x_t')
    xlabel('t')
    legend('xbar', 'x');
end

Time = toc;

end