function [xbar, RMSE, Time] = INPF_Forecast(Train_set, Test_set, L, Mleads)

% INPF_Forecast
% Integrated new product forecasting algorithm for Short Life Cycle
% Products.

% DESCRIPTION
% This function perform forecasts for SLCPs demand using the integrated new
% product forecasting algorithm (INPF).

% INPUTS
% Train_set:    An N by T matrix containing a set of N time series of
%               length T, the training set.
% Test_set:     An M by T matrix containing a set of M time series of 
%               length T required to forecast, the test set.
% L:            Number of Lags considered for leading indicators based
%               regression.
% M:            Number of leading indicators.

% OUTPUTS
% xbar:         An M by T-5 matrix containing the forecasts of the time
%               series in the Test_set.
% RMSE:         An M by 1 vector containing the results for root mean
%               square error of the forecasts. The RMSE consider only the
%               forecasts generated from period 5.
% Time:         Processing time.


tic

[N, ~] = size(Train_set);
lengths = zeros(N,1);
for i = 1:N
    lengths(i) = length(Train_set{i});
end

[n, ~] = size(Test_set);
xbar = cell(n,1); 
RMSE = zeros(n,1);

tt = 5; % Starting time for forecast

for i = 1:n         % Pronosticar para cada serie de tiempo del conjunto de prueba
    
    y = Test_set{i};
    xf = zeros(1,length(y));
    l = length(y);
    xf(1:tt) = inf;
    for t = tt:l    % Se pronostica a partir del periodo 4
        
        % Obtain the prior information
        % Logistic model
        [VarLPr, ~, ~, ~, ~, FitLPr] = NLLSE_SLogistic(y(1:t - 1), t, 0);
        % Gompertz model
        [VarGPr, ~, ~, ~, ~, FitGPr] = NLLSE_Gompertz(y(1:t - 1), t, 0);
        % Bass model
        [VarBPr, ~, ~, ~, ~, ~, FitBPr] = NLLSE_Bass(y(1:t - 1), t, 0);
        
        
        % Obtain the sampling information
        Trainl = Train_set(lengths>=t); Nl = size(Trainl,1);
        xx = zeros(Nl,t-1);
        M = min(Nl,Mleads);
        for r = 1:Nl
            x = Trainl{r};
            xx(r,:) = x(1:t-1);
        end
        clear Trainl;
        ypred = Leading_Search( xx(:, 1:t - 1), y(1:t - 1), L, M );
        YSmpl = [ repmat(y(1:t - 1), M, 1), ypred ];
        clear ypred;
        
        
        % Forecast demand for each leading indicator and difussion model
        Fits = zeros(M, 3);
        for j = 1:M
            % Logistic model
            [~, ~, ~, ~, ~, FitLSa] = NLLSE_SLogistic(YSmpl(j,:), t, 0);
            % Gompertz model
            [~, ~, ~, ~, ~, FitGSa] = NLLSE_Gompertz(YSmpl(j,:), t, 0);
            % Bass model
            [~, ~, ~, ~, ~, ~, FitBSa] = NLLSE_Bass(YSmpl(j,:), t, 0);
            
            Fits(j, :) = [FitLSa FitGSa FitBSa];
        end
        clear YSmpl;
        if M == 1
            Fits = repmat(Fits, 5,1) + 1e-10*rand(5,3);
        end
        VarSa = var(Fits);
        VarSa(VarSa == 0) = 1e-10;
        
        
        % Obtain the posterior life cycle projection
        SUMS = sum(Fits); clear Fits;
        MuLPo = (1/VarLPr)*FitLPr/( (1/VarLPr) + (M/VarSa(1))) + ...
            ( (M/VarSa(1))/((1/VarLPr) + (M/VarSa(1))) )*SUMS(1)/M;
        
        MuGPo = (1/VarGPr)*FitGPr/( (1/VarGPr) + (M/VarSa(2))) + ...
            ( (M/VarSa(2))/((1/VarGPr) + (M/VarSa(2))) )*SUMS(2)/M;
        
        MuBPo = (1/VarBPr)*FitBPr/( (1/VarBPr) + (M/VarSa(3))) + ...
            ( (M/VarSa(3))/((1/VarBPr) + (M/VarSa(3))) )*SUMS(3)/M;
        
        
        
        % Combine the ferecasts
        Pk = [(1/VarLPr) / (1/VarLPr + 1/VarGPr + 1/VarBPr), ...
              (1/VarGPr) / (1/VarLPr + 1/VarGPr + 1/VarBPr), ...
              (1/VarBPr) / (1/VarLPr + 1/VarGPr + 1/VarBPr)];
          
        xf(t) = MuLPo*Pk(1) + MuGPo*Pk(2) + MuBPo*Pk(3);
        
    end
    
    xbar{i} = xf;
    RMSE(i) = sqrt( sum( (xf(tt:end) - y(tt:end)).^2 )/(l - tt + 1) );
    RMSE(i) = RMSE(i)/mean(y(tt:end));
    
end

if size(Test_set, 1) == 1
    plot(Test_set{1}, 'k'); hold on; plot(xbar{1}, 'g');
    legend({'real', 'Forecast'});
    ylabel('x_t')
    xlabel('t')
end

Time = toc;

end