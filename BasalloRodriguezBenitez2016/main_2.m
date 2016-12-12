clear all
close all
clc
path(path,'C:\BasalloRodriguezBenitez2016\MLR');
path(path,'C:\BasalloRodriguezBenitez2016\WMLR');
path(path,'C:\BasalloRodriguezBenitez2016\SVR');
path(path,'C:\BasalloRodriguezBenitez2016\AFA');
path(path,'C:\BasalloRodriguezBenitez2016\AnFA');
path(path,'C:\BasalloRodriguezBenitez2016');

%--------------------------------------------------------------------------
%
% Before running this script we sugest review the README.txt file provided.
%
% This script performs a sensitivity analysis on the RMSE by varing the 
% training-set size. The results are evaluated ONLY for methods MLR, WMLR, 
% SVR, AFA, and according to RD1, RD2, RD3, and SD data sets. A plot shows
% the variation on RMSE at different training set sizes.
%
% "technique" is a cell, which comprises string values related to the
% forecasting algorithms to be evaluated. 
%
% EXAMPLE
% If you want to evaluate the performance of MLR and AFA, you must set:
% technique = {'MLR','AFA'};
%
% WARNINGS
% ========
% 1) To evaluate the SVR, you need to have installed the LIBSVM toolbox of
%    Matlab (See Chang, C.-C., & Lin, C.-J. (2011). LIBSVM: a library for
%    support vector machines. ACM Transactions on Inteligent Systems and
%    Technology).
%--------------------------------------------------------------------------

% technique = {'MLR', 'WMLR', 'SVR', 'AFA'};
technique = {'MLR'};

load RD1_dataset
for S = 1:7
    for r = 1:100
        
        X = unidrnd(1000, size(Train_set,1), 1);
        [~, IX] = sort(X);
        X = Train_set(IX(1:5*2^(S-1)));
        
        for i = 1:length(technique)
            switch technique{i}
                case 'MLR'
                    [~, EMLR(:,r)] = MLR_Forecast(X, Test_set, MLR_lags);
                case 'WMLR'
                    if S >= 2
                        [~, EWMLR(:,r)] = WMLR_Forecast(X, Test_set, WMLR_lags, Part);
                    end
                case 'SVR'
                    ESVR_Par = SVR_Parameters(2,:);
                    [~, ESVR(:,r)] = SVR_Forecast(X, Test_set, SVR_lags, SVR_Parameters(SVR_lags,:));
                case 'AFA'
                    [~, EAFA(:,r)] = AFA_Forecast(X, Test_set, AFA_TauSigma);
            end
        end
        
    end
    
    for i = 1:length(technique)
        switch technique{i}
            case 'MLR'
                RD1_MLR_RMSE(:, S) = mean(EMLR,2);
            case 'WMLR'
                if S >= 2
                    RD1_WMLR_RMSE(:, S) = mean(EWMLR,2);
                end
            case 'SVR'
                RD1_SVR_RMSE(:, S) = mean(ESVR,2);
            case 'AFA'
                RD1_AFA_RMSE(:, S) = mean(EAFA,2);
        end
    end
    
end
EMLR = []; ESVR = []; EAFA = []; EWMLR = [];


load SD1_dataset
for S = 1:7
    for r = 1:100
        
        X = unidrnd(1000, size(Train_set,1), 1);
        [~, IX] = sort(X);
        X = Train_set(IX(1:5*2^(S-1)));
        
        for i = 1:length(technique)
            switch technique{i}
                case 'MLR'
                    [~, EMLR(:,r)] = MLR_Forecast(X, Test_set, MLR_lags);
                case 'WMLR'
                    if S >= 2
                        [~, EWMLR(:,r)] = WMLR_Forecast(X, Test_set, WMLR_lags, Part);
                    end
                case 'SVR'
                    ESVR_Par = SVR_Parameters(2,:);
                    [~, ESVR(:,r)] = SVR_Forecast(X, Test_set, SVR_lags, SVR_Parameters(SVR_lags,:));
                case 'AFA'
                    [~, EAFA(:,r)] = AFA_Forecast(X, Test_set, AFA_TauSigma);
            end
        end
        
    end
    
    for i = 1:length(technique)
        switch technique{i}
            case 'MLR'
                SD1_MLR_RMSE(:, S) = mean(EMLR,2);
            case 'WMLR'
                if S >= 2
                    SD1_WMLR_RMSE(:, S) = mean(EWMLR,2);
                end
            case 'SVR'
                SD1_SVR_RMSE(:, S) = mean(ESVR,2);
            case 'AFA'
                SD1_AFA_RMSE(:, S) = mean(EAFA,2);
        end
    end
    
end
EMLR = []; ESVR = []; EAFA = []; EWMLR = [];


load RD2_dataset
for S = 1:7
    for r = 1:100
        
        X = unidrnd(1000, size(Train_set,1), 1);
        [~, IX] = sort(X);
        X = Train_set(IX(1:5*2^(S-1)));
        
        for i = 1:length(technique)
            switch technique{i}
                case 'MLR'
                    [~, EMLR(:,r)] = MLR_Forecast(X, Test_set, MLR_lags);
                case 'WMLR'
                    if S >= 2
                        [~, EWMLR(:,r)] = WMLR_Forecast(X, Test_set, WMLR_lags, Part);
                    end
                case 'SVR'
                    ESVR_Par = SVR_Parameters(2,:);
                    [~, ESVR(:,r)] = SVR_Forecast(X, Test_set, SVR_lags, SVR_Parameters(SVR_lags,:));
                case 'AFA'
                    [~, EAFA(:,r)] = AFA_Forecast(X, Test_set, AFA_TauSigma);
            end
        end
        
    end
    
    for i = 1:length(technique)
        switch technique{i}
            case 'MLR'
                RD2_MLR_RMSE(:, S) = mean(EMLR,2);
            case 'WMLR'
                if S >= 2
                    RD2_WMLR_RMSE(:, S) = mean(EWMLR,2);
                end
            case 'SVR'
                RD2_SVR_RMSE(:, S) = mean(ESVR,2);
            case 'AFA'
                RD2_AFA_RMSE(:, S) = mean(EAFA,2);
        end
    end
    
end
EMLR = []; ESVR = []; EAFA = []; EWMLR = [];


load RD3_dataset
for S = 1:7
    for r = 1:100
        
        X = unidrnd(1000, size(Train_set,1), 1);
        [~, IX] = sort(X);
        X = Train_set(IX(1:5*2^(S-1)));
        
        for i = 1:length(technique)
            switch technique{i}
                case 'MLR'
                    [~, EMLR(:,r)] = MLR_Forecast(X, Test_set, MLR_lags);
                case 'WMLR'
                    if S >= 2
                        [~, EWMLR(:,r)] = WMLR_Forecast(X, Test_set, WMLR_lags, Part);
                    end
                case 'SVR'
                    ESVR_Par = SVR_Parameters(2,:);
                    [~, ESVR(:,r)] = SVR_Forecast(X, Test_set, SVR_lags, SVR_Parameters(SVR_lags,:));
                case 'AFA'
                    [~, EAFA(:,r)] = AFA_Forecast(X, Test_set, AFA_TauSigma);
            end
        end
        
    end
    
    for i = 1:length(technique)
        switch technique{i}
            case 'MLR'
                RD3_MLR_RMSE(:, S) = mean(EMLR,2);
            case 'WMLR'
                if S >= 2
                    RD3_WMLR_RMSE(:, S) = mean(EWMLR,2);
                end
            case 'SVR'
                RD3_SVR_RMSE(:, S) = mean(ESVR,2);
            case 'AFA'
                RD3_AFA_RMSE(:, S) = mean(EAFA,2);
        end
    end
    
end
EMLR = []; ESVR = []; EAFA = []; EWMLR = [];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLOT OF THE RESULTS
XX = [5, 10, 20, 40, 80, 160, 320];
figure;
subplot(2,2,1)
for i = 1:length(technique)
    switch technique{i}
        case 'MLR'
            plot(XX, mean(RD1_MLR_RMSE), '*-b'); hold on;
        case 'WMLR'
            plot(XX, mean(RD1_WMLR_RMSE), '-->c'); hold on;
        case 'SVR'
            plot(XX, mean(RD1_SVR_RMSE), ':og'); hold on;
        case 'AFA'
            plot(XX, mean(RD1_AFA_RMSE), '-.sr'); hold on;
    end
end
xlim([0,400]); 
title('RD1 dataset')

subplot(2,2,2)
for i = 1:length(technique)
    switch technique{i}
        case 'MLR'
            plot(XX, mean(SD1_MLR_RMSE), '*-b'); hold on;
        case 'WMLR'
            plot(XX, mean(SD1_WMLR_RMSE), '-->c'); hold on;
        case 'SVR'
            plot(XX, mean(SD1_SVR_RMSE), ':og'); hold on;
        case 'AFA'
            plot(XX, mean(SD1_AFA_RMSE), '-.sr'); hold on;
    end
end
xlim([0,400]); 
title('SD1 dataset')

subplot(2,2,3)
for i = 1:length(technique)
    switch technique{i}
        case 'MLR'
            plot(XX, mean(RD2_MLR_RMSE), '*-b'); hold on;
        case 'WMLR'
            plot(XX, mean(RD2_WMLR_RMSE), '-->c'); hold on;
        case 'SVR'
            plot(XX, mean(RD2_SVR_RMSE), ':og'); hold on;
        case 'AFA'
            plot(XX, mean(RD2_AFA_RMSE), '-.sr'); hold on;
    end
end
xlim([0,400]);
title('RD2 dataset')

subplot(2,2,4)
for i = 1:length(technique)
    switch technique{i}
        case 'MLR'
            plot(XX, mean(RD3_MLR_RMSE), '*-b'); hold on;
        case 'WMLR'
            plot(XX, mean(RD3_WMLR_RMSE), '-->c'); hold on;
        case 'SVR'
            plot(XX, mean(RD3_SVR_RMSE), ':og'); hold on;
        case 'AFA'
            plot(XX, mean(RD3_AFA_RMSE), '-.sr'); hold on;
    end
end
xlim([0,400]); 
title('RD3 dataset')
ylabel('RMSE')
xlabel('Training set size')
hleg = legend('MLR', 'SVR', 'AFA');

