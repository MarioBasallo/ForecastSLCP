clear all
close all
clc
path(path,'C:\BasalloRodriguezBenitez2016\NAIVE');
path(path,'C:\BasalloRodriguezBenitez2016\MLR');
path(path,'C:\BasalloRodriguezBenitez2016\WMLR');
path(path,'C:\BasalloRodriguezBenitez2016\ANN');
path(path,'C:\BasalloRodriguezBenitez2016\AFA');
path(path,'C:\BasalloRodriguezBenitez2016\AnFA');
path(path,'C:\BasalloRodriguezBenitez2016\INPF');
path(path,'C:\BasalloRodriguezBenitez2016\SVR');
path(path,'C:\BasalloRodriguezBenitez2016');

%--------------------------------------------------------------------------
%
% Before running this script we sugest review the README.txt file provided.
%
% This script evaluates NAIVE, MLR, WMLR, SVR, ANN, AFA, AnFA, and INPF 
% forecasting methods by perform forecasts on RD1, RD2, RD3, and SD data 
% sets. The results are shown as plots except for the statistical 
% significance matrix which is printed in command window.
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
%    Technology)
% 2) You should be aware that AnFA and INPF method may take a few hours to 
%    generate results (1 or 2 hours depending of the machine).
%--------------------------------------------------------------------------


% Select the forecasting algorithms to be evaluated
% EXAMPLE: technique = {'NAIVE', 'MLR', 'WMLR', 'SVR', 'ANN', 'AFA', 'AnFA'};
technique = {'NAIVE', 'MLR', 'WMLR', 'SVR'};

load RD1_dataset
RMSE_RD1 = [];
TIME_RD1 = [];
RD1 = Test_set;
for i = 1:length(technique)
    switch technique{i}
        case 'NAIVE'
            [RD1_NAIVE_xbar, RD1_NAIVE_RMSE, RD1_NAIVE_Time] = NaiveForecast(Test_set, 1);
            RMSE_RD1 = [RMSE_RD1, RD1_NAIVE_RMSE];
            TIME_RD1 = [TIME_RD1; RD1_NAIVE_Time];
        case 'MLR'
            [RD1_MLR_xbar, RD1_MLR_RMSE, RD1_MLR_Time] = MLR_Forecast(Train_set, Test_set, MLR_lags);
            RMSE_RD1 = [RMSE_RD1, RD1_MLR_RMSE];
            TIME_RD1 = [TIME_RD1; RD1_MLR_Time];
        case 'WMLR'
            [RD1_WMLR_xbar, RD1_WMLR_RMSE, RD1_WMLR_Time] = WMLR_Forecast(Train_set, Test_set, WMLR_lags, Part);
            RMSE_RD1 = [RMSE_RD1, RD1_WMLR_RMSE];
            TIME_RD1 = [TIME_RD1; RD1_WMLR_Time];
        case 'SVR'
            [RD1_SVR_xbar, RD1_SVR_RMSE, RD1_SVR_Time] = SVR_Forecast(Train_set, Test_set, SVR_lags, SVR_Parameters(SVR_lags,:));
            RMSE_RD1 = [RMSE_RD1, RD1_SVR_RMSE];
            TIME_RD1 = [TIME_RD1; RD1_SVR_Time];
        case 'ANN'
            [RD1_ANN_xbar, RD1_ANN_RMSE, RD1_ANN_Time] = ANN_Forecast(Train_set, Test_set, ANN_lags, ANN_Parameters(ANN_lags,:));
            RMSE_RD1 = [RMSE_RD1, RD1_ANN_RMSE];
            TIME_RD1 = [TIME_RD1; RD1_ANN_Time];
        case 'AFA'
            [RD1_AFA_xbar, RD1_AFA_RMSE, RD1_AFA_Time] = AFA_Forecast(Train_set, Test_set, AFA_TauSigma);
            RMSE_RD1 = [RMSE_RD1, RD1_AFA_RMSE];
            TIME_RD1 = [TIME_RD1; RD1_AFA_Time];
        case 'AnFA'
            [RD1_AnFA_xbar, RD1_AnFA_RMSE, RD1_AnFA_Time] = AnFA_Forecast(Train_set, Test_set);
            RMSE_RD1 = [RMSE_RD1, RD1_AnFA_RMSE];
            TIME_RD1 = [TIME_RD1; RD1_AnFA_Time];
    end
end

load SD1_dataset
RMSE_SD1 = [];
TIME_SD1 = [];
SD1 = Test_set;
for i = 1:length(technique)
    switch technique{i}
        case 'NAIVE'
            [SD1_NAIVE_xbar, SD1_NAIVE_RMSE, SD1_NAIVE_Time] = NaiveForecast(Test_set, 1);
            RMSE_SD1 = [RMSE_SD1, SD1_NAIVE_RMSE];
            TIME_SD1 = [TIME_SD1; SD1_NAIVE_Time];
        case 'MLR'
            [SD1_MLR_xbar, SD1_MLR_RMSE, SD1_MLR_Time] = MLR_Forecast(Train_set, Test_set, MLR_lags);
            RMSE_SD1 = [RMSE_SD1, SD1_MLR_RMSE];
            TIME_SD1 = [TIME_SD1; SD1_MLR_Time];
        case 'WMLR'
            [SD1_WMLR_xbar, SD1_WMLR_RMSE, SD1_WMLR_Time] = WMLR_Forecast(Train_set, Test_set, WMLR_lags, Part);
            RMSE_SD1 = [RMSE_SD1, SD1_WMLR_RMSE];
            TIME_SD1 = [TIME_SD1; SD1_WMLR_Time];
        case 'SVR'
            [SD1_SVR_xbar, SD1_SVR_RMSE, SD1_SVR_Time] = SVR_Forecast(Train_set, Test_set, SVR_lags, SVR_Parameters(SVR_lags,:));
            RMSE_SD1 = [RMSE_SD1, SD1_SVR_RMSE];
            TIME_SD1 = [TIME_SD1; SD1_SVR_Time];
        case 'ANN'
            [SD1_ANN_xbar, SD1_ANN_RMSE, SD1_ANN_Time] = ANN_Forecast(Train_set, Test_set, ANN_lags, ANN_Parameters(ANN_lags,:));
            RMSE_SD1 = [RMSE_SD1, SD1_ANN_RMSE];
            TIME_SD1 = [TIME_SD1; SD1_ANN_Time];
        case 'AFA'
            [SD1_AFA_xbar, SD1_AFA_RMSE, SD1_AFA_Time] = AFA_Forecast(Train_set, Test_set, AFA_TauSigma);
            RMSE_SD1 = [RMSE_SD1, SD1_AFA_RMSE];
            TIME_SD1 = [TIME_SD1; SD1_AFA_Time];
        case 'AnFA'
            [SD1_AnFA_xbar, SD1_AnFA_RMSE, SD1_AnFA_Time] = AnFA_Forecast(Train_set, Test_set);
            RMSE_SD1 = [RMSE_SD1, SD1_AnFA_RMSE];
            TIME_SD1 = [TIME_SD1; SD1_AnFA_Time];
    end
end

load RD2_dataset
RMSE_RD2 = [];
TIME_RD2 = [];
RD2 = Test_set;
for i = 1:length(technique)
    switch technique{i}
        case 'NAIVE'
            [RD2_NAIVE_xbar, RD2_NAIVE_RMSE, RD2_NAIVE_Time] = NaiveForecast(Test_set, 1);
            RMSE_RD2 = [RMSE_RD2, RD2_NAIVE_RMSE];
            TIME_RD2 = [TIME_RD2; RD2_NAIVE_Time];
        case 'MLR'
            [RD2_MLR_xbar, RD2_MLR_RMSE, RD2_MLR_Time] = MLR_Forecast(Train_set, Test_set, MLR_lags);
            RMSE_RD2 = [RMSE_RD2, RD2_MLR_RMSE];
            TIME_RD2 = [TIME_RD2; RD2_MLR_Time];
        case 'WMLR'
            [RD2_WMLR_xbar, RD2_WMLR_RMSE, RD2_WMLR_Time] = WMLR_Forecast(Train_set, Test_set, WMLR_lags, Part);
            RMSE_RD2 = [RMSE_RD2, RD2_WMLR_RMSE];
            TIME_RD2 = [TIME_RD2; RD2_WMLR_Time];
        case 'SVR'
            [RD2_SVR_xbar, RD2_SVR_RMSE, RD2_SVR_Time] = SVR_Forecast(Train_set, Test_set, SVR_lags, SVR_Parameters(SVR_lags,:));
            RMSE_RD2 = [RMSE_RD2, RD2_SVR_RMSE];
            TIME_RD2 = [TIME_RD2; RD2_SVR_Time];
        case 'ANN'
            [RD2_ANN_xbar, RD2_ANN_RMSE, RD2_ANN_Time] = ANN_Forecast(Train_set, Test_set, ANN_lags, ANN_Parameters(ANN_lags,:));
            RMSE_RD2 = [RMSE_RD2, RD2_ANN_RMSE];
            TIME_RD2 = [TIME_RD2; RD2_ANN_Time];
        case 'AFA'
            [RD2_AFA_xbar, RD2_AFA_RMSE, RD2_AFA_Time] = AFA_Forecast(Train_set, Test_set, AFA_TauSigma);
            RMSE_RD2 = [RMSE_RD2, RD2_AFA_RMSE];
            TIME_RD2 = [TIME_RD2; RD2_AFA_Time];
        case 'AnFA'
            [RD2_AnFA_xbar, RD2_AnFA_RMSE, RD2_AnFA_Time] = AnFA_Forecast(Train_set, Test_set);
            RMSE_RD2 = [RMSE_RD2, RD2_AnFA_RMSE];
            TIME_RD2 = [TIME_RD2; RD2_AnFA_Time];
    end
end

load RD3_dataset
RMSE_RD3 = [];
TIME_RD3 = [];
RD3 = Test_set;
for i = 1:length(technique)
    switch technique{i}
        case 'NAIVE'
            [RD3_NAIVE_xbar, RD3_NAIVE_RMSE, RD3_NAIVE_Time] = NaiveForecast(Test_set, 1);
            RMSE_RD3 = [RMSE_RD3, RD3_NAIVE_RMSE];
            TIME_RD3 = [TIME_RD3; RD3_NAIVE_Time];
        case 'MLR'
            [RD3_MLR_xbar, RD3_MLR_RMSE, RD3_MLR_Time] = MLR_Forecast(Train_set, Test_set, MLR_lags);
            RMSE_RD3 = [RMSE_RD3, RD3_MLR_RMSE];
            TIME_RD3 = [TIME_RD3; RD3_MLR_Time];
        case 'WMLR'
            [RD3_WMLR_xbar, RD3_WMLR_RMSE, RD3_WMLR_Time] = WMLR_Forecast(Train_set, Test_set, WMLR_lags, Part);
            RMSE_RD3 = [RMSE_RD3, RD3_WMLR_RMSE];
            TIME_RD3 = [TIME_RD3; RD3_WMLR_Time];
        case 'SVR'
            [RD3_SVR_xbar, RD3_SVR_RMSE, RD3_SVR_Time] = SVR_Forecast(Train_set, Test_set, SVR_lags, SVR_Parameters(SVR_lags,:));
            RMSE_RD3 = [RMSE_RD3, RD3_SVR_RMSE];
            TIME_RD3 = [TIME_RD3; RD3_SVR_Time];
        case 'ANN'
            [RD3_ANN_xbar, RD3_ANN_RMSE, RD3_ANN_Time] = ANN_Forecast(Train_set, Test_set, ANN_lags, ANN_Parameters(ANN_lags,:));
            RMSE_RD3 = [RMSE_RD3, RD3_ANN_RMSE];
            TIME_RD3 = [TIME_RD3; RD3_ANN_Time];
        case 'AFA'
            [RD3_AFA_xbar, RD3_AFA_RMSE, RD3_AFA_Time] = AFA_Forecast(Train_set, Test_set, AFA_TauSigma);
            RMSE_RD3 = [RMSE_RD3, RD3_AFA_RMSE];
            TIME_RD3 = [TIME_RD3; RD3_AFA_Time];
        case 'AnFA'
            [RD3_AnFA_xbar, RD3_AnFA_RMSE, RD3_AnFA_Time] = AnFA_Forecast(Train_set, Test_set);
            RMSE_RD3 = [RMSE_RD3, RD3_AnFA_RMSE];
            TIME_RD3 = [TIME_RD3; RD3_AnFA_Time];
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Forecasting errors
ERRORS = {RMSE_RD1, RMSE_SD1, RMSE_RD2, RMSE_RD3};
n = length(technique);
ErrorMat = cell(5,n + 1);
ErrorMat(1:end,1) = {'Data set', 'RD1', 'SD', 'RD2', 'RD3'};
ErrorMat(1,2:end) = technique;
for i = 1:4
    for j = 1:n
        Errs = ERRORS{i};
        ErrorMat(i+1,j+1) = {strcat( num2str(mean(Errs(:,j))),' +/-  ', num2str(std(Errs(:,j))) )};
    end
end
Forecasting_Errors = ErrorMat


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Statistical signinificance matrix
n = length(technique);
CompMat = cell(n + 1,n + 1);
CompMat(1,2:end) = technique;
CompMat(2:end, 1) = technique;
for i = 2:n + 1
    for j = 2:n + 1
        if kruskalwallis([RMSE_RD1(:,i - 1), RMSE_RD1(:,j - 1)], [], 'off') <= 0.05
            if median(RMSE_RD1(:,i - 1)) < median(RMSE_RD1(:,j - 1))
                CompMat{i, j} = '1';
            else
                CompMat{i, j} = '0';
            end
        else
            CompMat{i, j} = '-';
        end
    end
end

for i = 2:n + 1
    for j = 2:n + 1
        if kruskalwallis([RMSE_SD1(:,i - 1), RMSE_SD1(:,j - 1)], [], 'off') <= 0.05
            if median(RMSE_SD1(:,i - 1)) < median(RMSE_SD1(:,j - 1))
                CompMat{i, j} = [CompMat{i,j}, '1'];
            else
                CompMat{i, j} = [CompMat{i,j}, '0'];
            end
        else
            CompMat{i, j} = [CompMat{i,j}, '-'];
        end
    end
end

for i = 2:n + 1
    for j = 2:n + 1
        if kruskalwallis([RMSE_RD2(:,i - 1), RMSE_RD2(:,j - 1)], [], 'off') <= 0.05
            if median(RMSE_RD2(:,i - 1)) < median(RMSE_RD2(:,j - 1))
                CompMat{i, j} = [CompMat{i,j}, '1'];
            else
                CompMat{i, j} = [CompMat{i,j}, '0'];
            end
        else
            CompMat{i, j} = [CompMat{i,j}, '-'];
        end
    end
end

for i = 2:n + 1
    for j = 2:n + 1
        if kruskalwallis([RMSE_RD3(:,i - 1), RMSE_RD3(:,j - 1)], [], 'off') <= 0.05
            if median(RMSE_RD3(:,i - 1)) < median(RMSE_RD3(:,j - 1))
                CompMat{i, j} = [CompMat{i,j}, '1'];
            else
                CompMat{i, j} = [CompMat{i,j}, '0'];
            end
        else
            CompMat{i, j} = [CompMat{i,j}, '-'];
        end
    end
end

Statistical_Significance_Matrix = CompMat


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Forecasting times
TIME = [TIME_RD1, TIME_SD1, TIME_RD2, TIME_RD3];
figure;
bar(TIME); colormap(summer);
ylabel('Processing time (seconds)')
xlabel('Forecasting algorithm')
set(gca, 'XTickLabel', technique)
hleg = legend('RD1', 'SD', 'RD2', 'RD3');
set(hleg, 'Location','NorthWest')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Forecasts of randomly selected time series
figure;
subplot(2,2,1)
r1 = unidrnd(size(RD1, 1), 1, 1);
plot(RD1{r1,:}, 'k', 'LineWidth', 2); hold on; 
for i = 1:length(technique)
    switch technique{i}
        case 'NAIVE'
            plot(RD1_NAIVE_xbar{r1,:}, '*-k'); hold on;
        case 'MLR'
            plot(RD1_MLR_xbar{r1,:}, '*-b'); hold on;
        case 'WMLR'
            plot(RD1_WMLR_xbar{r1,:}, '.m'); hold on;
        case 'SVR'
            plot(RD1_SVR_xbar{r1,:}, '-.sr'); hold on;
        case 'ANN'
            plot(RD1_ANN_xbar{r1,:}, '-py'); hold on;
        case 'AFA'
            plot(RD1_AFA_xbar{r1,:}, ':og'); hold on;
        case 'AnFA'
            plot(RD1_AnFA_xbar{r1,:}, '-->c'); hold on;
        case 'INPF'
            plot(RD1_INPF_xbar{r1,:}, '--<m'); hold on;
    end
end
xlim([0,length(RD1{r1,:}) + 1]);
ylim([0,max(max(RD1{r1,:}), max(RD1_MLR_xbar{r1,:})) + 1]); 
title('RD1 dataset')

subplot(2,2,2)
r1 = unidrnd(size(SD1, 1), 1, 1);
plot(SD1{r1,:}, 'k', 'LineWidth', 2); hold on; 
for i = 1:length(technique)
    switch technique{i}
        case 'NAIVE'
            plot(SD1_NAIVE_xbar{r1,:}, '*-k'); hold on;
        case 'MLR'
            plot(SD1_MLR_xbar{r1,:}, '*-b'); hold on;
        case 'WMLR'
            plot(SD1_WMLR_xbar{r1,:}, '.m'); hold on;
        case 'SVR'
            plot(SD1_SVR_xbar{r1,:}, '-.sr'); hold on;
        case 'ANN'
            plot(SD1_ANN_xbar{r1,:}, '-py'); hold on;
        case 'AFA'
            plot(SD1_AFA_xbar{r1,:}, ':og'); hold on;
        case 'AnFA'
            plot(SD1_AnFA_xbar{r1,:}, '-->c'); hold on;
        case 'INPF'
            plot(SD1_INPF_xbar{r1,:}, '--<m'); hold on;
    end
end
xlim([0,length(SD1{r1,:}) + 1]);
ylim([0,max(max(SD1{r1,:}), max(SD1_MLR_xbar{r1,:})) + 1]); 
title('SD1 dataset')

subplot(2,2,3)
r1 = unidrnd(size(RD2, 1), 1, 1);
plot(RD2{r1,:}, 'k', 'LineWidth', 2); hold on; 
for i = 1:length(technique)
    switch technique{i}
        case 'NAIVE'
            plot(RD2_NAIVE_xbar{r1,:}, '*-k'); hold on;
        case 'MLR'
            plot(RD2_MLR_xbar{r1,:}, '*-b'); hold on;
        case 'WMLR'
            plot(RD2_WMLR_xbar{r1,:}, '.m'); hold on;
        case 'SVR'
            plot(RD2_SVR_xbar{r1,:}, '-.sr'); hold on;
        case 'ANN'
            plot(RD2_ANN_xbar{r1,:}, '-py'); hold on;
        case 'AFA'
            plot(RD2_AFA_xbar{r1,:}, ':og'); hold on;
        case 'AnFA'
            plot(RD2_AnFA_xbar{r1,:}, '-->c'); hold on;
        case 'INPF'
            plot(RD2_INPF_xbar{r1,:}, '--<m'); hold on;
    end
end
xlim([0,length(RD2{r1,:}) + 1]);
ylim([0,max(max(RD2{r1,:}), max(RD2_MLR_xbar{r1,:})) + 1]); 
title('RD2 dataset')

subplot(2,2,4)
r1 = unidrnd(size(RD3, 1), 1, 1);
plot(RD3{r1,:}, 'k', 'LineWidth', 2); hold on; 
for i = 1:length(technique)
    switch technique{i}
        case 'NAIVE'
            plot(RD3_NAIVE_xbar{r1,:}, '*-k'); hold on;
        case 'MLR'
            plot(RD3_MLR_xbar{r1,:}, '*-b'); hold on;
        case 'WMLR'
            plot(RD3_WMLR_xbar{r1,:}, '.m'); hold on;
        case 'SVR'
            plot(RD3_SVR_xbar{r1,:}, '-.sr'); hold on;
        case 'ANN'
            plot(RD3_ANN_xbar{r1,:}, '-py'); hold on;
        case 'AFA'
            plot(RD3_AFA_xbar{r1,:}, ':og'); hold on;
        case 'AnFA'
            plot(RD3_AnFA_xbar{r1,:}, '-->c'); hold on;
        case 'INPF'
            plot(RD3_INPF_xbar{r1,:}, '--<m'); hold on;
    end
end
xlim([0,length(RD3{r1,:}) + 1]);
ylim([0,max(max(RD3{r1,:}), max(RD3_MLR_xbar{r1,:})) + 1]); 
title('RD3 dataset')
ylabel('RMSE')
xlabel('Training set size')
hleg = legend(['real', technique]);

