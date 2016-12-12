
README
======

Information here provided can be used to reproduce the results in the paper "Analogue based demand forecasting for short life cycle products: A regression method and a comprehensive assesment". The .mat files contain the data sets of time series used, and the values of the parameters required for forecasting methods. Such data sets divided into a training set, and a test set. The following is a description of the contents of .mat files.

Train_set: 	An N by T matrix containing N analogous or historical time series of length T, the training set.

Test_set: 	An M by T matrix  containing M current time series of length T to forecast, the test set.

MLR_lags: 	The number of lags p, necessary in multiple linear regression method.

WMLR_lags:  	The number of lags p, necessary in weigthed multiple linear regression method.

SVR_lags: 	The number of lags p, necessary in support vector regression method.

ANN_lags:	The number of lags p, necessary in support vector regression method.

SVR_Parameters: A p by 3 vector containing the Epsilon-SVR parameters for different values of number of lags (p) in rows, each column contain in order: epsilon, constant penalty, and gamma (the parameter of Gaussian kernel).

ANN_Parameters:	A p by 3 vector containing the number of neurons per hidden layer in the ANN model. Note that the results are presented at different values of the lag parameter.

AFA_TauSigma: 	The Tau-Sigma ratio required to AFA forecasting method.

Part: 		Optimal number of clusters for the WMLR method.



The following files allow reproducing the results.

1) main_1.m
   This script reproduces, in order, the following results:
     • Forecasting errors of methods.
     • The processing times of each forecasting method.
     • An illustration of forecasts for randomly selected time series.
2) main_2.m
   This script performs a sensitivity analysis considering different sizes of the training set.


Each .m file above mentioned brings an introduction that should be helpful to run with selected methods. Note that in these scripts, we only consider the MLR, WMLR, SVR, AFA, and, AnFA, methods, and omit the INPF method. The reason for doing this is due to with INPF method it is impossible to obtain forecasts along the entire life cycle, and this method requires consumes high processing time (however, note that INPF code is also available in the folder INPF).


Be aware that in the first lines of these .m files there is a variable “technique”. This variable can be set at any selection of the forecasting methods above discussed (NAIVE, MLR, WMLR, SVR, AFA, AnFA, and INPF). We emphasize that SVR method requires the LIBSVM toolbox be installed in your computer. In addition, note that the AnFA method can take a few hours (1 or 2 depending of the machine).


Finally, consider that the folder must be placed on the disk C on windows; otherwise, you have to change the path in every .m file.


