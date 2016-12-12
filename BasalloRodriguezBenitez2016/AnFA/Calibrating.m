function [V, fkopt, wopt] = Calibrating(f, g, graph)

% This function adjusts the scale of a time-series g in order to make
% as similar as possible to time series f. For more information see Szozda, 
% N. (2010). Analogous forecasting of products with a short life cycle. 
% Decision Making in Manufacturing and Services, 4 , 71–85.

% INPUTS
% f:	A 1 by T vector containing the current time series.
% g:    A 1 by T vector containing time series previously realized
%       (analogous).
% graph:A 0-1 value. If 1, then a plot is generated with the results.

% OUTPUTS
% V:    Scaled time series.
% fkopt:Minimm distance between time seres after scaling.
% wopt: Optimal value of the escale parameter.

% EEXAMPLE
%%% From this example you require to load the RD1_dataset.mat file.
% load RD1_dataset
% [V, fkopt, wopt] = Calibrating(Train_set(1,:), Train_set(2,:), 1)

k = length(f);  % It is required that k > 2

% Optimization results
options = optimset('Display','off');
[wopt, fkopt]  =  fminbnd( @Cal_coef, 0, max(f), options );

V = wopt*g;

    function fk = Cal_coef(w)
        % Calibrate the time series
        vi = w*g(1:k);
        
        % Compute the Euclidean distance
        dk = sum( sqrt( (vi(1:end - 1) - f(1:k-1)).^2 + (vi(2:end) - f(2:k)).^2 ) )/(k - 1);
        
        % Compute the similarity measure
        mk = Analogous_dist(f, vi);
        
        % Obtain the objective value
        fk = dk/(mk + 1);
    end

if graph == 1
    plot(f, 'k'); hold on; plot(g, 'r'); hold on; plot(V, 'g'); 
    legend({'f', 'g', 'Scaled g'});
end

end

