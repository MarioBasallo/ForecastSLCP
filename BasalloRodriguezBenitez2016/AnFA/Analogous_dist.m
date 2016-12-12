function [m, F] = Analogous_dist(f, g)

% DESCRIPTION
% This function returns the similarity measure between time series f and g.

% INPUTS
% f:	A 1 by T vector containing the current time series.
% g:    A 1 by T vector containing time series previously realized.

% OUTPUTS
% m:    Is the Cieslak-Jasinski distance (see Szozda, N. (2010): Analogous 
%       forecasting of products with a short life cycle. Decision Making in
%       Manufacturing and Services, 4 , 71–85.)
% F:    A value that relates the Euclidean and Cieslak-Jasinski distance
%       metrics.

% EXAMPLE
%%% From this example you require to red the RD1_dataset.mat file.
% load RD1_dataset
% [m, F] = Analogous_dist(Train_set(1,:), Train_set(2,:)))

n = length(f);
mi = zeros(1, n - 1);
for i = 1:n - 1
    alpha = acos( min(1, ( (i + 1 - i)*(i + 1 - i) + (f(i + 1) - f(i))*(g(i + 1) - g(i)) )/...
        ( sqrt( (i + 1 - i)^2 + (f(i + 1) - f(i))^2 )*sqrt( (i + 1 - i)^2 + (g(i + 1) - g(i))^2 ) ) ) );
    if (f(i + 1) - f(i))*(g(i + 1) - g(i)) >= 0  
        mi(i) = 1 - 2*alpha/pi;
    else
        mi(i) = - alpha/pi;
    end
end

m = ( 1/n )*sum(mi);
d = sum( sqrt( (g(1:end - 1) - f(1:end-1)).^2 + (g(2:end) - f(2:end)).^2 ) )/(n - 1);
F = d/(m+1);

end