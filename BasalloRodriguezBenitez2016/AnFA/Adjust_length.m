function [E, fkopt, deltopt] = Adjust_length(f, g, graph)

% DESCRIPTION
% This function adjusts the length of time series g in order to maximize the
% similarity with current time series f.

% INPUTS
% f:	A 1 by T vector containing the current time series.
% g:    A 1 by T vector containing time series previously realized
%       (analogous).
% graph:A 0-1 value. If 1, then a plot is generated with the
%       results.

% OUTPUTS
% E:    A vector containing the time series after adjusting the
%       length.
% fkopt:Is the minimm distance between time series obtained after adjusting
%       the length.
% deltopt: Optimal value of the parameter for length adjustment.

% EXAMPLE
%%% From this example you require read the RD1_dataset.mat file.
% load RD1_dataset
% [E, fkopt, deltopt] = Adjust_length(Train_set(1,:), Train_set(2,:), 1)

k = length(f);  % It is required that k > 2
Lg = length(g);

% Optimization results
options = optimset('Display', 'off');
[deltopt, fkopt]  =  fminbnd( @ Adjust, 0.0001, Lg/k, options );

[~, E] = Adjust(deltopt);

    function [fk, e] = Adjust(delta)
        
        % Length of the adjusted segment
        delta = real(delta);
        le = ceil(Lg/delta);
        e = zeros(1, le);
        
        % Adjust the length
        e(1) = min(1, delta)*g(1); nn = 1;
        count1 = min(1, delta); count2 = count1;
        while count1 < Lg
            while count2 < delta
                if floor( count1 ) < floor( count1 + (delta - count2) )
                    if ceil(count1) ~= count1
                        diff = ( ceil(count1) - count1 );
                        e(nn) = e(nn) + diff*g( ceil(count1) );
                    else
                        diff = 1;
                        if nn < le
                            e(nn) = e(nn) + diff*g( ceil(count1) + 1 );
                        end
                    end
                    count2 = count2 + diff;
                    count1 = count1 + diff;
                else
                    if ceil(count1) + 1 <= Lg
                        e(nn) = e(nn) + ( delta - count2 )*g( ceil(count1) + 1 );
                    end
                    diff = ( delta - count2 );
                    count2 = count2 + ( delta - count2 );
                    count1 = count1 + diff;
                end
            end
            count2 = 0; nn = nn + 1;
        end
        
        ei = e(1:k);
        
        % Compute the Euclidean distance
        dk = sum( sqrt( (ei(1:end - 1) - f(1:k-1)).^2 + (ei(2:end) - f(2:k)).^2 ) )/(k - 1);
        
        % Compute the similarity measure
        mk = Analogous_dist(f, ei);
        
        % Obtain the objective value
        fk = dk/(mk + 1);
        
    end

if graph == 1
    plot(f,'k'); hold on; plot(g, 'r'); hold on; plot(E(1:Lg), 'g');
    legend({'f', 'g', 'Adj. Length g'});
end

end