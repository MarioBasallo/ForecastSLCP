function Rho = Corr_Coef(x, y, L)

xx = x(1:end - L);    xbar = mean(xx);
yy = y(L + 1:end);    ybar = mean(yy);
Rho = abs( (( xx - xbar )*( yy - ybar )')/sqrt( sum( (xx - xbar).^2 )*sum( (yy - ybar).^2 ) ) );

if isnan(Rho)
    Rho = 0;
end

end