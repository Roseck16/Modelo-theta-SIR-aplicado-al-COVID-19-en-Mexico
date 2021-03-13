function dydt = odefun(t,y,N, alpha, beta, gamma, sigma)
    dydt = zeros(4,1);
    dydt(1) = (-alpha/N)*y(1)*(beta * y(4) + y(2));
    dydt(2) = gamma * y(4) - sigma * y(1);
    dydt(3) = sigma * y(2);
    dydt(4) = (-alpha/N)*y(1)*(beta * y(4) + y(2)) - gamma * y(4);
end

