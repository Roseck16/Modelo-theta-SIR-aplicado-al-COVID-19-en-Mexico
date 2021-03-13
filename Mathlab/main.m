alpha = 0.6;
beta = 0.0370;
gamma = 1.9560e-07;
sigma = 0;
N = 127792286;

path = "D:\Code\[Servicio Social]\Datos\Datos_2020_No_Oficial.csv";

% syms sus(t) inf(t) rec(t) exp(t)

% ode1 = diff(sus) == (-alpha / N)*sus*(beta * exp + inf);
% ode2 = diff(inf) == gamma * exp - sigma * inf;
% ode3 = diff(rec) == sigma * inf;
% ode4 = diff(exp) == (-alpha / N)*sus*(beta * exp + inf) - gamma*exp;

% odes = [ode1; ode2; ode3; ode4];
% disp(odes)

% cond1 = sus(0) == 102233829;
% cond2 = inf(0) == 5;
% cond3 = rec(0) == 0;
% cond4 = exp(0) == 25558452;

% conds = [cond1; cond2; cond3; cond4];
% disp(conds)

% sol = dsolve(odes, conds);
% disp(sol)

% tspan = linspace(0, 31, 31);
% sus0 = 102233829;
% inf0 = 5;
% rec0 = 0;
% exp0 = 25558452;
% initial_con = [sus0 inf0 rec0 exp0];

% [t, y] = ode45(@(t,y) odefun(t,y,N,alpha,beta,gamma,sigma), tspan, initial_con);

mex_model = SimpleModel("dia", 15, "marzo", "marzo");
mex_model = official_data(mex_model, path);

x0 = [alpha, beta, gamma, sigma];
lb = [0.0001, 0, 0.0001, 0.0001];
up = [Inf, 1, 0.1, 0.123969699532899];

x = optimizar_con_limites(mex_model, x0, lb, up);

mex_model = solucion(mex_model, x(1), x(2), x(3), x(4));

%graficar(mex_model, ["i"], true, false)