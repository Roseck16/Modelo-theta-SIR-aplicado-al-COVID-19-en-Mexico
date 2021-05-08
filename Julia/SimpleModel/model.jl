using Plots, DifferentialEquations

tspan = (0.0, 15.0)

function simple_model(du, u, p, t)
    du[1] = (-α/N)*u[1]*(β*u[4] + u[2])
    du[2] = γ*u[4] - σ*u[2]
    du[3] = σ*u[2]
    du[4] = (α/N)*u[1]*(β*u[4] + u[2]) - γ*u[4]
end

# Mexico con datos de https://datos.covid-19.conacyt.mx/
# de la fecha 2 de enero 2021
N = 128649565;
S_0 = 102919652
I_0 = 1466490; # 1 466 490 Infectados
R_0 = 1241959; # 1 241 959 Defunciones y recuperados
C_0 = 23021464;
γ = 0.0637;
σ = 0.0878; # Defunciones entre población total
α = 0.6;
β = 0.037;

# Susceptible, infected, removed, exposed
MEX = [102919652; 1466490; 1241959; 23021464]
prob_mex = ODEProblem(simple_model, MEX, tspan)
sol_mex = solve(prob_mex)

result = plot(
    sol_mex, 
    vars=(0,2),
    label = "",
    xlabel = "Tiempo (Semana)",
    ylabel = "Infectados en MEX"
)

savefig(result, "imagenes/simple_model/mexico")