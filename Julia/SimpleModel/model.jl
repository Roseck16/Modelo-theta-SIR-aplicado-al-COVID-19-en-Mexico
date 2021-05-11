using Plots, DifferentialEquations, Evolutionary
using Dates, DataFrames
import CSV: File as fl
using DifferentialEquations: Rosenbrock23, Rodas4, Euler
using JuMP, Alpine

struct Data{M<:Vector{Date}, N<:Vector{Int64}}
    """
    Struct that contains the data for day, cumulative number of infected, infected medical personal and deaths
    """
    days::M
    infec::N
    infec_medic::N
    dead::N
    imported::N

    function Data(path::String)
        data = DataFrame(fl(path))
        t = data."Fecha"
        i = data."Positivos"
        im = data."Positivos medicos"
        d = data."Fallecidos"
        e = data."Importados"
        new{Vector{Date}, Vector{Int64}}(t,i,im,d,e)
    end
end

function simple_model(du, u, p, t)
    #println(t)
    S, I, R, E = u
    α, β, γ, σ = p
    du[1] = dS = (-α/N)*S*(β*E + I)
    du[2] = dI = γ*E - σ*I
    du[3] = dR = σ*I
    du[4] = dE = (α/N)*S*(β*E + I) - γ*E
end

function check(dt,u,p,t)
    if any(isnan, u) || any(isinf, u)
        println(u)
        println("Tiempo: $t")
        println(p)
        return true
    end
    return false
end
#Rosenbrock23(autodiff=false)
function sol(f::Function, u0::M, tspan::N, p::M) where {V<:Float64, M<:Vector{V}, N<:Tuple{V,V}}
    prob = ODEProblem(f, u0, tspan, p)
    return solve(prob, adaptive=false, dt=1.0, unstable_check=check)
end

function distance(x)
    #x = [α, β, γ, σ]
    solution = sol(simple_model, u02, tspan2, x)
    dist = sqrt(
        sum(
            (solution[2,:] .- data.infec[200:300]) .^ 2
        )
    )
    return dist
end

path = "D:\\Code\\[Servicio Social]\\Datos\\Casos_Modelo_Theta_final.csv"
data = Data(path)

const N = 127575528.0
const u0 = [1.0, N-1, 0.0, 0.0]
const tspan = (1.0, 446.0)
u02 = [N*0.8, 384554, 475795, 24325687]
tspan2 = (200.0, 300.0)

ga = GA(populationSize=100,selection=uniformranking(3),
        mutation=gaussian(),crossover=uniformbin())

up_limit = 100000
low = [0.0, 0.0, 0.0, 0.0]
up = [up_limit, 1.0, up_limit, up_limit]
x0 = [0.5, 0.5, 0.5, 0.5]

simple_sol = sol(simple_model, u0, tspan2, x0)

min_sol = Evolutionary.optimize(distance, low, up, x0, ga)
min_param = Evolutionary.minimizer(min_sol)

solution = sol(simple_model, u0, tspan2, min_param)

plot(solution, vars=(0,2))
plot(data.infec)

