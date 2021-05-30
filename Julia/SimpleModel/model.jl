using Plots, Dates, DataFrames, DifferentialEquations, Flux, Optim, DiffEqFlux, DiffEqSensitivity
import CSV: File as fl
using BSON: @save, @load

struct Data{N<:Vector{Int64}}
    """
    Struct that contains the data needed for the model.
    """
    days::Vector{Date}
    population::N
    infec::N
    recovered::N
    susceptible::N
    exposed::N

    function Data(path::String)
        data = DataFrame(fl(path))
        t = data."Fecha"
        n = data."Poblacion"
        i = data."Positivos"
        r = data."Recuperados"
        sus = data."Susceptibles"
        e = data."Expuestos"
        new{Vector{Int64}}(t,n,i,r,sus,e)
    end
end

function simple_model!(du, u, p, t)
    #println(t)
    S, I, R, E = u
    α, β, γ, σ = p
    du[1] = dS = (-α/N)*S*(β*E + I)
    du[2] = dI = γ*E - σ*I
    du[3] = dR = σ*I
    du[4] = dE = (α/N)*S*(β*E + I) - γ*E
end

function graf_predictions(data::Data, solution::Any, labels::Matrix{String})
    xs = map(x -> round(Int, x),solution.t)
    ys = [solution[2,:], data.infec[xs]]
    
    plot(xs,ys, label=labels)
end

labels = ["Model" "Real data"]

# * Get the data
path = "D:\\Code\\[Servicio Social]\\Datos\\Casos_Modelo_Theta_final_complemented.csv"
const data = Data(path)

# * Time interval and intermediary points
tspan = (100.0, 446.0)
tsteps = 100.0:1.0:446

# * Inityal conditions
N = data.population[Int(tspan[1])]
u0 = [
    data.susceptible[Int(tspan[1])],
    data.infec[Int(tspan[1])],
    data.recovered[Int(tspan[1])],
    data.exposed[Int(tspan[1])]
]

# * SM equation parameter. p = [α, β, γ, σ]
p = [0.6, 0.037, 0.0018, 0.0165]
# p = [0.5754835798959406,
#     0.0033489694590588155,
#     5.043780700111756e-5,
#     0.0
# ]

# * Best minimum parameters found
@load "D:\\Code\\[Servicio Social]\\Julia\\SimpleModel\\minimum_parameters5.bson" p

# * Set up the ODE problem, then solve
prob = ODEProblem(simple_model!, u0, tspan, p)
sol = solve(prob)
@benchmark solve(prob)

plot(sol, vars=(0,3))

# * Plot the solution
graf_predictions(data, sol, labels)
#savefig("D:\\Code\\[Servicio Social]\\Julia\\SimpleModel\\simple_model_ode.png")

# * Optimize the model
function distance(p)
    sol = solve(prob, p=p, saveat=tsteps)
    days = map(x -> round(Int, x),sol.t)
    distance = sqrt(sum(abs2, sol[2,:].- data.infec[days]))
    return distance, sol
end

callback = function (p, l, pred)
    display(l)
    plt = graf_predictions(data, pred, labels)
    display(plt)
    return false
end

result_ode = DiffEqFlux.sciml_train(
    distance, p,
    ADAM(0.0001),
    cb = callback,
    maxiters = 1000
)

p = result_ode.minimizer

@save "D:\\Code\\[Servicio Social]\\Julia\\SimpleModel\\minimum_parameters5.bson" p

# * Predictions
tspan = (446.0, 2000.0)
N = data.population[Int(tspan[1])]
u0 = [
    data.susceptible[Int(tspan[1])],
    data.infec[Int(tspan[1])],
    data.recovered[Int(tspan[1])],
    data.exposed[Int(tspan[1])]
]

prob = ODEProblem(simple_model!, u0, tspan, p)
sol = solve(prob)

plot(sol, vars=(0,2))

function graf_results(solution, labels)
    xs = map(x -> round(Int, x),solution.t)
    ys = [
        solution[1,:],
        solution[2,:],
        solution[3,:],
        solution[4,:]
    ]

    plot(xs, ys, label=labels)
end

label_results = ["Susceptible" "Infected" "Recovered" "Exposed"]

graf_results(sol, label_results)