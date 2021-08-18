include("GetModelParameters.jl")
include("ThetaModel.jl")
using Flux, Optim, DiffEqFlux, DiffEqSensitivity, ReverseDiff
using BenchmarkTools, Plots
#using BSON: @save, @load

# * Get the data
path = "D:\\Code\\[Servicio Social]\\Datos\\Casos_Modelo_Theta_v3.csv"
const data = Data(path)

lambdas = [
    "2020-03-20",
    "2020-03-23",
    "2020-03-30",
    "2020-04-21",
    "2020-06-01"
]
const dates = day_to_index(lambdas, data, Float64)

# * Time interval and intermediary points
tspan = (1.0, 446.0)
tsteps = 1.0:446.0

# * Initial conditions
N = Float64(data.population[Int(tspan[1])])
u0 = [
    data.susceptible[Int(tspan[1])],
    data.exposed[Int(tspan[1])],
    data.infec[Int(tspan[1])], # Infected 
    data.infec_u[Int(tspan[1])], # Infected undetected
    #data.infec_u[Int(tspan[1])]*0.3, # Infected undetected that will die
    data.hospitalized[Int(tspan[1])],
    data.hospitalized[Int(tspan[1])]*0.3,
    #data.quarentine[Int(tspan[1])],
    #data.recovered[Int(tspan[1])],
    #data.recovered[Int(tspan[1])]*1.3,
    #data.dead[Int(tspan[1])]*0.3,
    #data.dead[Int(tspan[1])]
]

# * θ-M equation parameters. 
γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, γ_Q = 5.5, 5.0, 9.0, 14.2729, 5.0, 36.0450

β_I0, c_E, c_u, ρ0, k2, c3, c5, ω_u0 =  0.4992, 0.3806, 0.3293, 0.7382, 0.5, 0.2, 0.8, 0.42
β_e0, β_I0_min = (c_E, c_u) .* β_I0

max_ω, min_ω = (0.804699241804244, 0.12785018214405364) # Random numbers
delays = Delays(γ_Hd, γ_E + γ_I)

times = Times(tspan, data, delays)

# Known ms
const ms_val = [1.0,1.0, 0.0, 0.0, 0.0, 0.0]

# ms and indexes for optimization
const ms_val_tracked = [
    ReverseDiff.TrackedReal(1.0, 0.0),
    ReverseDiff.TrackedReal(1.0, 0.0),
    ReverseDiff.TrackedReal(0.0, 0.0),
    ReverseDiff.TrackedReal(0.0, 0.0),
    ReverseDiff.TrackedReal(0.0, 0.0)
]

ms = Msλs(
    convert(Float64,times.t0), dates, 
    ms_val,
    [k2], [c3, c5]
)

# * List of parameters at time *t*

p = parameters_lists(times, data, delays, ms, dates, max_ω, min_ω, ω_u0, ρ0, β_I0, β_e0, β_I0_min, γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, γ_Q)

# * Test if the parameter functions are working and their running time

@benchmark parameters_lists(times, data, delays, ms, dates, max_ω, min_ω, ω_u0, ρ0, β_I0, β_e0, β_I0_min, γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, γ_Q)

# * Set up the ODE problem and solve
prob = ODEProblem(ThetaModel!, u0, tspan, p)

# AutoVern7(Rodas5())
sol = solve(prob, tstops=tsteps)

@benchmark solve(prob, tstops=tsteps)

# * Plot the solution
labels = ["Model" "Real data"]

plot(data.infec)
    
graf_predictions(data, sol, labels)

# * Optimize the model

callback = function(p, l, pred)
    display(l)
    plt = graf_predictions(data, pred, labels::Matrix{String})
    display(plt)
    return false
end

result_ode = DiffEqFlux.sciml_train(
    distance, p,
    ADAM(0.5),
    cb = callback,
    maxiters=10
)