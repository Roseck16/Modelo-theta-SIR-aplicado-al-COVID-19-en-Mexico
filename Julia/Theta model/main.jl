include("ThetaModel.jl")
#include("GetModelParameters.jl")
using Plots, Flux, Optim, DiffEqFlux, DiffEqSensitivity, ReverseDiff
using BenchmarkTools
#using BSON: @save, @load

# const saved = Dict{String,Float64}(
#     "γ_d" => 13.123753454738198,
#     #"γ_E" => 1/5.5,
#     #"γ_I" => 1/5,
#     "ω" => 0.014555,
#     "ω_0" =>0.50655
# )

# * Get the data
path = "D:\\Code\\[Servicio Social]\\Datos\\Casos_Modelo_Theta_final_complemented.csv"
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
tsteps = 1.0:1.0:446.0

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

# t0, tMAX, t_iCFR, t_θ0, t_η
const times = [0, 446, 23, 29, 13]

γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, γ_Q = 5.5, 5.0, 9.0, 14.2729, 5.0, 36.0450

β_I0, c_E, c_u, ρ0, k2, c3, c5, ω_u0 =  0.4992, 0.3806, 0.3293, 0.7382, 1.0, 1.0, 1.0, 0.42
β_e0, β_I0_min = (c_E, c_u) .* β_I0

# Known ms and indexes
const ms_val = [1.0,1.0, 0.0, 0.0, 0.0, 0.0]
const ms_index = [1,2,5]

# ms and indexes for optimization
const ms_val_tracked = [
    ReverseDiff.TrackedReal(1.0, 0.0),
    ReverseDiff.TrackedReal(1.0, 0.0),
    ReverseDiff.TrackedReal(0.0, 0.0),
    ReverseDiff.TrackedReal(0.0, 0.0),
    ReverseDiff.TrackedReal(0.0, 0.0)
]

# * Test if the parameter functions are working and their running time

@benchmark Msλs(
    convert(Float64,times[1]), dates, 
    ms_val,
    ms_index,
    [k2], [c3, c5]
)

@benchmark TimeParams(
    1, data, 
    times, 
    delays,
    ρ0, ω_0, ω_CFR0, θ_0, _ω=ω
)

@benchmark βs(
    1, 
    ω, θ, ω_u0, η, ρ, 
    ms, dates,
    γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, 
    β_I0, β_e0, β_I0_min
)

@benchmark paramsModel(γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, γ_Q, ρ0, k2, c3, c5, β_I0, β_e0, β_I0_min, ω_u0)

p = paramsModel(γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, γ_Q, ρ0, k2, c3, c5, β_I0, β_e0, β_I0_min, ω_u0)

# * Set up the ODE problem and solve
prob = ODEProblem(ThetaModel!, u0, tspan, p)

# AutoVern7(Rodas5())
sol = solve(prob, tstops=tsteps)

@benchmark solve(prob, tstops=tsteps)

# * Plot the solution
labels = ["Model" "Real data"]

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
    ADAM(0.01),
    cb = callback,
    maxiters=50
)
