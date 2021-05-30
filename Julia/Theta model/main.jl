include("ThetaModel.jl")
#include("GetModelParameters.jl")
using Plots, Flux, Optim, DiffEqFlux, DiffEqSensitivity, ReverseDiff
using BenchmarkTools
#using BSON: @save, @load

function graf_predictions(data::Data, solution, labels::Matrix{String})
    xs = map(x -> round(Int, x),solution.t)
    ys = [solution[3,:], data.infec[xs]]
    plot(xs,ys, label=labels)
end

const t0, tMAX, t_iCFR, t_θ0, t_η = 0, 446, 23, 29, 13
const saved = Dict{String,Float64}(
    "γ_d" => 13.123753454738198,
    #"γ_E" => 1/5.5,
    #"γ_I" => 1/5,
    "m0" => 1.0,
    "m1" => 1.0,
    "m4" => 0.0,
    "ω" => 0.014555,
    "ω_0" =>0.50655
)

const lambdas = [
    "2020-03-20",
    "2020-03-23",
    "2020-03-30",
    "2020-04-21",
    "2020-06-01"
]

labels = ["Model" "Real data"]

# Get the data
path = "D:\\Code\\[Servicio Social]\\Datos\\Casos_Modelo_Theta_final_complemented.csv"
const data = Data(path)

const dates = day_to_index(lambdas, data)

# Time interval and intermediary points
tspan = (1.0, 446.0)
tsteps = 1.0:1.0:446.0

# Inityal conditions
N = Float64(data.population[Int(tspan[1])])
u0 = [
    data.susceptible[Int(tspan[1])],
    data.exposed[Int(tspan[1])],
    data.infec[Int(tspan[1])], # Infected 
    data.infec_u[Int(tspan[1])], # Infected undetected
    #data.infec_u[Int(tspan[1])]*0.3, # Infected undetected that will die
    data.hospitalized[Int(tspan[1])],
    data.hospitalized[Int(tspan[1])]*0.3,
    data.quarentine[Int(tspan[1])],
    data.recovered[Int(tspan[1])],
    data.recovered[Int(tspan[1])]*1.3,
    #data.dead[Int(tspan[1])]*0.3,
    data.dead[Int(tspan[1])]
]

# θ-M equation parameters. 
γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, γ_Q = 5.5, 5.0, 9.0, 14.2729, 5.0, 36.0450
β_I0, c_E, c_u, ρ0, k2, c3, c5, ω_u0 =  0.4992, 0.3806, 0.3293, 0.7382, 1.0, 1.0, 1.0, 0.42
β_e0, β_I0_min = (c_E, c_u) .* β_I0
γ_d = trunc(Int64,get(saved, "γ_d", 0))
ω = get(saved, "ω", 0.0)

ms = Msλs(dates, [k2], [c3, c5])

ω_0 = get(saved, "ω_0") do
    get_ω(t_θ0, ms, λs, max_ω, min_ω)
end
ω_CFR0 = get(saved, "ω_CFR0") do
    get_ω_CFR(Int(t_θ0), data, Int(t_iCFR), γ_d)
end
θ_0 = get(saved, "θ0") do
    ω_0 / ω_CFR0
end

ω_CFR = get_ω_CFR(1, data, t_iCFR, γ_d)
θ = 1 <= t_θ0 ? ω_0 / ω_CFR0 : ω / ω_CFR
ρ = get_ρ(ω_0, ω, θ_0, θ, ρ0)
η = get_η(1, data, t0, tMAX, t_η, γ_E, γ_I)

p = [
    γ_d, γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, γ_Q, # gammas
    k2, c3, c5,
    ω_0, ω_CFR0, θ_0, ω,
    β_I0, c_E, c_u, ρ0, ω_u0
]

# Set up the ODE problem
prob = ODEProblem(ThetaModel!, u0, tspan, p)

# AutoVern7(Rodas5())
sol = solve(prob, tstops=tsteps, unstable_check=check)

@benchmark solve(prob, tstops=tsteps, unstable_check=check)
λs = Float64.(dates)

function multi_βs(
    time,
    ω, θ, ω_u0, η, ρ,
    ms, λs,
    γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, 
    β_I0, β_e0, β_I0_min
    )
    for t in time[1]:time[2]
        βs(
        t,
        ω, θ, ω_u0, η, ρ,
        ms, λs,
        γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, 
        β_I0, β_e0, β_I0_min
        )
    end
end

function multi_TimeParams(
    time,
    ω, θ, ω_u0, η, ρ,
    ms, λs,
    γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, 
    β_I0, β_e0, β_I0_min
    )
    last_t = trunc(Int64,time[1])
    for t in time[1]:0.1:time[2]
        βs(
        t,
        ω, θ, ω_u0, η, ρ,
        ms, λs,
        γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, 
        β_I0, β_e0, β_I0_min
        )
    end
end

@code_warntype get_β_Iu0(θ, β_I0, β_I0_min)
@benchmark get_β_Iu0(θ, β_I0, β_I0_min)
@code_warntype βs(
    1.0,
    ω, θ, ω_u0, η, ρ,
    ms, λs,
    γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, 
    β_I0, β_e0, β_I0_min
)
@benchmark βs(
    1.0,
    ω, θ, ω_u0, η, ρ,
    ms, λs,
    γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, 
    β_I0, β_e0, β_I0_min
)

function mul_round(a)
    for i in 1.0:0.01:a
        trunc(Int64, i)
    end
end

function adder(a::Int64)
    a + 2 * 3
end

function mul_round2(a)
    for i in 1.0:0.01:a
        round(Int64, i)
    end
end

@benchmark mul_round(10.0)
@benchmark mul_round2(10.0)

for i in 100.0:200.0
    if i in dates
        println(i)
    end
end

# Plot the solution
graf_predictions(data, sol, labels)

callback = function(p, l, pred)
    display(l)
    plt = graf_predictions(data, pred, labels)
    display(plt)
    return false
end

result_ode = DiffEqFlux.sciml_train(
    distance, p,
    ADAM(0.0001),
    cb = callback,
    maxiters=200
)
