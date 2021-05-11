using DifferentialEquations, Evolutionary
using DifferentialEquations: Rosenbrock23
include("GetModelParameters.jl")

function day_to_index(day::Integer)
    return day
end

function day_to_index(day::AbstractFloat)
    return round(Int, day)
end

function day_to_index(
    date::String, 
    data::Data,
    dateformat::T=DateFormat("y-m-d")
    ) where {T<:DateFormat}

    day = Date(date, dateformat)
    return findfirst(x -> x==day, data.days)
end

function day_to_index(
    date::Vector{String},
    data::Data,
    dateformat::T=DateFormat("y-m-d")
    ) where {T<:DateFormat}

    return map(x -> day_to_index(x, data, dateformat), date)
end

function day_to_index(date::Date, data::Data)
    return findfirst(x -> x==date, data.days)
end

function day_to_index(date::Vector{Date}, data::Data)
    return map(x -> day_to_index(x, data), date)
end


#struct ThetaModel
#    solution::Any

# function ode!(du, u, p, t)
#     S, E, I, Iu, IDu, hr, hd, Q, Rd, Ru, Du, D = u
#     γ_d, γ_E, γ_I, γ_Iu, γ_IDu, γ_Hr, γ_Hd, γ_Q, θ, ω, ω_u, ρ, τ1, τ2, β_e, β_I, β_Iu, β_IDu, β_hr, β_hd, N = p[1](t)

#     du[1] = -(S/N) * (β_e * E + β_I * I + β_Iu * Iu + β_IDu * IDu + β_hr * hr + β_hd * hd)
#     du[2] = (S/N) * (β_e * E + β_I * I + β_Iu * Iu + β_IDu * IDu + β_hr * hr + β_hd * hd) - γ_E * E + τ1 - τ2
#     du[3] = γ_E * E - γ_I * I
#     du[4] = (1 - θ - ω_u) * γ_I * I - γ_Iu * Iu
#     du[5] = ω_u * γ_I * I - γ_IDu * IDu
#     du[6] = ρ * (θ - ω) * γ_I * I - γ_Hr * hr
#     du[7] = ω * γ_I * I - γ_Hd * hd
#     du[8] = (1 - ρ) * (θ - ω) * γ_I * I + γ_Hr * hr - γ_Q * Q
#     du[9] = γ_Q * Q
#     du[10] = γ_Iu * Iu
#     du[11] = γ_IDu * IDu
#     du[12] = γ_Hd * hd
# end

function ode!(du, u, p, t)
    fun_params = p[1]
    opt_params = p[2]
    saved_params = p[3]
    γ_d, γ_E, γ_I, γ_Iu, γ_IDu, γ_Hr, γ_Hd, γ_Q, θ, ω, ω_u, ρ, τ1, τ2, β_e, β_I, β_Iu, β_IDu, β_hr, β_hd, N = fun_params(t, opt_params ,saved_params)

    S, E, I, Iu, IDu, hr, hd, Q, Rd, Ru, Du, D = u

    du[1] = -(S/N) * (β_e * E + β_I * I + β_Iu * Iu + β_IDu * IDu + β_hr * hr + β_hd * hd)
    du[2] = (S/N) * (β_e * E + β_I * I + β_Iu * Iu + β_IDu * IDu + β_hr * hr + β_hd * hd) - γ_E * E + τ1 - τ2
    du[3] = γ_E * E - γ_I * I
    du[4] = (1 - θ - ω_u) * γ_I * I - γ_Iu * Iu
    du[5] = ω_u * γ_I * I - γ_IDu * IDu
    du[6] = ρ * (θ - ω) * γ_I * I - γ_Hr * hr
    du[7] = ω * γ_I * I - γ_Hd * hd
    du[8] = (1 - ρ) * (θ - ω) * γ_I * I + γ_Hr * hr - γ_Q * Q
    du[9] = γ_Q * Q
    du[10] = γ_Iu * Iu
    du[11] = γ_IDu * IDu
    du[12] = γ_Hd * hd
end

function check(dt,u,p,t)
    if any(isnan, u) || any(isinf, u)
        println(p[2])
        println(u)
        println("Tiempo: $t")
        println(p[1](t, p[2], p[3]))
        return true
    end
    return false
end

function sol(f::Function, u0::M, tspan::N, p::Vector{Any}) where {V<:Float64, M<:Vector{V}, N<:Tuple{V,V}}
    prob = ODEProblem(f, u0, tspan, p)
    return solve(prob,Rosenbrock23(autodiff=false), adaptive=false, dt=1.0, unstable_check=check)
end

function distance(x::Vector{Float64})
    γ_Iu, γ_E, γ_I, γ_IDu, γ_Q, γ_Hr, γ_Hd, β_I0, c_E, c_u, c_IDu, ρ0, k2, c3, c5, ω_u0 = x
    optim = Dict(
        "γ_Iu" => γ_Iu,
        "γ_Q" => γ_Q,
        "γ_Hr" => γ_Hr,
        "γ_Hd" => γ_Hd,
        "γ_E" => γ_E,
        "γ_I" => γ_I,
        "γ_IDu" => γ_IDu,
        "β_I0" => β_I0,
        "c_E" => c_E,
        "c_u" => c_u,
        "c_IDu" => c_IDu,
        "ρ0" => ρ0,
        "ks" => [k2],
        "cs" => [c3,c5],
        "ω_u0" => ω_u0
    )

    #q = convert(Float64,length(data.days))
    
    infected = sol(ode!, u0, tspan, [full_params, optim, saved])
    
    distance = sqrt(
        sum(
            (infected[3,:] .- data.infec) .^ 2
        )
    )
    ## this operation returns a result even if the 
    ## dimensions are different
    # lazy_distance = sqrt(
    #     sum(
    #         map((x,y) -> (x-y)^2, infected[3,:], data.infec)
    #     )
    # )
    return distance
end

function minim(x0::M, low::M, up::M, parameters::GA) where {M<:Vector{Float64}}
    return Evolutionary.optimize(distance, low, up, x0, parameters)
end

