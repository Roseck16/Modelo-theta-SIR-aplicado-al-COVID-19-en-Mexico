using DifferentialEquations
include("GetModelParameters.jl")

function ThetaModel!(du::M, u::M, p::M, t::Float64) where {M<:Vector{Float64}}
    S, E, I, Iu, hr, hd = u
    γ_d, γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, γ_Q, k2, c3, c5, ω_0, ω_CFR0, θ_0, ω, β_I0, β_e0, β_I0_min, ρ0, ω_u, θ, η, ρ, τ1, τ2, β_e, β_I, β_Iu, β_hr, β_hd = p

    delays = rounder!(γ_d, γ_E+γ_I)
    actual = round(Int64, t)
    # * Calculate new time-parameters only when the rounded value of *t* has changed
    if t in tsteps::StepRangeLen
        ω, θ, η, ρ, τ1, τ2 = TimeParams(
            actual, data,
            times, delays,
            ρ0, ω_0, ω_CFR0, θ_0, _ω=ω
        )
    end

    # * Calculate new βs only when *t* hits one of the values in *dates*

    if t in dates
        ms = Msλs(
            convert(Float64,times[1]), dates, 
            ms_val,
            ms_index,
            [k2], [c3, c5]
        )
        β_e, β_I, β_Iu, β_hr, β_hd = βs(
            actual, 
            ω, θ, ω_u, η, ρ, 
            ms, dates, 
            γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, 
            β_I0, β_e0, β_I0_min
        )
    end

    # * ODE system
    # du[1] = -(S/N) * b1
    # du[2] = (S/N) * b1 - E/γ_E + τ1 - τ2
    du[1] = -(S / N::Float64) * (β_e * E + β_I * I + β_Iu * Iu + β_hr * hr + β_hd * hd)
    du[2] = (S / N::Float64) * (β_e * E + β_I * I + β_Iu * Iu + β_hr * hr + β_hd * hd) - E/γ_E + τ1 - τ2
    du[3] = E/γ_E - I/γ_I
    du[4] = (1 - θ - ω_u) * I/γ_I - Iu/γ_Iu
    # du[5] = ω_u * I/γ_I
    du[5] = ρ * (θ - ω) * I/γ_I - hr/γ_Hr
    du[6] = ω * I/γ_I - hd/γ_Hd
    # du[7] = (1 - ρ) * (θ - ω) * I/γ_I + hr/γ_Hr - γ_Q * Q
    # du[8] = Q/γ_Q
    # du[9] = Iu/γ_Iu
    # du[10] = hd/γ_Hd
end

# * Model for when the input parameters are ReverseDiff.TrackedReal()
function ThetaModel!(du, u, p, t)
    #S, E, I, Iu, hr, hd, Q, _, _, _ = u
    S, E, I, Iu, hr, hd = u
    γ_d, γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, γ_Q, k2, c3, c5, ω_0, ω_CFR0, θ_0, ω, β_I0, β_e0, β_I0_min, ρ0, ω_u, θ, η, ρ, τ1, τ2, β_e, β_I, β_Iu, β_hr, β_hd = p

    delays = rounder!(γ_d, γ_E+γ_I)
    actual = round(Int64, t)
    # * Calculate new time-parameters only when the rounded value of *t* has changed
    if t in tsteps::StepRangeLen
        ω, θ, η, ρ, τ1, τ2 = TimeParams(
            actual, data,
            times, delays,
            ρ0, ω_0, ω_CFR0, θ_0, _ω=ω
        )
    end

    # * Calculate new βs only when *t* hits one of the values in *dates*

    if t in dates
        ms = Msλs(
            convert(Float64,times[1]), dates, 
            ms_val_tracked,
            ms_index,
            [k2], [c3, c5]
        )
        β_e, β_I, β_Iu, β_hr, β_hd = βs(
            actual, 
            ω, θ, ω_u, η, ρ, 
            ms, dates, 
            γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, 
            β_I0, β_e0, β_I0_min
        )
    end

    # * ODE system
    # du[1] = -(S/N) * b1
    # du[2] = (S/N) * b1 - E/γ_E + τ1 - τ2
    du[1] = -(S / N::Float64) * (β_e * E + β_I * I + β_Iu * Iu + β_hr * hr + β_hd * hd)
    du[2] = (S / N::Float64) * (β_e * E + β_I * I + β_Iu * Iu + β_hr * hr + β_hd * hd) - E/γ_E + τ1 - τ2
    du[3] = E/γ_E - I/γ_I
    du[4] = (1.0 - θ - ω_u) * I/γ_I - Iu/γ_Iu
    # du[5] = ω_u * I/γ_I
    du[5] = ρ * (θ - ω) * I/γ_I - hr/γ_Hr
    du[6] = ω * I/γ_I - hd/γ_Hd
    # du[7] = (1 - ρ) * (θ - ω) * I/γ_I + hr/γ_Hr - γ_Q * Q
    # du[8] = Q/γ_Q
    # du[9] = Iu/γ_Iu
    # du[10] = hd/γ_Hd
end

"""
    check(dt,u,p,t)
If the model finds some of the solutions to be unstable (NaN or Inf), call this function for debugging. It prints the solution values, the time and the parameters used when the solution returned NaN
"""
function check(dt, u, p, t)
    if any(isnan, u) || any(isinf, u)
        println(p)
        println(u)
        println("Tiempo: $t")
        return true
    end
    return false
end

# TODO: update this function according to the model function
function paramsModel(γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, γ_Q, ρ0, k2, c3, c5, β_I0, β_e0, β_I0_min, ω_u0)
    γ_d = 13.123753454738198
    delays = rounder!(γ_d, γ_E+γ_I)
    ω = 0.014555
    ω_0 = 0.50655
    ω_CFR0 = 147.71428571428572
    θ_0 = ω_0 / ω_CFR0
    
    _, θ, η, ρ, τ1, τ2 = TimeParams(
        1, data, 
        times, 
        delays,
        ρ0, ω_0, ω_CFR0, θ_0, _ω=ω
        )
    ρ = get_ρ(ω_0, ω, θ_0, θ, ρ0)
    ms = Msλs(
        convert(Float64,times[1]), dates, 
        ms_val,
        ms_index,
        [k2], [c3, c5]
    )
    β_e, β_I, β_Iu, β_hr, β_hd = βs(
        1, 
        ω, θ, ω_u0, η, ρ, 
        ms, dates,
        γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, 
        β_I0, β_e0, β_I0_min
    )
    return [
        γ_d, γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, γ_Q,
        k2, c3, c5,
        ω_0, ω_CFR0, θ_0, ω,
        β_I0, β_e0, β_I0_min, ρ0, ω_u0,
        θ, η, ρ, τ1, τ2,
        β_e, β_I, β_Iu, β_hr, β_hd
    ]
end

function distance(x)

    # Assing the values in ´x´ to a variable
    γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, γ_Q, β_I0, c_E, c_u, ρ0, k2, c3, c5, ω_u0 = x
    β_e0, β_I0_min = (c_E, c_u) .* β_I0

    # Get some of the parameters that are known
    p = paramsModel(γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, γ_Q, ρ0, k2, c3, c5, β_I0, β_e0, β_I0_min, ω_u0)
    # Solve the ODE problem and calculate the difference between the solution and the real data
    
    sol = solve(
        prob::ODEProblem, 
        p=p, 
        tstops=tsteps::StepRangeLen,
        verbose=false
    )

    days = map(x -> round(Int64, x), sol.t)
    distance = sqrt(sum(abs2, sol[3,:] .- data.infec[days]))
    return distance, sol
end

function graf_predictions(data::Data, solution::OrdinaryDiffEq.ODECompositeSolution, labels::Matrix{String})
    xs = map(x -> round(Int64, x),solution.t)
    ys = [solution[3,:], data.infec[xs]]
    plot(xs,ys, label=labels)
end