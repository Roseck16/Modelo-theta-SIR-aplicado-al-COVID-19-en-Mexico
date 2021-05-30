using DifferentialEquations
include("GetModelParameters.jl")

function ThetaModel!(du::M, u::M, p::M, t::Float64) where {M<:Vector{Float64}}
    S, E, I, Iu, hr, hd, Q, _, _, _ = u
    γ_d, γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, γ_Q, k2, c3, c5, ω_0, ω_CFR0, θ_0, ω, β_I0, c_E, c_u, ρ0, ω_u = p

    #t >= 2.0 ? error("Its time") : println("Hello")
    ms = Msλs(dates, [(k2)], [(c3), (c5)])
    ω, θ, η, ρ, τ1, τ2 = TimeParams(
        trunc(Int64, t), data,
        t0, tMAX, t_iCFR, t_θ0, t_η,
        trunc(Int64,γ_d), trunc(Int64,γ_E), trunc(Int64,γ_I),
        ρ0,
        ω_0, ω_CFR0, θ_0, _ω=ω
        )
    
    β_e, β_I, β_Iu, β_hr, β_hd = βs(
        trunc(Int64, t),
        ω, θ, ω_u, η, ρ,
        ms, Float64.(λs),
        trunc(Int64,γ_E), trunc(Int64,γ_I), trunc(Int64,γ_Iu), trunc(Int64,γ_Hr), trunc(Int64,γ_Hd), 
        β_I0, c_E, c_u
        )
        # b1 = β_e * E + β_I * I + β_Iu * Iu + β_IDu * IDu + β_hr * hr + β_hd * hd
    
    # ODE system
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

function ThetaModel!(du, u, p, t)
    S, E, I, Iu, hr, hd, Q, _, _, _ = u
    γ_d, γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, γ_Q, k2, c3, c5, ω_0, ω_CFR0, θ_0, ω, β_I0, c_E, c_u, ρ0, ω_u = p

    t >= 1.5 ? error("Its time") : display(typeof(p))
    ms, λs = Msλs(dates, [ReverseDiff.value(k2)], [ReverseDiff.value(c3), ReverseDiff.value(c5)])
    
    ω, θ, η, ρ, τ1, τ2 = TimeParams(
        ReverseDiff.value(t), data,
        ReverseDiff.value(t0), ReverseDiff.value(tMAX), ReverseDiff.value(t_iCFR), ReverseDiff.value(t_θ0), ReverseDiff.value(t_η),
        ReverseDiff.value(γ_d), ReverseDiff.value(γ_E), ReverseDiff.value(γ_I),
        ReverseDiff.value(ρ0),
        ReverseDiff.value(ω_0), ReverseDiff.value(ω_CFR0), ReverseDiff.value(θ_0), _ω=ReverseDiff.value(ω)
        )
    
    β_e, β_I, β_Iu, β_hr, β_hd = βs(
        ReverseDiff.value(t),
        ReverseDiff.value(ω), ReverseDiff.value(θ), ReverseDiff.value(ω_u), ReverseDiff.value(η), ReverseDiff.value(ρ),
        ReverseDiff.value(ms), ReverseDiff.value(λs),
        ReverseDiff.value(γ_E), ReverseDiff.value(γ_I), ReverseDiff.value(γ_Iu), ReverseDiff.value(γ_Hr), ReverseDiff.value(γ_Hd), 
        ReverseDiff.value(β_I0), ReverseDiff.value(c_E), ReverseDiff.value(c_u)
        )
        # b1 = β_e * E + β_I * I + β_Iu * Iu + β_IDu * IDu + β_hr * hr + β_hd * hd
    
    # ODE system
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

function distance(x)
    
    # Assing the values in ´x´ to a variable
    γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, γ_Q, β_I0, c_E, c_u, ρ0, k2, c3, c5, ω_u0 = x
    
    # Get some of the parameters that are known
    ω = get(saved, "ω", 0.0)::Float64
    γ_d = trunc(Int64,get(saved, "γ_d", 0))
    
    # These are non time-dependent parameters that we can already calculate or retrieve from a Dictionary
    ω_0 = get(saved, "ω_0") do
        get_ω(t_θ0, ms, λs, max_ω, min_ω)
    end
    ω_CFR0 = get(saved, "ω_CFR0") do
        get_ω_CFR(t_θ0, data, t_iCFR, γ_d)
    end
    θ_0 = get(saved, "θ0") do
        get_θ(t_θ0, t_θ0, ω_0, ω, ω_CFR0, 0.0)
    end
    
    p = [
        γ_d, γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, γ_Q, # gammas
        k2, c3, c5,
        ω_0, ω_CFR0, θ_0, ω, # Arguments for time parameters
        β_I0, c_E, c_u, ρ0, ω_u0
    ]
    # Solve the ODE problem and calculate the difference between the solution and the real data
    
    sol = solve(prob, p=p, tstops=tsteps, unstable_check=check)
    
    days = map(x -> trunc(x), sol.t)
    distance = sqrt(sum(abs2, sol[3,:] .- data.infec[days]))
    return distance, sol
end