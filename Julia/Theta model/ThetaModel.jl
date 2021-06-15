using DifferentialEquations
#include("GetModelParameters.jl")

# * Model for when the input parameters are ReverseDiff.TrackedReal
function ThetaModel!(du, u, p, t)
    #S, E, I, Iu, hr, hd, Q, _, _, _ = u
    S, E, I, Iu, hr, hd = u
    
    # * Get the time parameters from *p*
    actual = round(Int64, t)
    
    γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, γ_Q = p[actual,1], p[actual,2], p[actual,3], p[actual,4], p[actual,5], p[actual,6]

    ω_u = p[actual, 10]

    ω, τ1, τ2, θ, ρ = p[actual, 7], p[actual,8], p[actual,9], p[actual,11], p[actual,12]
    β_e, β_I, β_Iu, β_hr, β_hd = p[actual,13], p[actual,14], p[actual,15], p[actual,16], p[actual,17]

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

function distance(x)

    #* Assing the values in ´x´ to a variable
    γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, γ_Q, β_I0, c_E, c_u, ρ0, k2, c3, c5, ω_u0, max_ω, min_ω = x
    β_e0, β_I0_min = (c_E, c_u) .* β_I0

    #* Get some of the parameters that are known
    delays = Delays(γ_Hd, γ_E + γ_I)
    times = Times(tspan::Tuple{Float64,Float64}, data, delays)

    ms = Msλs(
        convert(Float64,times.t0), dates, 
        ms_val,
        [k2], [c3, c5]
    )
    p = parameters_lists(times, data, delays, ms, dates, max_ω, min_ω, ω_u0, ρ0, β_I0, β_e0, β_I0_min, γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, γ_Q)
    # Solve the ODE problem and calculate the difference between the solution and the real data
    
    sol = solve(
        prob::ODEProblem, 
        p=p,
        #unstable_check=check,
        tstops=tsteps::StepRangeLen,
        verbose=false
    )

    days = map(x -> round(Int64, x), sol.t)
    distance = sqrt(sum(abs2, sol[3,:] .- data.infec[days]))
    return distance, sol
end

function graf_predictions(data::Data, solution, labels::Matrix{String})
    xs = map(x -> round(Int64, x),solution.t)
    ys = [solution[3,:], data.infec[xs]]
    plot(xs,ys, label=labels)
end