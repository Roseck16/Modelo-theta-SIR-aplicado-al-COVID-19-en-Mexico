import CSV:File as fl
using Zygote: @ignore
using Dates, DataFrames

"""
Struct that contains all the data needed for the model.
"""
struct Data{N <: Vector{Int64}}
    
    days::Vector{Date}
    hospitalized::N
    quarentine::N
    imported::N
    exported::N
    infec::N
    infec_u::N
    dead::N
    infec_medic::N
    recovered::N
    population::N
    susceptible::N
    exposed::N

    function Data(path::String)
        data = DataFrame(fl(path))
        new{Vector{Int64}}(
            data."Fecha",
            data."Hospitalizados",
            data."En cuarentena",
            data."Importados",
            data."Exportados",
            data."Positivos",
            data."Positivos no detectados",
            data."Fallecidos",
            data."Positivos medicos",
            data."Recuperados",
            data."Poblacion",
            data."Susceptibles",
            data."Expuestos"
        )
    end
end

function rounder!(vals...)
    round.(Int64, vals)
end

function day_to_index(day::Integer)
    return day
end

function day_to_index(day::AbstractFloat)
    return round(Int64, day)
end

function day_to_index(
    date::String, 
    data::Data,
    out_type=Int64,
    dateformat::T=DateFormat("y-m-d")
    ) where {T <: DateFormat}

    day = Date(date, dateformat)
    return convert(out_type, findfirst(x -> x == day, data.days))
end

function day_to_index(
    date::Vector{String},
    data::Data,
    out_type=Int64,
    dateformat::T=DateFormat("y-m-d")
    ) where {T <: DateFormat}

    return map(x -> day_to_index(x, data,out_type, dateformat), date)
end

function day_to_index(date::Date, data::Data)
    return findfirst(x -> x == date, data.days)
end

function day_to_index(date::Vector{Date}, data::Data)
    return map(x -> day_to_index(x, data), date)
end

function get_t_η(data::Data, delay::M) where M
    q = length(data.infec)
    for index in 1:q
        delay_index = index + delay
        val = delay_index >= q ? data.infec[q] : data.infec[delay_index]
        if val !== 0
            return index
        end
    end
    return 0
end

function Msλs_calculate(ms, start_index, t0, dates, ks)
    k = ks[start_index]
    t = dates[start_index]
    tr = if start_index === 1
        t0
    else
        dates[start_index-1]
    end
    diff_ms = ms[start_index] - ms[start_index + 1]
    ex = exp(-k * (t - tr))
    return diff_ms * ex + ms[start_index + 1]
end

"""
    Msλs_loop(ms, ms_index, dates, ks, cs)
Inner loop of the `Msλs` function

# Arguments
- `ms`
- `ms_index::Vector{Int64}` : Indexes in the `ms` vector of the available m values.
- `dates`
- `ks`
- `cs`
"""

function Msλs_loop!(ms, ms_index, start_index, t0, dates, ks, cs)
    for index in 1:length(ms)
        if index ∉ ms_index
            # These m's depend on another m
            if index === 4
                ms[index] = ms[3] * cs[1]
            elseif index === 6
                ms[index] = ms[3] * cs[2]
            else
                ms[index] = Msλs_calculate(ms, start_index[1], t0, dates, ks)
                start_index[1] += 1
            end
        end
    end
end

function Msλs(t0, dates, ms, ms_index, ks, cs)
    # If we have 4 dates, we need an extra one because we always set m0 = m1 to start with the first date.
    start_index = [1]

    @ignore Msλs_loop!(ms, ms_index, start_index, t0, dates, ks, cs)
    return ms
end

function get_ð_less(t_η::M, data::Data, delay::M) where M
    infec = data.infec
    infec_medic = data.infec_medic
    q = length(infec)

    index = t_η + delay
    # if the index plus the delay is greater than the size of the data vectors, choose the last items
    num = index >= q ? infec_medic[q] : infec_medic[index]
    den = index >= q ? infec[q] : infec[index]
    return num / den
end

function get_ð_greater_loop(q, index, num1, den1, infec, infec_medic)
    for r in 1:q
        den2 = if index - r >= q
            infec[q - r]
        elseif index - r <= 1
            infec[1]
        else
            infec[index - r]
        end
        if (den1 - den2) !== 0
            num2 = if index - (r + 7) >= q
                infec_medic[q - (r + 7)]
            elseif index - (r + 7) <= 1 
                infec_medic[1]
            else
                infec_medic[index - (r + 7)]
            end
            return (num1 - num2) / (den1 - den2)
        end
    end
end

function get_ð_greater(t::M, data::Data, delay::M) where M
    infec = data.infec
    infec_medic = data.infec_medic
    q = length(infec)

    index1 = t + delay
    num1 = index1 >= q ? infec_medic[q] : infec_medic[index1]
    den1 = index1 >= q ? infec[q] : infec[index1]
    return get_ð_greater_loop(q, index1, num1, den1, infec, infec_medic)
end
    
function get_η_case1!(t0, t_η, data, delay)
    total = 0.0
    for i in 0:6
        total += if t0 + i <= t_η
            get_ð_less(t_η, data, delay)
        else
            get_ð_greater(t0 + i, data, delay)
        end
    end
    return total
end

function get_η_case2!(t, t_η, data, delay)
    total = 0.0
    for i in -3:3
        total += if t + i <= t_η
            get_ð_less(t_η, data, delay)
        else
            get_ð_greater(t + i, data, delay)
        end
    end
    return total
end

function get_η_case3!(t0, tMAX, t_η, data, delay)
    total = 0.0
    for i in 0:6
        total += if t0 + tMAX - i <= t_η
            get_ð_less(t_η, data, delay)
        else
            get_ð_greater(t0 + tMAX - i, data, delay)
        end
    end
    return total
end

function get_η(t::M, data::Data, t0::M, tMAX::M, t_η::M, delay::M) where M
    _sum = if t < t0 + 3
        get_η_case1!(t0, t_η, data, delay)
    elseif t0 + 3 <= t && t <= t0 + tMAX - 3
        get_η_case2!(t, t_η, data, delay)
    elseif t > t0 + tMAX - 3
        get_η_case3!(t0, tMAX, t_η, data, delay)
    else
        error("None of the 't' values matched in 'get_n': $(t)")
    end

    return _sum / 7.0
end

"""
    get_ρ(ω_0, ω, θ_0, θ, ρ0)

# Arguments
- `ω_0::Float64` : Value of ω_0.
- `ω::Float64` : Value of ω.
- `θ_0::Float64` : Value of θ_0.
- `θ::Float64` : Value of θ.
- `ρ0::Float64` : Value of ρ0.

"""
function get_ρ(ω_0, ω, θ_0, θ, ρ0)
    dif = θ - ω
    dif_0 = θ_0 - ω_0

    if dif >= dif_0
        ρ0 * dif_0 / dif
    elseif dif < dif_0
        1 - ((1 - ρ0) / dif_0) * dif
    else
        error("Value error: $dif, $dif_0")
    end
end

function get_iCFR_less(t_iCFR::M, data::Data, γ_d::M) where M
    infec = data.infec
    dead = data.dead
    delay = γ_d
    q = length(dead)

    index = t_iCFR + delay
    d_r = index >= q ? dead[q] : dead[index]
    c_r = infec[t_iCFR]
    return d_r / c_r
end

function get_iCFR_greater_loop(q, t, index, infec, dead)
    for r in 1:q
        den = infec[t] - infec[t - r]
        if den !== 0
            num2 = if index - r >= q
                dead[q]
            elseif index - r <= 1
                dead[1]
            else
                dead[index - r]
            end
            return den, num2
        end
    end
end

function get_iCFR_greater(t::M, data::Data, γ_d::M) where M
    infec = data.infec
    dead = data.dead
    delay = γ_d
    q = length(dead)

    index1 = t + delay
    num1 = index1 >= q ? dead[q] : dead[index1]
    den, num2 = get_iCFR_greater_loop(q, t, index1, infec, dead)
    return (num1 - num2) / den
end

function get_ω_CFR_loop1!(days_range,_sum, t, data, t_iCFR, γ_d)
    for i in 0:days_range[1]
        _sum[1] += if t - i <= t_iCFR
            get_iCFR_less(t_iCFR, data, γ_d)
        else
            get_iCFR_greater(t - i, data, γ_d)
        end
    end
end

function get_ω_CFR_loop2!(t, data, t_iCFR, γ_d)
    days_range = [6]
    while true
        _sum = [0.0]
        get_ω_CFR_loop1!(days_range,_sum, t, data, t_iCFR, γ_d)
        result = _sum[1] / Float64(days_range[1] + 1)
        if result <= 0.015
            days_range[1] += 1.0
        else
            return result
        end
    end
end

function get_ω_CFR(t::M, data::Data, t_iCFR::M, γ_d::M) where M
    if t <= t_iCFR
        return get_iCFR_less(t_iCFR, data, γ_d)
    else
        return get_ω_CFR_loop2!(t, data, t_iCFR, γ_d)
    end
end

function get_ω_loop(q, t, ms, λs)
    for i in 1:q-1
        if t <= λs[i + 1]
            return index <= 1 ? ms[1] : ms[i - 1]
        elseif i + 1 >= q
            return ms[end]
        end
    end
end

function get_ω(t::M, ms::N, λs::N, max_ω::O, min_ω::O) where {M,N,O}
    q = length(λs)
    m = get_ω_loop(q, t, ms, λs)
    return (m * max_ω) + ((1.0 - m) * min_ω)
end

"""
    TimeParams(t, data, times, delays, ρ0, ω_0, ω_CFR0, θ_0; _ω, ms, λs, max_ω, min_ω)

Calculate time-dependent parameters. If the value of _ω is not given, then it would be necessary to input the extra parameters ms, λs, max_ω, min_ω.

# Arguments
- `times::Vector{Int64}` : Vector with the values of t0, tMAX, t_iCFR, t_θ0, t_η
- `delays::Vector{Int64}` : Vector with the values of γ_d and γ_E + γ_I
"""

function TimeParams(
    t::M, data::Data, times, delays, 
    ρ0::N, # This parameter is optimized
    ω_0::N, ω_CFR0::N, θ_0::N; # These parameters can be calculated from others
    _ω=nothing, # If the value of ω is given, these 
    ms=nothing, # parameters are optional
    λs=nothing,
    max_ω=nothing,
    min_ω=nothing
    ) where {M,N}
    ω = isnothing(_ω) ? get_ω(t, ms, λs, max_ω, min_ω) : _ω
    ω_CFR = get_ω_CFR(t, data, times[3], delays[1])
    θ = t <= times[4] ? ω_0 / ω_CFR0 : ω / ω_CFR
    η = get_η(t, data, times[1], times[2], times[5], delays[2])
    ρ = get_ρ(ω_0, ω, θ_0, θ, ρ0)
    
    return ω, θ, η, ρ, 0, data.imported[t]
end

function index_βs(t, ms, dates)
    for i in 1:length(dates)
        if t <= dates[i + 1]
            return i <= 1 ? ms[1] : ms[i - 1]
        elseif i + 1 >= length(dates)
            return ms[end]
        end
    end
end

function get_β_Iu0(θ, val1, val2)
    θ >= 0.0 || error("θ is not positive")
    θ < 1.0 && return val1
    θ === 1.0 && return val2
    error("θ greater than 1")
end

# βs(
#     ::Int64, 
#     ::ReverseDiff.TrackedReal{Float64, Float64, ReverseDiff.TrackedArray{Float64, Float64, 1, Vector{Float64}, Vector{Float64}}}, 
#     ::ReverseDiff.TrackedReal{Float64, Float64, Nothing}, 
#     ::ReverseDiff.TrackedReal{Float64, Float64, ReverseDiff.TrackedArray{Float64, Float64, 1, Vector{Float64}, Vector{Float64}}}, 
#     ::Float64, 
#     ::ReverseDiff.TrackedReal{Float64, Float64, Nothing}, 
#     ::Vector{ReverseDiff.TrackedReal{Float64, Float64, Nothing}}, 
#     ::Vector{Float64}, 
#     ::ReverseDiff.TrackedReal{Float64, Float64, ReverseDiff.TrackedArray{Float64, Float64, 1, Vector{Float64}, Vector{Float64}}}, 
#     ::ReverseDiff.TrackedReal{Float64, Float64, ReverseDiff.TrackedArray{Float64, Float64, 1, Vector{Float64}, Vector{Float64}}}, 
#     ::ReverseDiff.TrackedReal{Float64, Float64, ReverseDiff.TrackedArray{Float64, Float64, 1, Vector{Float64}, Vector{Float64}}}, 
#     ::ReverseDiff.TrackedReal{Float64, Float64, ReverseDiff.TrackedArray{Float64, Float64, 1, Vector{Float64}, Vector{Float64}}}, 
#     ::ReverseDiff.TrackedReal{Float64, Float64, ReverseDiff.TrackedArray{Float64, Float64, 1, Vector{Float64}, Vector{Float64}}}, 
#     ::ReverseDiff.TrackedReal{Float64, Float64, ReverseDiff.TrackedArray{Float64, Float64, 1, Vector{Float64}, Vector{Float64}}}, 
#     ::ReverseDiff.TrackedReal{Float64, Float64, ReverseDiff.TrackedArray{Float64, Float64, 1, Vector{Float64}, Vector{Float64}}}, 
#     ::ReverseDiff.TrackedReal{Float64, Float64, ReverseDiff.TrackedArray{Float64, Float64, 1, Vector{Float64}, Vector{Float64}}}
# )

function βs(
    t, 
    ω, θ, ω_u0, η, ρ, 
    ms, λs,
    γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, 
    β_I0, β_e0, β_I0_min
    # β_I0::N, c_E::N, c_u::N
    )

    # β_e0, β_I0_min = (c_E, c_u) .* β_I0

    m = index_βs(t, ms, λs)
    β_Iu0 = get_β_Iu0(θ, β_I0, β_I0_min)

    β_e, β_I, β_Iu = (β_e0, β_I0, β_Iu0) .* m

    num = η * (β_e * γ_E + β_I * γ_I + (1.0 - θ - ω_u0) * β_Iu * γ_Iu + ω_u0)

    den = (1.0 - η) * (ρ * (θ - ω) * γ_Hr + ω * γ_Hd)

    β_hr = num / den

    return β_e, β_I, β_Iu, β_hr, β_hr
end