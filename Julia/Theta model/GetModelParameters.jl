import CSV:File as fl
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
    return trunc(day)
end

function day_to_index(
    date::String, 
    data::Data,
    dateformat::T=DateFormat("y-m-d")
    ) where {T <: DateFormat}

    day = Date(date, dateformat)
    return findfirst(x -> x == day, data.days)
end

function day_to_index(
    date::Vector{String},
    data::Data,
    dateformat::T=DateFormat("y-m-d")
    ) where {T <: DateFormat}

    return map(x -> day_to_index(x, data, dateformat), date)
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

function Msλs(dates::Vector{Int64}, ks::M, cs::M) where M
    # If we have 4 dates, we need an extra one because we always set m0 = m1 to start with the first date.
    q = length(dates) + 1
    start_index = 1

    ms = zeros(q)

    for index in 1:q
        m = "m$(index - 1)"
        if haskey(saved, m)::Bool
            ms[index] = get(saved, m, 0.0)::Float64
        elseif index in [4, 6]
            ms[index] = cs[1] * ms[3]
        else
            k = ks[start_index]
            t = Float64(dates[start_index])
            if start_index == 1
                tr = get(saved, "t0", 0.0)::Float64
            else
                tr = Float64(dates[start_index - 1])
            end

            diff_ms = ms[start_index] - ms[start_index + 1]

            ex = exp(-k * (t - tr))::Float64
            ms[index] = diff_ms * ex + ms[start_index + 1]

            start_index += 1
        end
    end
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
    
function get_η_case1!(_sum, t0, t_η, data, delay)
    for i in 0:6
        _sum[1] += if t0 + i <= t_η
            get_ð_less(t_η, data, delay)
        else
            get_ð_greater(t0 + i, data, delay)
        end
    end
end

function get_η_case2!(_sum, t, t_η, data, delay)
    for i in -3:3
        _sum[1] += if t + i <= t_η
            get_ð_less(t_η, data, delay)
        else
            get_ð_greater(t + i, data, delay)
        end
    end
end

function get_η_case3!(_sum, t0, tMAX, t_η, data, delay)
    for i in 0:6
        _sum[1] += if t0 + tMAX - i <= t_η
            get_ð_less(t_η, data, delay)
        else
            get_ð_greater(t0 + tMAX - i, data, delay)
        end
    end
end

function get_η(t::M, data::Data, t0::M, tMAX::M, t_η::M, delay::M) where M
    _sum = [0.0]
    if t < t0 + 3
        get_η_case1!(_sum, t0, t_η, data, delay)
    elseif t0 + 3 <= t && t <= t0 + tMAX - 3
        get_η_case2!(_sum, t, t_η, data, delay)
    elseif t > t0 + tMAX - 3
        get_η_case3!(_sum, t0, tMAX, t_η, data, delay)
    else
        error("None of the 't' values matched in 'get_n': $(t)")
    end

    return _sum[1] / 7.0
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
function get_ρ(ω_0::M, ω::M, θ_0::M, θ::M, ρ0::M) where M
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
    for i in 0.0:days_range[1]
        _sum[1] += if t - i <= t_iCFR
            get_iCFR_less(t_iCFR, data, γ_d)
        else
            get_iCFR_greater(t - i, data, γ_d)
        end
    end
end

function get_ω_CFR_loop2!(t, data, t_iCFR, γ_d)
    days_range = [6.0]
    while true
        _sum = [0.0]
        get_ω_CFR_loop1!(days_range,_sum, t, data, t_iCFR, γ_d)
        result = _sum[1] / (days_range[1] + 1.0)
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

function get_ω(t::M, ms::N, λs::N, max_ω::O, min_ω::O) where {M,N,O}
    q = length(λs)
    m = get_ω_loop(q, t, ms, λs)
    return (m * max_ω) + ((1.0 - m) * min_ω)
end

function TimeParams(
    t::M, data::Data,
    t0::M, tMAX::M, t_iCFR::M, t_θ0::M, t_η::M,
    γ_d::M, γ_E::M, γ_I::M, 
    ρ0::N, # This parameter is optimized
    ω_0::N, ω_CFR0::N, θ_0::N; # These parameters can be calculated from others
    _ω=nothing, # If the value of ω is given, these 
    ms=nothing, # parameters are optional
    λs=nothing,
    max_ω=nothing,
    min_ω=nothing
    ) where {M,N}
    delay1 = round(Int, γ_E + γ_I)
    ω = isnothing(_ω) ? get_ω(t, ms, λs, max_ω, min_ω) : _ω
    ω_CFR = get_ω_CFR(t, data, t_iCFR, γ_d)
    θ = t <= t_θ0 ? ω_0 / ω_CFR0 : ω / ω_CFR
    η = get_η(t, data, t0, tMAX, t_η, delay1)
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

function βs(
    t::M, 
    ω::M, θ::M, ω_u0::M, η::M, ρ::M, 
    ms::O, λs::O,
    γ_E::M, γ_I::M, γ_Iu::M, γ_Hr::M, γ_Hd::M, 
    β_I0::M, β_e0::M, β_I0_min::M
    # β_I0::N, c_E::N, c_u::N
    ) where {M,O}

    # β_e0, β_I0_min = (c_E, c_u) .* β_I0

    m = index_βs(t, ms, λs)
    β_Iu0 = get_β_Iu0(θ, β_I0, β_I0_min)

    β_e, β_I, β_Iu = (β_e0, β_I0, β_Iu0) .* m

    num = η * (β_e * γ_E + β_I * γ_I + (1.0 - θ - ω_u0) * β_Iu * γ_Iu + ω_u0)

    den = (1.0 - η) * (ρ * (θ - ω) * γ_Hr + ω * γ_Hd)

    β_hr = num / den

    return β_e, β_I, β_Iu, β_hr, β_hr
end