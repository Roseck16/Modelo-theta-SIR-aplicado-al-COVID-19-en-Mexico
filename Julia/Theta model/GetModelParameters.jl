using Zygote: @ignore
using CSV, Dates, DataFrames, Interpolations

"""
Struct that contains all the data needed for the model.
"""
struct Data{N <: Vector{Int64}}
    days::Vector{Dates.Date}
    hospitalized::N
    quarentine::N
    imported::N
    exported::N
    infec::N
    infec_u::N
    dead::N
    infec_medic
    recovered::N
    population::N
    susceptible::N
    exposed::N

    function Data(path::String)
        data = DataFrame(CSV.File(path))
        new{Vector{Int64}}(
            data."Date",
            data."Hospitalized",
            data."Quarentine",
            data."Imported",
            data."Exported",
            data."Positive",
            data."Positive_undetected",
            data."Dead",
            data."Positive_medic",
            data."Recovered",
            data."Population",
            data."Susceptible",
            data."Exposed"
        )
    end
end

struct Times
    t0
    tMAX
    t_η
    t_iCFR
    t_θ0
    function Times(tspan, data, delays)
        t_iCFR = get_t_iCFR(data, delays.γ_Hd)
        t_η = get_t_η(data, delays.γ_E_I)
        new(
            Int(tspan[1]), Int(tspan[2]),
            t_η, t_iCFR, t_iCFR+6
        )
    end
end

struct Delays
    γ_Hd
    γ_E_I
    function Delays(γ_Hd, γ_E_I)
        new(round(Int64, γ_Hd), round(Int64,γ_E_I))
    end
end

#region day_to_index methods
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
#endregion

function get_t_η(data::Data, delay::M) where M
    q = length(data.infec)
    for index in 1:q
        delay_index = index + delay
        val1 = delay_index >= q ? data.infec_medic[q] : data.infec_medic[delay_index]
        val2 = delay_index >= q ? data.infec[q] : data.infec[delay_index]
        if val1 !== missing && val1 !== 0  && val2 !== 0
            return index
        end
    end
    return 0
end

function get_t_iCFR(data, delay)
    q = length(data.infec)
    for i in 7:q
        val1 = data.infec[i] - data.infec[i-1]
        val2 = ifelse(i + delay >= q, data.dead[q], data.dead[i + delay])
        if val1 !== 0 && val2 !== 0
            return i
        end
    end
end

function Msλs(t0, dates, ms, ks, cs)
    # If we have 4 dates, we need an extra one because we always set m0 = m1 to start with the first date.
    m3 = Msλs_calculate(ms, 1, t0, dates, ks)
    ms_real = [
        ms[1],
        ms[2],
        m3 * cs[1],
        ms[5],
        m3 * cs[2]
    ]
    return ms_real
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

function get_ω(t::M, ms::N, λs, max_ω::O, min_ω::O) where {M,N,O}
    q = length(λs)
    m = get_ω_loop(q, t, ms, λs)
    return (m * max_ω) + ((1.0 - m) * min_ω)
end

function get_ω_loop(q, t, ms, λs)
    for i in 1:q-1
        if t <= λs[i + 1]
            return i <= 1 ? ms[1] : ms[i - 1]
        elseif i + 1 >= q
            return ms[end]
        end
    end
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
        1.0 - ((1.0 - ρ0) / dif_0) * dif
    else
        error("Value error: $dif, $dif_0")
    end
end

function get_ρs(i, times::Times, ω_0, θ_0, ωs, θs, ρ0, ρs)
    new_ρs = vcat(ρs, get_ρ(ω_0, ωs[i], θ_0, θs[i], ρ0))
    if i+1 > times.tMAX
        return new_ρs
    else
        get_ρs(i+1, times, ω_0, θ_0, ωs, θs, ρ0, new_ρs)
    end
end

"""
    get_ω_CFR(t, data, t_iCFR, tMAX, delay, iCFRs)

Calculate the value of ω_CFR for the given time. Method divided in various functions for rapid computation.

# Arguments
- t: Time to calculate the value of ω_CFR
- data: All the data available
- t_iCFR: Value of t_iCFR
- tMAX: Maximum time
- delay: Value to use for the delay
- iCFRs: Vector of all the iCFR values

# SubFunctions
-   get_iCFR_less(t_iCFR, data, delay)
-   get_ω_CFR_loop(t, data, t_iCFR, tMAX, delay, iCFRs)
-   get_ω_CFR_sum(days_range, t, data, t_iCFR, t0, tMAX, delay, iCFRs)
-   get_iCFR_greater(t, data, delay)
-   get_iCFR_loop(t, infec, dead, delay)
-   get_t_iCFR(data, γ_d)
"""
function get_ω_CFR(t, t_iCFR, iCFRs)
    if t <= t_iCFR
        return iCFRs[t_iCFR]
    else
        return get_ω_CFR_loop(t, iCFRs)
    end
end

function get_ω_CFRs(i, times::Times, iCFRs, ω_CFRs)
    new_ω_CFRs = vcat(ω_CFRs, get_ω_CFR(i, times.t_iCFR, iCFRs))
    if i+1 > times.tMAX
        return new_ω_CFRs
    else
        get_ω_CFRs(i+1, times, iCFRs, new_ω_CFRs)
    end
end

#region #* Methods to calculate the ω_CFR value
function get_ω_CFR_loop(t, iCFRs)
    days_range = 6
    while true
        _sum = get_ω_CFR_sum(days_range, t, iCFRs)
        result = _sum / Float64(days_range + 1)        
        if result <= 0.01
            days_range += 1
        else
            return result
        end
    end
end

function get_ω_CFR_sum(days_range, t, iCFRs)
    _sum = 0
    for i in 0:days_range
        _sum += iCFRs[t-i]
        #_sum += get_iCFR(t-i, t0, tMAX, delay, iCFRs)
    end
    return _sum
end

function get_iCFRs(t0, tMAX, t_iCFR, data, delay)
    iCFRs = zeros(tMAX)
    for i in eachindex(iCFRs)
        value = get_iCFR(i, t0, tMAX, t_iCFR, data, delay, iCFRs)
        iCFRs[i] = value
    end
    return iCFRs
end

function get_iCFR(t, t0, tMAX, t_iCFR, data, delay, iCFRs)
    if t <= t_iCFR
        get_iCFR_less(t_iCFR, data, delay)
    elseif t_iCFR < t < tMAX - delay
        get_iCFR_greater(t, data, delay)
    else
        ts = t0:tMAX
        interp_linear = LinearInterpolation(ts, iCFRs)
        interp_linear(t)
    end
end

function get_iCFR(t, times::Times, data::Data, delay, iCFRs)
    if t <= times.t_iCFR
        get_iCFR_less(times.t_iCFR, data, delay)
    elseif times.t_iCFR < t < times.tMAX - delay
        get_iCFR_greater(t, data, delay)
    else
        ts = times.t0:length(iCFRs)
        interp_linear = LinearInterpolation(ts, iCFRs, extrapolation_bc=Line())
        interp_linear(t)
    end
end

function get_iCFR_less(t, data, delay)
    infec = data.infec
    dead = data.dead
    q = length(dead)

    index = t + delay
    d_r = index >= q ? dead[q] : dead[index]
    c_r = infec[t]
    return d_r / c_r
end

function get_iCFR_greater(t, data, delay)
    infec = data.infec
    dead = data.dead
    return get_iCFR_loop(t, infec, dead, delay)
end

function get_iCFR_loop(t, infec, dead, delay)
    for r in eachindex(infec)
        den = infec[t] - infec[t - r]
        if den !== 0
            index = t + delay
            num = dead[index] - dead[index - r]
            result = num/den
            return ifelse(result <= 1.0, result, 1.0)
        end
    end
end
#endregion

function get_η(t, times::Times, ðs)
    if t < times.t0 + 6
        get_η_loop(times.t0, 0, 12, ðs)
    elseif times.t0 + 6 <= t <= times.tMAX-times.t0 - 6
        get_η_loop(t, -6, 6, ðs)
    elseif t > times.tMAX-times.t0 - 6
        get_η_loop(times.tMAX-times.t0, 0, 12, ðs, -)
    else
        error("None of the 't' values matched in 'get_n': $(t)")
    end
end

function get_ηs(i, times, ðs, ηs)
    new_ηs = vcat(ηs, get_η(i, times, ðs))
    if i+1 > times.tMAX
        return new_ηs
    else
        get_ηs(i+1, times, ðs, new_ηs)
    end
end

#region #* Methods to calculate η

function get_η_loop(t, infValue, supValue, ðs, operation=+)
    total = 0.0
    for i in infValue:supValue
        total += ðs[operation(t,i)]
    end
    return total / 13.0
end

function get_ðs(times, data, delay)
    ðs = zeros(tMAX)
    for i in eachindex(ðs)
        value = get_ð(i, times, data, delay, ðs)
        ðs[i] = value
    end
    return ðs
end

function get_ð(t, times, data, delay, ðs)
    if t <= times.t_η
        get_ð_less(times.t_η, data, delay)
    elseif times.t_η < t < times.tMAX - delay && data.infec_medic[t+delay] !== missing
        get_ð_greater(t, data, delay)
    else
        ts = times.t0:length(ðs)
        interp_linear = LinearInterpolation(ts, ðs, extrapolation_bc=Line())
        interp_linear(t)
    end
end

function get_ð_less(t::M, data::Data, delay::M) where M
    cr = data.infec
    hr = data.infec_medic
    q = length(cr)

    index = ifelse(t + delay >= q, q, t + delay)
    # if the index plus the delay is greater than the size of the data vectors, choose the last items
    num = hr[index]
    den = cr[index]
    return num / den
end

function get_ð_greater(t::M, data::Data, delay::M) where M
    cr = data.infec
    hr = data.infec_medic
    q = length(cr)

    index1 = t + delay
    #num1 = index1 >= q ? infec_medic[q] : infec_medic[index1]
    #den1 = index1 >= q ? infec[q] : infec[index1]
    return get_ð_greater_loop(index1, cr, hr)
end

function get_ð_greater_loop(index, cr, hr)
    for r in 1:length(cr)
        den = cr[index] - cr[index-r]
        if den !== 0 && hr[index-r] !== missing
            num = hr[index] - hr[index-r]
            result = num/den
            return ifelse(result <= 1.0, result, 1.0)
        end
    end
end
#endregion

function get_θ(t, times::Times, ωs, ω_CFRs)
    if t <= times.t_θ0
        ω0 = ωs[times.t_θ0]
        ω_CFR0 = ω_CFRs[times.t_θ0]
        return ω0 / ω_CFR0
    else
        ω = ωs[t]
        ω_CFR = ω_CFRs[t]
        return ω / ω_CFR
    end
end

function get_θs(i, times::Times, ωs, ω_CFRs, θs)
    new_θs = vcat(θs, get_θ(i, times, ωs, ω_CFRs))
    if i+1 > times.tMAX
        return new_θs
    else
        get_θs(i+1, times, ωs, ω_CFRs, new_θs)
    end
end
"""
    TimeParams(t, times, values, data, ω_CFR0, ρ0)

Calculate time-dependent parameters. If the value of _ω is not given, then it would be necessary to input the extra parameters ms, λs, max_ω, min_ω.

# Arguments
- `times::Vector{Int64}` : Vector with the values of t0, tMAX, t_η, t_iCFR, t_θ0
- `delays::Vector{Int64}` : Vector with the values of γ_d and γ_E + γ_I
"""
function TimeParams(t::M, times::Times, time_values, data::Data, ρ0::N) where {M,N}
    η = time_values[t, 4]
    ω = time_values[t, 5]
    θ = if t <= times.t_θ0
        ω_0 = time_values[times.t_θ0, 5]
        ω_CFR0 = time_values[times.t_θ0, 3]
        ω_0 / ω_CFR0
    else
        ω_CFR = time_values[t, 3]
        ω / ω_CFR
    end
    ρ = get_ρ(ω_0, ω, θ_0, θ, ρ0)
    
    return ω, θ, η, ρ, 0, data.imported[t]
end

function index_βs(t, ms, dates)::Float64
    for i in 1:length(dates)
        if i + 1 >= length(dates)
            return ms[end]
        elseif t <= dates[i + 1]
            if i <= 1
                return ms[1]
            else
                ms[i - 1]
            end
            #return ifelse(i <= 1, ms[1], ms[i - 1])
        end
    end
end

function get_β_Iu0(θ, val1, val2)
    θ >= 0.0 || error("θ is not positive")
    θ < 1.0 && return val1
    #θ == 1.0 && return val2
    return val2
    #error("θ greater than 1: $(θ)")
end

function get_β(t, ms, λs, β_I0, β_e0, β_I0_min, ω_u0, ω, θ, η, ρ, γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd)
    β_Iu0 = get_β_Iu0(θ, β_I0, β_I0_min)
    m = index_βs(t, ms, λs)
    β_e, β_I, β_Iu = [β_e0, β_I0, β_Iu0] .* m

    num = η * (β_e * γ_E + β_I * γ_I + (1.0 - θ - ω_u0) * β_Iu * γ_Iu)
    den = (1.0 - η) * (ρ * (θ - ω) * γ_Hr + ω * γ_Hd)

    β_hr = num / den

    return [β_e β_I β_Iu β_hr β_hr]
end

function get_βs(i, times::Times, ms, λs, β_I0, β_e0, β_I0_min, ω_u0, γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, ωs, θs, ηs, ρs, βs)
    new_βs = vcat(
        βs,
        get_β(i, ms, λs, β_I0, β_e0, β_I0_min, ω_u0, ωs[i], θs[i], ηs[i], ρs[i], γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd)
    )
    if i+1 > times.tMAX
        return new_βs
    else
        get_βs(i+1, times, ms, λs, β_I0, β_e0, β_I0_min, ω_u0, γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, ωs, θs, ηs, ρs, new_βs)
    end
end

"""
    parameters_lists(times, data, delays, ms, λs, max_ω, min_ω, ω_u0, ρ0, β_I0, β_e0, β_I0_min, γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd)

Returns a matrix with the values:
    γ_E γ_I γ_Iu γ_Hr γ_Hd γ_Q ω τ1 τ2 ω_u0 θs ρs βs
"""
function parameters_lists(times::Times, data::Data, delays::Delays, ms, λs, max_ω, min_ω, ω_u0, ρ0, β_I0, β_e0, β_I0_min, γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, γ_Q)

    iCFR = @ignore get_iCFR(1, times, data, delays.γ_Hd, [0])
    ð = @ignore get_ð(1, times, data, delays.γ_E_I, [0])
    ω = get_ω(1, ms, λs, max_ω, min_ω)
    τ2 = convert(typeof(max_ω), data.imported[1])
    time_values = [γ_E γ_I γ_Iu γ_Hr γ_Hd γ_Q ω zero(max_ω) τ2 ω_u0]

    time_values, iCFRs, ðs = parameters_lists_loop1(2, times, data, delays, γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, γ_Q, ω_u0, ms, λs, max_ω, min_ω, iCFR, ð, time_values)

    time_values = parameters_lists_loop2(times, ms, λs, ω_u0, ρ0, β_I0, β_e0, β_I0_min, γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, iCFRs, ðs, time_values)
    
    return time_values
end


function parameters_lists_loop1(i, times, data, delays, γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, γ_Q, ω_u0, ms, λs, max_ω, min_ω, iCFRs, ðs, destination)
    iCFR = @ignore get_iCFR(i, times, data, delays.γ_Hd, iCFRs)
    ð = @ignore get_ð(i, times, data, delays.γ_E_I, ðs)
    ω = get_ω(i, ms, λs, max_ω, min_ω)
    τ2 = convert(typeof(max_ω), data.imported[i])
    new_iCFRs = vcat(iCFRs, iCFR)
    new_ðs = vcat(ðs, ð)
    new_params = vcat(
        destination, 
        [γ_E γ_I γ_Iu γ_Hr γ_Hd γ_Q ω zero(max_ω) τ2 ω_u0]
    )

    if i+1 > times.tMAX
        return new_params, new_iCFRs, new_ðs
    else
        parameters_lists_loop1(i+1, times, data, delays, γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, γ_Q, ω_u0, ms, λs, max_ω, min_ω, new_iCFRs, new_ðs, new_params)
    end
end

function parameters_lists_loop2(times, ms, λs, ω_u0, ρ0, β_I0, β_e0, β_I0_min, γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, iCFRs, ðs, destination)

    ω_CFRs = get_ω_CFRs(2, times, iCFRs, get_ω_CFR(1, times.t_iCFR, iCFRs))
    ηs = get_ηs(2, times, ðs, get_η(1, times, ðs))
    θs = get_θs(2, times, destination[:,7], ω_CFRs, get_θ(1, times, destination[:,7], ω_CFRs))
    ω_θ0 = destination[times.t_θ0,7]
    ρs = get_ρs(
        2, times::Times, 
        ω_θ0, θs[times.t_θ0], destination[:,7], θs, ρ0, 
        get_ρ(ω_θ0, destination[1,7], θs[times.t_θ0], θs[1], ρ0)
    )
    βs = get_βs(
        2, times, ms, λs, β_I0, β_e0, β_I0_min, ω_u0, γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd, destination[:,9], θs, ηs, ρs, 
        get_β(1, ms, λs, β_I0, β_e0, β_I0_min, ω_u0, destination[1,9], θs[1], ηs[1], ρs[1], γ_E, γ_I, γ_Iu, γ_Hr, γ_Hd)
    )

    return hcat(destination, θs, ρs, βs)
end