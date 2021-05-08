#module GetModelParameters

import CSV: File as fl
using Dates, DataFrames

struct Data{M<:Vector{Date}, N<:Vector{Int64}}
    """
    Struct that contains the data for day, cumulative number of infected, infected medical personal and deaths
    """
    days::M
    infec::N
    infec_medic::N
    dead::N
    imported::N

    function Data(path::String)
        data = DataFrame(fl(path))
        t = data."Fecha"
        i = data."Positivos"
        im = data."Positivos medicos"
        d = data."Fallecidos"
        e = data."Importados"
        new{Vector{Date}, Vector{Int64}}(t,i,im,d,e)
    end
end

struct γs{M<:Float64}
    γ_d::M
    γ_E::M
    γ_I::M
    γ_Iu::M
    γ_IDu::M
    γ_Hr::M
    γ_Hd::M
    γ_Q::M

    function γs(opt::Dict{String, Any}, saved::Dict{String, Real})
        γ_d = get_data("γ_d", opt, saved)
        γ_E = get_data("γ_E", opt, saved)
        γ_I = get_data("γ_I", opt, saved)
        γ_Iu = get_data("γ_Iu", opt, saved)
        γ_IDu = get_data("γ_IDu", opt, saved)
        γ_Hr = get_data("γ_Hr", opt, saved)
        γ_Hd = get_data("γ_Hd", opt, saved)
        γ_Q = get_data("γ_Q", opt, saved)
        new{Float64}(γ_d, γ_E, γ_I, γ_Iu, γ_IDu, γ_Hr, γ_Hd, γ_Q)
    end
end

struct StaticParams{M<:Real}
    t0::M
    tMAX::M
    t_iCFR::M
    t_θ0::M 
    t_η::M

    function get_t_η(data::Data, gammas::γs)
        q = length(data.infec)
        for index in 1:q
            delay_index = index + round(Int, 1/gammas.γ_E + 1/gammas.γ_I)
            val = delay_index >= q ? val = data.infec[q] : data.infec[delay_index]
            if val != 0
                return index
            end
        end
    end

    function StaticParams(opt::Dict{String, Any}, saved::Dict{String, Real}, data::Data , gammas::γs)
        t0 = get_data("t0", opt, saved)
        tMAX = get_data("tMAX", opt, saved)
        t_iCFR = get_data("t_iCFR", opt, saved)
        t_θ0 = get_data("t_θ0", opt, saved)
        t_η = get_t_η(data, gammas)
        new{Real}(t0, tMAX, t_iCFR, t_θ0, t_η)
    end
end

struct Msλs{M<:Vector{Real}}
    ms::M
    λs::M

    function Msλs(opt::Dict{String, Any}, saved::Dict{String, Real}, dates::Vector{Int64})
        ks = get_data("ks", opt, saved)
        cs = get_data("cs", opt, saved)
        # If we have 4 dates, we need an extra one because we always set m0 = m1 to start with the first date.
        q = length(dates)+1
        start_index = 1

        ms = zeros(q)

        for index in 1:q
            m = "m$(index-1)"
            if haskey(saved, m)
                ms[index] = get(saved, m, nothing)
            elseif index in [4, 6]
                ms[index] = cs[1] * ms[3]
            else
                k = ks[start_index]
                t = dates[start_index]
                if start_index == 1
                    tr = get_data("t0", opt, saved, 0)
                else
                    tr = dates[start_index-1]
                end

                diff_ms = ms[start_index] - ms[start_index+1]

                ex = exp(-k * (t - tr))
                ms[index] = diff_ms * ex + ms[start_index+1]

                start_index += 1
            end
        end
        new{Vector{Real}}(ms, dates)
    end
end

struct TimeParams{M<:Real}
    ω_0::M
    ω::M
    θ_0::M
    θ::M
    ω_u0::M
    η::M
    ρ::M
    τ1::M
    τ2::M

    function get_ð(t::Integer, data::Data, stpms::StaticParams, gammas::γs)
        infec = data.infec
        infec_medic = data.infec_medic
        t_η = stpms.t_η
        q = length(infec)
        delay = round(Int, 1/gammas.γ_E + 1/gammas.γ_I)

        if t <= t_η
            index = t_η + delay
            # if the index plus the delay is greater than the size of the data vectors, choose the last items
            num = index >= q ? infec_medic[q] : infec_medic[index]
            den = index >= q ? infec[q] : infec[index]
            return num/den
        elseif t > t_η
            index1 = t + delay
            num1 = index1 >= q ? infec_medic[q] : infec_medic[index1]
            
            for r in 1:q
                #index2 = t - r + delay
                den2 = index1-r >= q ? infec[q-r] : index1-r <= 1 ? infec[1] : infec[index1-r]
                num2 = index1-(r+7) >= q ? infec_medic[q-7] : index1-(r+7) <= 1 ? infec_medic[1] : infec_medic[index1-(r+7)]
            den1 = index1 >= q ? infec[q] : infec[index1]
                if (den1 - den2) != 0
                    return (num1 - num2) / (den1 - den2)
                end
            end
        else
            error("Error: Linear interpolation needed for 'get_ñ'")
        end
    end

    function get_η(t::Integer, data::Data, stpms::StaticParams, gammas::γs)
        t0 = stpms.t0
        tMAX = stpms.tMAX
        _sum = 0

        if t < t0 + 3
            for i in 0:6
                _sum += get_ð(t0+i, data, stpms, gammas)
            end
        elseif t0 + 3 <= t && t <= t0 + tMAX - 3
            for i in -3:3
                _sum += get_ð(t+i, data, stpms, gammas)
            end
        elseif t > t0 + tMAX - 3
            for i in 0:6
                _sum += get_ð(t0+tMAX-i, data, stpms, gammas)
            end
        else
            error("None of the 't' values matched in 'get_n': $(t)")
        end

        return _sum / 7
    end

    function get_iCFR(t::Integer, data::Data, stpms::StaticParams, gammas::γs)
        infec = data.infec
        dead = data.dead
        delay = round(Int, 1/gammas.γ_d)
        t_iCFR = stpms.t_iCFR
        q = length(dead)

        if t <= t_iCFR
            index = t_iCFR + delay
            d_r = index >= q ? dead[q] : dead[index]
            c_r = infec[t_iCFR]
            return d_r / c_r
        elseif t > t_iCFR
            index1 = t + delay
            num1 = index1 >= q ? dead[q] : dead[index1]
            for r in 1:q
                #index2 = t - r + delay
                den = infec[t] - infec[t-r]

                if den != 0
                    num2 = index1-r >= q ? dead[q] : index1-r <= 1 ? dead[1] : dead[index1-r]
                    return (num1 - num2) / den
                end
            end
        else
            error("Error: Linear interpolation needed for 'get_iCFR'")
        end
    end

    function get_ω_CFR(t::Integer, data::Data, stpms::StaticParams, gammas::γs)
        t_iCFR = stpms.t_iCFR
        days_range = 6

        if t <= t_iCFR
            return get_iCFR(t_iCFR, data, stpms, gammas)
        elseif t > t_iCFR
            while true
                _sum = 0
                for i in 0:days_range
                    _sum += get_iCFR(t-i, data, stpms, gammas)
                end
                result = _sum / (days_range + 1)
                if result <= 0.015
                    days_range += 1
                else
                    return result
                end
            end
        end
    end

    function get_ω(t::Integer, opt::Dict{String, Any}, ms::Msλs)
        λs = ms.λs
        q = length(λs)
        max_ω = get_data("max_ω", opt, saved)
        min_ω = get_data("min_ω", opt, saved)
        m = 0

        for index in 1:q-1
            if t <= λs[index+1]
                m = index <= 1 ? ms.ms[1] : ms.ms[index-1]
                break
            elseif index+1 >= q
                m = ms.ms[end]
                break
            end
        end
        return (m * max_ω) + ((1 - m) * min_ω)
    end

    """
        get_θ(t, values, data, stpms, gammas)
    ...
    # Arguments
    - `t::Integer`: Value representing the time.
    - `values::Vector{Real}`: Vector containing the values  ω_0, ω, ω_CFR0 and ω_CFR.
    - `data::Data`: Struct Data.
    - `stpms::StaticParams`: Struct StaticParams.
    - `gammas::γs`: Struct γs
    ...
    """
    function get_θ(t::Integer, values::Vector{Float64}, stpms::StaticParams)
        
        t_θ0 = stpms.t_θ0

        if t <= t_θ0
            num = values[1]#get(values, "ω_0", nothing)
            den = values[3]#get(values, "ω_CFR0", nothing)
        elseif t > t_θ0
            num = values[2]#get(values, "ω", nothing)
            den = values[4]#get(values, "ω_CFR", nothing)
        end
        return num / den
    end

    """
        get_ρ(ωs, θs, ρ0)
    ...
    # Arguments
    - `ωs::Vector{Real}`: Vector containing the values ω_0 and ω.
    - `θs::Vector{Real}`: Vector containing the values θ_0 and θ.
    - `ρ0::Real`: Value of ρ0.
    ...
    """
    function get_ρ(ωs::N, θs::N, ρ0::M) where {M<:Float64, N<:Vector{M}}
        dif = θs[2] - ωs[2]
        dif_0 = θs[1] - ωs[1]

        if dif >= dif_0
            ρ0 * dif_0 / dif
        elseif dif < dif_0
            1 - ((1-ρ0)/dif_0) * dif
        else
            error("Value error: $dif, $dif_0")
        end
    end

    function TimeParams(
        t::Integer,
        opt::Dict{String, Any},
        saved::Dict{String, Real},
        data::Data,
        stpms::StaticParams,
        ms::Msλs,
        gammas::γs,
        )
        ωs = @sync begin
            ω_0 = @async get_data(
                "ω_0", opt, saved, 
                get_ω,
                stpms.t_θ0, opt, ms
            )
            ω = @async get_data(
                "ω", opt, saved, 
                get_ω,
                t, opt, ms
            )
            ω_CFR0 = @async get_data(
                "ω_CFR0", opt, saved,
                get_ω_CFR,
                stpms.t_θ0, data, stpms, gammas
            )
            ω_CFR = @async get_data(
                "ω_CFR", opt, saved,
                get_ω_CFR,
                t, data, stpms, gammas
            )
            [fetch(ω_0), fetch(ω), fetch(ω_CFR0), fetch(ω_CFR)]
        end
        
        θs = @sync begin
            θ_0 = @async get_θ(stpms.t_θ0, ωs, stpms)
            θ = @async get_θ(t, ωs, stpms)
            [fetch(θ_0), fetch(θ)]
        end
        
        ρ0 = get_data("ρ0", opt, saved)
        
        values = @sync begin
            ω_u0 = @async get_data("ω_u0", opt, saved)
            η = @async get_η(t, data, stpms, gammas)
            ρ = @async get_ρ([ωs[1], ωs[2]], θs, ρ0)
            [fetch(ω_u0), fetch(η), fetch(ρ)]
        end
        
        new{Real}(ωs[1], ωs[2], θs[1], θs[2], values[1], values[2], values[3], 0, data.imported[t])
    end
end

struct βs{M<:Float64}
    β_e::M
    β_I::M
    β_Iu::M
    β_IDu::M
    β_hr::M
    β_hd::M

    function βs(t::Integer, opt::Dict{String, Any}, saved::Dict{String, Real}, tmpms::TimeParams, ms::Msλs, gammas::γs)
        β_I0 = get_data("β_I0", opt, saved)
        c_E = get_data("c_E", opt, saved)
        c_u = get_data("c_u", opt, saved)
        c_IDu = get_data("c_IDu", opt, saved)
    
        β_e0, β_I0_min, β_IDu0 = (c_E, c_u, c_IDu) .* β_I0
    
        λs = ms.λs
        θ = tmpms.θ
        q = length(λs)
        m = 0

        for index in 1:q-1
            if t <= λs[index+1]
                m = index <= 1 ? ms.ms[1] : ms.ms[index-1]
            elseif index + 1 >= q
                m = ms.ms[end]
            end
        end

        if θ >= 0 && θ < 1
            β_Iu0 = β_I0
        elseif θ == 1
            β_Iu0 = β_I0_min
        else
            error("θ greater than 1: $θ at time $t")
        end
        β_e, β_I, β_Iu, β_IDu = (β_e0, β_I0, β_Iu0, β_IDu0) .* m

        #num = tmpms.η * (β_e/gammas.γ_E + β_I/gammas.γ_I + (1 - θ - tmpms.ω_u0) * β_Iu/gammas.γ_Iu + tmpms.ω_u0 * (β_IDu/gammas.γ_IDu))
        num = tmpms.η * (β_e/gammas.γ_E + β_I/gammas.γ_I + (1 - θ - tmpms.ω_u0) * β_Iu/gammas.γ_Iu + tmpms.ω_u0)

        den = (1 - tmpms.η) * (tmpms.ρ * (θ - tmpms.ω) * (1/gammas.γ_Hr) + tmpms.ω * (1/gammas.γ_Hd))

        β_hr = num / den

        new{Float64}(β_e, β_I, β_Iu, β_IDu, β_hr, β_hr)
    end
end

function full_params(t, opt_params, saved_params)
    day = day_to_index(t)
    gammP = γs(opt_params, saved_params)
    staticP = StaticParams(opt_params, saved_params, data, gammP)
    msP = Msλs(opt_params, saved_params, dates)
    timeP = TimeParams(day, opt_params, saved_params, data, staticP, msP, gammP)
    betasP = βs(day, opt_params, saved_params, timeP, msP, gammP)
    return [
        gammP.γ_d, gammP.γ_E, gammP.γ_I, gammP.γ_Iu, gammP.γ_IDu, gammP.γ_Hr, gammP.γ_Hd, gammP.γ_Q,
        timeP.θ, timeP.ω, timeP.ω_u0, timeP.ρ, timeP.τ1, timeP.τ2, 
        betasP.β_e, betasP.β_I, betasP.β_Iu, betasP.β_IDu, betasP.β_hr, betasP.β_hd, 
        N
    ]
    # if any(isnan, vals)
    #     error("The error are the parameters. $timeP, $betasP")
    # else
    #     return vals
    # end
end

"""
    get_data(key, opt, saved, def_ret)

Return the value stored in the collections *saved* or *opt* for the given key. If no value is found, return the value given in the optional parameter def_ret (default to 'nothing').


# Arguments
- `key::String`: the key to search for.
- `opt::Dict{String, Real}`: Dictionary that contains the optimized values.
- `saved::Dict{String, Real}`: Dictionary that contains saved values already found or taken as constant.
- `def_ret::Union{Nothing, Real}`: Optional. Default value to return.

"""
function get_data(key::String, opt::Dict{String, Any}, saved::Dict{String, Real}, def_ret::P=nothing) where { P<:Union{Nothing, Real}}
    if haskey(saved, key)
        get(saved, key, def_ret)
    elseif haskey(opt, key)
        get(opt, key, def_ret)
    else
        def_ret
    end
end

"""
    get_data[key, opt, saved, def_ret::Function, args...]

The default value can also be the result of a function
"""
function get_data(key::String, opt::Dict{String, Any}, saved::Dict{String, Real}, def_ret::Function, args...)
    if haskey(saved, key)
        get(saved, key, nothing)
    elseif haskey(opt, key)
        get(opt, key, nothing)
    else
        def_ret(args...)
    end
end