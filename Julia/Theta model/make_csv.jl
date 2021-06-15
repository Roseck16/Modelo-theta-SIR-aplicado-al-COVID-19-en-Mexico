using CSV: propertynames, getproperty
using CSV, DataFrames, Dates

#* Functions
function find_date(cases_list, date)
    for (index, dict) in enumerate(cases_list)
        dict === nothing && return false, 1
        dict["Date"] === date && return true, index
    end
end

function if_found!(cases_list, found, index, last_index, values, date)
    if found
        cases_list[index]["Hos"] += values[1]
        cases_list[index]["Quad"] += values[2]
        cases_list[index]["Imp"] += values[3]
        cases_list[index]["Exp"] += values[4]
        cases_list[index]["Pos"] += values[5]
        cases_list[index]["Pos_u"] += values[6]
        cases_list[index]["Dead"] += values[7]
        cases_list[index]["Dead_u"] += values[8]
    else
        cases_list[last_index] = Dict{String, Union{Dates.Date,Int64}}(
            "Date" => date,
            "Hos" => values[1],
            "Quad" => values[2],
            "Imp" => values[3],
            "Exp" => values[4],
            "Pos" => values[5],
            "Pos_u" => values[6],
            "Dead" => values[7],
            "Dead_u" => values[8]
        )
        return 1
    end
    return 0
end

function make_csv_loop(cases_list, data)
    last_index = 1
    for row in data
        date = Date(row.:FECHA_INGRESO, DateFormat("y-m-d"))
        type = row.:TIPO_PACIENTE
        _imp = row.:NACIONALIDAD
        _exp = row.:ENTIDAD_RES
        _class = row.:CLASIFICACION_FINAL
        _fall = row.:FECHA_DEF

        hos = ifelse(_class === "3" && type === "2", 1, 0)
        qua = ifelse(_class === "3" && type === "1", 1, 0)
        imp = ifelse(_class === "3" && _imp === "2", 1, 0)
        exp = ifelse(_class === "3" && _exp === "NA", 1, 0)
        pos = ifelse(_class === "3", 1, 0)
        pos_u = ifelse(_class in ("1","2","4","5","6"), 1, 0)
        dead = ifelse(_fall !== "9999-99-99" && _class === "3", 1, 0)
        dead_u = ifelse(_fall !== "9999-99-99" && _class in ("1","2","4","5","6"), 1, 0)
        values = [hos, qua, imp, exp, pos, pos_u, dead, dead_u]
        
        found, index = find_date(cases_list, date)
        last_index += if_found!(cases_list, found, index, last_index, values, date)
    end
end

function make_dict(path, size=500)
    cases_list = Array{Union{Nothing, Dict{String, Union{Dates.Date, Int64}}}}(nothing, size)
    data = CSV.Rows(path)
    make_csv_loop(cases_list, data)
    filter!(x -> !isnothing(x), cases_list)
    return cases_list
end

"""
    make_dict2(df)
Load a DataFrame `df` and sum every row to get the total cumulative data.
"""
function make_dict2(df)
    cases_list = Array{Union{Nothing, Dict{String, Union{Dates.Date, Int64}}}}(nothing, size(df)[1])
    hos, qua, imp, exp, pos, pos_u, dead, dead_u = zeros(Int64, 8)
    headers = names(df)

    for (index, row) in enumerate(eachrow(df))
        hos += getproperty(row, headers[2])
        qua += getproperty(row, headers[3])
        imp += getproperty(row, headers[4])
        exp += getproperty(row, headers[5])
        pos += getproperty(row, headers[6])
        pos_u += getproperty(row, headers[7])
        dead += getproperty(row, headers[8])
        dead_u += getproperty(row, headers[9])

        cases_list[index] = Dict{String, Union{Dates.Date, Int64}}(
            headers[1] => getproperty(row, headers[1]),
            headers[2] => hos,
            headers[3] => qua,
            headers[4] => imp,
            headers[5] => exp,
            headers[6] => pos,
            headers[7] => pos_u,
            headers[8] => dead,
            headers[9] => dead_u
        )
    end
    return cases_list
end

function make_csv(target, list)
    df = DataFrame(Date=Dates.Date[], Hos=Int[], Quad=Int[], Imp=Int[], Exp=Int[], Pos=Int[], Pos_u=Int[], Dead=Int[], Dead_u=Int[])
    for dict in list
        push!(df, dict)
    end

    CSV.write(target, df)
end 

#* Implementation
source = "D:\\Code\\[Servicio Social]\\Datos oficiales\\210321COVID19MEXICO.csv"
target = "D:\\Code\\[Servicio Social]\\Datos\\Casos_Modelo_Theta_v3.csv"

cases_dict = make_dict(source)

make_csv(target, cases_dict)

#! Before continuing, open the just created CSV and sort it by date.

final_data = CSV.File(target) |> DataFrame

cases_list_final = make_dict2(final_data)
make_csv(target, cases_list_final)