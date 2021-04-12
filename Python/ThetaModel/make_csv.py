#%%
import pandas as pd
import numpy as np
import csv
#%%

def revisar_fecha(lista_dics, fecha):
    for index, dic in enumerate(np.trim_zeros(lista_dics)):
        if dic['Fecha'] == fecha:
            return False, index
    return True, 0

def crear_dict(path, dias=400, chunk=100):
    lista_dics = np.zeros(dias, dtype=object)
    data = pd.read_csv(path, encoding="ISO-8859-1", engine="c", chunksize=chunk)
    ultimo_indice = 0

    for ch in data:
        for index, row in ch.iterrows():

            fecha = row['FECHA_INGRESO']
            _tipo = row['TIPO_PACIENTE']
            _imp = row['NACIONALIDAD']
            _exp = row['ENTIDAD_RES']
            _clas = row['CLASIFICACION_FINAL']
            _fall = row['FECHA_DEF']

            hos = 1 if _tipo == 2 and _clas == 3 else 0
            qua = 1 if _tipo == 1 and _clas == 3 else 0
            imp = 1 if _imp == 2 and _clas == 3 else 0
            exp = 1 if _exp == 'NA' and _clas == 3 else 0
            pos = 1 if _clas == 3 else 0
            pos_u = 1 if _clas in (1,2,4,5,6) else 0
            fall = 1 if _fall != '9999-99-99' else 0

            val, index = revisar_fecha(lista_dics, fecha)

            if not val:
                lista_dics[index]['Hos'] += hos
                lista_dics[index]['Qua'] += qua
                lista_dics[index]['Imp'] += imp
                lista_dics[index]['Exp'] += exp
                lista_dics[index]['Pos'] += pos
                lista_dics[index]['Pos_u'] += pos_u
                lista_dics[index]['Fall'] += fall
            else:
                lista_dics[ultimo_indice] = {
                    'Fecha' : fecha, 
                    "Hos" : hos,
                    "Qua" : qua,
                    "Imp" : imp,
                    "Exp" : exp,
                    "Pos" : pos,
                    "Pos_u" : pos_u,
                    "Fall" : fall
                }
                ultimo_indice += 1

    return np.trim_zeros(lista_dics)

def crear_csv(dic, labels, nombre_destino):
    try:
        with open(nombre_destino, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=labels)
            writer.writeheader()
            for elem in dic:
                writer.writerow(elem)
    except IOError:
        print("I/O error:", IOError)
#%%

destino = "../../Datos/new/Casos_Modelo_Theta2.csv"
fuente = "../../Datos/new/210321COVID19MEXICO.csv"

dicc = crear_dict(fuente, dias=500)
#%%
labels = ['Fecha', 'Hos', 'Qua', 'Imp', 'Exp', 'Pos', 'Pos_u', 'Fall']
crear_csv(dicc, labels, destino)

#%%
# Abrir manualmente el csv y ordenarlo por fecha antes de proceder
# con el siguiente código

final_data = pd.read_csv(destino)

#%%
def crear_dict_2(df):
    n = len(df)
    lista_dicts = np.zeros(n, dtype=object)
    hos, qua, imp, exp, pos, pos_u, fall = 0, 0, 0, 0, 0, 0, 0

    lab = [i for i in df]

    for index, row in df.iterrows():
        hos += row[lab[1]]
        qua += row[lab[2]]
        imp += row[lab[3]]
        exp += row[lab[4]]
        pos += row[lab[5]]
        pos_u += row[lab[6]]
        fall += row[lab[7]]

        lista_dicts[index] = {
            lab[0] : row[lab[0]],
            lab[1] : hos,
            lab[2] : qua,
            lab[3] : imp,
            lab[4] : exp,
            lab[5] : pos,
            lab[6] : pos_u,
            lab[7] : fall
        }

    return lista_dicts

labels2 = [i for i in final_data]
destino_final = "../../Datos/new/Casos_Modelo_Theta_final.csv"

d = crear_dict_2(final_data)
crear_csv(d, labels2, destino_final)
# %%
