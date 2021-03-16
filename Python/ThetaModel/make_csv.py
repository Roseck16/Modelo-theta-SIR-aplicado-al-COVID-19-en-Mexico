import pandas as pd
import numpy as np
import math
import csv
from scipy.integrate import odeint

path = "Datos/210103COVID19MEXICO.csv"
eng="c", 
enc="ISO-8859-1"

data = pd.read_csv(path, encoding=enc, engine="c", chunksize=20)

c = data.get_chunk()
c['FECHA_INGRESO']

days = []

for index, row in c.iterrows():

    fecha = row['FECHA_INGRESO']
    hos = row['TIPO_PACIENTE']
    imp = row['NACIONALIDAD']
    clas = row['CLASIFICACION_FINAL']
    fall = row['FECHA_DEF']

    if fecha not in days:
        days[fecha] = {"Hos" : 0, "Imp" : 0, "Pos" : 0, "Pos_u" : 0, "Fall" : 0}

    if hos == 2 and clas == 3:
        days[fecha]["Hos"] += 1
    if imp == 2 and clas == 3:
        days[fecha]["Imp"] += 1
    if clas == 3:
        days[fecha]["Pos"] += 1
    if clas in (1,2,4,5,6):
        days[fecha]["Pos_u"] += 1
    if fall != '9999-99-99':
        days[fecha]["Fall"] += 1


# hospitalizados, importados, positivos, positivos no detectados, fallecidos
days = [
    {'Fecha' : "2020-04-02", "Hos" : 0, "Imp" : 0, "Pos" : 1, "Pos_u" : 0, "Fall" : 0},
    {'Fecha' : "2020-03-26", "Hos" : 1, "Imp" : 0, "Pos" : 1, "Pos_u" : 0, "Fall" : 1},
    {'Fecha' : "2020-03-28", "Hos" : 1, "Imp" : 0, "Pos" : 1, "Pos_u" : 0, "Fall" : 1},
    {'Fecha' : "2020-03-31", "Hos" : 0, "Imp" : 0, "Pos" : 1, "Pos_u" : 0, "Fall" : 0}
]

for dics in days:
    if dics['Fecha'] == "2020-03-31":
        print(dics)

a = np.zeros(400, dtype=object)
a[0] = {'Fecha' : "2020-04-02", "Hos" : 0, "Imp" : 0, "Pos" : 1, "Pos_u" : 0, "Fall" : 0}
np.trim_zeros(a)

def revisar_fecha(lista_dics, fecha):
    for dic in lista_dics:
        if dic['Fecha'] == fecha:
            return False
    return True