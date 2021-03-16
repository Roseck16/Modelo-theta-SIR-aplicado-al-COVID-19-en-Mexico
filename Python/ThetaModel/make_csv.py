import pandas as pd
import math
from scipy.integrate import odeint

path = "Datos/210103COVID19MEXICO.csv"
eng="c", 
enc="ISO-8859-1"

data = pd.read_csv(path, encoding=enc, engine="c", chunksize=20)

c = data.get_chunk()
c['FECHA_INGRESO']

days = {}

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
        if 
    

# hospitalizados, importados, positivos, positivos no detectados, fallecidos
days = {
    "2020-04-02" : {"Hos" : 0, "Imp" : 0, "Pos" : 1, "Pos_u" : 0, "Fall" : 0},
    "2020-03-26" : {"Hos" : 1, "Imp" : 0, "Pos" : 1, "Pos_u" : 0, "Fall" : 1},
    "2020-03-28" : {"Hos" : 1, "Imp" : 0, "Pos" : 1, "Pos_u" : 0, "Fall" : 1},
    "2020-03-31" : {"Hos" : 0, "Imp" : 0, "Pos" : 1, "Pos_u" : 0, "Fall" : 0}
}

if True:
    days["2020-03-31"]["Hos"] += 1