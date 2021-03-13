import numpy as np
import pandas as pd
from Python.model import SimpleModel


# -----------------------
"""
Datos del inicio del año 2020
(Febrero 28, 2020)
"""
path = "Datos\Datos_2020_No_Oficial.csv"

# Constante u optimizado alpha, beta, gamma, sigma
constants = [0.6, 0.037, 0.0, 0.0]
#opt1 = [0.613107, 0.120047, 0.004564, 0.086723]

mex_model = SimpleModel(
    tiempo="semana", 
    #num_datos=10, 
    mes_inicio="diciembre", 
    mes_fin="enero",
    minimizer="Powell"
)
#mex_model.add_sir_data(sir)
mex_model.official_data(path)

x_opt = mex_model.optimize(x0=np.array(constants))
x_const_opt = mex_model.optimize_constrain(
    x0=np.array(constants),
    limit1=(0, None),
    limit2=(0,1),
    limit3=(0, None),
    limit4=(0, None),
)

print("alpha, beta: {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(
    x_opt[0], x_opt[1], x_opt[2], x_opt[3])
)
print("alpha, beta, gamma, sigma con restricciones: {}, {}, {}, {}".format(
    x_const_opt[0], x_const_opt[1], x_const_opt[2], x_const_opt[3])
)

#mex_model.solucion(x_opt[0], x_opt[1])
mex_model.solucion(x_const_opt[0], x_const_opt[1], x_const_opt[2], x_const_opt[3])
#mex_model.solucion(constants[0], constants[1])

# Introducir una lista como ["s", "i", "r", "e"]
mex_model.grafica(["i"], compare=True)

# read = pd.read_csv(path)
# i = read[read["Fecha"] == "2021-01-31"]

#cons = np.insert(x_const_opt, 0, 127612179)
cons = np.array([127612179, 0.0000,    0.1427,    0.4242,    0.0000])
#initial = [102089743, 2041380, 1775427, 21705629]

mex_model.predecir(100, cons)
mex_model.grafica(["i"], prediccion=True)

# Pregunta: los valores de alpha, beta, gamma, sigma obtenidos en un rango de día o semana,
# involucra que las predicciones que se hagan con esos valores también serán en día o semana?