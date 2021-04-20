#%%
import pandas as pd
import numpy as np
import math
from get_parameters_class import SaveData
from ThetaModel import ThetaModel
# %%
full_data = "../../Datos/new/210321COVID19MEXICO.csv"
cumulative_data = "../../Datos/new/Casos_Modelo_Theta_final.csv"
save_file = '../../Datos/new/variables.pkl'

#%%
#data = pd.read_csv(full_data, encoding="ISO-8859-1", engine="c", chunksize=100)
cu_data = pd.read_csv(cumulative_data)
#%%
fechas = np.array([
    "2020-03-20",
    "2020-03-23",
    "2020-03-30",
    "2020-04-21",
    "2020-06-01",
])

d_i_f_im = [
    cu_data["Fecha"], 
    cu_data["Positivos"], 
    cu_data["Fallecidos"], 
    cu_data["Positivos medicos"]

]

#%%
loader = SaveData(save_file)
# %%
data = {
    "gamma_d" :1/13.123753454738198,
    "gamma_E" : 1/5.5,
    "gamma_I" : 1/5,
    "t_iCFR" : 22,
    "t_theta0" : 22+6,
    "m0" : 1,
    "m1" : 1,
    "m4" : 0,
}
#%%
#loader.save_data(data)
# %%
#data2 = loader.load_data()
# %%
bounds = np.array([
    [0, 500],
    [0, 500],
    [0, 500],
    [0, 500],
    [0, 500],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
])
# %%
model = ThetaModel(data, fechas)
# %%
model.conseguir_datos(cumulative_data)
# %%
result = model.minimize(bounds)
# %%
params = '../../Datos/new/params.pkl'
sol = '../../Datos/new/sol.pkl'

loader1 = SaveData(params)
loader2 = SaveData(sol)

parameters = loader1.load_data()
sol  = loader2.load_data()
# %%
