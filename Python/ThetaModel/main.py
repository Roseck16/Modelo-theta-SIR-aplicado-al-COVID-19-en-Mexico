#%%
import pandas as pd
import numpy as np
import math
from get_parameters_class import SaveData
from ThetaModel import ThetaModel
# %%
full_data = "../../Datos oficiales/210321COVID19MEXICO.csv"
cumulative_data = "../../Datos/Casos_Modelo_Theta_final.csv"
save_file = '../../Datos/saved/variables.pkl'

#%%
#data = pd.read_csv(full_data, encoding="ISO-8859-1", engine="c", chunksize=100)
#cu_data = pd.read_csv(cumulative_data)
#%%
fechas = np.array([
    "2020-03-20",
    "2020-03-23",
    "2020-03-30",
    "2020-04-21",
    "2020-06-01",
])

# d_i_f_im = [
#     cu_data["Fecha"], 
#     cu_data["Positivos"], 
#     cu_data["Fallecidos"], 
#     cu_data["Positivos medicos"]

# ]

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
    "m2" : 1,
    "m3" : 0,
    "m4" : 0,
    "ms_dict" : {79: 1.0, 82: 1.0, 89: 1.0, 111: 0.0, 152: 0.0}
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

# parameters= {
#     'max_num_iteration': None,\
#     'population_size':100,\
#     'mutation_probability':0.1,\
#     'elit_ratio': 0.01,\
#     'crossover_probability': 0.5,\
#     'parents_portion': 0.3,\
#     'crossover_type':'uniform',\
#     'max_iteration_without_improv':None,\
# }
# %%
model = ThetaModel(data, fechas)
# %%
model.conseguir_datos(cumulative_data)
# %%
result = model.minimize(bounds, funtimeout=120.0, vartype='real')
# %%
params = '../../Datos/saved/params.pkl'
sol = '../../Datos/saved/sol.pkl'

loader1 = SaveData(params)
loader2 = SaveData(sol)

parameters = loader1.load_data()
sol  = loader2.load_data()
# %%
