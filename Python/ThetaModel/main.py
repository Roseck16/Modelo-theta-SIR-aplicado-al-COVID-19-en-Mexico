#%%
import pandas as pd
# %%
fuente = "../../Datos/new/Casos_Modelo_Theta_final.csv"

data = pd.read_csv(fuente)
# %%
for indice,fecha in enumerate(data['Fecha']):
    print(indice, fecha)
# %%
