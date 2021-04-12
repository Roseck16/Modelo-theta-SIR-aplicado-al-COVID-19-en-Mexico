#%%
import pandas as pd
import numpy as np
import pickle as pl
from datetime import datetime, timedelta
# %%
full_data = "../../Datos/new/210321COVID19MEXICO.csv"
cumulative_data = "../../Datos/new/Casos_Modelo_Theta_final.csv"
save_file = '../../Datos/new/variables.pkl'

# %%

data = pd.read_csv(full_data, encoding="ISO-8859-1", engine="c", chunksize=100)
cu_data = pd.read_csv(cumulative_data)
save_file = '../../Datos/new/variables.pkl'
#%%
fechas = np.array([
    "2020-03-20",
    "2020-03-23",
    "2020-03-30",
    "2020-04-21",
    "2020-06-01",
])
#%%
d_i_f = [cu_data["Fecha"], cu_data["Positivos"], cu_data["Fallecidos"]] 
dias = d_i_f[0]
# %%


# %%
class GetParameters:

    def __init__(self, data, fecha, **kwargs):
        """
        Parametros
        ----------
        data: list(DataFrame)
            Lista con los datos con la fecha, número acumulado de infectados y
            de fallecidos
        fecha: String
            Fecha en el formato "YYYY-MM-DD"
        """
        self.dias = data[0]
        self.infec = data[1]
        self.fall = data[2]
        self.fecha = fecha
        self.t0 = datetime.strptime(dias.iloc[0], '%Y-%m-%d')
        self.gamma_d = kwargs.get('gamma_d', None)

    def diferencia_dias(self, dia1, dia2, abs_val=True):
        d1 = datetime.strptime(dia1, '%Y-%m-%d')
        d2 = datetime.strptime(dia2, '%Y-%m-%d')
        diff = abs(d1 - d2) if abs_val else d1 - d2
        return diff/timedelta(days=1)

    def save_data(self, data, path):
        with open(path, 'wb') as f:
            pl.dump(data, f)

    def load_data(self, path):
        with open(path, 'wb') as f:
            data = pl.load(f)
            return data

    def get_gamma_d(self, path, chunk=100):
        data = pd.read_csv(path, encoding="ISO-8859-1", engine="c", chunksize=chunk)
        dias = 0
        numero = 0

        for ch in data:
            for _, datos in ch.iterrows():
                if datos['FECHA_DEF'] != '9999-99-99':
                    sint = datos['FECHA_SINTOMAS']
                    fall = datos['FECHA_DEF']
                    dias += self.diferencia_dias(sint, fall)
                    numero += 1
        self.gamma_d = 1/(dias/numero)

    def get_t_iCFR(self, gamma_d=1/13.123753454738198):
        n = len(self.infec)

        for index in range(7, n):
            # Según el artículo, t_iCFR >= 7 días,
            # por lo que empezamos desde el número 7
            den = self.infec.iloc[index] - self.infec.iloc[index - 1]
            num = self.fall.iloc[index + round(1/gamma_d)]
            if num != 0 and den != 0:
                return index
        print("Error: Fecha ideal no encontrada")

    def get_omega(self, fecha, ms, max_omega, min_omega):
        return ms*max_omega + (1 - ms)*min_omega

    def get_iCFR(self, fecha, gamma_d, t_iCFR):
        """
        Parametros
        ----------
        gamma_d: número
            Tasa de transición de un caso infectado a su muerte.
        t_iCFR: número
            Fecha de iCFR (Instantaneous case fatality ratio), la cual es
            simplemente un valor numérico que representa un índice en una
            colección de datos.
        """
        dias = self.dias
        infec = self.infec
        fall = self.fall

        t = dias.loc[dias == fecha].index.values[0] if type(fecha) == str else fecha

        if t <= t_iCFR:

            d_r = fall.iloc[t_iCFR + round(1/gamma_d)]
            c_r = infec.iloc[t_iCFR]

            return d_r / c_r
        
        elif t > t_iCFR:
            for r in range(1, len(infec)):
                den = infec.iloc[t] - infec.iloc[t - r]

                if den != 0:
                    num1 = fall.iloc[t + round(1/gamma_d)]
                    num2 = fall.iloc[t - r + round(1/gamma_d)]

                    return (num1 - num2) / den
        #else: TODO

    def get_w_CFR(self, fecha, gamma_d, t_iCFR):
        dias = self.dias

        t = dias.loc[dias == fecha].index.values[0] if type(fecha) == str else fecha
        rango_dias = 7

        if t <= t_iCFR:
            return self.get_iCFR(t_iCFR, gamma_d, t_iCFR)
        elif t > t_iCFR:
            while True:
                suma = 0
                for i in range(rango_dias):
                    suma += self.get_iCFR(t-i, gamma_d, t_iCFR)
                resultado = suma/rango_dias
                if resultado <= 0.01:
                    rango_dias += 1
                else:
                    return resultado

    def get_theta(self, fecha, gamma_d, t_iCFR, omega):
        dias = self.dias

        t_theta0 = t_iCFR + 6
        t = dias.loc[dias == fecha].index.values[0]

        if t <= t_theta0:
            num = omega
            den = self.get_w_CFR(t_theta0, gamma_d, t_iCFR)
            return num/den

        elif t > t_theta0:
            num = omega
            den = self.get_w_CFR(t, gamma_d, t_iCFR)
            return num/den

    def get_p(self, p0):

    def get_ms(self, ks, lambdas, **kwargs):
        q = len(lambdas) + 1
        ms = np.zeros(q)

        for indice in range(q):
            m = "m{}".format(indice)
            if m in kwargs.keys():
                ms[indice] = kwargs.get(m)
            else:
                k = ks[indice-2]
                t = lambdas[indice - 2]
                t_menos = self.t0 if indice == 2 else lambdas[indice - 3]

                diff_ms = ms[indice-2] - ms[indice-1]
                diff_dias = self.diferencia_dias(t, t_menos, abs_val=False)

                ex = np.exp(-k * diff_dias)
                ms[indice] = diff_ms * ex + ms[indice-1]
        return ms

    def get_ms2(self, cs, ms):
        ms[3] = cs[0] * ms[2]
        ms[5] = cs[1] * ms[2]
        return ms

# %%

days = [datetime.strptime(i, '%Y-%m-%d') for i in fechas]

# %%

if days[0] < days[1]:
    print("Yes")
# %%
def simple_fun(**kwargs):
    print(kwargs.keys())
    print(kwargs.items())
# %%
simple_fun(a=1,b=2,c=3, hello="Hello")
# %%
