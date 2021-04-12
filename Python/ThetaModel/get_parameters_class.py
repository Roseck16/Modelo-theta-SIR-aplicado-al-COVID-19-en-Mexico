#%%
import pandas as pd
import numpy as np
import pickle as pl
from datetime import datetime, timedelta
# %%
class SaveData:

    def __init__(self, path):
        self.path = path

    def save_data(self, data):
        with open(self.path, 'wb') as f:
            pl.dump(data, f)

    def load_data(self):
        with open(self.path, 'rb') as f:
            return pl.load(f)

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
        self.t0 = datetime.strptime(self.dias.iloc[0], '%Y-%m-%d')
        if "gamma_d" in kwargs.keys():
            self.gamma_d = kwargs.get('gamma_d')
            self.get_t_iCFR()
            self.t_theta0 = self.t_iCFR + 6

    def diferencia_dias(self, dia1, dia2, abs_val=True):
        d1 = datetime.strptime(dia1, '%Y-%m-%d')
        d2 = datetime.strptime(dia2, '%Y-%m-%d')
        diff = abs(d1 - d2) if abs_val else d1 - d2
        return diff/timedelta(days=1)

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
        self.get_t_iCFR()

    def get_t_iCFR(self):
        n = len(self.infec)

        for index in range(7, n):
            # Según el artículo, t_iCFR >= 7 días,
            # por lo que empezamos desde el número 7
            den = self.infec.iloc[index].value() - self.infec.iloc[index - 1]
            num = self.fall.iloc[index + round(1/self.gamma_d)]
            if num != 0 and den != 0:
                self.t_iCFR = index
                self.t_theta0 = index + 6
                return
        print("Error: Fecha ideal no encontrada")

    def get_ms(self, ks, cs, lambdas, **kwargs):
        q = len(lambdas) + 1
        ms = np.zeros(q)

        for indice in range(q):
            m = "m{}".format(indice)
            if m in kwargs.keys():
                ms[indice] = kwargs.get(m)
            elif indice == 3:
                ms[3] = cs[0] * ms[2]
            elif indice == 5:
                ms[5] = cs[1] * ms[2]
            else:
                k = ks[indice-2]
                t = lambdas[indice - 2]
                t_menos = self.t0 if indice == 2 else lambdas[indice - 3]

                diff_ms = ms[indice-2] - ms[indice-1]
                diff_dias = self.diferencia_dias(t, t_menos, abs_val=False)

                ex = np.exp(-k * diff_dias)
                ms[indice] = diff_ms * ex + ms[indice-1]

        ms_dict = {}
        lambdas_indices = list(map(lambda x: dias.loc[dias == x].index.values[0], lambdas))
        for fecha, m in zip(lambdas_indices, ms):
            ms_dict[fecha_indice] = m
        self.ms = ms_dict

    def get_w(self, fecha, max_omega, min_omega):
        t = self.dias.loc[self.dias == fecha].index.values[0] if type(fecha) == str else fecha
        lambdas = list(self.ms.keys())
        q = len(lambdas)

        for indice in range(q):
            if indice+1 >= q or t <= lambdas[indice+1]:
                m = ms[lambdas[indice]]
                val = m * max_omega + (1 - m) * min_omega
                if t == self.t_theta0:
                    self.w0 = val
                    return
                else:
                    self.w = val
                    return

    def get_iCFR(self, fecha):
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
        gamma_d = self.gamma_d
        t_iCFR = self.t_iCFR

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

    def get_w_CFR(self, fecha):
        t = self.dias.loc[self.dias == fecha].index.values[0] if type(fecha) == str else fecha
        rango_dias = 7
        gamma_d = self.gamma_d
        t_iCFR = self.t_iCFR

        if t <= t_iCFR:
            return self.get_iCFR(t_iCFR)
        elif t > t_iCFR:
            while True:
                suma = 0
                for i in range(rango_dias):
                    suma += self.get_iCFR(t-i)
                resultado = suma/rango_dias
                if resultado <= 0.01:
                    rango_dias += 1
                else:
                    return resultado

    def get_theta(self, fecha, omega=0, **kwargs):
        t = self.dias.loc[self.dias == fecha].index.values[0] if type(fecha) == str else fecha
        theta0 = self.t_theta0

        if omega != 0:
            num = omega
        else:
            ms = self.ms
            max_omega = kwargs.get('max_omega')
            min_omega = kwargs.get('min_omega')

        if t <= theta0:
            if omega == 0:
                self.get_w(theta0, ms, max_omega, min_omega)
                num = self.w0
            den = self.get_w_CFR(theta0)
            self.theta_0 = num/den

        elif t > self.t_theta0:
            if omega == 0:
                self.get_w(t, ms, max_omega, min_omega)
                num = self.w
            den = self.get_w_CFR(t)
            self.theta = num/den

    def get_p(self, fecha, p0):
        print("'get_' not implemented")

# %%

days = [datetime.strptime(i, '%Y-%m-%d') for i in fechas]

# %%
full_data = "../../Datos/new/210321COVID19MEXICO.csv"
cumulative_data = "../../Datos/new/Casos_Modelo_Theta_final.csv"
save_file = '../../Datos/new/variables.pkl'

#%%
data = pd.read_csv(full_data, encoding="ISO-8859-1", engine="c", chunksize=100)
cu_data = pd.read_csv(cumulative_data)
#%%
fechas = np.array([
    "2020-03-20",
    "2020-03-23",
    "2020-03-30",
    "2020-04-21",
    "2020-06-01",
])

ms = np.array([
    2.0,
    5.0,
    7.0,
    9.0,
    12.0
])
#%%
d_i_f = [cu_data["Fecha"], cu_data["Positivos"], cu_data["Fallecidos"]]
dias = d_i_f[0]

# %%
params = GetParameters(d_i_f, "2020-02-20", gamma_d = 1/13.123753454738198)
#%%
loader = SaveData(save_file)
# %%
data = {
    "gamma_d" :1/13.123753454738198, 
    "t_iCFR" : 22
    #""
}
loader.save_data(data)
# %%
data2 = loader.load_data()
# %%
def simple_fun(a=0):
    if a != 0:
        b = 5
    print("a + b = {}".format(a+b))
# %%
fechas_indices = list(map(lambda x: dias.loc[dias == x].index.values[0], fechas))
#%%
my_dic = {}
# %%
for fecha, num in zip(fechas_indices, ms):
    my_dic[fecha] = num
# %%
for key in my_dic.keys():
    print(key)
# %%
list_dic = list(my_dic.keys())
# %%
for indice in range(len(list_dic)):
    if indice + 1 >= len(list_dic) or 160 < list_dic[indice + 1]:
        print(indice)
        break
# %%
