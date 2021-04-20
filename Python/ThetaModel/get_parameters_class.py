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

    def __init__(self, data, fecha, saved, **kwargs):
        """
        Parametros
        ----------
        data: list(DataFrame)
            Lista con los datos con la fecha, número acumulado de infectados,
            de fallecidos y de personal médico infectado; en ese orden.
        fecha: String
            Fecha en el formato "YYYY-MM-DD"
        """
        self.dias = data[0]
        self.infec = data[1]
        self.fall = data[2]
        self.infec_medic = data[3]
        self.fecha = self.day_to_index(fecha)
        self.t0 = 0
        self.tMAX = self.dias.loc[self.dias == self.dias.iloc[-1]].index.values[0]
        self.gamma_d = self.get_data_dict("gamma_d", saved, **kwargs)
        self.gamma_E = self.get_data_dict("gamma_E", saved, **kwargs)
        self.gamma_I = self.get_data_dict("gamma_I", saved, **kwargs)
        self.gamma_Iu = self.get_data_dict("gamma_Iu", saved, **kwargs)
        self.gamma_IDu = self.get_data_dict("gamma_IDu", saved, **kwargs)
        self.gamma_Hr = self.get_data_dict("gamma_Hr", saved, **kwargs)
        self.gamma_Hd = self.get_data_dict("gamma_Hd", saved, **kwargs)
        self.gamma_Q = self.get_data_dict("gamma_Q", saved, **kwargs)
        self.gammas = np.array([
            self.gamma_E, self.gamma_I, self.gamma_Iu, self.gamma_IDu, self.gamma_Hr, self.gamma_Hd, self.gamma_Q
        ])
        self.t_iCFR = self.get_data_dict("t_iCFR", saved, **kwargs)
        self.iCFR = self.get_data_dict("iCFR", saved, **kwargs)
        self.t_n = 22
        self.t_theta0 = self.get_data_dict("t_theta0", saved, **kwargs)
        if self.t_iCFR == None:
            self.get_t_iCFR()

        self.w, self.w0, self.theta = None, None, None
        self.w_u0 = self.get_data_dict("w_u0", saved, **kwargs)
        self.p, self.n = None, None

    def get_data_dict(self, name, dict_name, **kwargs):
        if name in dict_name.keys():
            return dict_name.get(name)
        elif name in kwargs.keys():
            return kwargs.get(name)
        else:
            return None

    def diferencia_dias(self, dia1, dia2, abs_val=True):
        d1 = datetime.strptime(dia1, '%Y-%m-%d')
        d2 = datetime.strptime(dia2, '%Y-%m-%d')
        diff = abs(d1 - d2) if abs_val else d1 - d2
        return diff/timedelta(days=1)

    def day_to_index(self, day):
        if type(day) == str:
            return self.dias.loc[self.dias == day].index.values[0]
        elif type(day) == np.ndarray:
            return list(
                map(
                    lambda x: self.dias.loc[self.dias == x].index.values[0], day
                )
            )
        elif type(day) in (int, np.int64):
            return day
        else:
            raise TypeError("Unexpected type for variable 'day': {}".format(type(day)))

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
            den = self.infec.iloc[index] - self.infec.iloc[index - 1]
            num = self.fall.iloc[index + round(1/self.gamma_d)]
            if num != 0 and den != 0:
                self.t_iCFR = index
                self.t_theta0 = index + 6
                return
        print("Error: Fecha ideal no encontrada")

    def get_t_n(self):
        n = len(self.infec_medic)

        for index in range(n):
            val = self.infec_medic[index + round(1/self.gamma_E + 1/self.gamma_I)]
            if val != 0:
                self.t_n = index
                break

    def get_ms(self, saved, lambdas, **kwargs):
        ks = kwargs.get('ks')
        cs = kwargs.get('cs')
        lambdas = self.day_to_index(lambdas)
        q = len(lambdas) + 1
        ms = np.zeros(q)

        for indice in range(q):
            m = "m{}".format(indice)
            if m in saved.keys():
                ms[indice] = saved.get(m)
            elif indice == 3:
                ms[3] = cs[0] * ms[2]
            elif indice == 5:
                ms[5] = cs[1] * ms[2]
            else:
                k = ks[indice-2]
                t = lambdas[indice - 2]
                t_menos = self.t0 if indice == 2 else lambdas[indice - 3]

                diff_ms = ms[indice-2] - ms[indice-1]
                diff_dias = t - t_menos

                ex = np.exp(-k * diff_dias)
                ms[indice] = diff_ms * ex + ms[indice-1]

        ms_dict = {}
        for fecha, m in zip(lambdas, ms):
            ms_dict[fecha] = m
        self.ms = ms_dict
        self.lambdas = lambdas

    def get_ñ(self, fecha):
        infec = self.infec
        infec_medic = self.infec_medic
        t_n = self.t_n

        t = self.day_to_index(fecha)
        
        if t <= t_n:
            num = infec_medic.iloc[t_n + round(1/self.gamma_E + 1/self.gamma_I)]
            den = infec.iloc[t_n + round(1/self.gamma_E + 1/self.gamma_I)]
            return num / den

        elif t > t_n:
            for r in range(1, len(infec)):
                den1 = infec.iloc[t + round(1/self.gamma_E + 1/self.gamma_I)] 
                den2 = infec.iloc[t - r + round(1/self.gamma_E + 1/self.gamma_I)]

                if den1 - den2 != 0:
                    num1 = infec_medic.iloc[t + round(1/self.gamma_E + 1/self.gamma_I)]
                    num2 = infec_medic.iloc[t - r + round(1/self.gamma_E + 1/self.gamma_I)]

                    return (num1 - num2) / (den1 - den2)
        else:
            raise Exception("Error: Linear interpolation needed for 'get_ñ'")

    def get_n(self, fecha):
        t = self.day_to_index(fecha)
        t0 = self.t0
        tMAX = self.tMAX
        result = 0

        if t < t0 + 3:
            for i in range(7):
                result += self.get_ñ(t0+i)
        elif t0 + 3 <= t and t <= t0 + tMAX - 3:
            for i in range(-3, 4):
                result += self.get_ñ(t+i)
        elif t > t0 + tMAX - 3:
            for i in range(7):
                result += self.get_ñ(t0+tMAX-i)
        else:
            raise ValueError("None of the 't' values matched in 'get_n': {}".format(t))
        
        self.n = result / 7

    def get_betas(self, fecha, saved, **kwargs):
        fecha = self.day_to_index(fecha)
        b_I0 = kwargs.get('b_I0')
        c_E = kwargs.get('c_E')
        c_u = kwargs.get('c_u')
        c_IDu = kwargs.get('c_IDu')
        if self.p == None:
            self.get_p(fecha, saved, **kwargs)
            self.get_n(fecha)
        n = self.n
        p = self.p

        b_e0 = c_E * b_I0
        b_I0_min = c_u * b_I0
        b_IDu0 = c_IDu * b_I0

        lambdas = self.lambdas
        q = len(lambdas)
        for indice in range(q):
            if indice+1 >= q or fecha <= lambdas[indice+1]:
                m = self.ms[lambdas[indice]]
                if self.theta >= 0 or self.theta < 1:
                    b_Iu0 = b_I0
                elif self.theta == 1:
                    b_Iu0 = b_I0_min

                b_e, b_I, b_Iu, b_IDu = b_e0 * m, b_I0 * m, b_Iu0 * m, b_IDu0 * m

                num = n * (b_e/self.gamma_E + b_I/self.gamma_I + (1-self.theta-self.w_u0) * (b_Iu/self.gamma_Iu) + self.w_u0 * (b_IDu/self.gamma_IDu))
                
                den = (1 - n) * (p*(self.theta - self.w)*(1/self.gamma_Hr) + self.w*(1/self.gamma_Hd))

                b_hr = num / den
                
                self.betas = np.array([b_e, b_I, b_Iu, b_IDu, b_hr, b_hr])
                break

    def get_w(self, fecha, **kwargs):
        t = self.day_to_index(fecha)
        lambdas = self.lambdas
        q = len(lambdas)
        max_omega = kwargs.get('max_omega')
        min_omega = kwargs.get('min_omega')

        for indice in range(q):
            if indice+1 >= q or t <= lambdas[indice+1]:
                m = self.ms[lambdas[indice]]
                val = m * max_omega + (1 - m) * min_omega
                if t == self.t_theta0:
                    self.w0 = val
                return val

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
        infec = self.infec
        fall = self.fall
        gamma_d = self.gamma_d
        t_iCFR = self.t_iCFR

        t = self.day_to_index(fecha)

        if t <= t_iCFR:

            if self.iCFR is None:

                d_r = fall.iloc[t_iCFR + round(1/gamma_d)]
                c_r = infec.iloc[t_iCFR]

                self.iCFR = d_r / c_r

            return self.iCFR
        
        elif t > t_iCFR:
            for r in range(1, len(infec)):
                den = infec.iloc[t] - infec.iloc[t - r]

                if den != 0:
                    num1 = fall.iloc[t + round(1/gamma_d)]
                    num2 = fall.iloc[t - r + round(1/gamma_d)]

                    return (num1 - num2) / den
        else:
            raise Exception("Error: Linear interpolation needed for 'get_iCFR'")


    def get_w_CFR(self, fecha):
        t = self.day_to_index(fecha)
        rango_dias = 7
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

    def get_theta(self, fecha, saved, **kwargs):
        t = self.day_to_index(fecha)
        theta0 = self.t_theta0

        if t <= theta0:
            if 'omega0' in saved.keys():
                self.w0 = saved.get('omega0')
            else:
                num = self.get_w(theta0, **kwargs)
            den = self.get_w_CFR(theta0)
            self.t_theta0 = num/den

        elif t > self.t_theta0:
            if 'omega' in saved.keys():
                self.w = saved.get('omega')
            else:
                self.get_w(t, **kwargs)
            num = self.w
            den = self.get_w_CFR(t)
        
        return num / den

    def get_p(self, fecha, saved, **kwargs):
        if self.theta == None:
            self.theta = self.get_theta(fecha, saved, **kwargs)
        if self.w == None:
            self.w = self.get_w(fecha, **kwargs)
        if self.t_theta0 == None:
            self.t_theta0 = self.get_theta(self.t_theta0, saved, **kwargs)
        if self.w0 == None:
            self.w0 = self.get_w(self.t_theta0, **kwargs)
        #self.get_w(fecha, **kwargs)
        #self.get_w(self.t_theta0, **kwargs)
        diff = self.theta - self.w
        diff_0 = self.t_theta0 - self.w0
        p0 = kwargs.get("p0")

        if diff >= diff_0:
            self.p = p0 * diff_0 / diff
        elif diff < diff_0:
            self.p = 1 - ((1-p0)/diff_0) * diff[0, 500],