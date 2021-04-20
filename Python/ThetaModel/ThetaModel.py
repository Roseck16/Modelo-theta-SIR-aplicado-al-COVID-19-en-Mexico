#%%
import pandas as pd
import math
import numpy as np
from get_parameters_class import SaveData, GetParameters
from scipy.integrate import odeint
from geneticalgorithm import geneticalgorithm as ga

#%%
class ThetaModel:
    """
    Clase que define el modelo theta-SIR
    """

    def __init__(self, saved, lambdas, N=127575529):
        """
        Parámetros 
        ----------
        fecha_inicio: string
            Texto con la fecha inicial que se desea estudiar.
            Ejemplo: fecha_inicio = '2020-03-01'

        fecha_fin: string
            Texto con la fecha final que se desea estudiar.
            Ejemplo: fecha_fin = '2020-12-31'

        N: número
            Número de personas antes de la pandemia.
        """
        self.saved = saved
        self.lambdas = lambdas
        self.N = N

    def conseguir_datos(self, path):
        """
        
        """
        data = pd.read_csv(path)
        d_i_f_im = [
            data["Fecha"], 
            data["Positivos"], 
            data["Fallecidos"], 
            data["Positivos medicos"],
            data["Importados"]
        ]
        self.dias = d_i_f_im[0]
        self.positivos = d_i_f_im[1]
        self.data =  d_i_f_im

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

    def modelo_theta_sir(self, y, t, betas, gammas, extras):
        """
        Definición del modelo Theta-SIR. vectores 'betas', 'gammas'
        y 'extras' tienen que tener un orden específico que se basa
        en el orden en el que aparecen en las ecuaciones del artículo.
        Ese orden se describe a continuación.

        Parametros
        ----------

        betas: ndarray
            vector con los valores de los coeficientes beta. Su orden debe ser el siguiente:
            betas = [B_E, B_I, B_{I_U}, B_{I_{D_u}}, B_{H_R}, B_{H_D}]

        gammas: ndarray
            vector con los valores de los coeficientes gamma. Su orden debe ser el siguiente:
            gammas = [Gamma_E, Gamma_I, Gamma_{I_U}, Gamma_{I_{D_u}}, Gamma_{H_D}, Gamma_{H_R}, Gamma_Q]
        extras: ndarray
            vector con los valores de parámetros faltantes. Su orden debe ser el siguiente:
            extras = [N, theta, omega_u, P, omega]
        """
        be, bi, biu, bidu, bhr, bhd = [i for i in betas]
        ge, gi, giu, gidu, ghr, ghd, gq = [i for i in gammas]
        N, theta, wu, p, w, t1, t2 = [i for i in extras]
        S, E, I, I_u, I_du, H_r, H_d, Q, R_d, R_u, D_u, D = y
        dydt = [
            -(S/N) * (be*E + bi*I + biu*I_u + bidu*I_du + bhr*H_r + bhd*H_d), # 1 Susceptibles
            (S/N) * (be*E + bi*I + biu*I_u + bidu*I_du + bhr*H_r + bhd*H_d) - ge*E + t1 - t2, # 2 Expuestos
            ge*E - gi*I, # 3 Infectados
            (1 - theta - wu) * gi*I - giu*I_u, # 4 Infectados no detectados
            wu*gi*I - gidu*I_du, # 5 Infectados no detectados que morirán
            p*(theta - w)*gi*I - ghr*H_r, # 6 Hospitalizados que se van a recuperar
            w*gi*I - ghd*H_d, # 7 Hospitalizados que van a fallecer
            # ---------------------------
            (1 - p) * (theta - w) * gi*I + ghr*H_r - gq*Q, # 8 En cuarentena
            gq*Q, # 9 Recuperados después de ser detectados infectados
            giu*I_u, # 10 Recuperados después de ser detectados infectados pero no detectados
            gidu*I_du, # 11 Fallecidos por COVID-19
            ghd*H_d, # 12 Fallecidos por COVID-19 pero no detectados
        ]
        return dydt

    def solucion(self, t, betas, gammas, extras):
        return odeint(
            self.modelo_theta_sir, 
            [1, self.N-1, 0,0,0,0,0,0,0,0,0,0], 
            t, 
            args=(betas, gammas, extras)
        )

    def distance(self, X):
        """
        Declara alpha y beta como variables y hace una predicción usando
        el modelo.
        Luego resta término a término el vector de predicciones y el vector
        de datos reales, los eleva al cuadrado, suma el resultado y regresa
        el resultado.
        """
        gamma_Iu, gamma_IDu, gamma_Hr, gamma_Hd, gamma_Q, max_omega, min_omega, c3, c5, k1, b_I0, c_E, c_u, c_IDu, p0, w_u0 = X
        params = GetParameters(self.data, 0, self.saved, gamma_Iu=gamma_Iu, gamma_IDu=gamma_IDu, gamma_Hr=gamma_Hr, gamma_Hd=gamma_Hd, gamma_Q=gamma_Q, w_u0=w_u0)

        ks = np.array([k1])
        cs = np.array([c3, c5])

        params.get_ms(self.saved, self.lambdas, ks=ks, cs=cs)
        gammas = params.gammas
        q = len(self.dias)

        sols = np.zeros((q, 12))
        for indice in range(q):
            t = self.dias.iloc[indice]
            
            params.get_betas(t, self.saved, b_I0=b_I0, c_E=c_E, c_u=c_u, c_IDu=c_IDu, p0=p0,max_omega=max_omega, min_omega=min_omega)
            betas = params.betas
            extras = np.array([
                self.N, params.theta, params.w_u0, params.p, params.w, self.data[4].iloc[indice],0
            ])
            ind = self.solucion(np.array([indice, indice + 1]), betas, gammas, extras)
            params_path = '../../Datos/new/params.pkl'
            sol_path = '../../Datos/new/sol.pkl'
            loader1 = SaveData(params_path)
            loader2 = SaveData(sol_path)
            loader1.save_data(params)
            loader2.save_data(ind)

            sols[indice] = ind[0]
            
        pred = sols[:,2]

        dis = 0
        for index in range(q):
            dis += (pred[index] - self.positivos.iloc[index])**2

        return math.sqrt(dis)

    def minimize(self, bounds, dims=16, params=None):
        if params is None:
            model = ga(
                function=self.distance,\
                dimension=dims,\
                variable_type='int',\
                variable_boundaries=bounds
            )
        else:
            model = ga(
                function=self.distance,\
                dimension=dims,\
                variable_type='int',\
                variable_boundaries=bounds,\
                algorithm_parameters=params
            )
        model.run()
        return model