import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize

class SimpleModel:
    """
    Clase que define el modelo simple trabajado hasta ahora.
    implementa el modelo, la función de optimización para 
    encontrar alpha y beta, así como una función de graficado
    """

    def __init__(self, 
        tiempo="semana",
        num_datos=None,
        mes_inicio="abril", 
        mes_fin="diciembre",
        minimizer="SLSQP"):

        self.mes_inicio = mes_inicio
        self.mes_fin = mes_fin
        self.num_datos = num_datos
        self.tiempo = tiempo
        self.minimizer = minimizer

        inicio = 0
        if mes_inicio == "marzo":
            inicio = 2
        elif mes_inicio == "abril":
            inicio = 33
        elif mes_inicio == "mayo":
            inicio = 63
        elif mes_inicio == "junio":
            inicio = 94
        elif mes_inicio == "julio":
            inicio = 124
        elif mes_inicio == "agosto":
            inicio = 155
        elif mes_inicio == "septiembre":
            inicio = 186
        elif mes_inicio == "octubre":
            inicio = 216
        elif mes_inicio == "noviembre":
            inicio = 247
        elif mes_inicio == "diciembre":
            inicio = 277
        elif mes_inicio == "enero":
            inicio = 308

        self.inicio = inicio

        fin = 32

        if mes_fin == "abril":
            fin = 62
        elif mes_fin == "mayo":
            fin = 93
        elif mes_fin == "junio":
            fin = 123
        elif mes_fin == "julio":
            fin = 154
        elif mes_fin == "agosto":
            fin = 185
        elif mes_fin == "septiembre":
            fin = 215
        elif mes_fin == "octubre":
            fin = 246
        elif mes_fin == "noviembre":
            fin = 276
        elif mes_fin == "diciembre":
            fin = 307
        elif mes_fin == "enero":
            fin = 338

        self.fin = fin

        assert inicio < fin, "El mes final debe ser después del mes inicial"

        # Tiempo tomado
        if num_datos != None:
            self.t = np.linspace(0, num_datos, num_datos)
        else:
            if tiempo == "semana":
                semanas = round((fin - inicio + 1) / 7)
                self.t = np.linspace(0, semanas, semanas)
            elif tiempo == "dia":
                dias = fin - inicio + 1
                self.t = np.linspace(0,dias, dias)

    # def add_sir_data(self, data):
    #     self.sus = data[0]
    #     self.inf = data[1]
    #     self.rec = data[2]
    #     self.exp = data[3]

    def official_data(self, path):
        """
        Obtiene a partir del archivo CSV los parametros iniciales de 
        gamma, sigma, poblacion, susceptibles, infectados, recuperados 
        y expuestos para el tiempo inicial escogido, del archivo 
        CSV con todos los datos.

        También colecciona en una lista los datos del número de infectados
        de acuerdo a los meses escogidos o a la cantidad de datos que se requieran
        """
        self.full_data = pd.read_csv(path)
        self.gamma = self.full_data.iloc[self.inicio, 9].item() # I / C
        self.sigma = self.full_data.iloc[self.inicio, 10].item() # Tasa de mortalidad (Fallecidos / I)
        
        self.N = self.full_data.iloc[self.inicio, 7].item() # Población
        self.sus = self.full_data.iloc[self.inicio, 8].item() # Susceptibles
        self.inf = self.full_data.iloc[self.inicio, 4].item() # Infectados
        self. rec = self.full_data.iloc[self.inicio, 3].item() # Removidos
        self.exp = self.full_data.iloc[self.inicio, 6].item() # Expuestos
        data = []
        
        if self.num_datos != None:
            if self.tiempo == "semana":
                for semana in range(self.inicio, self.inicio + self.num_datos, 7):
                    data.append(self.full_data.iloc[semana, 4].item())

                self.data = np.array([data]).reshape((len(data),))

            elif self.tiempo == "dia":
                for dia in range(self.inicio, self.inicio + self.num_datos):
                    data.append(self.full_data.iloc[dia, 4].item())

            self.data = np.array([data]).reshape((len(data),))

        else:
            if self.tiempo == "semana":
                for semana in range(self.inicio,self.fin+1, 7):
                    data.append(self.full_data.iloc[semana, 4].item())
                    # Esta condicion es agregada porque algunos meses tienen dias extras,
                    # que no cuentan como semanas, y por lo tanto no entren en los datos
                    if len(data) + 1 > len(self.t):
                        break

                self.data = np.array([data]).reshape((len(data),))

            elif self.tiempo == "dia":
                for dia in range(self.inicio, self.fin+1):
                    data.append(self.full_data.iloc[dia, 4].item())

                self.data = np.array([data]).reshape((len(data),))

    def simple_model(self, y, t, N, alpha, beta, gamma, sigma):
        sus, inf, rec, expo = y
        dydt = [
            (-alpha/N)*sus*(beta*expo + inf),
            gamma*expo - sigma*inf,
            sigma*inf,
            (alpha/N)*sus*(beta*expo + inf) - gamma*expo
        ]
        return dydt

    def solucion(self, alpha, beta, gamma, sigma):
        self.sol = odeint(
            self.simple_model, 
            [self.sus, self.inf, self.rec, self.exp], 
            self.t, 
            args=(self.N, alpha, beta, gamma, sigma)
        )

    def predecir(self, tiempo, cons):
        """
        Hace una predicción en el tiempo a partir de unas condiciones iniciales 
        y con constantes dadas, donde 'cons' y 'cond_in' son vectores.
        Argumentos:
        tiempo: Int o Float mayor que cero
        cons: Vector de la forma [N, alpha, beta, gamma, sigma],
        cond_in: Vector de la forma [susceptibles, infectados, removidos, expuestos]
        """
        self.t_pred = np.linspace(0, tiempo, tiempo)
        self.pred = odeint(
            self.simple_model,
            [102089743, 2041380, 1775427, 21705629], # datos de 21 de Febrero 2021
            self.t_pred,
            args=(cons[0], cons[1], cons[2], cons[3], cons[4])
        )

    def distance(self, X):
        """
        Declara alpha y beta como variables y hace una predicción usando
        el modelo.
        Luego resta término a término el vector de predicciones y el vector
        de datos reales, los eleva al cuadrado, suma el resultado y regresa
        el resultado.
        """
        alpha, beta, gamma, sigma = X
        self.solucion(alpha, beta, gamma, sigma)
        pred = self.sol[:,1]

        dis = 0
        for index in range(len(pred)):
            dis += (pred[index] - self.data[index])**2

        return math.sqrt(dis)

    def optimize_constrain(self, x0 , limit1, limit2, limit3, limit4):
        """
        Obtiene alpha y beta con optimización con:
        x0: estimación inicial de alpha y beta
        limit1: límites de alpha
        limit2: límites de beta
        """
        return minimize(
            self.distance,
            x0,
            method="{}".format(self.minimizer),
            bounds=[limit1, limit2, limit3, limit4]
        ).x

    def optimize(self, x0):
        """
        Obtiene alpha y beta con optimización sin restricciones.
        x0: estimación inicial de alpha y beta
        """
        return minimize(
            self.distance,
            x0,
            method="BFGS"
        ).x

    def grafica(self, desired, compare=True, save=False, prediccion=False):
        if prediccion:
            t = self.t_pred
            sol = self.pred
        else:
            t = self.t
            sol = self.sol

        if "s" in desired:
            plt.plot(t, sol[:,0], 'b', label="Susceptibles") # Susceptibles
        if "i" in desired:
            plt.plot(t, sol[:,1], 'r', label='Infectados') # Infectados
        if "r" in desired:
            plt.plot(t, sol[:,2], 'g', label="Recuperados") # Recuperados
        if "e" in desired:
            plt.plot(t, sol[:,3], 'y', label="Expuestos") # Expuestos
        if desired == "all":
            plt.plot(t, sol[:,0], 'r', label="Susceptibles")
            plt.plot(t, sol[:,1], 'b', label='Infectados')
            plt.plot(t, sol[:,2], 'g', label="Recuperados")
            plt.plot(t, sol[:,3], 'y', label="Expuestos")
                
        if prediccion:
            plt.legend(loc='best')
            plt.xlabel("Tiempo")
            if save:
                plt.savefig("imagenes\simple_model\mexico_2020_{}_predecidos.png".format(desired))
            else:
                plt.show()
        else:
            if compare:
                plt.plot(self.t, self.data, 's', label='Reales')

            plt.legend(loc='best')
            plt.xlabel('Tiempo ({})'.format(self.tiempo))
            #plt.ylabel('Infectados en México')
            if self.mes_inicio == self.mes_fin:
                plt.title("Gráfica para el mes de {}".format(self.mes_inicio))
            else:
                plt.title("Gráfica para los meses de {} - {}".format(self.mes_inicio, self.mes_fin))
            plt.grid()
            if save:
                plt.savefig("imagenes\simple_model\mexico_2020_{}.png".format(desired))
            else:
                plt.show()