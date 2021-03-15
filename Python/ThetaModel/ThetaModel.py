import pandas as pd
import math
from scipy.integrate import odeint

path = "Datos/210103COVID19MEXICO.csv"

class SimpleModel:
    """
    Clase que define el modelo theta-SIR
    """

    def __init__(self, iniciado=True):

        if iniciado:
            self.hospitalizados = 505173
            self.importados = 4773
            self.positivos = 1390663
            self.positivos_no_detectados = 458134
            self.fallecidos = 177191

    def conseguir_datos(self, 
        path, 
        chun=100, 
        eng='c', 
        enc="ISO-8859-1"):
        """
        ADVERTENCIA: Proceso lento. No ingresar un valor para el argumento
        "chun" mayor a 500 o se realentizará el sistema.
        Obtiene a partir del archivo CSV los valores de personas infectadas
        hospotalizadas, casos importados, casos positivos, casos positivos no
        detectados, y fallecidos detectados (son los disponibles en el archivo)
        CSV usado
        """
        data = pd.read_csv(path, encoding=enc, engine=eng, chunksize=chun)
        hos = 0
        impo = 0
        pos = 0
        pos_u = 0
        fall = 0

        for ch in data:
            hos += (ch['TIPO_PACIENTE'] == 2).sum()
            impo += ((ch['NACIONALIDAD'] == 2) & (ch['CLASIFICACION_FINAL'] == 3)).sum()
            pos += (ch['CLASIFICACION_FINAL'] == 3).sum()
            pos_u += (ch['CLASIFICACION_FINAL'].isin((1,2,4,5,6))).sum()
            fal += (ch['FECHA_DEF'] != '9999-99-99').sum()

        self.hospitalizados = hos
        self.importados = impo
        self.positivos = pos
        self.positivos_no_detectados = pos_u
        self.fallecidos = fall

    def modelo_theta_sir(self, y, t, betas, gammas, extras):
        """
        Definición del modelo Theta-SIR. vectores 'betas', 'gammas'
        y 'extras' tienen que tener un orden específico que se basa
        en el orden en el que aparecen en las ecuaciones del artículo.
        Ese orden se describe a continuación.
        Input:
        betas:  vector con los valores de los coeficientes beta. Su orden
                debe ser el siguiente:
                    betas = [B_E, B_I, B_{I_U}, B_{I_{D_u}}, B_{H_R}, B_{H_D}]
        gammas: vector con los valores de los coeficientes gamma. Su orden
                debe ser el siguiente:
                    gammas = [Gamma_E, Gamma_I, Gamma_{I_U}, Gamma_{I_{D_u}}, 
                            Gamma_{H_D}, Gamma_{H_R}, Gamma_Q]
        extras: vector con los valores de parámetros faltantes. Su orden
                debe ser el siguiente:
                    extras = [N, theta, omega_u, P, omega]
        """
        be, bi, biu, bidu, bhr, bhd = [i for i in betas]
        ge, gi, giu, gidu, ghr, ghd, gq = [i for i in gammas]
        N, theta, wu, p, w = [i for i in extras]
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

