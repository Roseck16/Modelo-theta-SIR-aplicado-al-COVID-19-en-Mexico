import pandas as pd
import math
from scipy.integrate import odeint

path = "Datos/210103COVID19MEXICO.csv"

class ThetaModel:
    """
    Clase que define el modelo theta-SIR
    """

    def __init__(self, fecha_inicio, fecha_fin, N=127575529):
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
        self.fi = fecha_inicio
        self.ff = fecha_fin
        self.N = N

        if iniciado:
            self.hospitalizados = 505173
            self.importados = 4773
            self.positivos = 1390663
            self.positivos_no_detectados = 458134
            self.fallecidos = 177191
            self.N = 127575529

    def conseguir_datos(self, 
        path, 
        chun=100,
        enc="ISO-8859-1"):
        """
        ADVERTENCIA: Proceso lento. No ingresar un valor para el argumento
        "chun" mayor a 500 o se realentizará el sistema.
        Obtiene a partir del archivo CSV los valores de personas infectadas
        hospotalizadas, casos importados, casos positivos, casos positivos no
        detectados, y fallecidos detectados (son los disponibles en el archivo)
        CSV usado
        """
        data = pd.read_csv(path, encoding=enc, engine="c", chunksize=chun)
        hos = 0
        impo = 0
        pos = 0
        pos_u = 0
        fall = 0

        for ch in data:
            hos += ((ch['TIPO_PACIENTE'] == 2) & (ch['CLASIFICACION_FINAL'] == 3)).sum()
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
            extras = [N, tau_1, tau_2, theta, omega_u, P, omega]
        """
        be, bi, biu, bidu, bhr, bhd = [i for i in betas]
        ge, gi, giu, gidu, ghr, ghd, gq = [i for i in gammas]
        N, t1, t2, theta, wu, p, w = [i for i in extras]
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

