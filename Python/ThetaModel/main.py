#%%
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from GetModelParameters import *
from ThetaModel import *
from lmfit import minimize, Parameters, Parameter, report_fit
#%% #* Size of plots
W = 15
H = 10
# %%
path = "D:\\Code\\[Servicio Social]\\Datos\\Casos_Modelo_Theta_v3.csv"
data = Data(path)
# %%
lambdas = [
    "2020-03-20",
    "2020-03-23",
    "2020-03-30",
    "2020-04-21",
    "2020-06-01"
]
dates = get_lambdas(lambdas, data)
# %%
tspan = np.arange(0, 446)
#%%
N = data.population[tspan[0]]
u0 = [
    data.susceptible[tspan[0]],
    data.exposed[tspan[0]],
    data.infec[tspan[0]],
    data.infec_u[tspan[0]],
    data.hospitalized[tspan[0]],
    data.hospitalized[tspan[0]]*0.3,
]

ms_val = np.array([1.0,1.0, 0.0, 0.0, 0.0, 0.0])
#%%
gammas = Gammas(5.5, 5.0, 9.0, 14.2729, 5.0, 36.0450)

beta_I0, c_E, c_u, rho0, omega_u0, k2, c3, c5 = 0.4992, 0.3806, 0.3293, 0.7382, 0.42, 0.5, 0.2, 0.8
beta_I0_min, beta_e0 = c_E * beta_I0, c_u * beta_I0
omega = 0.014555

max_omega, min_omega = 0.804699241804244, 0.12785018214405364
#region
delays = Delays(gammas)

times = Times(tspan, data, delays)

ms = get_Ms(
    tspan[0], dates, ms_val, 
    np.array([k2]),
    np.array([c3, c5])
)
# %%
p = parameters_list(times, data, gammas, delays, ms, dates, omega, omega_u0, rho0, beta_I0, beta_I0_min, beta_e0)
#%% #* Solve the model with some initial parameters
sol = solve_ivp(thetaModel, tspan, u0, args=(N, gammas, p), dense_output=True)
u = sol.sol(tspan)
#%% 
#%% #* Plot the solution with the real data of infecteds
fig = plt.figure()
fig.set_figwidth(W)
fig.set_figheight(H)
plt.plot(tspan, u.T[:,2])
plt.plot(tspan, data.infec)
plt.grid()
plt.show()
#endregion
#%% #* Optimization using genetic algorithm
bounds_gammas = np.array([[0, 200]] * 6)
bounds_01 = np.array([[0, 1]] * 7)
bounds_0inf = np.array([[0, 500]] * 3)

bounds = np.vstack((bounds_gammas, bounds_01, bounds_0inf))

#%% #* Define the parameters for the genetic algorithm
parameters= {
    'max_num_iteration': 1000,
    'population_size':100,
    'mutation_probability':0.3,
    'elit_ratio': 0.01,
    'crossover_probability': 0.5,
    'parents_portion': 0.3,
    'crossover_type':'uniform',
    'max_iteration_without_improv':200
}
# %% #* Define the genetic algorithm
model = OptimizeModel(path, 0, 446, lambdas, ms_val)
# %%
genetic_opt = model.genetic_alg(bounds, parameters, 100.0)

#%%
genetic_opt.run()

#%% #* Optimization using 'leastsq' algorithm
#params = Parameters()
# params.add_many(
#     ('gamma_E', 5.5, True, 0, 200),
#     ('gamma_I', 5.0, True, 0, 200),
#     ('gamma_Iu', 9.0, True, 0, 400),
#     ('gamma_Hr', 14.2729, True, 0, 400),
#     ('gamma_Hd', 5.0, True, 0, 200),
#     ('gamma_Q', 36.0450, True, 0, 400),
#     ('c3', 0.2, True, 0, 1),
#     ('c5', 0.8, True, 0, 1),
#     ('omega_u0', 0.42, True, 0, 1),
#     ('max_omega', 0.804699241804244, True, 0, 1),
#     ('min_omega', 0.12785018214405364, True, 0, 1),
#     ('c_E', 0.4942, True, 0, 1),
#     ('c_u', 0.3293, True, 0, 1),
#     ('beta_I0', 0.4992, True, 0, 500),
#     ('rho0', 0.7382, True, 0, 500),
#     ('k2', 0.5, True, 0, 500)
# )
# params.add_many(
#     ('gamma_E', 5.5, True, 0, 200),
#     ('gamma_I', 5.0, True, 0, 200),
#     ('gamma_Iu', 9.0, True, None, None),
#     ('gamma_Hr', 14.2729, True, None, None),
#     ('gamma_Hd', 5.0, True, 0, 200),
#     ('gamma_Q', 36.0450, True, None, None),
#     ('c3', 0.2, True, None, None),
#     ('c5', 0.8, True, None, None),
#     ('omega_u0', 0.42, True, None, None),
#     ('max_omega', 0.804699241804244, True, None, None),
#     ('min_omega', 0.12785018214405364, True, None, None),
#     ('c_E', 0.4942, True, None, None),
#     ('c_u', 0.3293, True, None, None),
#     ('beta_I0', 0.4992, True, None, None),
#     ('rho0', 0.7382, True, None, None),
#     ('k2', 0.5, True, None, None)
# )
# %%
def modelSol(y, t, N, params):

    gammas = Gammas(params['gamma_E'].value, params['gamma_I'].value, params['gamma_Iu'].value, params['gamma_Hr'].value, params['gamma_Hd'].value, params['gamma_Q'].value)
    delays = Delays(gammas)

    times = Times(tspan, data, delays)
    ms = get_Ms(
        tspan[0], dates, ms_val,
        np.array([params['k2'].value]),
        np.array([params['c3'].value, params['c5'].value])
    )

    beta_I0_min, beta_e0 = params['c_E'] * params['beta_I0'], params['c_u'] * params['beta_I0']

    p = parameters_list(times, data, gammas, delays, ms, dates, params['max_omega'], params['min_omega'], params['omega_u0'], params['rho0'], params['beta_I0'], beta_I0_min, beta_e0)

    sol = solve_ivp(thetaModel, t, y, args=(N, gammas, p), dense_output=True)
    x = sol.sol(tspan)
    return x

def residual(params, u0, N, ts, data):
    model = modelSol(u0, ts, N, params)
    #graf_model(model, data)
    return (model.T[:,2] - data.infec).ravel()

#%%
result = minimize(residual, params, args=(u0, N, tspan, data), method='leastsq')
# %%
report_fit(result)
# %%
gammas = Gammas(
    191.891877,
    96.2037522,
    259.224291,
    -5053.42822,
    147.355871,
    36.0450000
    )

c3, c5, omega_u0, max_omega, min_omega = 0.20000000, 79.0292694, -30.4328221, -54.0873843, -4.14139384
c_E, c_u, beta_I0, rho0, k2 = 0.49420000, 0.32930000, 0.49920000, -43.3341960, 0.50000000

beta_I0_min, beta_e0 = c_E * beta_I0, c_u * beta_I0

delays = Delays(gammas)

times = Times(tspan, data, delays)

ms = get_Ms(
    tspan[0], dates, ms_val, 
    np.array([k2]),
    np.array([c3, c5])
)
# %%
# %%
p = parameters_list(times, data, gammas, delays, ms, dates, max_omega, min_omega, omega_u0, rho0, beta_I0, beta_I0_min, beta_e0)
#%% #* Solve the model with some initial parameters
sol = solve_ivp(thetaModel, tspan, u0, args=(N, gammas, p), dense_output=True)
u = sol.sol(tspan)
#%%
graf_model(u, data, tspan)

#%%
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from GetModelParameters import *
from ThetaModel import *
from lmfit import minimize, Parameters, Parameter, report_fit

W = 15
H = 10
# %%
path_it = "D:\\Code\\[Servicio Social]\\Datos\\timeseries.csv"
control_measures = "D:\\Code\\[Servicio Social]\\Datos\\controlmeasures.csv"

class DataIt:
    def __init__(self, path):
        data = pd.read_csv(path, delimiter=';')
        self.days = data['Date'].values
        self.infec = data['Cumulative Cases'].values
        self.dead = data['Cumulative Deaths'].values
        self.infec_medic = data['Healthcare workers'].values
        self.imported = data['Imported E'].values
        self.exported = data['Evacuated'].values


data = DataIt(path_it)
# %%
lambdas = [
    "19-Jan-2020",
    "23-Feb-2020",
    "11-Mar-2020",
    "22-Mar-2020",
    "04-May-2020",
    "18-May-2020",
    "03-Jun-2020",
    "22-Jul-2020"
]

dates = get_lambdas(lambdas, data, "%d-%b-%Y")
# %%
control = pd.read_csv(control_measures, delimiter=';')

#ms = np.array([1., 1.    , 0.5332, 0.1369, 0.    , 0.0549, 0.0577, 0.0578])
ms = control['m'].values
ks = control['kappa'].values

N = 60317000
u0 = [
    N-1,
    1,
    0,
    0,
    0,
    0
]
# %%

tspan = np.arange(185)
gammas = Gammas(5.5, 5.000000001, 9.0, 14.2729, 5.0, 36.0450)

beta_I0, c_E, c_u, rho0, omega_u0, k2, c3, c5 = 0.4992, 0.3806, 0.3293, 0.7382, 0.42, 0.5, 0.2, 0.8
beta_I0_min, beta_e0 = c_E * beta_I0, c_u * beta_I0

delays = Delays(gammas)

times = Times(tspan, data, delays)

omega = 0.014555
# %%
p = parameters_list(times, data, gammas, delays, ms, dates, omega, omega_u0, rho0, beta_I0, beta_I0_min, beta_e0)
# %%
sol = solve_ivp(thetaModel, (0, 184), u0, args=(N, gammas, p), t_eval=tspan)
# %%
sol.status
# %%
fig = plt.figure()
fig.set_figwidth(W)
fig.set_figheight(H)
plt.plot(tspan, data.infec)
plt.plot(sol.t, sol.y[2,:])
plt.grid()
plt.show()
# %%
graf_model(u, data, tspan)
# %%
