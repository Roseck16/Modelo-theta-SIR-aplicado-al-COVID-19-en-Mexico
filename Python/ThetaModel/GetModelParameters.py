import pandas as pd
import numpy as np
import warnings
from scipy import interpolate
from operator import add, sub
from datetime import datetime

class Data:
    def __init__(self, path):
        data = pd.read_csv(path)
        self.days = data.iloc[:,0].values
        self.hospitalized = data.iloc[:,1].values
        self.quarentine = data.iloc[:,2].values
        self.imported = data.iloc[:,3].values
        self.exported = data.iloc[:,4].values
        self.infec = data.iloc[:,5].values
        self.infec_u = data.iloc[:,6].values
        self.dead = data.iloc[:,7].values
        self.infec_medic = data.iloc[:,8].values
        self.recovered = data.iloc[:,9].values
        self.population = data.iloc[:,10].values
        self.susceptible = data.iloc[:,11].values
        self.exposed = data.iloc[:,12].values

class Gammas:
    def __init__(self, gamma_E, gamma_I, gamma_Iu, gamma_Hr, gamma_Hd, gamma_Q):
        self.gamma_E = gamma_E
        self.gamma_I = gamma_I
        self.gamma_Iu = gamma_Iu
        self.gamma_Hr = gamma_Hr
        self.gamma_Hd = gamma_Hd
        self.gamma_Q = gamma_Q

    def __fields__(self):
        return np.array([self.gamma_E, self.gamma_I, self.gamma_Iu, self.gamma_Hr, self.gamma_Hd, self.gamma_Q])

class Delays:
    def __init__(self, gammas):
        self.gamma_Hd = round(gammas.gamma_Hd)
        self.gamma_E_I = round(gammas.gamma_E + gammas.gamma_I)

    def __fields__(self):
        return np.array([self.gamma_Hd, self.gamma_E_I])

class Times:
    def __init__(self, tspan, data, delays):
        t_iCFR = get_t_iCFR(tspan[0], data, delays.gamma_Hd)
        t_eta = get_t_eta(tspan[0], data, delays.gamma_E_I)
        self.t0 = round(tspan[0])
        self.tMAX = round(tspan[-1])
        self.t_eta = t_eta
        self.t_iCFR = t_iCFR
        self.t_theta0 = t_iCFR+6

    def __fields__(self):
        return np.array([self.t0, self.tMAX, self.t_eta, self.t_iCFR, self.t_theta0])

def days_between(d1, d2, format="%Y-%m-%d"):
    d1 = datetime.strptime(d1, format)
    d2 = datetime.strptime(d2, format)
    return abs((d1 - d2).days)

def get_lambdas(dates, data, format="%Y-%m-%d"):
    d1 = data.days[0]
    return np.array(list(map(lambda x: days_between(x, d1, format), dates)))

def get_Ms(t0, dates, ms, ks, cs):
    m3 = Ms_calculate(ms, 0, t0, dates, ks)
    ms_real = np.array([
        ms[1],
        ms[2],
        m3 * cs[0],
        ms[5],
        m3 * cs[1]
    ])
    return ms_real

def Ms_calculate(ms, start_index, t0, dates, ks):
    k = ks[start_index]
    t = dates[start_index]
    tr = t0 if start_index == 0 else dates[start_index-1]

    diff_ms = ms[start_index] - ms[start_index+1]
    ex = np.exp(-k * (t-tr))
    return diff_ms * ex + ms[start_index+1]

def get_t_iCFR(t0, data, delay):
    q = len(data.infec)
    start = t0 if t0 > 7 else 7
    for i in range(start,q):
        val1 = data.infec[i] - data.infec[i-1]
        val2 = data.dead[i + delay] if i+delay <= q else data.dead[q-1]
        if val1 != 0 and val2 != 0:
            return i
    warnings.warn("No index found for t_iCFR, returning 0")
    return 0

def get_t_eta(t0, data, delay):
    q = len(data.infec)
    for i in range(t0+1,q):
        delay_i = i + delay
        val1 = data.infec_medic[delay_i] if delay_i <= q else data.infec_medic[q-1]
        val2 = data.infec[delay_i] if delay_i <= q else data.infec[q-1]
        if not np.isnan(val1) and val1 != 0 and val2 != 0:
            return i
    warnings.warn("No index found for t_eta, returning 0")
    return 0

def get_omega(t, ms, lambdas, max_omega, min_omega):
    q = len(lambdas)
    m = get_omega_loop(q, t, ms, lambdas)
    return (m * max_omega) + ((1.0 - m) * min_omega)

def get_omega_loop(q, t, ms, lambdas):
    for i in range(q):
        if i + 1 >= q:
            return ms[-1]
        elif t <= lambdas[i+1]:
            return ms[i-1] if i <= 1 else ms[1]

def get_rho(omega_theta0, theta_0, omega, theta, rho0):
    diff = theta - omega
    diff_0 = theta_0 - omega_theta0

    if diff >= diff_0:
        return rho0 * diff_0 / diff
    elif diff < diff_0:
        return 1.0 - ((1.0 - rho0) / diff_0) * diff
    else:
        raise ValueError("Value error: {}, {}".format(diff, diff_0))

def get_rhos(i, times, omega_theta0, theta_0, rho0, omegas, thetas, rhos):
    new_rhos = np.vstack((
        rhos, get_rho(omega_theta0, theta_0, omegas[i], thetas[i], rho0)
    ))
    if i+1 > times.tMAX:
        return new_rhos
    else:
        return get_rhos(i+1, times, omega_theta0, theta_0, rho0, omegas, thetas, new_rhos)

def get_omega_CFR(t, t_iCFR, iCFRs):
    if t <= t_iCFR:
        return iCFRs[t_iCFR]
    else:
        return get_omega_CFR_loop(t, iCFRs)

def get_omega_CFRs(i, times, iCFRs, omega_CFRs):
    new_omega_CFRs = np.vstack((
        omega_CFRs, get_omega_CFR(i, times.t_iCFR, iCFRs)
    ))
    if i+1 > times.tMAX:
        return new_omega_CFRs
    else:
        return get_omega_CFRs(i+1, times, iCFRs, new_omega_CFRs)

#region #* Methods to calculate the omega_CFR value
def get_omega_CFR_loop(t, iCFRs):
    days_range = 7
    while True:
        _sum = get_omega_CFR_sum(days_range, t, iCFRs)
        result = _sum / days_range
        if result <= 0.01:
            days_range += 1
        else:
            return result

def get_omega_CFR_sum(days_range, t, iCFRs):
    _sum = 0
    for i in range(days_range):
        _sum += iCFRs[t-i]
    return _sum

def get_iCFR(t, times, data, delay, iCFRs):
    if t <= times.t_iCFR:
        return get_iCFR_less(times.t_iCFR, data, delay)
    elif times.t_iCFR < t < times.tMAX - delay:
        return get_iCFR_greater(t, data, delay)
    else:
        size_array = range(times.t0, len(iCFRs))
        ts = np.array(size_array)
        inter_iCFRs = iCFRs.reshape(ts.shape)
        f = interpolate.interp1d(ts, inter_iCFRs, fill_value="extrapolate")
        return f(t)

def get_iCFR_less(t, data, delay):
    infec = data.infec
    dead = data.dead
    q = len(dead)

    index = t + delay if t+delay < q else q-1
    d_r = dead[index]
    c_r = infec[t]
    return d_r / c_r

def get_iCFR_greater(t, data, delay):
    infec = data.infec
    dead = data.dead
    return get_iCFR_greater_loop(t, infec, dead, delay)

def get_iCFR_greater_loop(t, infec, dead, delay):
    for r in range(len(infec)):
        den = infec[t] - infec[t - r]
        if den != 0:
            index = t + delay
            num = dead[index] - dead[index - r]
            result = num / den
            return min(result, 1.0)

#endregion

def get_eta(t, times, eths):
    if t < times.t0 + 3:
        return get_eta_loop(times.t0, 0, 7, eths)
    elif times.t0 + 3 <= t <= times.tMAX+times.t0 - 3:
        return get_eta_loop(t, -3, 4, eths)
    elif t > times.tMAX+times.t0 - 3:
        return get_eta_loop(times.tMAX+times.t0, 0, 7, eths, sub)
    else:
        raise ValueError("None of the 't' values matched in 'get_eta': {}".format(t))

def get_etas(i, times, eths, etas):
    new_etas = np.vstack((
        etas, get_eta(i, times, eths)
    ))
    if i+1 > times.tMAX:
        return new_etas
    else:
        return get_etas(i+1, times, eths, new_etas)

#region #* Methods to calculate eta
def get_eta_loop(t, infVal, supVal, eths, operationFun=add):
    total = 0.0
    for i in range(infVal,supVal):
        total += eths[operationFun(t,i)]
    return total / 7.0

def get_eth(t, times, data, delay, eths):
    if t <= times.t_eta:
        return get_eth_less(times.t_eta, data, delay)
    elif times.t_eta < t < times.tMAX - delay and data.infec_medic[t+delay] != 0:
        return get_eth_greater(t, data, delay)
    else:
        ts = np.arange(times.t0, len(eths))
        inter_eths = eths.reshape(ts.shape)
        f = interpolate.interp1d(ts, inter_eths, fill_value="extrapolate")
        return f(t)

def get_eth_less(t, data, delay):
    cr = data.infec
    hr = data.infec_medic
    q = len(cr)
    index = t + delay if t + delay < q else q-1

    num = hr[index]
    den = cr[index]
    return num / den

def get_eth_greater(t, data, delay):
    cr = data.infec
    hr = data.infec_medic
    q = len(cr)

    index1 = t + delay if t + delay < q else q-1
    return get_eth_greater_loop(index1, cr, hr)

def get_eth_greater_loop(index, cr, hr):
    for r in range(1, len(cr)):
        den = cr[index] - cr[index-r]
        if hr[index - r] != 0 and den != 0:
            num = hr[index] - hr[index-r]
            result = num/den
            return min(result, 1.0)
#endregion

def get_theta(t, times, omegas, omega_CFRs):
    if t <= times.t_theta0:
        omega0 = omegas[times.t_theta0]
        omega_CFR0 = omega_CFRs[times.t_theta0]
        return omega0 / omega_CFR0
    else:
        omega = omegas[t]
        omega_CFR = omega_CFRs[t]
        return min(omega / omega_CFR, 1.0)

def get_thetas(i, times, omegas, omega_CFRs, thetas):
    new_thetas = np.vstack((
        thetas, get_theta(i, times, omegas, omega_CFRs)
    ))
    if i+1 > times.tMAX:
        return new_thetas
    else:
        return get_thetas(i+1, times, omegas, omega_CFRs, new_thetas)

def get_beta(t, gammas, ms, lambdas, beta_I0, beta_I0_min, beta_e0, omega_u0, omega, theta, eta, rho):
    beta_Iu0 = get_beta_Iu0(theta, beta_I0, beta_I0_min)
    m = index_betas(t, ms, lambdas)
    
    beta_e, beta_I, beta_Iu = np.array([beta_e0, beta_I0, beta_Iu0]) * m

    num = eta * (
        beta_e * gammas.gamma_E + beta_I * gammas.gamma_I + (1.0 - theta - omega_u0) * beta_Iu * gammas.gamma_Iu
    )
    den = (1.0 - eta) * (
        rho * (theta - omega) * gammas.gamma_Hr + omega * gammas.gamma_Hd
    )
    beta_hr = num / den

    return np.array([beta_e, beta_I, beta_Iu, beta_hr, beta_hr])

def get_betas(i, times, gammas, ms, lambdas, beta_I0, beta_I0_min, beta_e0, omega_u0, omegas, thetas, etas, rhos, betas):
    new_betas = np.vstack((
        betas,
        get_beta(i, gammas, ms, lambdas, beta_I0, beta_I0_min, beta_e0, omega_u0, omegas[i], thetas[i,0], etas[i,0], rhos[i,0])
    ))
    if i+1 > times.tMAX:
        return new_betas
    else:
        return get_betas(i+1, times, gammas, ms, lambdas, beta_I0, beta_I0_min, beta_e0, omega_u0, omegas, thetas, etas, rhos, new_betas)

def get_beta_Iu0(theta, val1, val2):
    if 0.0 <= theta < 1.0:
        return val1
    else:
        # txt = "Theta is greater than 1.0: {}".format(theta)
        # warnings.warn(txt)
        return val2

def index_betas(t, ms , dates):
    for i in range(len(dates)):
        if i + 1 >= len(dates):
            return ms[-1]
        elif t <= dates[i+1]:
            if i <= 1:
                return ms[1]
            else:
                return ms[i-1]

#%%
def parameters_list(
    times, data, gammas, delays, 
    ms, dates, omega, omega_u0, rho0, beta_I0, beta_I0_min, beta_e0
    ):

    iCFR = get_iCFR(times.t0, times, data, delays.gamma_Hd, [0])
    eth = get_eth(times.t0, times, data, delays.gamma_E_I, [0])
    #omega = get_omega(times.t0, ms, dates, max_omega, min_omega)
    tau1 = data.exported[times.t0]
    tau2 = data.imported[times.t0]
    # time_values = [
    #     gammas.gamma_E, gammas.gamma_I, gammas.gamma_Iu, gammas.gamma_Hr, gammas.gamma_Hd, gammas.gamma_Q, omega, 0, tau2, omega_u0
    # ]
    time_values = [
        omega, tau1, tau2, omega_u0
    ]

    time_values, iCFRs, eths = parameters_list_recursion1(times.t0+1, times, data, delays, omega_u0, ms, dates, omega, iCFR, eth, time_values)

    time_values = parameters_list_recursion2(times, gammas, ms, dates, omega_u0, rho0, beta_I0, beta_I0_min, beta_e0, iCFRs, eths, time_values)

    return time_values

def parameters_list_recursion1(
    i, times, data, delays, 
    omega_u0, ms, dates, omega, 
    iCFRs, eths, params
    ):
    try:
        iCFR = get_iCFR(i, times, data, delays.gamma_Hd, iCFRs)
        eth = get_eth(i, times, data, delays.gamma_E_I, eths)
        #omega = get_omega(i, ms, dates, max_omega, min_omega)
        tau1 = data.exported[i]
        tau2 = data.imported[i]
    except:
        print(i)
        print(times.__fields__())
        print(delays.__fields__())
        print(iCFRs)
        print(eths)
        raise TypeError("Something went wrong")

    new_iCFRs = np.vstack((iCFRs, iCFR))
    new_eths = np.vstack((eths, eth))
    new_params = np.vstack((
        params,
        [omega, tau1, tau2, omega_u0]
    ))

    if i+1 > times.tMAX:
        return new_params, new_iCFRs, new_eths
    else:
        return parameters_list_recursion1(i+1, times, data, delays, omega_u0, ms, dates, omega, new_iCFRs, new_eths, new_params)

def parameters_list_recursion2(
    times, gammas, ms, dates, 
    omega_u0, rho0, beta_I0, beta_I0_min, beta_e0, 
    iCFRs, eths, params
    ):

    omega_CFRs = get_omega_CFRs(
        1, times, iCFRs, 
        get_omega_CFR(0, times.t_iCFR, iCFRs)
        )

    etas = get_etas(
        1, times, eths,
        get_eta(0, times, eths)
    )

    thetas = get_thetas(
        1, times, params[:,0], omega_CFRs,
        get_theta(0, times, params[:,0], omega_CFRs)
    )

    omega_theta0 = params[times.t_theta0, 0]

    rhos = get_rhos(
        1, times, omega_theta0,
        thetas[times.t_theta0], rho0,
        params[:, 0], thetas,
        get_rho(omega_theta0, thetas[times.t_theta0,0], params[0,0], thetas[0,0], rho0)
    )

    betas = get_betas(
        1, times, gammas, ms, dates,
        beta_I0, beta_I0_min, beta_e0,
        omega_u0, params[:,0], thetas, etas, rhos,
        get_beta(1, gammas, ms, dates, beta_I0, beta_I0_min, beta_e0, omega_u0, params[0, 0], thetas[0,0], etas[0,0], rhos[0,0])
    )

    return np.hstack((params, thetas, rhos, betas))
# %%
