#%%
import numpy as np
import pandas as pd
from datetime import datetime
#%%
class DataIt:
    def __init__(self, path):
        data = pd.read_csv(path, delimiter=';')
        self.days = data['Date'].values
        self.infec = data['Cumulative Cases'].values
        self.dead = data['Cumulative Deaths'].values
        self.infec_medic = data['Healthcare workers'].values
        self.imported = data['Imported E'].values
        self.exported = data['Evacuated'].values

def days_between(d1, d2, format="%d-%b-%Y"):
        d1 = datetime.strptime(d1, format)
        d2 = datetime.strptime(d2, format)
        return abs((d1 - d2).days)

class GetParams():

    def __init__(self, data, Nhist, dateI, dateF, format="%d-%b-%Y"):
        self.data = data
        self.Nhist = len(data.days) # Number of historical events (its the number of days in the dataset)
        try:
            self.NhistC = Nhist - np.where(data.days == dateI)[0][0]
        except IndexError:
            raise IndexError("Date not found in the dateset.")
        self.nvariants = 1
        self.dateinit = dateI
        self.dateend = dateF
        self.dmax = days_between(dateI, dateF, format)
        self.coefbetaH = np.zeros(self.dmax)
        self.ncm = 50 # Number of control measures
        self.datecm = np.zeros(self.ncm, dtype=int)
        self.lambdas = np.zeros(self.ncm, dtype=int)
        self.csvm = np.zeros(self.ncm + 1)
        self.csvkappa = np.zeros(self.ncm + 1)
        self.dt = 1.0/6.0

    def setLambdas(self, dates, ms, ks, format="%d-%b-%Y"):
        self.csvm[0] = 1.0
        self.csvkappa[0] = 0.0

        for i in range(len(dates)):
            self.datecm[i] = dates[i]
            self.csvm[i + 1] = ms[i]
            self.csvkappa[i + 1] = ks[i]

        ngivencm = len(dates)
        while self.ncm - ngivencm > 0:
            self.datecm[ngivencm] = "01-Jan-2050"
            ngivencm += 1

        for i in range(self.ncm):
            self.lambdas[i] = days_between(self.dateinit / self.dt, self.datecm[i], format)

    def computeCFR(self, NhistC, delay) -> np.ndarray:
        icfr = np.zeros(NhistC)
        cfr  = np.zeros(NhistC+2)

        firstIndex = 1
        while (self.data.infec[firstIndex] - self.data.infec[firstIndex-1]) == 0 or (self.data.dead[firstIndex + delay]) == 0:
            firstIndex += 1

        drold = self.data.dead[firstIndex + delay]
        crold = self.data.dead[firstIndex]

        # First loop for when *t* < t_iCFR
        # We add 1 here to firstIndex to make it inclusive in the range
        for d in range(firstIndex+1):
            icfr[d] = drold / crold
            cfr[d] = icfr[d]

        # icfr main loop
        self.icfr_main_loop(NhistC, delay, icfr, firstIndex, drold, crold)

        # cfr main loop
        self.cfr_loop(NhistC, delay, icfr, cfr, firstIndex)

        cfr[NhistC] = firstIndex

        return cfr

    def cfr_loop(self, NhistC, delay, icfr, cfr, firstIndex):
        d = firstIndex+1
        while d + delay < NhistC:
            if d < 6:
                cfr[d] = cfr[0]
            else:
                cfr[d] = icfr[d] + icfr[d - 1] + icfr[d - 2] + icfr[d - 3] + icfr[d - 4] + icfr[d - 5] + icfr[d - 6]

                ibef = 7
                while cfr[d] / ibef < 0.01 and d >= ibef:
                    cfr[d] = cfr[d] + icfr[d] - ibef
                    ibef += 1

                cfr[d] = cfr[d] / ibef
            
            cfr[NhistC+1] = d
            d += 1

    def icfr_main_loop(self, NhistC, delay, icfr, firstIndex, drold, crold):
        countz = 0
        d = firstIndex+1
        while d+delay < NhistC:
            if self.data.infec[d] != crold:
                icfr[d] = self.data.dead[d+delay] - drold / self.data.infec[d] - crold
                drold = self.data.dead[d+delay]
                crold = self.data.infec[d]

                i = countz
                while i > 0:
                    icfr[d-i] = icfr[d - countz - 1] + (icfr[d] - icfr[d - countz - 1]) * (countz - i + 1) / (countz + 1)
                    i -= 1

                countz = 0
            else:
                countz += 1
            d += 1

    def evaluateTheta(tfor, tstep, fr, cfr):
        pass

    def evaluateBeta(self, tstep, betai0, coef, cmeas, cgamma, cfrate, ctheta, p, wu, ieta) -> np.ndarray:

        mbetae = cmeas[0] * coef[0] * betai0
        mbetai = cmeas[0] * betai0

        betainf = coef[1] * betai0
        betaiu = betainf + ((betai0 - betainf) / (1 - cfrate)) * (1 - ctheta - wu) * (1 - ctheta)
        mbetaiu = cmeas[0] * betaiu

        mbetahr = (ieta * (
            (
                (mbetai[0] / cgamma[1]) + (mbetae[0] / cgamma[0]) + (1 - ctheta - wu) 
            ) * ( 
                (mbetaiu[0] / cgamma[2])
            )
        )
        ) / (
        	(1 - ieta) * ((p * (ctheta - cfrate) / cgamma[3]) + cfrate / cgamma[4])
        )

        mbetahr = mbetahr * self.coefbetaH[tstep]
        mbetahd = mbetahr

        return np.array([mbetae, mbetai, mbetaiu, mbetahd, mbetahd])

    def computeEta(self, delay) -> np.ndarray:
        eta = np.zeros(self.NhistC)
        etaprom = np.zeros(self.NhistC)
        firstIndex = 0
        while self.data.infec_medic[firstIndex + delay - 1] == 0:
            firstIndex += 1

        hrold = self.data.infec_medic[firstIndex + delay - 1]
        crold = self.data.infec[firstIndex + delay - 1]
        for d in range(firstIndex):
            eta[d] = hrold/ crold
        
        countz = 0
        d = firstIndex
        while d + delay < self.NhistC:
            if self.data.infec_medic[d + delay] != 0 and self.data.infec[d + delay] != crold and self.data.infec_medic[d + delay] - hrold < self.data.infec[d + delay]:
                eta[d] = (self.data.infec_medic[d + delay] - hrold) / (self.data.infec[d + delay] - crold)

                hrold = self.data.infec_medic[d + delay]
                crold = self.data.infec[d + delay]

                i = countz
                while i > 0:
                    eta[d - i] = eta[d - countz - 1] + (eta[d] - eta[d - countz - 1]) * (countz - i + 1) / (countz + 1)

                    i -= 1

                countz = 0
            else:
                countz += 1

            d += 1

        d = self.NhistC - delay - countz
        while d < self.NhistC:
            eta[d] = eta[self.NhistC - delay - countz - 1]
            d += 1

        i = 3
        while i < self.NhistC - 3:
            etaprom[i] = (eta[i - 3] + eta[i - 2] + eta[i - 1] + eta[i] + eta[i + 1] + eta[i + 2] + eta[i + 3]) / 7.0
            i += 1

        etaprom[0] = etaprom[3]
        etaprom[1] = etaprom[3]
        etaprom[2] = etaprom[3]
        etaprom[self.NhistC - 3] = etaprom[self.NhistC - 4]
        etaprom[self.NhistC - 2] = etaprom[self.NhistC - 4]
        etaprom[self.NhistC - 1] = etaprom[self.NhistC - 4]

        return etaprom

#%%


# %%
