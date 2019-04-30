import os
import numpy as np
import tables
from scipy import stats

# ===========================================================
# Load in NS mass/radius EOS priors of Steiner et al. (2018)
#   Adapted from code courtesy of Adelle Goodwin (2019)
# ===========================================================

pyburst_path = os.environ['PYBURST']
path = os.path.join(pyburst_path, 'files', 'temp')

class MrPrior:
    def __init__(self, filename='qlmxb_threep_base_all_out', path=path):
        filepath = os.path.join(path, filename)
        self.raw_file = tables.open_file(filepath)  # base model

        table = self.raw_file.root.markov_chain0.data
        markovobject = self.raw_file.get_node("/markov_chain0", "data")

        Data2 = {}
        for name in markovobject:
            Data2['{}'.format(name)] = name.read()
        # get radius data
        Radius2 = {}
        for key in Data2:
            if "/markov_chain0/data/R_" in key:
                Radius2['{}'.format(key)] = (Data2['{}'.format(key)])

    # now just get radii we want:
    R_mcmc2 = []
    for i in range(0, 100):
        R_mcmc2.append(Radius2["/markov_chain0/data/R_{} (EArray(291827,)) ''".format(i)])
    # exclude (mask) bad data where Rx has been set to zero as M_max is less than mass
    for i in range(0, len(R_mcmc2)):
        R_mcmc2[i] = np.ma.array(R_mcmc2[i], mask=False)
        for j in range(0, len(R_mcmc2[0])):
            if R_mcmc2[i][j] == 0.0:
                R_mcmc2[i].mask[j] = True

    # create grid points for mass
    Marray = np.linspace(0.2, 3.0, 97)  # exclude last 3 mass grid points (low statistics)
    mu = []
    sigma = []
    for i in range(0, 97):
        mu.append(np.mean(R_mcmc2[i]))
        sigma.append(np.std(R_mcmc2[i]))


# make prior function:
# required arrays: mu, sigma and Marray
def mr_prior(M, R):
    # hard mass limits: M = 0.2-2.5, R = 9.5-16
    # exclude values outside of domain of interpolation:
    if M > 2.5 or M < 0.2:
        return -np.inf
    if R > 16 or R < 9.5:
        return -np.inf
    else:
        # extract closest mass grid coordinate:
        index = np.searchsorted(Marray, M)

        for i in [index]:
            y = stats.norm.pdf(R, mu[i], sigma[i])

            return y
