import numpy as np
import matplotlib.pyplot as plt

from ..grids import grid_analyser
from ..mcmc import burstfit, mcmc_tools


class Best:
    """Testing comparisons of LC from 'best' MCMC sample,
    against observed LC
    """
    def __init__(self, source='biggrid2', version=48):
        self.grid = grid_analyser.Kgrid('test_bg2', load_lc=True)
        self.bfit = burstfit.BurstFit(source, version, re_interp=False)

        self.best_params = mcmc_tools.get_max_lhood_params(source, version=version,
                                                           n_walkers=960, n_steps=10000)

        red_idx = self.bfit.param_idxs['redshift']
        fb_idx = self.bfit.param_idxs['f_b']
        self.redshift = self.best_params[red_idx]
        self.f_b = self.best_params[fb_idx]

    def plot(self, burst):
        tshifts = [8, 8, 8]
        fig, ax = plt.subplots()

        obs_burst = self.bfit.obs[burst]
        obs_x = np.array(obs_burst.time)
        obs_y = np.array(obs_burst.flux)
        obs_y_u = np.array(obs_burst.flux_err)

        model = self.grid.mean_lc[1][burst+1]
        lum_to_flux = self.redshift * 4 * np.pi * self.f_b * 1e45

        m_x = (model[:, 0] * self.redshift) + tshifts[burst]
        m_y = model[:, 1] / lum_to_flux
        m_y_u = model[:, 2] / lum_to_flux

        ax.errorbar(obs_x, obs_y, yerr=obs_y_u, ls='none', capsize=3)
        ax.plot(m_x, m_y)

        plt.show(block=False)
