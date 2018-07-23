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

    def plot(self):
        tshifts = [8, 8, 8]
        n_bursts = len(tshifts)
        fig, ax = plt.subplots(n_bursts, 1, sharex=True, figsize=(10, 12))

        for burst in range(n_bursts):
            obs_burst = self.bfit.obs[burst]
            obs_x = np.array(obs_burst.time)
            obs_y = np.array(obs_burst.flux)
            obs_y_u = np.array(obs_burst.flux_err)

            model = self.grid.mean_lc[1][burst+1]
            lum_to_flux = self.redshift * 4 * np.pi * self.f_b * 1e45

            m_x = (model[:, 0] * self.redshift) + tshifts[burst]
            m_y = model[:, 1] / lum_to_flux
            m_y_u = model[:, 2] / lum_to_flux
            m_y_upper = m_y + m_y_u
            m_y_lower = m_y - m_y_u

            ax[burst].fill_between(m_x, m_y_lower, m_y_upper, color='0.7')
            ax[burst].plot(m_x, m_y, color='black')
            ax[burst].errorbar(obs_x, obs_y, yerr=obs_y_u, ls='none', capsize=3, color='C1')

        ax[-1].set_xlabel('Time (s)', fontsize=20)
        ax[1].set_ylabel(r'Flux (erg cm$^{-2}$ s$^{-1}$)', fontsize=20)
        ax[-1].set_xlim([-10, 200])
        plt.tight_layout()
        plt.show(block=False)
