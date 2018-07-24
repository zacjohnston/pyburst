import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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

        self.n_bursts = len(self.grid.mean_lc[1])
        self.shifted_lc = None
        self.interp_lc = None
        self.extract_lc()

    def extract_lc(self):
        # NOTE: this overwrites mean_lc. Need to do deep copy
        shifted_lc = self.grid.mean_lc[1]
        lc_interp = {}

        lum_to_flux = self.redshift * 4 * np.pi * self.f_b * 1e45
        tshifts = [8, 8, 8]

        for burst in range(1, self.n_bursts+1):
            shifted_lc[burst][:, 0] *= self.redshift
            shifted_lc[burst][:, 0] += tshifts[burst-1]
            shifted_lc[burst][:, 1] *= 1 / lum_to_flux
            shifted_lc[burst][:, 2] *= 1 / lum_to_flux

            lc_interp[burst] = {}
            lc_interp[burst]['flux'] = interp1d(shifted_lc[burst][:, 0], shifted_lc[burst][:, 1])
            lc_interp[burst]['flux_err'] = interp1d(shifted_lc[burst][:, 0], shifted_lc[burst][:, 2])

        self.shifted_lc = shifted_lc
        self.interp_lc = lc_interp
    
    def plot(self, residuals=True):
        fig, ax = plt.subplots(self.n_bursts, 2, sharex=True, figsize=(20, 12))

        for burst in range(self.n_bursts):
            obs_burst = self.bfit.obs[burst]
            obs_x = np.array(obs_burst.time)
            obs_y = np.array(obs_burst.flux)
            obs_y_u = np.array(obs_burst.flux_err)

            model = self.shifted_lc[burst+1]

            m_x = model[:, 0]
            m_y = model[:, 1]
            m_y_u = model[:, 2]
            m_y_upper = m_y + m_y_u
            m_y_lower = m_y - m_y_u

            # ====== Plot lightcurves ======
            ax[burst][0].fill_between(m_x, m_y_lower, m_y_upper, color='0.7')
            ax[burst][0].plot(m_x, m_y, color='black')
            ax[burst][0].errorbar(obs_x, obs_y, yerr=obs_y_u, ls='none', capsize=3, color='C1')

            # ====== Plot residuals ======
            if residuals:
                y_residuals = obs_y - self.interp_lc[burst+1]['flux'](obs_x)
                ax[burst][1].fill_between(m_x, -m_y_u, m_y_u, color='0.7')
                ax[burst][1].plot([-1e3, 1e3], [0, 0], color='black')
                ax[burst][1].errorbar(obs_x, y_residuals, yerr=obs_y_u, ls='none', capsize=3, color='C1')

        ax[-1][0].set_xlabel('Time (s)', fontsize=20)
        ax[1][0].set_ylabel(r'Flux (erg cm$^{-2}$ s$^{-1}$)', fontsize=20)
        ax[-1][0].set_xlim([-10, 200])
        plt.tight_layout()
        plt.show(block=False)
