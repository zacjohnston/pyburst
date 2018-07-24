import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize

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
        self.t_shifts = None

        self.extract_lc()

    def extract_lc(self):
        # NOTE: this overwrites mean_lc. Need to do deep copy
        shifted_lc = self.grid.mean_lc[1]
        lc_interp = {}

        lum_to_flux = self.redshift * 4 * np.pi * self.f_b * 1e45
        # tshifts = [8, 8, 8]

        for burst in range(1, self.n_bursts+1):
            shifted_lc[burst][:, 0] *= self.redshift
            # shifted_lc[burst][:, 0] += tshifts[burst-1]
            shifted_lc[burst][:, 1] *= 1 / lum_to_flux
            shifted_lc[burst][:, 2] *= 1 / lum_to_flux

            lc_interp[burst] = {}
            lc_interp[burst]['flux'] = interp1d(shifted_lc[burst][:, 0], shifted_lc[burst][:, 1])
            lc_interp[burst]['flux_err'] = interp1d(shifted_lc[burst][:, 0], shifted_lc[burst][:, 2])

        self.shifted_lc = shifted_lc
        self.interp_lc = lc_interp

    def fit_tshift(self, burst, n_points=500):
        """Finds LC tshift that minimises chi^2
        """
        obs = self.bfit.obs[burst]
        model = self.shifted_lc[burst + 1]

        min_tshift = (obs.time[-1].value + 0.5*obs.dt[-1].value
                      - model[-1, 0])
        max_tshift = (obs.time[0].value + 0.5*obs.dt[0].value
                      - model[0, 0])

        t = np.linspace(min_tshift, max_tshift, n_points)
        chi2 = np.zeros_like(t)

        for i in range(n_points):
            chi2[i] = self.compare(t[i], burst=burst)

        min_idx = np.argmin(chi2)
        return t[min_idx]

    def compare(self, tshift, burst):
        """Returns chi^2 of model vs. observed lightcurves
        """
        obs_burst = self.bfit.obs[burst]
        obs_x = np.array(obs_burst.time + 0.5*obs_burst.dt)
        obs_flux = np.array(obs_burst.flux)
        obs_flux_err = np.array(obs_burst.flux_err)

        model = self.interp_lc[burst+1]
        model_flux = model['flux'](obs_x - tshift)
        model_flux_err = model['flux_err'](obs_x - tshift)

        return np.sum((obs_flux - model_flux)**2 / np.sqrt(obs_flux_err**2 + model_flux_err**2))

    def plot(self, residuals=True):
        fig, ax = plt.subplots(self.n_bursts, 2, sharex=True, figsize=(20, 12))

        for burst in range(self.n_bursts):
            obs_burst = self.bfit.obs[burst]
            obs_x = np.array(obs_burst.time + 0.5*obs_burst.dt)
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
