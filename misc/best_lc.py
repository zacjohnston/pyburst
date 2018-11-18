import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from pyburst.grids import grid_analyser
from pyburst.mcmc import burstfit, mcmc_tools


class Best:
    """Testing comparisons of LC from 'best' MCMC sample,
    against observed LC
    """
    def __init__(self, source='grid4', source2='grid4', version=1,
                 runs=(9, 10, 11), batches=(5, 5, 5), n_walkers=960, n_steps=1000):
        self.grid = grid_analyser.Kgrid(source2)
        self.bfit = burstfit.BurstFit(source, version, re_interp=False)
        self.runs = runs
        self.batches = batches
        self.best_params = mcmc_tools.get_max_lhood_params(source, version=version,
                                                           n_walkers=n_walkers,
                                                           n_steps=n_steps)
        red_idx = self.bfit.param_idxs['redshift']
        fb_idx = self.bfit.param_idxs['f_b']
        self.redshift = self.best_params[red_idx]
        self.f_b = self.best_params[fb_idx]

        self.n_epochs = len(runs)
        self.shifted_lc = {}
        self.interp_lc = {}
        self.t_shifts = None

        self.extract_lc()
        self.get_all_tshifts()

    def extract_lc(self):
        lum_to_flux = self.redshift * 4 * np.pi * self.f_b * 1e45

        for batch in np.unique(self.batches):
            self.grid.load_mean_lightcurves(batch)

        for i in range(self.n_epochs):
            epoch = i+1
            run = self.runs[i]
            batch = self.batches[i]
            self.shifted_lc[epoch] = np.array(self.grid.mean_lc[batch][run])
            self.shifted_lc[epoch][:, 0] *= self.redshift
            self.shifted_lc[epoch][:, 1:3] *= 1 / lum_to_flux

            t = self.shifted_lc[epoch][:, 0]
            flux = self.shifted_lc[epoch][:, 1]
            flux_err = self.shifted_lc[epoch][:, 2]

            self.interp_lc[epoch] = {}
            self.interp_lc[epoch]['flux'] = interp1d(t, flux, bounds_error=False,
                                                     fill_value=0)
            self.interp_lc[epoch]['flux_err'] = interp1d(t, flux_err, bounds_error=False,
                                                         fill_value=0)

    def get_all_tshifts(self):
        """Gets best t_shift for all bursts
        """
        t_shifts = np.full(self.n_epochs, np.nan)

        for i in range(self.n_epochs):
            t_shifts[i] = self.fit_tshift(burst=i)

        self.t_shifts = t_shifts

    def fit_tshift(self, burst, n_points=500):
        """Finds LC tshift that minimises chi^2
        """
        obs = self.bfit.obs[burst]
        model = self.shifted_lc[burst + 1]
        # TODO: Bug when model LC shorter than obs LC
        # min_tshift = (obs.time[-1].value + 0.5*obs.dt[-1].value
        #               - model[-1, 0])
        # max_tshift = (obs.time[0].value + 0.5*obs.dt[0].value
        #               - model[0, 0])
        min_tshift = -60
        max_tshift = 60
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
        fig, ax = plt.subplots(self.n_epochs, 2, sharex=True, figsize=(20, 12))

        for burst in range(self.n_epochs):
            obs_burst = self.bfit.obs[burst]
            obs_x = np.array(obs_burst.time + 0.5*obs_burst.dt)
            obs_y = np.array(obs_burst.flux)
            obs_y_u = np.array(obs_burst.flux_err)

            model = self.shifted_lc[burst+1]
            t_shift = self.t_shifts[burst]

            m_x = model[:, 0] + t_shift
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
                y_residuals = obs_y - self.interp_lc[burst+1]['flux'](obs_x - t_shift)
                ax[burst][1].fill_between(m_x, -m_y_u, m_y_u, color='0.7')
                ax[burst][1].errorbar(obs_x, y_residuals, yerr=obs_y_u, ls='none', capsize=3, color='C1')
                ax[burst][1].plot([-1e3, 1e3], [0, 0], color='black')

        ax[-1][0].set_xlabel('Time (s)', fontsize=20)
        ax[1][0].set_ylabel(r'Flux (erg cm$^{-2}$ s$^{-1}$)', fontsize=20)
        ax[-1][0].set_xlim([-10, 200])
        plt.tight_layout()
        plt.show(block=False)
