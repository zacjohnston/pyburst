import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# kepler_grids
from pyburst.grids import grid_analyser, grid_strings
from pyburst.mcmc import burstfit

# Concord
try:
    import ctools
except ModuleNotFoundError:
    print("pyburst/MCMC: Concord not installed, some functionality won't be available")

obs_sources = {
    'sample5': 'gs1826',
    'sample2': '4u1820',
}


class Ksample:
    """Testing comparisons of LC from 'best' MCMC sample,
    against observed LC
    """
    def __init__(self, source, mcmc_source, mcmc_version, batches, runs=None,
                 verbose=True, single_burst_lc=False, burst_i=None):
        self.source = source
        self.obs_source = obs_sources[source]
        self.grid = grid_analyser.Kgrid(self.source)
        self.bfit = burstfit.BurstFit(mcmc_source, version=mcmc_version, re_interp=False)
        self.obs = ctools.load_obs(self.obs_source)
        self.batches = batches
        self.params = load_param_sample(self.source, self.batches)
        self.verbose = verbose
        self.single_burst_lc = single_burst_lc
        self.burst_i = burst_i

        if runs is None:
            sub_batch = self.grid.get_params(self.batches[0])
            self.runs = np.array(sub_batch['run'])
        else:
            self.runs = runs

        self.n_epochs = len(batches)
        self.loaded_lc = {}
        self.shifted_lc = {}
        self.interp_lc = {}
        self.t_shifts = None

        self.load_model_lc()
        self.extract_lc()
        self.interp_obs_lc()
        self.get_all_tshifts()

    def printv(self, string):
        if self.verbose:
            print(string)

    def interp_obs_lc(self):
        """Creates interpolated lightcurve of observed burst epochs
        """
        self.interp_lc['obs'] = {}
        for epoch_i in range(self.n_epochs):
            if self.verbose:
                sys.stdout.write('\rInterpolating observed burst lightcurves: '
                                 f'{epoch_i + 1}/{self.n_epochs}')

            obs_burst = self.obs[epoch_i]
            obs_x = np.array(obs_burst.time + 0.5 * obs_burst.dt)
            obs_flux = np.array(obs_burst.flux)
            obs_flux_err = np.array(obs_burst.flux_err)

            self.interp_lc['obs'][epoch_i] = {}
            self.interp_lc['obs'][epoch_i]['flux'] = interp1d(obs_x, obs_flux,
                                                              bounds_error=False,
                                                              fill_value=0)
            self.interp_lc['obs'][epoch_i]['flux_err'] = interp1d(obs_x, obs_flux_err,
                                                                  bounds_error=False,
                                                                  fill_value=0)
        if self.verbose:
            sys.stdout.write('\n')

    def load_model_lc(self):
        """Loads model burst lightcurves
        """
        self.printv('Loading model lightcurves')
        # TODO: Use multiple burst LCs from each model (with discard)
        if self.single_burst_lc:
            if self.burst_i is None:
                raise ValueError('Must supply burst_i if single_burst_lc=True')

            for batch in self.batches:
                self.loaded_lc[batch] = {}
                self.grid.load_burst_lightcurves(batch, burst=self.burst_i)
                lc_batch = self.grid.burst_lc[batch]

                for run in lc_batch:
                    burst_lc = lc_batch[run][self.burst_i]

                    padded_lc = np.zeros([len(burst_lc), 3])
                    padded_lc[:, :2] = burst_lc

                    self.loaded_lc[batch][run] = padded_lc
        else:
            for batch in self.batches:
                self.grid.load_mean_lightcurves(batch)
                self.loaded_lc[batch] = self.grid.mean_lc[batch]

    def extract_lc(self):
        """Extracts mean lightcurves from models and shifts to observer according to
            sample parameters
        """
        for batch in self.batches:
            self.shifted_lc[batch] = {}
            self.interp_lc[batch] = {}

            for i, run in enumerate(self.runs):
                if self.verbose:
                    sys.stdout.write('\rExtracting and shifting model lightcurves: '
                                     f'Batch {batch} : run {run}/{len(self.runs)}')
                self.shifted_lc[batch][run] = np.array(self.loaded_lc[batch][run])

                lc = self.shifted_lc[batch][run]
                t = lc[:, 0]
                lum = lc[:, 1:3]

                params = self.params[i]
                params_dict = self.bfit.get_params_dict(params)

                lc[:, 0] = 3600 * self.bfit.shift_to_observer(values=t, bprop='dt',
                                                              params=params_dict)
                lc[:, 1:3] = self.bfit.shift_to_observer(values=lum, bprop='peak',
                                                         params=params_dict)

                flux = lum[:, 0]
                flux_err = lum[:, 1]

                self.interp_lc[batch][run] = {}
                self.interp_lc[batch][run]['flux'] = interp1d(t, flux, bounds_error=False,
                                                              fill_value=0)
                self.interp_lc[batch][run]['flux_err'] = interp1d(t, flux_err,
                                                                  bounds_error=False,
                                                                  fill_value=0)
            if self.verbose:
                sys.stdout.write('\n')

    def get_all_tshifts(self):
        """Gets best t_shift for all bursts
        """
        t_shifts = np.full((self.n_epochs, len(self.runs)), np.nan)

        for epoch_i in range(self.n_epochs):
            for run_i, run in enumerate(self.runs):
                if self.verbose:
                    sys.stdout.write('\rOptimising time shifts: '
                                     f'epoch {epoch_i + 1}, run {run}/{len(self.runs)}')
                t_shifts[epoch_i, run_i] = self.fit_tshift(run=run, epoch_i=epoch_i)

            if self.verbose:
                sys.stdout.write('\n')

        self.t_shifts = t_shifts

    def fit_tshift(self, run, epoch_i, n_points=500):
        """Finds LC tshift that minimises chi^2

        Note: assumes epoch_i correspond to index of batches
        """
        # TODO: Bug when model LC shorter than obs LC
        # batch = self.batches[epoch_i]
        # obs = self.bfit.obs[epoch_i]
        # model = self.shifted_lc[batch][run]
        # min_tshift = (obs.time[-1].value + 0.5*obs.dt[-1].value
        #               - model[-1, 0])
        # max_tshift = (obs.time[0].value + 0.5*obs.dt[0].value
        #               - model[0, 0])
        min_tshift = -60
        max_tshift = 60
        t = np.linspace(min_tshift, max_tshift, n_points)
        chi2 = np.zeros_like(t)

        for i in range(n_points):
            chi2[i] = self.chi_squared(t[i], epoch_i=epoch_i, run=run)

        min_idx = np.argmin(chi2)
        return t[min_idx]

    def chi_squared(self, tshift, epoch_i, run):
        """Returns chi^2 of model vs. observed lightcurves
        """
        obs_burst = self.obs[epoch_i]
        obs_x = np.array(obs_burst.time + 0.5*obs_burst.dt)
        obs_flux = np.array(obs_burst.flux)
        obs_flux_err = np.array(obs_burst.flux_err)

        batch = self.batches[epoch_i]
        model = self.interp_lc[batch][run]
        model_flux = model['flux'](obs_x - tshift)
        model_flux_err = model['flux_err'](obs_x - tshift)

        return np.sum((obs_flux - model_flux)**2 / np.sqrt(obs_flux_err**2 + model_flux_err**2))

    def plot(self, residuals=True, shaded=False, alpha_lines=0.3, alpha_shaded=0.7,
             fontsize=16):
        n_subplots = {True: 2, False: 1}.get(residuals)
        fig, ax = plt.subplots(self.n_epochs, n_subplots, sharex=True, figsize=(14, 10))
        y_scale=1e-8

        if residuals:
            lc_ax = ax[:, 0]
            res_ax = ax[:, 1]
        else:
            lc_ax = ax[:]
            res_ax = None

        for epoch_i in range(self.n_epochs):
            batch = self.batches[epoch_i]
            obs_burst = self.obs[epoch_i]
            obs_x = np.array(obs_burst.time + 0.5*obs_burst.dt)
            obs_y = np.array(obs_burst.flux) / y_scale
            obs_y_u = np.array(obs_burst.flux_err) / y_scale

            for run_i, run in enumerate(self.runs):
                model = self.shifted_lc[batch][run]
                t_shift = self.t_shifts[epoch_i, run_i]

                m_x = model[:, 0] + t_shift
                m_y = model[:, 1] / y_scale
                m_y_u = model[:, 2] / y_scale
                m_y_upper = m_y + m_y_u
                m_y_lower = m_y - m_y_u

                # ====== Plot model lightcurves ======
                if shaded:
                    lc_ax[epoch_i].fill_between(m_x, m_y_lower, m_y_upper, color='0.7',
                                                alpha=alpha_shaded)
                lc_ax[epoch_i].plot(m_x, m_y, color='black', alpha=alpha_lines)

                # ====== Plot residuals ======
                if residuals:
                    res_ax[-1].set_xlabel('Time (s)', fontsize=fontsize)
                    res_ax[1].set_ylabel(r'Residuals ($10^{-8}$ erg cm$^{-2}$ s$^{-1}$)',
                                         fontsize=fontsize)
                    # y_residuals = m_y - self.interp_lc['obs'][epoch_i]['flux'](m_x)
                    y_residuals = self.interp_lc[batch][run]['flux'](obs_x-t_shift) - obs_y
                    y_residuals_err = self.interp_lc[batch][run]['flux_err'](obs_x-t_shift)

                    res_ax[epoch_i].plot(obs_x, y_residuals, color='black', alpha=alpha_lines)

                    if shaded:
                        res_ax[epoch_i].fill_between(obs_x, y_residuals - y_residuals_err,
                                                     y_residuals + y_residuals_err,
                                                     color='0.7', alpha=alpha_shaded)

                    res_ax[epoch_i].errorbar(obs_x, np.zeros_like(obs_x), yerr=obs_y_u,
                                             ls='none', capsize=3, color='C1')

            # ====== Plot observed lightcurves ======
            lc_ax[epoch_i].errorbar(obs_x, obs_y, yerr=obs_y_u, ls='none', capsize=3, color='C1')

        lc_ax[-1].set_xlabel('Time (s)', fontsize=fontsize)
        lc_ax[1].set_ylabel(r'Flux ($10^{-8}$ erg cm$^{-2}$ s$^{-1}$)', fontsize=fontsize)
        lc_ax[-1].set_xlim([-10, 200])
        plt.tight_layout()
        plt.show(block=False)


def plot_batch(source, batch, error=False):
    kgrid = grid_analyser.Kgrid(source=source, linregress_burst_rate=False,
                                load_lc=True)

    table = kgrid.get_params(batch)

    fig, ax = plt.subplots()
    for row in table.itertuples():
        kgrid.add_lc_plot(ax, batch=batch, run=row.run, label=f'{row.run}', error=error)

    plt.tight_layout()
    plt.show(block=False)


def load_param_sample(source, batches):
    filename = f'param_sample_{source}_{batches[0]}-{batches[-1]}.txt'
    path = grid_strings.get_source_path(source)
    filepath = os.path.join(path, filename)

    param_sample = np.loadtxt(filepath)
    return param_sample
