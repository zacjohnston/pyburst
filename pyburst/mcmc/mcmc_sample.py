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
                 verbose=True, n_bursts=1,
                 fit_tail_only=False, n_points=None):
        self.source = source
        self.n_epochs = len(batches)
        self.obs_source = obs_sources[source]
        self.grid = grid_analyser.Kgrid(self.source)
        self.bfit = burstfit.BurstFit(mcmc_source, version=mcmc_version, re_interp=False)
        self.obs = ctools.load_obs(self.obs_source)
        self.batches = batches
        self.params = load_param_sample(self.source, self.batches)
        self.verbose = verbose
        self.n_bursts = n_bursts  # no. bursts to get from each model

        self.xlims = {'gs1826': (-10, 170),
                      '4u1820': (-2, 27),
                      }.get(self.obs_source)

        self.epochs = {'gs1826': (1998, 2000, 2007),
                       }.get(self.obs_source)

        if runs is None:  # assume all batches have corresponding runs
            sub_batch = self.grid.get_params(self.batches[0])
            self.runs = np.array(sub_batch['run'])
        else:
            self.runs = runs

        self.n_runs = len(self.runs)
        self.n_bursts_batch = self.n_runs * self.n_bursts

        self.n_points = n_points
        if self.n_points is None:
            self.n_points = {'gs1826': 200,
                             '4u1820': 500}.get(self.obs_source)

        self.peak_i = np.zeros(self.n_epochs, dtype=int)
        if fit_tail_only:
            self.get_peak_indexes()

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

    def get_peak_indexes(self):
        """Get indexes for peak bins of each obs epoch
        """
        for epoch_i in range(self.n_epochs):
            self.peak_i[epoch_i] = np.argmax(self.obs[epoch_i].flux)

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
        if self.n_bursts is 1:  # use mean lightcurves
            for batch in self.batches:
                self.grid.load_mean_lightcurves(batch)
                self.loaded_lc[batch] = self.grid.mean_lc[batch]
        else:
            for batch in self.batches:
                burst_count = 1
                self.loaded_lc[batch] = {}  # to contain every burst
                self.grid.load_burst_lightcurves(batch)
                lc_batch = self.grid.burst_lc[batch]

                for run_n in lc_batch:
                    n_bursts_run = int(self.grid.get_summ(batch, run_n).num)
                    burst_start = n_bursts_run + 1 - self.n_bursts
                    if burst_start < 1:
                        raise ValueError(f'Fewer than n_bursts in model '
                                         f'run={run_n}, batch={batch}')

                    for burst in range(burst_start, n_bursts_run+1):
                        burst_lc = lc_batch[run_n][burst-1]
                        padded_lc = np.zeros([len(burst_lc), 3])
                        padded_lc[:, :2] = burst_lc

                        self.loaded_lc[batch][burst_count] = padded_lc
                        burst_count += 1

    def extract_lc(self):
        """Extracts mean lightcurves from models and shifts to observer according to
            sample parameters
        """
        for batch in self.batches:
            self.shifted_lc[batch] = {}
            self.interp_lc[batch] = {}

            for burst, burst_lc in self.loaded_lc[batch].items():
                run = int(np.floor(burst / self.n_bursts))
                if self.verbose:
                    sys.stdout.write('\rExtracting and shifting model lightcurves: '
                                     f'Batch {batch} : '
                                     f'burst {burst}/{self.n_bursts_batch}')

                self.shifted_lc[batch][burst] = np.array(burst_lc)
                lc = self.shifted_lc[batch][burst]
                t = lc[:, 0]
                lum = lc[:, 1:3]

                params = self.params[run-1]
                params_dict = self.bfit.get_params_dict(params)

                lc[:, 0] = 3600 * self.bfit.shift_to_observer(values=t, bprop='dt',
                                                              params=params_dict)
                lc[:, 1:3] = self.bfit.shift_to_observer(values=lum, bprop='peak',
                                                         params=params_dict)

                flux = lum[:, 0]
                flux_err = lum[:, 1]

                self.interp_lc[batch][burst] = {}
                self.interp_lc[batch][burst]['flux'] = interp1d(t, flux, bounds_error=False,
                                                                fill_value=0)
                self.interp_lc[batch][burst]['flux_err'] = interp1d(t, flux_err,
                                                                    bounds_error=False,
                                                                    fill_value=0)
            if self.verbose:
                sys.stdout.write('\n')

    def get_all_tshifts(self):
        """Gets best t_shift for all bursts
        """
        t_shifts = np.full((self.n_epochs, self.n_bursts_batch), np.nan)

        for epoch_i in range(self.n_epochs):
            for i in range(self.n_bursts_batch):
                burst = i + 1
                if self.verbose:
                    sys.stdout.write('\rOptimising time shifts: '
                                     f'epoch {epoch_i + 1}, burst {burst}/{self.n_bursts_batch}')
                t_shifts[epoch_i, i] = self.fit_tshift(burst=burst, epoch_i=epoch_i)

            if self.verbose:
                sys.stdout.write('\n')

        self.t_shifts = t_shifts

    def fit_tshift(self, burst, epoch_i):
        """Finds LC tshift that minimises chi^2

        Note: assumes epoch_i correspond to index of batches
        """
        min_tshift = -60
        max_tshift = 60
        t = np.linspace(min_tshift, max_tshift, self.n_points)
        chi2 = np.zeros_like(t)

        for i in range(self.n_points):
            chi2[i] = self.chi_squared(t[i], epoch_i=epoch_i, burst=burst)

        min_idx = np.argmin(chi2)
        return t[min_idx]

    def chi_squared(self, tshift, epoch_i, burst):
        """Returns chi^2 of model vs. observed lightcurves
        """
        obs_burst = self.obs[epoch_i]
        peak_i = int(self.peak_i[epoch_i])
        obs_x = np.array(obs_burst.time + 0.5*obs_burst.dt)[peak_i:]
        obs_flux = np.array(obs_burst.flux)[peak_i:]
        obs_flux_err = np.array(obs_burst.flux_err)[peak_i:]

        batch = self.batches[epoch_i]
        model_interp = self.interp_lc[batch][burst]
        model_flux = model_interp['flux'](obs_x - tshift)
        model_flux_err = model_interp['flux_err'](obs_x - tshift)

        return np.sum((obs_flux - model_flux)**2 / np.sqrt(obs_flux_err**2 + model_flux_err**2))

    def plot(self, residuals=True, shaded=False, alpha_lines=0.5,
             alpha_shaded=0.7, xlims=None,
             k_color='C9', obs_color='black', errorbars=False,
             sub_figsize=None, linewidth=1, display=True,
             all_ylabels=True, epoch_text=True, bounds=False):
        """Plot lightcurve comparison between observed and sample models
        """
        subplot_cols = {True: 2, False: 1}.get(residuals)
        if xlims is None:
            xlims = self.xlims
        if sub_figsize is None:
            sub_figsize = (6 * subplot_cols, 2.33 * self.n_epochs)
        fig, ax = plt.subplots(self.n_epochs, subplot_cols, sharex=True,
                               figsize=sub_figsize)
        y_scale = 1e-8
        ylabel = r'Flux ($10^{-8}$ erg cm$^{-2}$ s$^{-1}$)'

        if residuals:
            lc_ax = ax[:, 0]
            res_ax = ax[:, 1]
            res_ax[-1].set_xlabel('Time (s)')
        else:
            lc_ax = ax[:]
            res_ax = None

        for epoch_i in range(self.n_epochs):
            batch = self.batches[epoch_i]
            obs_burst = self.obs[epoch_i]
            obs_x = np.array(obs_burst.time + 0.5*obs_burst.dt)
            obs_y = np.array(obs_burst.flux) / y_scale
            obs_y_u = np.array(obs_burst.flux_err) / y_scale

            # ====== Labelling =====
            if all_ylabels:
                lc_ax[epoch_i].set_ylabel(ylabel)

            if epoch_text:
                lc_ax[epoch_i].text(0.95, 0.9, str(self.epochs[epoch_i]),
                                    transform=lc_ax[epoch_i].transAxes,
                                    fontsize=16, va='top', ha='right')
            for i in range(self.n_bursts_batch):
                burst = i + 1
                model = self.shifted_lc[batch][burst]
                t_shift = self.t_shifts[epoch_i, i]

                m_x = model[:, 0] + t_shift
                m_y = model[:, 1] / y_scale
                m_y_u = model[:, 2] / y_scale
                m_y_upper = m_y + m_y_u
                m_y_lower = m_y - m_y_u

                # ====== Plot model lightcurves ======
                if shaded:
                    lc_ax[epoch_i].fill_between(m_x, m_y_lower, m_y_upper,
                                                color='0.7', alpha=alpha_shaded)
                if bounds:
                    lc_ax[epoch_i].plot(m_x, m_y_lower, ls='-', color='0.',
                                        alpha=alpha_shaded, linewidth=0.5)
                    lc_ax[epoch_i].plot(m_x, m_y_upper, ls='-', color='0.',
                                        alpha=alpha_shaded, linewidth=0.5)

                lc_ax[epoch_i].plot(m_x, m_y, color=k_color, alpha=alpha_lines,
                                    linewidth=linewidth)

                # ====== Plot residuals ======
                if residuals:
                    res_ax[epoch_i].set_ylabel(r'Residuals '
                                               r'($10^{-8}$ erg cm$^{-2}$ s$^{-1}$)')
                    y_residuals = (self.interp_lc[batch][burst]['flux'](obs_x-t_shift)
                                   / y_scale - obs_y)
                    y_residuals_err = (self.interp_lc[batch][burst]['flux_err']
                                       (obs_x-t_shift)) / y_scale

                    res_ax[epoch_i].plot(obs_x, y_residuals, color=k_color,
                                         alpha=alpha_lines, zorder=0,
                                         linewidth=linewidth)

                    if shaded:
                        res_ax[epoch_i].fill_between(obs_x, y_residuals - y_residuals_err,
                                                     y_residuals + y_residuals_err,
                                                     color='0.7', alpha=alpha_shaded)

            # ====== Plot observed lightcurves ======
            lc_ax[epoch_i].step(obs_burst.time, obs_y,
                                where='post', color=obs_color)

            if errorbars:
                lc_ax[epoch_i].errorbar(obs_x, obs_y, yerr=obs_y_u, ls='none',
                                        capsize=3, color=obs_color, zorder=10)
            if residuals:
                res_ax[epoch_i].errorbar(obs_x, np.zeros_like(obs_x), yerr=obs_y_u,
                                         ls='none', capsize=3, color=obs_color,
                                         zorder=10, linewidth=0.5*linewidth)

        if not all_ylabels:
            lc_ax[1].set_ylabel(ylabel, labelpad=10)

        lc_ax[-1].set_xlabel('Time (s)')
        lc_ax[-1].set_xlim(xlims)
        # plt.tight_layout()
        if display:
            plt.show(block=False)
        return fig


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
