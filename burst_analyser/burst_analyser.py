import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from scipy import interpolate, integrate
from scipy.signal import argrelextrema
from scipy.stats import linregress
from functools import reduce

# kepler_grids
from . import burst_tools
from ..grids import grid_tools, grid_strings

GRIDS_PATH = os.environ['KEPLER_GRIDS']
MODELS_PATH = os.environ['KEPLER_MODELS']


def default_plt_options():
    """Initialise default plot parameters"""
    params = {'mathtext.default': 'regular',
              'font.family': 'serif', 'text.usetex': False}
    plt.rcParams.update(params)


default_plt_options()

# TODO: Generalise to non-batch organised models
# TODO: param description docstring


class NoBursts(Exception):
    pass


class BurstRun(object):
    def __init__(self, run, batch, source, verbose=True, basename='xrb',
                 reload=False, save_lum=True, analyse=True, plot=False,
                 min_regress=20, min_discard=1, exclude_outliers=False,
                 exclude_short_wait=True):
        # min_regress : int
        #   minimum number of bursts to use in linear regression (self.linregress)
        # min_discard : int
        #   minimum no. of bursts to discard when averaging

        self.flags = {'loaded': False,
                      'analysed': False,
                      'too_few_bursts': False,
                      'short_waits': False,
                      'outliers': False,
                      'regress_too_few_bursts': False,
                      }

        self.options = {'verbose': verbose,
                        'reload': reload,
                        'save_lum': save_lum,
                        'exclude_outliers': exclude_outliers,
                        'exclude_short_wait': exclude_short_wait,
                        }
        self.run = run
        self.batch = batch
        self.source = source
        self.basename = basename
        self.run_str = grid_strings.get_run_string(run, basename)
        self.batch_str = grid_strings.get_batch_string(batch, source)
        self.model_str = grid_strings.get_model_string(run, batch, source)

        self.batch_models_path = grid_strings.get_batch_models_path(batch, source)
        self.analysis_path = grid_strings.get_source_subdir(source, 'burst_analysis')
        self.source_path = grid_strings.get_source_path(source)
        self.plots_path = grid_strings.get_source_subdir(source, 'plots')

        self.lum = None
        self.lumf = None
        self.new_lum = None
        self.load()

        self.bursts = pd.DataFrame()
        self.candidates = None
        self.summary = {}
        self.n_bursts = None
        self.n_short_wait = None
        self.n_outliers = None
        self.n_outliers_unique = None
        self.bprops = ['dt', 'fluence', 'peak', 'length']
        self.shocks = []

        # ===== linregress things =====
        self.regress_bprops = ['dt', 'fluence', 'peak']
        self.min_regress = min_regress
        self.min_discard = min_discard
        self.n_regress = None
        self.slopes = {}    # NOTE: starts at min_discard
        self.slopes_err = {}
        self.residuals = {}
        self.discard = None
        self.converged = None

        if analyse:
            self.analyse()
        if plot:
            self.plot_model()

    def printv(self, string):
        if self.options['verbose']:
            print(string)

    def print_warn(self, string):
        full_string = f"\nWARNING: {string}\n"
        self.printv(full_string)

    def set_converged_too_few(self):
        self.converged = False
        self.flags['regress_too_few_bursts'] = True

    def clean_bursts(self, exclude_min_regress=False):
        """Returns subset of self.bursts that are not outliers, short_waits, or min_discard
        """
        mask = np.invert(self.bursts['short_wait']) & np.invert(self.bursts['outlier'])
        mask.iloc[:self.min_discard] = False

        if exclude_min_regress:
            return self.bursts[mask].iloc[:-self.min_regress]
        else:
            return self.bursts[mask]

    def short_waits(self):
        """Returns subset of self.bursts that are classified as short_wait
        """
        mask = self.bursts['short_wait']
        return self.bursts[mask]

    def not_short_waits(self):
        """Returns subset of self.bursts that are NOT classified as short_wait
        """
        mask = np.invert(self.bursts['short_wait'])
        return self.bursts[mask]

    def outliers(self, unique=False):
        """Returns subset of self.bursts that are outliers

        unique : bool
            whether to exclude bursts already identified as short_waits or min_discard
        """
        if unique:
            mask = self.bursts['outlier'] & np.invert(self.bursts['short_wait'])
            mask.iloc[:self.min_discard] = False
        else:
            mask = self.bursts['outlier']

        return self.bursts[mask]

    def not_outliers(self):
        return self.bursts[np.invert(self.bursts['outlier'])]

    def regress_bursts(self):
        """Return subset of self.bursts to use for linregress
        """
        pass

    def load(self):
        """Load luminosity data from kepler simulation
        """
        self.lum = burst_tools.load(run=self.run, batch=self.batch, source=self.source,
                                    basename=self.basename, save=self.options['save_lum'],
                                    reload=self.options['reload'])

        self.lumf = interpolate.interp1d(self.lum[:, 0], self.lum[:, 1])
        self.flags['loaded'] = True

    def analyse(self):
        """Performs complete analysis of model.
        """
        self.ensure_analysed_is(False)
        self.identify_bursts()
        self.get_fluences()
        self.identify_outliers()
        self.get_bprop_slopes()
        self.discard = self.get_discard()
        self.get_means()
        self.flags['analysed'] = True

    def ensure_analysed_is(self, analysed):
        """Checks that model has (or hasn't) been analysed
        """
        strings = {True: 'Model not yet analysed. Run self.analyse() first',
                   False: 'Model has already been analysed. Reload model first'}

        if self.flags['analysed'] != analysed:
            if self.flags['too_few_bursts']:
                string = 'Too few bursts for analysis'
            else:
                string = strings[analysed]
            raise AttributeError(string)

    def identify_bursts(self):
        """Extracts peaks, times, and recurrence times of bursts

         Pipeline:
         ---------
           1. Get maxima above minimum threshold
           2. Discard shock peaks
           3. Get largest peaks in some radius
           4. Identify short-wait bursts (below some fraction of mean dt)
           5. Get start/end times (discard final burst if cut off)
        """
        self.printv('Identifying bursts')
        self.get_burst_candidates()

        try:
            self.get_burst_peaks()
        except NoBursts:
            return

        if self.n_bursts > 1:
            dt = np.diff(self.bursts['t_peak'])
            self.bursts['dt'] = np.concatenate(([np.nan], dt))  # Recurrence times (s)
            self.identify_short_wait_bursts()

        self.get_burst_starts()
        self.get_burst_ends()
        self.bursts.reset_index(inplace=True, drop=True)

        self.bursts['length'] = self.bursts['t_end'] - self.bursts['t_start']
        self.bursts['n'] = np.arange(self.n_bursts) + 1  # burst ID (starting from 1)

    def get_burst_candidates(self):
        """Identify potential bursts, while removing shocks in lightcurve
        """
        maxima_change = 999
        self.candidates = [1]
        count = 0
        while maxima_change != 0:
            old_maxima = len(self.candidates)
            self.candidates = self.get_lum_maxima()
            self.remove_shocks(self.candidates)

            maxima_change = old_maxima - len(self.candidates)
            count += 1

        print(f'Maxima iterations: {count}')
        self.shocks = np.array(self.shocks)

    def get_lum_maxima(self):
        """Returns all maxima in luminosity above lum_thresh
        """
        lum_thresh = 1e36  # minimum threshold luminosity
        thresh_i = np.where(self.lum[:, 1] > lum_thresh)[0]
        lum_cut = self.lum[thresh_i]

        maxima_i = argrelextrema(lum_cut[:, 1], np.greater)[0]
        return lum_cut[maxima_i]

    def remove_shocks(self, maxima):
        """Cut out convective shocks (extreme spikes in luminosity).
        Identifies spikes, and replaces them with interpolation from neighbours.

        parameters
        ----------
        maxima : nparray(n,2)
            local maxima to check (t, lum)
        """
        radius = 2  # radius of neighbour zones to compare against
        tolerance = 2.0  # maxima should not be more than this factor larger than neighbours
        self.remove_zeros()

        # ----- Discard if maxima more than [tolerance] larger than all neighbours -----
        shocks = False
        for max_i in maxima:
            t, lum = max_i
            idx = np.searchsorted(self.lum[:, 0], t)

            left = self.lum[idx-radius: idx, 1]  # left neighbours
            right = self.lum[idx+1: idx+radius+1, 1]  # right neighbours
            neighbours = np.concatenate([left, right])

            if True in (lum > tolerance*neighbours):
                if self.options['verbose']:
                    if not shocks:
                        print('Shocks detected and removed: consider verifying'
                              ' with self.plot_model(shocks=True)')

                new_lum = 0.5 * (left[-1] + right[0])  # mean of two neighbours
                self.lum[idx, 1] = new_lum
                max_i[1] = new_lum
                self.shocks.append([idx, t, lum])
                shocks = True

    def remove_zeros(self):
        """During shocks, kepler can also give zero luminosity (for some reason...)
        """
        replace_with = 1e35
        zeros = np.where(self.lum[:, 1] == 0.0)
        n_zeros = len(zeros)
        self.printv(f'Removed {n_zeros} zeros from luminosity')
        self.lum[zeros, 1] = replace_with

    def get_burst_peaks(self):
        """Keep largest maxima within some time-window
        """
        t_radius = 60  # burst peak must be largest maxima within t_radius (s)
        peaks = []

        for maxi in self.candidates:
            t, lum = maxi
            i_left = np.searchsorted(self.lum[:, 0], t - t_radius)
            i_right = np.searchsorted(self.lum[:, 0], t + t_radius)

            maxx = np.max(self.lum[i_left:i_right, 1])
            if maxx == lum:
                peaks.append(maxi)

        peaks = np.array(peaks)
        self.n_bursts = len(peaks)

        if self.n_bursts < 2:
            self.flags['too_few_bursts'] = True
            message = {0: 'No bursts in this model',
                       1: 'Only one burst detected'}[self.n_bursts]
            self.print_warn(message)

            if self.n_bursts == 0:
                raise NoBursts

        self.bursts['t_peak'] = peaks[:, 0]  # times of burst peaks (s)
        self.bursts['t_peak_i'] = np.searchsorted(self.lum[:, 0], self.bursts['t_peak'])
        self.bursts['peak'] = peaks[:, 1]  # Peak luminosities (erg/s)

    def get_burst_starts(self):
        """Finds first point in lightcurve that reaches a given fraction of the peak
        """
        pre_time = 30  # time (s) before burst peak that should always contain burst rise
        start_frac = 0.25  # Burst start as fraction of peak lum
        peak_frac = 10

        self.bursts['t_pre'] = self.bursts['t_peak'] - pre_time  # time before burst (s)
        self.bursts['t_pre_i'] = np.searchsorted(self.lum[:, 0], self.bursts['t_pre'])
        self.bursts['lum_pre'] = self.lum[self.bursts['t_pre_i'], 1]

        self.bursts['t_start'] = np.full(self.n_bursts, np.nan)
        self.bursts['t_start_i'] = np.zeros(self.n_bursts, dtype=int)

        for burst in self.bursts.itertuples():
            rise_steps = burst.t_peak_i - burst.t_pre_i
            if (rise_steps < 50
                    or burst.peak / burst.lum_pre < peak_frac):
                self.printv(f"Removing micro-burst (t={burst.t_peak:.0f} s)")
                self.delete_burst(burst.Index)
                continue

            lum_slice = self.lum[burst.t_pre_i:burst.t_peak_i]
            pre_lum = lum_slice[0, 1]
            peak_lum = lum_slice[-1, 1]
            start_lum = pre_lum + start_frac * (peak_lum - pre_lum)

            slice_i = np.searchsorted(lum_slice[:, 1], start_lum)
            t_start = lum_slice[slice_i, 0]
            self.bursts.loc[burst.Index, 't_start'] = t_start
            self.bursts.loc[burst.Index, 't_start_i'] = np.searchsorted(self.lum[:, 0], t_start)

    def get_burst_ends(self):
        """Finds first point in lightcurve > min_length after peak that falls
        to a given fraction of luminosity
        """
        end_frac = 0.01  # end of burst defined when luminosity falls to this fraction of peak
        min_length = 5  # minimum length of burst after peak (s)
        self.bursts['t_end'] = np.full(self.n_bursts, np.nan)
        self.bursts['t_end_i'] = np.zeros(self.n_bursts, dtype=int)

        for burst in self.bursts.itertuples():
            lum_slice = self.lum[burst.t_peak_i:]
            pre_lum = self.lum[burst.t_pre_i, 1]

            peak_t, peak_lum = lum_slice[0]
            lum_diff = peak_lum - pre_lum

            time_from_peak = lum_slice[:, 0] - peak_t
            threshold_lum = pre_lum + (end_frac * lum_diff)
            thresh_i = np.where(lum_slice[:, 1] < threshold_lum)[0]

            min_length_i = np.where(time_from_peak > min_length)[0]
            intersection = list(set(thresh_i).intersection(min_length_i))

            if len(intersection) == 0:
                if burst.Index == (self.n_bursts - 1):
                    self.printv('File ends during burst. Discarding final burst')
                    self.delete_burst(burst.Index)
                    continue
                else:
                    raise RuntimeError(f'Failed to find end of burst {burst.Index + 1} '
                                       + f'(t={peak_t:.0f s})')
            else:
                end_i = np.min(intersection)
                t_end = lum_slice[end_i, 0]
                self.bursts.loc[burst.Index, 't_end'] = t_end
                self.bursts.loc[burst.Index, 't_end_i'] = np.searchsorted(self.lum[:, 0], t_end)

    def delete_burst(self, burst_i):
        """Removes burst from self.bursts table
        """
        self.bursts = self.bursts.drop(burst_i)
        self.n_bursts -= 1

    def identify_short_wait_bursts(self):
        """Identify bursts which have unusually short recurrence times
        """
        min_dt_frac = 0.5
        mean_dt = np.mean(self.bursts['dt'][1:])
        self.bursts['short_wait'] = self.bursts['dt'] < min_dt_frac * mean_dt
        self.n_short_wait = len(self.short_waits())

        if self.n_short_wait > 0:
            self.printv(f'{self.n_short_wait} short-waiting time bursts detected')
            self.flags['short_waits'] = True

    def get_fluences(self):
        """Calculates burst fluences by integrating over burst luminosity
        """
        self.bursts['fluence'] = np.zeros(self.n_bursts)
        for burst in self.bursts.itertuples():
            lum_slice = self.lum[burst.t_pre_i:burst.t_end_i]
            self.bursts.loc[burst.Index, 'fluence'] = integrate.trapz(y=lum_slice[:, 1],
                                                                      x=lum_slice[:, 0])

    def identify_outliers(self):
        """Identify outlier bursts

        Note: bursts up to min_discard and short_waits will not be considered
                in the calculation of the mean
        """
        if self.flags['too_few_bursts']:
            self.printv('Too few bursts to get outliers')
            return

        dt = self.bursts['dt']
        percentiles = burst_tools.get_quartiles(dt[self.min_discard:])
        outliers = (dt < percentiles[0]) | (dt > percentiles[-1])
        outliers[:self.min_discard] = True
        self.bursts['outlier'] = outliers
        self.n_outliers = len(self.outliers())
        self.n_outliers_unique = len(self.outliers(unique=True))

    def get_bprop_slopes(self):
        """Calculate slopes for properties as the burst sequence progresses
        """
        self.get_n_regress()
        if self.n_regress > 0:
            for bprop in self.regress_bprops:
                slopes, slopes_err = self.linregress(bprop)
                self.residuals[bprop] = np.abs(slopes / slopes_err)
                self.slopes[bprop], self.slopes_err[bprop] = slopes, slopes_err
        else:
            self.set_converged_too_few()
            self.printv(f'Too few bursts to do linregress. '
                        + f'Have {self.n_bursts}, need {self.min_regress + self.min_discard}')

    def get_n_regress(self):
        """Determine number of potential discards to try with linregress
        """
        self.n_regress = self.n_bursts + 1 - self.min_regress - self.min_discard

        if self.options['exclude_short_wait']:
            self.n_regress -= self.n_short_wait

        if self.options['exclude_outliers']:
            self.n_regress -= len(self.outliers(unique=True))

    def linregress(self, bprop):
        """Do linear regression on given bprop for different number of burst discards
        """
        n = self.n_regress
        y = self.bursts[bprop]
        x = np.arange(len(y))

        if self.options['exclude_outliers']:
            idxs = np.array(self.outlier_i)

        slope = np.full(n, np.nan)
        slope_err = np.full(n, np.nan)

        for i in range(n):
            i0 = self.min_discard + i
            lin = linregress(x[i0:], y[i0:])
            slope[i] = lin[0]
            slope_err[i] = lin[-1]

        return slope, slope_err

    def get_discard(self):
        """Returns min no. of bursts to discard, to achieve zero slope in bprops
        """
        if self.flags['regress_too_few_bursts']:
            self.printv('Too few bursts to find self.discard, using min_discard')
            return self.min_discard

        zero_slope_i = []
        for bprop in self.regress_bprops:
            zero_slope_i += [self.min_discard + np.where(self.residuals[bprop] < 1)[0]]

        valid_discards = reduce(np.intersect1d, zero_slope_i)

        if len(valid_discards) == 0:
            self.print_warn('Bursts not converged, using min_discard')
            self.converged = False
            return self.min_discard
        else:
            self.converged = True
            return valid_discards[0]

    def get_means(self):
        """Calculate mean burst properties
        """
        sec_day = 8.64e4

        if self.flags['too_few_bursts']:
            self.printv("Too few bursts to get average properties")
            for bprop in (self.bprops + ['rate']):
                self.summary[f'mean_{bprop}'] = np.nan
                self.summary[f'std_{bprop}'] = np.nan
        else:
            for bprop in self.bprops:
                values = self.bursts[bprop][self.discard:]

                if self.options['exclude_outliers']:
                    idxs = np.array(self.outlier_i) - self.discard
                    if bprop == 'dt':
                        idxs -= 1
                    idxs = idxs[np.where(idxs >= 0)[0]]  # discard negatives
                    values = np.delete(values, idxs)

                self.summary[f'mean_{bprop}'] = np.mean(values)
                self.summary[f'std_{bprop}'] = np.std(values)

            # ===== calculate burst rate =====
            dt = self.summary['mean_dt']
            u_dt = self.summary['std_dt']
            self.summary['mean_rate'] = sec_day / dt  # burst rate (per day)
            self.summary['std_rate'] = sec_day * u_dt / dt**2

    def show_save_fig(self, fig, display, save, plot_name,
                      path=None, extra='', extension='png'):
        """Displays and/or Saves given figure

        parameters
        ----------
        fig : plt.Figure object
        display : bool
        save : bool
            save the figure to file (to fold in checking from other functions)
        plot_name : str
            type of plot being saved
        path : str (optional)
            path of diretcory to save to.
            If not provided, assumes there exists a folder [source]/plots/[plot_name]
        extra : str (optional)
            optional string to attach to filename
        extension : str (optional)
        """

        if save:
            filename = f'{plot_name}_{self.model_str}{extra}.{extension}'

            if path is None:
                path = os.path.join(self.source_path, 'plots', plot_name)
            filepath = os.path.join(path, filename)

            self.printv(f'Saving figure: {filepath}')
            grid_tools.try_mkdir(path, skip=True)
            fig.savefig(filepath)

        if display:
            plt.show(block=False)
        else:
            plt.close(fig)

    def save_burst_lightcurves(self, path=None):
        """Saves burst lightcurves to txt files. Excludes 'pre' bursts
        """
        self.ensure_analysed_is(True)
        if path is None:  # default to model directory
            path = self.batch_models_path

        for i in range(self.n_bursts):
            bnum = i + 1

            i_start = self.bursts['t_pre_i'][i]
            i_zero = self.bursts['t_start_i'][i]
            i_end = self.bursts['t_end_i'][i]

            t = self.lum[i_start:i_end, 0] - self.lum[i_zero, 0]
            lum = self.lum[i_start:i_end, 1]
            uncertainty = 0.02
            u_lum = lum * uncertainty

            lightcurve = np.array([t, lum, u_lum]).transpose()
            header = 'time luminosity u_luminosity'
            b_file = f'b{bnum}.txt'
            filepath = os.path.join(path, b_file)

            np.savetxt(filepath, lightcurve, header=header)

    def plot_model(self, bursts=True, display=True, save=False, log=True,
                   burst_stages=False, candidates=False, legend=False, time_unit='h',
                   short_wait=False, shocks=False, fontsize=14, title=True,
                   show_all=False):
        """Plots overall model lightcurve, with detected bursts
        """
        self.ensure_analysed_is(True)
        timescale = {'s': 1, 'm': 60, 'h': 3600, 'd': 8.64e4}.get(time_unit, 1)
        time_label = {'s': 's', 'm': 'min', 'h': 'hr', 'd': 'day'}.get(time_unit, 's')

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlabel(f'Time ({time_label})', fontsize=fontsize)

        if show_all:
            burst_stages = True
            candidates = True
            short_wait = True
            shocks = True

        if title:
            ax.set_title(self.model_str)

        if log:
            yscale = 1
            ax.set_yscale('log')
            ax.set_ylim([1e34, 1e40])
            ax.set_ylabel(f'Luminosity (erg s$^{-1}$)', fontsize=fontsize)
        else:
            ax.set_ylabel(f'Luminosity ($10^{38}$ erg s$^{-1}$)', fontsize=fontsize)
            yscale = 1e38

        ax.plot(self.lum[:, 0]/timescale, self.lum[:, 1]/yscale, c='black')

        if candidates:  # NOTE: candidates may be modified if a shock was removed
            t = self.candidates[:, 0] / timescale
            y = self.candidates[:, 1] / yscale
            ax.plot(t, y, marker='o', c='C0', ls='none', label='candidates')

        if short_wait:
            if self.flags['short_waits']:
                t = self.bursts['short_wait_peaks'][:, 0] / timescale
                y = self.bursts['short_wait_peaks'][:, 1] / yscale
                ax.plot(t, y, marker='o', c='C4', ls='none', label='short-wait')

        if burst_stages:
            for stage in ('t_pre', 't_start', 't_end'):
                t = self.bursts[stage] / timescale
                y = self.lumf(t) / yscale
                label = {'t_pre': 'stages'}.get(stage, None)

                ax.plot(t, y, marker='o', c='C2', ls='none', label=label)

        if shocks:  # plot shocks that were removed
            for i, shock in enumerate(self.shocks):
                idx = int(shock[0])
                shock_lum = shock[2]

                shock_slice = self.lum[idx-1:idx+2, :]
                shock_slice[1, 1] = shock_lum
                ax.plot(shock_slice[:, 0]/timescale, shock_slice[:, 1]/yscale, c='C3',
                        label='shocks' if (i == 0) else '_nolegend_')
        if bursts:
            ax.plot(self.bursts['t_peak']/timescale, self.bursts['peak']/yscale, marker='o', c='C1', ls='none',
                    label='peaks')

        if legend:
            ax.legend(loc=4)
        self.show_save_fig(fig, display=display, save=save, plot_name='model')

    def plot_convergence(self, bprops=('dt', 'fluence', 'peak'), discard=None,
                         show_values=True, legend=False, show_first=False,
                         display=True, save=False, fix_xticks=False):
        """Plots individual and average burst properties along the burst sequence
        """
        self.ensure_analysed_is(True)
        if discard is None:
            discard = self.discard

        if self.n_bursts < discard+2:
            print('Too few bursts to plot convergence')
            return

        y_units = {'tDel': 'hr', 'dt': 'hr', 'fluence': '10^39 erg',
                   'peak': '10^38 erg/s'}
        y_scales = {'tDel': 3600, 'dt': 3600,
                    'fluence': 1e39, 'peak': 1e38}

        b_start = {True: 1, False: 2}.get(show_first)
        fig, ax = plt.subplots(3, 1, figsize=(6, 8))

        for i, bprop in enumerate(bprops):
            y_unit = y_units.get(bprop)
            y_scale = y_scales.get(bprop, 1.0)
            ax[i].set_ylabel(f'{bprop} ({y_unit})')

            if fix_xticks:
                ax[i].set_xticks(np.arange(b_start, self.n_bursts+1))
                if i != len(bprops)-1:
                    ax[i].set_xticklabels([])

            b_vals = self.bursts[bprop]
            nv = len(b_vals)

            for j in range(discard+1, nv+1):
                b_slice = b_vals[discard:j]
                mean = np.mean(b_slice)
                std = np.std(b_slice)

                ax[i].errorbar(j, mean/y_scale, yerr=std/y_scale, ls='none',
                               marker='o', c='C0', capsize=3,
                               label='cumulative mean' if j == discard+1 else '_nolegend_')

            self.printv(f'{bprop}: mean={mean:.3e}, std={std:.3e}, frac={std/mean:.3f}')
            if show_values:
                ax[i].plot(np.arange(b_start, nv+1), b_vals[b_start-1:]/y_scale,
                           marker='o', c='C1', ls='none', label='bursts')
        if legend:
            ax[0].legend(loc=1)

        ax[0].set_title(self.model_str)
        ax[-1].set_xlabel('Burst number')
        plt.tight_layout()
        self.show_save_fig(fig, display=display, save=save, plot_name='convergence')

    def plot_linregress(self, display=True, save=False):
        if self.flags['regress_too_few_bursts']:
            self.printv("Can't plot linregress: bursts not converged")
            return
        fig, ax = plt.subplots(3, 1, figsize=(6, 8))
        x = np.arange(self.n_regress) + self.min_discard
        fontsize = 14

        for i, bprop in enumerate(self.regress_bprops):
            y = self.slopes[bprop]
            y_err = self.slopes_err[bprop]
            ax[i].set_ylabel(bprop, fontsize=fontsize)
            ax[i].errorbar(x, y, yerr=y_err, ls='none', marker='o', capsize=3)
            ax[i].plot([0, self.n_bursts], [0, 0], ls='--')

        ax[-1].set_xlabel('Discarded bursts', fontsize=fontsize)
        ax[0].set_title(self.model_str)
        plt.tight_layout()
        self.show_save_fig(fig, display=display, save=save, plot_name='linregress')

    def plot_lightcurves(self, bursts, save=False, display=True, log=False,
                         zero_time=True, fontsize=14):
        """Plot individual burst lightcurve

        parameters
        ----------
        bursts : [int]
            list of burst indices to plot
        save : bool
        display : bool
        log : bool
        zero_time : bool
        fontsize : int
        """
        self.ensure_analysed_is(True)
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.set_ylabel('Luminosity ($10^{38}$ erg s$^{-1}$)', fontsize=fontsize)
        ax.set_xlabel('Time (s)', fontsize=fontsize)
        if log:
            ax.set_yscale('log')
            ax.set_ylim([1e34, 1e39])

        for burst in bursts:
            self.add_lightcurve(burst, ax, zero_time=zero_time)

        ax.set_xlim(xmin=-5)
        plot_path = os.path.join(self.plots_path, 'lightcurves', self.batch_str)

        self.show_save_fig(fig, display=display, save=save, plot_name='lightcurve',
                           path=plot_path, extra='')

    def add_lightcurve(self, burst, ax, zero_time=True):
        """Add a lightcurve to the provided matplotlib axis

        parameters
        ----------
        burst : int
            index of burst to add (e.g. 0 for first burst)
        ax : matplotlib axis
            axis object to add lightcurves to
        zero_time : bool
        """
        yscale = 1e38
        if burst > self.n_bursts - 1\
                or burst < 0:
            raise ValueError(f'Burst index ({burst}) out of bounds '
                             f'(n_bursts={self.n_bursts})')

        i_start = self.bursts['t_pre_i'][burst]
        i_end = self.bursts['t_end_i'][burst]
        x = self.lum[i_start:i_end, 0]
        y = self.lum[i_start:i_end, 1]

        if zero_time:
            x = x - self.bursts['t_start'][burst]
        ax.plot(x, y / yscale, c='C0', label=f'{burst}')

    def save_all_lightcurves(self, **kwargs):
        for burst in range(self.n_bursts):
            self.plot_lightcurves(burst, save=True, display=False, **kwargs)


