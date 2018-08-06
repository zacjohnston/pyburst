import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from scipy import interpolate, integrate
from scipy.signal import argrelextrema
from scipy.stats import linregress

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
                      'converged': False,
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
        self.slopes = {}    # NOTE: starts at min_discard
        self.slopes_err = {}
        self.residuals = {}
        self.discard = None

        if analyse:
            self.analyse()
        if plot:
            self.plot()

    def printv(self, string):
        if self.options['verbose']:
            print(string)

    def print_warn(self, string):
        full_string = f"\nWARNING: {string}\n"
        self.printv(full_string)

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

    def clean_bursts(self, exclude_short_wait=None, exclude_outliers=None,
                     exclude_min_regress=False, exclude_discard=False):
        """Returns subset of self.bursts that are not in min_discard,
            and (depending on exclude options), not outliers or short_waits

        parameters
        ----------
        exclude_short_wait : bool (optional)
            if not provided, fall back on self.options
        exclude_outliers : bool (optional)
            if not provided, fall back on self.options
        exclude_min_regress : bool (optional)
        """
        if exclude_short_wait is None:
            exclude_short_wait = self.options['exclude_short_wait']
        if exclude_outliers is None:
            exclude_outliers = self.options['exclude_outliers']

        mask = np.full(self.n_bursts, True)
        mask[:self.min_discard] = False

        if exclude_short_wait:
            mask = mask & np.invert(self.bursts['short_wait'])
        if exclude_outliers:
            mask = mask & np.invert(self.bursts['outlier'])
        if exclude_discard:
            mask[:self.discard] = False

        if exclude_min_regress:
            return self.bursts[mask].iloc[:-self.min_regress + 1]
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

    def set_converged_too_few(self):
        self.flags['converged'] = False
        self.flags['regress_too_few_bursts'] = True

    def load(self):
        """Load luminosity data from kepler simulation
        """
        self.lum = burst_tools.load(run=self.run, batch=self.batch, source=self.source,
                                    basename=self.basename, save=self.options['save_lum'],
                                    reload=self.options['reload'])

        self.lumf = interpolate.interp1d(self.lum[:, 0], self.lum[:, 1])
        self.flags['loaded'] = True

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

        self.get_burst_starts()
        self.get_burst_ends()

        self.identify_short_wait_bursts()
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
                              ' with self.plot(shocks=True)')

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

        self.bursts['lum_start'] = self.lum[self.bursts['t_start_i'], 1]

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
                                       + f'(t={peak_t:.0f} s)')
            else:
                end_i = np.min(intersection)
                t_end = lum_slice[end_i, 0]
                self.bursts.loc[burst.Index, 't_end'] = t_end
                self.bursts.loc[burst.Index, 't_end_i'] = np.searchsorted(self.lum[:, 0], t_end)

        self.bursts['lum_end'] = self.lum[self.bursts['t_end_i'], 1]

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
            self.printv(f'{self.n_short_wait} short-wait bursts detected')
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

        clean_dt = self.clean_bursts(exclude_outliers=False)['dt']
        percentiles = burst_tools.get_quartiles(clean_dt)

        outliers = (self.bursts['dt'] < percentiles[0]) | (self.bursts['dt'] > percentiles[-1])
        outliers[:self.min_discard] = True

        self.bursts['outlier'] = outliers
        self.n_outliers = len(self.outliers())
        self.n_outliers_unique = len(self.outliers(unique=True))

    def get_bprop_slopes(self):
        """Calculate slopes for properties as the burst sequence progresses
        """
        bursts_regress = self.clean_bursts(exclude_min_regress=True)
        bursts_regress_full = self.clean_bursts(exclude_min_regress=False)

        if len(bursts_regress) > 0:
            for bprop in self.regress_bprops:
                self.bursts[f'slope_{bprop}'] = np.full(self.n_bursts, np.nan)
                self.bursts[f'slope_{bprop}_err'] = np.full(self.n_bursts, np.nan)

                for burst in bursts_regress.itertuples():
                    regress_slice = bursts_regress_full[burst.Index:]
                    lin = linregress(regress_slice['n'], regress_slice[bprop])

                    self.bursts.loc[burst.Index, f'slope_{bprop}'] = lin[0]
                    self.bursts.loc[burst.Index, f'slope_{bprop}_err'] = lin[-1]

        else:
            minimum = (self.min_regress + self.min_discard
                       + self.n_short_wait + self.n_outliers_unique)
            self.print_warn(f'Too few bursts to do linregress. '
                            + f'Have {self.n_bursts}, need at least {minimum} '
                            + '(assuming no further outliers/short_waits occur)')
            self.set_converged_too_few()

    def get_discard(self):
        """Returns min no. of bursts to discard, to achieve zero slope in bprops
        """
        if self.flags['regress_too_few_bursts']:
            self.printv('Too few bursts to find self.discard, defaulting to min_discard')
            return self.min_discard

        bursts = self.clean_bursts(exclude_min_regress=True)
        zero_slope = np.full(len(bursts), True)

        for bprop in self.regress_bprops:
            residuals = np.abs(bursts[f'slope_{bprop}'] / bursts[f'slope_{bprop}_err'])
            zero_slope = zero_slope & (residuals < 1)

        for burst_i, flat in zero_slope.iteritems():
            if flat:
                self.flags['converged'] = True
                return burst_i
        else:
            self.print_warn('Bursts not yet converged, using largest discard to satisfy min_regress')
            self.flags['converged'] = False
            return bursts.index[-1]

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
            bursts = self.clean_bursts(exclude_discard=True)
            for bprop in self.bprops:
                values = bursts[bprop]
                self.summary[f'mean_{bprop}'] = np.mean(values)
                self.summary[f'std_{bprop}'] = np.std(values)

            self.summary['mean_rate'] = sec_day / self.summary['mean_dt']  # burst rate (per day)
            self.summary['std_rate'] = sec_day * self.summary['std_dt'] / self.summary['mean_dt']**2

    def plot(self, peaks=True, display=True, save=False, log=True,
             burst_stages=False, candidates=False, legend=False, time_unit='h',
             short_wait=False, shocks=False, fontsize=14, title=True,
             outliers=False, show_all=False):
        """Plots overall model lightcurve, with detected bursts
        """
        timescale = {'s': 1, 'm': 60, 'h': 3600, 'd': 8.64e4}.get(time_unit, 1)
        time_label = {'s': 's', 'm': 'min', 'h': 'hr', 'd': 'day'}.get(time_unit, 's')
        markersize = 10
        markeredgecolor = '0'

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlabel(f'Time ({time_label})', fontsize=fontsize)

        if show_all:
            burst_stages = True
            candidates = True
            short_wait = True
            shocks = True
            outliers = True

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

        if not self.flags['analysed']:
            self.print_warn('Model not analysed. Only plotting raw lightcurve')
            self.show_save_fig(fig, display=display, save=save, plot_name='model')
            return

        if candidates:  # NOTE: candidates may be modified if a shock was removed
            x = self.candidates[:, 0] / timescale
            y = self.candidates[:, 1] / yscale
            ax.plot(x, y, marker='o', c='C4', ls='none', label='Candidates',
                    markersize=markersize, markeredgecolor=markeredgecolor)

        if peaks:
            ax.plot(self.bursts['t_peak']/timescale, self.bursts['peak']/yscale, marker='o', c='C1', ls='none',
                    label='Bursts', markeredgecolor=markeredgecolor, markersize=markersize)

        if outliers:
            bursts = self.outliers()
            x = bursts['t_peak'] / timescale
            y = bursts['peak'] / yscale
            ax.plot(x, y, marker='o', c='C9', ls='none', label='Outliers',
                    markeredgecolor=markeredgecolor, markersize=markersize)

        if short_wait:
            if self.flags['short_waits']:
                bursts = self.short_waits()
                x = bursts['t_peak'] / timescale
                y = bursts['peak'] / yscale
                ax.plot(x, y, marker='o', c='C0', ls='none', label='Short-wait',
                        markersize=markersize, markeredgecolor=markeredgecolor)

        if burst_stages:
            for stage in ('pre', 'start', 'end'):
                x = self.bursts[f't_{stage}'] / timescale
                y = self.bursts[f'lum_{stage}'] / yscale
                label = {'pre': 'Burst stages'}.get(stage, None)
                ax.plot(x, y, marker='o', c='C2', ls='none', label=label,
                        markersize=markersize, markeredgecolor=markeredgecolor)

        if shocks:  # plot shocks that were removed
            for i, shock in enumerate(self.shocks):
                idx = int(shock[0])
                shock_lum = shock[2]

                shock_slice = self.lum[idx-1:idx+2, :]
                shock_slice[1, 1] = shock_lum
                ax.plot(shock_slice[:, 0]/timescale, shock_slice[:, 1]/yscale, c='C3',
                        label='shocks' if (i == 0) else '_nolegend_')

        if legend:
            ax.legend(loc=1, framealpha=1, edgecolor='0')
        self.show_save_fig(fig, display=display, save=save, plot_name='model')

    def plot_convergence(self, bprops=('dt', 'fluence', 'peak'), discard=None,
                         legend=False, display=True, save=False, fix_xticks=False):
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

        fig, ax = plt.subplots(3, 1, figsize=(6, 8))
        bursts = self.clean_bursts()
        bursts_discard = self.clean_bursts(exclude_discard=True)

        for i, bprop in enumerate(bprops):
            y_unit = y_units.get(bprop)
            y_scale = y_scales.get(bprop, 1.0)
            ax[i].set_ylabel(f'{bprop} ({y_unit})')

            if fix_xticks:
                ax[i].set_xticks(self.bursts['n'])
                if i != len(bprops)-1:
                    ax[i].set_xticklabels([])

            for burst in bursts_discard.itertuples():
                bslice = bursts_discard.loc[:burst.Index][bprop]
                mean = np.mean(bslice)
                std = np.std(bslice)
                ax[i].errorbar(burst.Index, mean/y_scale, yerr=std/y_scale, ls='none',
                               marker='o', c='C0', capsize=3,
                               label='cumulative mean' if burst.Index == bursts_discard.index[0] else '_nolegend_')

            self.printv(f'{bprop}: mean={mean:.3e}, std={std:.3e}, frac={std/mean:.3f}')
            ax[i].plot(bursts['n'], bursts[bprop] / y_scale, marker='o',
                       c='C1', ls='none', label='Bursts')
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
        bursts = self.clean_bursts(exclude_min_regress=True)
        x = bursts['n']
        fontsize = 14

        for i, bprop in enumerate(self.regress_bprops):
            y = bursts[f'slope_{bprop}']
            y_err = bursts[f'slope_{bprop}_err']
            ax[i].set_ylabel(bprop, fontsize=fontsize)
            ax[i].errorbar(x, y, yerr=y_err, ls='none', marker='o', capsize=3)
            ax[i].plot([0, self.n_bursts], [0, 0], ls='--')

        ax[-1].set_xlabel('Discarded bursts', fontsize=fontsize)
        ax[0].set_title(self.model_str)
        plt.tight_layout()
        self.show_save_fig(fig, display=display, save=save, plot_name='linregress')

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
