import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from scipy import interpolate, integrate
from scipy.signal import argrelextrema
import multiprocessing as mp
import time

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


class BurstRun(object):
    def __init__(self, run, batch, source, verbose=True, basename='xrb',
                 re_load=False, savelum=True, analyse=True, plot=False):
        self.run = run
        self.batch = batch
        self.source = source
        self.basename = basename
        self.run_str = grid_strings.get_run_string(run, basename)
        self.batch_str = grid_strings.get_batch_string(batch, source)
        self.model_str = grid_strings.get_model_string(run, batch, source)
        self.verbose = verbose

        self.batch_models_path = grid_strings.get_batch_models_path(batch, source)
        self.analysis_path = grid_strings.get_source_subdir(source, 'burst_analysis')
        self.source_path = grid_strings.get_source_path(source)
        self.plots_path = grid_strings.get_source_subdir(source, 'plots')

        self.loaded = False
        self.lum = None
        self.lumf = None
        self.load(savelum=savelum, re_load=re_load)

        self.analysed = False
        self.bursts = {}
        self.n_bursts = None
        self.outliers = np.array(())
        self.secondary_bursts = np.array(())
        self.shocks = []
        self.short_waits = False

        if analyse:
            self.analyse()

        if plot:
            self.plot_model()

    def printv(self, string):
        if self.verbose:
            print(string)

    def load(self, savelum=True, re_load=False):
        """Load luminosity data from kepler simulation
        """
        self.lum = burst_tools.load(run=self.run, batch=self.batch, source=self.source,
                                    basename=self.basename, save=savelum, re_load=re_load)
        self.lumf = interpolate.interp1d(self.lum[:, 0], self.lum[:, 1])
        self.loaded = True

    def analyse(self):
        """Analyses all quantities of the model.
        """
        self.ensure_analysed_is(False)
        self.identify_bursts()
        self.find_fluence()
        self.analysed = True

    def ensure_analysed_is(self, analysed):
        """Checks that model has (or hasn't) been analysed
        """
        strings = {True: 'Model not yet analysed. Run self.analyse() first',
                   False: 'Model has already been analysed. Reload model first'}

        if self.analysed != analysed:
            raise AttributeError(strings[analysed])

    def remove_zeros(self):
        """During shocks, kepler can also give zero luminosity (for some reason...)
        """
        replace_with = 1e35
        zeros = np.where(self.lum[:, 1] == 0.0)
        n_zeros = len(zeros)
        self.printv(f'Removed {n_zeros} zeros from luminosity')
        self.lum[zeros, 1] = replace_with

    def remove_shocks(self, maxima):
        """Cut out convective shocks (extreme spikes in luminosity).
        Identifies spikes, and replaces them with interpolation from neighbours.
        maxima : nparray(n,2)
            local maxima to check (t, lum)
        """
        radius = 1  # radius of neighbour zones to compare against
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
                if self.verbose:
                    if not shocks:
                        print('Shocks detected and removed: consider verifying'
                              ' with self.plot_model(shocks=True)')

                new_lum = 0.5 * (left[-1] + right[0])  # mean of two neighbours
                self.lum[idx, 1] = new_lum
                max_i[1] = new_lum
                self.shocks.append([idx, t, lum])
                shocks = True

        self.shocks = np.array(self.shocks)

    def identify_bursts(self):
        """Extracts times, separations, and mean separation of bursts
        """
        pre_time = 30  # time (s) before burst peak that should always contain burst rise
        min_dt_frac = 0.5  # minimum recurrence time (as fraction of mean)
        self.printv('Identifying bursts')

        # ===== get maxima in luminosity curve =====
        candidates = self.get_lum_maxima()
        self.remove_shocks(candidates)

        # ===== determine bursts from maxima =====
        peaks = self.get_burst_peaks(candidates)
        peak_idxs = self.get_peak_idxs(peaks)
        n_bursts = len(peaks)

        # ===== get dt, and discard short-wait bursts =====
        if n_bursts > 1:
            dt = np.diff(peaks[:, 0])
            mean_dt = np.mean(dt)
            short_wait = (dt < min_dt_frac * mean_dt)

            if True in short_wait:
                short_idxs = np.where(short_wait)[0] + 1
                n_short = len(short_idxs)
                self.printv(f'{n_short} short waiting-time burst detected. Discarding')
                self.bursts['short_wait_peaks'] = peaks[short_idxs]
                self.short_waits = True

                peaks = np.delete(peaks, short_idxs, axis=0)
                peak_idxs = np.delete(peak_idxs, short_idxs)
                dt = np.delete(dt, short_idxs-1)
                n_bursts -= n_short
        else:
            dt = [np.nan]
            if n_bursts == 0:
                print('\nWARNING: No bursts in this model!\n')
            else:
                self.printv('Only one burst detected')

        # ===== find burst start and end =====
        t_pre = peaks[:, 0] - pre_time
        t_pre_idx = np.searchsorted(self.lum[:, 0], t_pre)
        t_start = []
        t_start_idx = []
        t_end = []
        t_end_idx = []

        for i, pre_idx in enumerate(t_pre_idx):
            start_idx = self.get_burst_start_idx(pre_idx, peak_idxs[i])
            end_idx = self.get_burst_end_idx(peak_idxs[i])

            if end_idx is None:
                self.printv('Discarding final burst')
                dt = np.delete(dt, -1)
                peaks = np.delete(peaks, -1, axis=0)
                peak_idxs = np.delete(peak_idxs, -1)
                t_pre_idx = np.delete(t_pre_idx, -1)
                t_pre = np.delete(t_pre, -1)
                n_bursts -= 1
            else:
                t_start_idx.append(start_idx)
                t_start.append(self.lum[start_idx, 0])
                t_end.append(self.lum[end_idx, 0])
                t_end_idx.append(end_idx)

        t_start = np.array(t_start)
        t_start_idx = np.array(t_start_idx, dtype=int)
        t_end = np.array(t_end)
        t_end_idx = np.array(t_end_idx, dtype=int)

        self.bursts['candidates'] = candidates
        self.bursts['t_pre'] = t_pre  # pre_time before burst peak (s)
        self.bursts['t_start'] = t_start  # time of reaching start_frac of burst peak
        self.bursts['t_peak'] = peaks[:, 0]  # times of burst peaks (s)
        self.bursts['t_end'] = t_end  # Time of burst end (end_frac of peak) (s)
        self.bursts['length'] = t_end - t_start  # Burst lengths (s)

        self.bursts['t_pre_idx'] = t_pre_idx
        self.bursts['t_start_idx'] = t_start_idx
        self.bursts['peak_idx'] = peak_idxs
        self.bursts['t_end_idx'] = t_end_idx

        self.bursts['peak'] = peaks[:, 1]  # Peak luminosities (erg/s)
        self.bursts['dt'] = dt  # Recurrence times (s)
        self.n_bursts = n_bursts

    def get_lum_maxima(self):
        """Returns all maxima in luminosity above lum_thresh
        """
        lum_thresh = 1e36  # minimum threshold luminosity
        thresh_idxs = np.where(self.lum[:, 1] > lum_thresh)[0]
        lum_cut = self.lum[thresh_idxs]

        maxima_idxs = argrelextrema(lum_cut[:, 1], np.greater)[0]
        return lum_cut[maxima_idxs]

    def get_burst_peaks(self, maxima):
        """Identify which maxima are burst peaks
        """
        t_radius = 60  # burst peak must be largest maxima within t_radius (s)
        peaks = []

        for maxi in maxima:
            t, lum = maxi
            i_left = np.searchsorted(self.lum[:, 0], t - t_radius)
            i_right = np.searchsorted(self.lum[:, 0], t + t_radius)

            maxx = np.max(self.lum[i_left:i_right, 1])
            if maxx == lum:
                peaks.append(maxi)

        return np.array(peaks)  # Each entry contains (t, lum)

    def get_peak_idxs(self, peaks):
        """Returns array of indexes from self.lum corresponding to given peaks
        """
        idxs = []
        for peak in peaks:
            t = peak[0]
            idx = np.searchsorted(self.lum[:, 0], t)
            idxs.append(idx)

        return np.array(idxs, dtype=int)

    def get_burst_start_idx(self, pre_idx, peak_idx):
        """Finds first point in lightcurve that reaches a given fraction of the peak
        """
        start_frac = 0.25  # Burst start as fraction of peak lum

        lum_slice = self.lum[pre_idx:peak_idx]
        peak_lum = self.lum[peak_idx, 1]

        start_i = np.where(lum_slice[:, 1] > start_frac*peak_lum)[0][0]
        start_t = lum_slice[start_i, 0]
        return np.searchsorted(self.lum[:, 0], start_t)

    def get_burst_end_idx(self, peak_idx):
        """Finds first point in lightcurve > min_length after peak that falls
        to a given fraction of luminosity
        """
        end_frac = 0.005  # end of burst defined when luminosity falls to this fraction of peak
        min_length = 5  # minimum length of burst after peak (s)

        peak_t = self.lum[peak_idx, 0]
        peak_lum = self.lum[peak_idx, 1]
        lum_slice = self.lum[peak_idx:]

        time_from_peak = lum_slice[:, 0] - peak_t
        thresh_idx = np.where(lum_slice[:, 1] < end_frac*peak_lum)[0]
        min_length_idx = np.where(time_from_peak > min_length)[0]
        intersection = list(set(thresh_idx).intersection(min_length_idx))

        if len(intersection) == 0:
            self.printv('File ends during burst')
            return None
        else:
            end_idx = np.min(intersection)   # this is index fro lum_slice
            end_t = lum_slice[end_idx, 0]   # need to get idx for full self.lum array
            return np.searchsorted(self.lum[:, 0], end_t)

    def find_fluence(self):
        """Calculates burst fluences by integrating over burst luminosity
        """
        fluences = np.zeros(self.n_bursts)
        for i in range(self.n_bursts):
            t0 = self.bursts['t_pre_idx'][i]
            t1 = self.bursts['t_end_idx'][i]
            fluences[i] = integrate.trapz(y=self.lum[t0:t1 + 1, 1], x=self.lum[t0:t1 + 1, 0])

        self.bursts['fluence'] = fluences  # Burst fluence (ergs)

    # def identify_outliers(self):

    def save_bursts(self, path=None):
        """Saves burst lightcurves to txt files. Excludes 'pre' bursts
        """
        if path is None:  # default to model directory
            path = self.batch_models_path

        n = self.bursts['num']
        for i in range(n):
            bnum = i + 1

            i_start = self.bursts['t_pre_idx'][i]
            i_zero = self.bursts['t_start_idx'][i]
            i_end = self.bursts['t_end_idx'][i]

            time = self.lum[i_start:i_end, 0] - self.lum[i_zero, 0]
            lum = self.lum[i_start:i_end, 1]
            uncertainty = 0.02
            u_lum = lum * uncertainty

            lightcurve = np.array([time, lum, u_lum]).transpose()
            header = 'time luminosity u_luminosity'
            b_file = f'b{bnum}.txt'
            filepath = os.path.join(path, b_file)

            np.savetxt(filepath, lightcurve, header=header)

    def plot_model(self, bursts=True, display=True, save=False, log=True,
                   burst_stages=False, candidates=False, legend=False, time_unit='h',
                   short_wait=False, shocks=False):
        """Plots overall model lightcurve, with detected bursts
        """
        self.ensure_analysed_is(True)
        timescale = {'s': 1, 'm': 60, 'h': 3600, 'd': 8.64e4}.get(time_unit, 1)
        time_label = {'s': 's', 'm': 'min', 'h': 'hr', 'd': 'day'}.get(time_unit, 's')
        yscale = 1e38

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title(f'{self.model_str}')
        ax.set_xlabel(f'Time ({time_label})')
        ax.set_ylabel(f'Luminosity ( {yscale} erg/s)')

        ax.plot(self.lum[:, 0]/timescale, self.lum[:, 1]/yscale, c='black')

        if log:
            ax.set_yscale('log')
            ax.set_ylim([1e-4, 1e2])

        if candidates:  # NOTE: candidates may be modified if a shock was removed
            t = self.bursts['candidates'][:, 0] / timescale
            y = self.bursts['candidates'][:, 1] / yscale
            ax.plot(t, y, marker='o', c='C0', ls='none', label='candidates')

        if short_wait:
            if self.short_waits:
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
            first_shock = True
            for shock in self.shocks:
                idx = int(shock[0])
                shock_lum = shock[2]

                shock_slice = self.lum[idx-1:idx+2, :]
                shock_slice[1, 1] = shock_lum
                ax.plot(shock_slice[:, 0]/timescale, shock_slice[:, 1]/yscale, c='C3',
                        label='shocks' if first_shock else '_nolegend_')
                first_shock = False
        if bursts:
            ax.plot(self.bursts['t_peak']/timescale, self.bursts['peak']/yscale, marker='o', c='C1', ls='none',
                    label='peaks')

        if legend:
            ax.legend(loc=4)
        if display:
            plt.show(block=False)
        if save:
            filename = f'model_{self.model_str}.png'
            filepath = os.path.join(self.source_path, 'plots', 'burst_analysis', filename)
            fig.savefig(filepath)
        return fig

    def plot_lightcurve(self, burst, save=False, display=True, log=False,
                        zero_time=True):
        """Plot individual burst lightcurve
        """
        if burst > self.n_bursts - 1\
                or burst < 0:
            raise ValueError(f'Burst index ({burst}) out of bounds '
                             f'(n_bursts={self.n_bursts})')

        fig, ax = plt.subplots()
        ax.set_title(f'Burst {burst}')
        ax.set_ylabel('Lum (erg/s)')
        ax.set_xlabel('Time (s)')

        if log:
            ax.set_yscale('log')
            ax.set_ylim([1e34, 1e39])

        i_start = self.bursts['t_pre_idx'][burst]
        i_end = self.bursts['t_end_idx'][burst]
        x = self.lum[i_start:i_end, 0]
        y = self.lum[i_start:i_end, 1]

        if zero_time:
            x -= self.bursts['t_start'][burst]
        ax.plot(x, y, c='C0')

        if display:
            plt.show(block=False)
        if save:
            filename = f'burst_{self.model_str}_{burst:02}.png'
            plot_dir = os.path.join(self.plots_path, 'lightcurves', self.batch_str)
            filepath = os.path.join(plot_dir, filename)

            grid_tools.try_mkdir(plot_dir, skip=True)
            fig.savefig(filepath)
        return fig

    def save_all_lightcurves(self, **kwargs):
        for burst in range(self.n_bursts):
            fig = self.plot_lightcurve(burst, save=True, display=False, **kwargs)
            plt.close(fig)


def multithread_extract(batches, source, plot=True):
    args = []
    for batch in batches:
        args.append([batch, source, plot])

    t0 = time.time()
    with mp.Pool(processes=8) as pool:
        pool.starmap(extract_burstfit_1808, args)
    t1 = time.time()
    dt = t1 - t0
    print(f'Time taken: {dt:.1f} s')


def extract_burstfit_1808(batches, source, plot=True, skip_bursts=1):
    source_path = grid_strings.get_source_path(source)
    batches = grid_tools.expand_batches(batches, source)

    b_ints = ('batch', 'run', 'num')
    bprops = ('dt', 'fluence', 'length', 'peak')
    col_order = ['batch', 'run', 'num', 'dt', 'u_dt', 'fluence', 'u_fluence',
                 'length', 'u_length', 'peak', 'u_peak']

    for batch in batches:
        batch_str = f'{source}_{batch}'
        analysis_path = os.path.join(source_path, 'burst_analysis', batch_str)
        grid_tools.try_mkdir(analysis_path, skip=True)

        filename = f'burst_analysis_{batch_str}.txt'
        filepath = os.path.join(analysis_path, filename)

        data = {}
        for bp in bprops:
            u_bp = f'u_{bp}'
            data[bp] = []
            data[u_bp] = []

        for b in b_ints:
            data[b] = []

        n_runs = grid_tools.get_nruns(batch, source)
        for run in range(1, n_runs + 1):
            sys.stdout.write(f'\r{source}_{batch} xrb{run:02}')
            burstfit = BurstRun(run, batch, source)

            data['batch'] += [batch]
            data['run'] += [run]
            data['num'] += [burstfit.n_bursts]

            for bp in bprops:
                u_bp = f'u_{bp}'
                mean = np.mean(burstfit.bursts[bp][skip_bursts:])
                std = np.std(burstfit.bursts[bp][skip_bursts:])

                data[bp] += [mean]
                data[u_bp] += [std]

            if plot:
                fig = burstfit.plot_model(display=False, save=True)
                plt.close(fig)

        table = pd.DataFrame(data)
        table = table[col_order]
        table_str = table.to_string(index=False, justify='left', col_space=12)

        with open(filepath, 'w') as f:
            f.write(table_str)


def check_n_bursts(batches, source, kgrid):
    """Compares n_bursts detected with kepler_analyser against burstfit_1808
    """
    mismatch = np.zeros(4)
    filename = f'mismatch_{source}_{batches[0]}-{batches[-1]}.txt'
    filepath = os.path.join(GRIDS_PATH, filename)

    for batch in batches:
        summ = kgrid.get_summ(batch)
        n_runs = len(summ)

        for i in range(n_runs):
            run = i + 1
            n_bursts1 = summ.iloc[i]['num']
            sys.stdout.write(f'\r{source}_{batch} xrb{run:02}')

            burstfit = BurstRun(run, batch, source, verbose=False)
            burstfit.analyse()
            n_bursts2 = burstfit.n_bursts

            if n_bursts1 != n_bursts2:
                m_new = np.array((batch, run, n_bursts1, n_bursts2))
                mismatch = np.vstack((mismatch, m_new))

        np.savetxt(filepath, mismatch)
    return mismatch


def plot_convergence(bfit, bprop='dt', discard=1, show_values=True):
    fig, ax = plt.subplots()
    b_vals = bfit.bursts[bprop]
    nv = len(b_vals)

    for i in range(discard, nv):
        b_slice = b_vals[discard:i+1]
        mean = np.mean(b_slice)
        std = np.std(b_slice)

        print(f'mean: {mean:.3e}, std={std:.3e}, frac={std/mean:.3f}')
        # if i != 1:
        #     change = (mean - mean_old) / mean
        #     print(f'Change: {change:.3e}')

        ax.errorbar([i], [mean], yerr=std, ls='none', marker='o', c='C0', capsize=3)
        # mean_old = mean
    if show_values:
        ax.plot(np.arange(nv), b_vals, marker='o', c='C1', ls='none')

    plt.show(block=False)


def plot_bprop(bfit, bprop):
    fig, ax = plt.subplots()
    b_vals = bfit.bursts[bprop]
    nv = len(b_vals)

    ax.plot(np.arange(nv), b_vals, ls='none', marker='o', c='C0')
    plt.show(block=False)


def multi_batch_plot(batches, source, multithread=True):
    if multithread:
        args = []
        for batch in batches:
            args.append((batch, source))
        with mp.Pool(processes=8) as pool:
            pool.starmap(save_batch_plots, args)
    else:
        for batch in batches:
            save_batch_plots(batch, source)


def save_batch_plots(batch, source, **kwargs):
    runs = grid_tools.get_nruns(batch, source)
    runs = grid_tools.expand_runs(runs)
    for run in runs:
        model = BurstRun(run, batch, source, analyse=True)
        fig = model.plot_model(display=False, save=True, **kwargs)
        plt.close(fig)
