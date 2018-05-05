import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from scipy import interpolate, integrate
import multiprocessing as mp

# kepler_grids
from . import burst_tools
from ..grids import grid_tools, grid_strings

# mdot
import burstfit_1808

GRIDS_PATH = os.environ['KEPLER_GRIDS']
MODELS_PATH = os.environ['KEPLER_MODELS']


class BurstRun(object):
    def __init__(self, run, batch, source, verbose=True, basename='xrb',
                 re_load=False, savelum=True, analyse=True):
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

        if analyse:
            self.analyse()

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
        replace_with = 1e34
        zeros = np.where(self.lum[:, 1] == 0.0)
        n_zeros = len(zeros)
        self.printv(f'Removed {n_zeros} zeros from luminosity')
        self.lum[zeros, 1] = replace_with

    def remove_shocks(self, maxima_idx):
        """Cut out convective shocks (extreme spikes in luminosity).
        Identifies spikes, and replaces them with interpolation from neighbours.
        maxima_idx : []
            list of indices of local maxima to check
        """
        radius = 5  # radius of neighbour zones to compare against
        tolerance = 5e2  # maxima should not be more than this factor larger than neighbours
        self.remove_zeros()

        # ----- Discard if maxima more than [tolerance] larger than mean of neighbours -----
        shocks = False
        for i in maxima_idx:
            left = self.lum[i - radius:i, 1]  # left neighbours
            right = self.lum[i + 1:i + radius + 1, 1]  # right neighbours
            mean = np.mean(np.concatenate([left, right]))

            l = self.lum[i, 1]
            if l > tolerance * mean:
                if self.verbose:
                    if not shocks:
                        print('SHOCKS DETECTED AND REMOVED: Consider verifying')
                        print('    Time(hr)    Lum(erg)   Factor')
                        shocks = True

                    time = self.lum[i, 0] / 3600
                    factor = l / mean
                    self.printv(f'    {time:.2f}      {l:.2e}   {factor:.2f}')

                # --- replace with mean of two neighbours ---
                self.lum[i, 1] = 0.5 * (left[-1] + right[0])

    def identify_bursts(self):
        """Extracts times, separations, and mean separation of bursts
        """
        self.printv('Identifying burst times and peaks in KEPLER model')
        bmax = 1000
        btimes = np.full(bmax, np.nan)  # Detected burst peak times
        candidates = np.full(10000, np.nan)  # Possible burst peaks

        cnum = 0  # candidate counter
        tol = 1  # No. of neighbour zones to consider for maxima
        maxl = 1e38  # typical max peak
        lum_thresh = 0.02  # cut-off luminosity for finding bursts, as fraction of maxl
        t_radius = 30  # minimum time (s) that each burst should be separated by
        pre_time = 30  # time (s) before burst peak to start integrating fluence from
        start_frac = 0.25  # Burst start as fraction of peak lum
        end_frac = 0.005  # end of burst defined when luminosity falls to this fraction of peak
        min_length = 5  # minimum length of burst after peak (s)

        # ===== cut out everything below threshold =====
        b_idx = np.where(self.lum[:, 1] > lum_thresh * maxl)[0]
        n = len(self.lum[:, 0])

        # ===== pick local maxima (larger than +/- tol neighbours) =====
        for i in b_idx[:-1]:
            lum_i = self.lum[i, 1]
            if lum_i > self.lum[i-tol, 1] \
                    and lum_i > self.lum[i+tol, 1]:
                candidates[cnum] = i
                cnum += 1

        candidates = candidates[~np.isnan(candidates)]
        self.remove_shocks(candidates.astype('int'))

        # ===== burst peak if only maxima within t_radius (s) =====
        bnum = 0
        for j, i in enumerate(candidates):
            t = self.lum[int(i), 0]
            t0 = np.searchsorted(self.lum[:, 0], t-t_radius)
            t1 = np.searchsorted(self.lum[:, 0], t+t_radius)
            lum_can = self.lum[int(i), 1]

            if lum_can == np.max(self.lum[t0:t1, 1]):
                btimes[bnum] = t
                bnum += 1
            else:
                candidates[j] = np.nan

        candidates = candidates[~np.isnan(candidates)].astype(int)
        btimes = btimes[~np.isnan(btimes)]

        # ====== Find burst start (start_frac% of peak), and burst end (2% of peak) ======
        tpre_idx = np.searchsorted(self.lum[:, 0], btimes - pre_time)  # Pre-burst indexes
        t_start = np.ndarray(bnum)  # Times of burst starts
        t_start_idx = np.ndarray(bnum, dtype=int)  # Indexes of burst starts
        t_end = np.ndarray(bnum)  # Times of burst ends
        t_end_idx = np.ndarray(bnum, dtype=int)  # Indexes of burst ends

        for i, s_idx in enumerate(tpre_idx):
            j = s_idx
            peak = self.lum[candidates[i], 1]

            while self.lum[j, 1] < start_frac * peak:
                j += 1

            t_start[i] = self.lum[j, 0]
            t_start_idx[i] = j

            # End must be min_length seconds after peak (avoid spikes)
            while self.lum[j, 1] > end_frac*peak \
                    or (self.lum[j, 0] - t_start[i]) < min_length:
                if j + 1 == n:
                    if self.verbose:
                        print('WARNING: File ends during burst tail')
                        print('Length/Fluence of last burst may be invalid')
                    break
                j += 1

            t_end[i] = self.lum[j, 0]
            t_end_idx[i] = j

        if len(btimes) > 1:
            dt = np.diff(btimes)
        else:
            dt = [np.nan]

        self.bursts['t'] = btimes  # Time of peaks (s)
        self.bursts['tpre'] = btimes - pre_time
        self.bursts['tstart'] = t_start
        self.bursts['tend'] = t_end  # Time of burst end (2% of peak) (s)
        self.bursts['length'] = t_end - t_start  # Burst lengths (s)

        self.bursts['idx'] = candidates  # .lum indexes of peaks,
        self.bursts['tpre_idx'] = tpre_idx
        self.bursts['tstart_idx'] = t_start_idx  # Array indices
        self.bursts['tend_idx'] = t_end_idx

        self.bursts['peak'] = self.lum[candidates, 1]  # Peak luminosities (erg/s)
        self.bursts['num'] = len(btimes)  # Number of bursts
        self.n_bursts = len(btimes)
        self.bursts['dt'] = dt  # Recurrence times (s)

        if self.bursts['num'] == 0:
            print('\nWARNING: No bursts in this model!\n')

    def find_fluence(self):
        """Calculates burst fluences by integrating over burst luminosity
        """
        n = self.bursts['num']
        fluences = np.ndarray(n)

        for i in range(n):
            t0 = self.bursts['tpre_idx'][i]
            t1 = self.bursts['tend_idx'][i]
            fluences[i] = integrate.trapz(y=self.lum[t0:t1 + 1, 1], x=self.lum[t0:t1 + 1, 0])

        self.bursts['fluence'] = fluences  # Burst fluence (ergs)

    def save_bursts(self, path=None):
        """Saves burst lightcurves to txt files. Excludes 'pre' bursts
        """
        if path is None:  # default to model directory
            path = self.batch_models_path

        n = self.bursts['num']
        for i in range(n):
            bnum = i + 1

            i_start = self.bursts['tpre_idx'][i]
            i_zero = self.bursts['tstart_idx'][i]
            i_end = self.bursts['tend_idx'][i]

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
                   burst_stages=False):
        """Plots overall model lightcurve, with detected bursts
        """
        fig, ax = plt.subplots()
        ax.plot(self.lum[:, 0], self.lum[:, 1], c='C0')
        ax.set_title(f'{self.model_str}')

        if log:
            ax.set_yscale('log')
            ax.set_ylim([1e34, 1e39])
        if bursts:
            ax.plot(self.bursts['t'], self.bursts['peak'], marker='o', c='C1', ls='none')
        if burst_stages:
            for stage in ('tpre', 'tstart', 'tend'):
                t = self.bursts[stage]
                y = self.lumf(t)
                ax.plot(t, y, marker='o', c='C2', ls='none')

        if display:
            plt.show(block=False)
        if save:
            filename = f'model_{self.model_str}.png'
            filepath = os.path.join(self.source_path, 'plots', 'burst_analysis', filename)
            fig.savefig(filepath)
        return fig


def multithread_extract(batches, source):
    args = []
    for batch in batches:
        args.append([batch, source])

    with mp.Pool(processes=8) as pool:
        pool.starmap(extract_burstfit_1808, args)


def extract_burstfit_1808(batches, source, skip_bursts=1):
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

        filename = f'summary_{batch_str}.txt'
        filepath = os.path.join(analysis_path, filename)

        data = {}
        for bp in bprops:
            u_bp = f'u_{bp}'
            data[bp] = []
            data[u_bp] = []

        for b in b_ints:
            data[b] = []

        n_runs = grid_tools.get_nruns(batch, source)

        load_dir = f'{source}_{batch}_input/'
        load_path = os.path.join(GRIDS_PATH, 'analyser', source, load_dir)

        for run in range(1, n_runs + 1):
            sys.stdout.write(f'\r{source}_{batch} xrb{run:02}')

            burstfit = BurstRun(run, batch, source)
            # burstfit = burstfit_1808.BurstRun(run, flat_run=True, truncate=False,
            #                                   runs_home=load_path, extra_b=0, pre_t=0,
            #                                   verbose=False, load_analyser=True)
            # burstfit.analyse_all()
            # burstfit.ensure_observer_frame_is(False)

            data['batch'] += [batch]
            data['run'] += [run]
            data['num'] += [burstfit.bursts['num']]

            for bp in bprops:
                u_bp = f'u_{bp}'
                mean = np.mean(burstfit.bursts[bp][skip_bursts:])
                std = np.std(burstfit.bursts[bp][skip_bursts:])

                data[bp] += [mean]
                data[u_bp] += [std]

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
        load_dir = f'{source}_{batch}_input/'
        load_path = os.path.join(GRIDS_PATH, 'analyser', source, load_dir)

        for i in range(n_runs):
            run = i + 1
            n_bursts1 = summ.iloc[i]['num']
            sys.stdout.write(f'\r{source}_{batch} xrb{run:02}')

            burstfit = BurstRun(run, batch, source,
                                verbose=False, load_analyser=True)
            burstfit.analyse_all()
            n_bursts2 = burstfit.bursts['num']

            if n_bursts1 != n_bursts2:
                m_new = np.array((batch, run, n_bursts1, n_bursts2))
                mismatch = np.vstack((mismatch, m_new))

        np.savetxt(filepath, mismatch)
    return mismatch


def plot_convergence(bfit, bprop='dt', start=1, show_values=True):
    fig, ax = plt.subplots()
    b_vals = bfit.bursts[bprop]
    nv = len(b_vals)

    for i in range(start, nv + 1):
        b_slice = b_vals[start - 1:i]
        mean = np.mean(b_slice)
        std = np.std(b_slice)

        print(f'mean: {mean:.3e}, std={std:.3e}')
        # if i != 1:
        #     change = (mean - mean_old) / mean
        #     print(f'Change: {change:.3e}')

        ax.errorbar([i], [mean], yerr=std, ls='none', marker='o', c='C0', capsize=3)
        if show_values:
            ax.plot([i], b_vals[i - 1], marker='o', c='C1')
        mean_old = mean

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
