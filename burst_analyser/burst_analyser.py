import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kepdump
import sys
import os
from scipy import interpolate, integrate

# Custom modules required
from printing import *
import plotting
import b_utils

# kepler_grids
from ..grids import grid_tools

GRIDS_PATH = os.environ['KEPLER_GRIDS']
MODELS_PATH = os.environ['KEPLER_MODELS']


def extract_burstfit_1808(batches, source, skip_bursts=1):
    source_path = pyprint.get_source_path(source)

    # ===== setup container for burst data =====
    data = {}
    b_ints = ('batch', 'run', 'num')
    bprops = ('dt', 'fluence', 'length', 'peak')
    col_order = ['batch', 'run', 'num', 'dt', 'u_dt', 'fluence', 'u_fluence',
                 'length', 'u_length', 'peak', 'u_peak']

    for bp in bprops:
        u_bp = f'u_{bp}'
        data[bp] = []
        data[u_bp] = []

    for b in b_ints:
        data[b] = []

    for batch in batches:
        n_runs = grid_tools.get_nruns(batch, source)

        load_dir = f'{source}_{batch}_input/'
        load_path = os.path.join(GRIDS_PATH, 'analyser', source, load_dir)

        for run in range(1, n_runs + 1):
            sys.stdout.write(f'\r{source}_{batch} xrb{run:02}')

            burstfit = burstfit_1808.BurstRun(run, flat_run=True, truncate=False,
                                              runs_home=load_path, extra_b=0,
                                              verbose=False, pre_t=0, load_analyser=True)
            burstfit.analyse_all()
            burstfit.ensure_observer_frame_is(False)

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

    filename = f'burstfit_V2_extract_{batches[0]}-{batches[-1]}.txt'
    filepath = os.path.join(source_path, filename)
    with open(filepath, 'w') as f:
        f.write(table_str)


class BurstRun(object):
    """

    """

    def __init__(self,
                 run,
                 verbose=True,
                 basename='xrb',
                 re_load=False,
                 savelum=True,
                 load_analyser=False):

        self.run = run
        self.run_str = basename + str(run)
        self.verbose = verbose
        self.path = runs_home + self.run_str + '/'
        self.analysed = False  # Has the model been analysed yet
        self.bursts = {}  # Kepler burst properties

        self.load(savelum=savelum, basename=basename, re_load=re_load,
                  load_analyser=load_analyser)

    def analyse_all(self):
        """
        Analyses all quantities of the model.
        """
        self.ensure_analysed_is(False)

        self.identify_bursts()
        self.lum_old = np.array(self.lum)  # re-save lum with shocks/zeros removed

        if self.truncate:
            self.truncate_eddington()

        self.find_fluence()

        if not self.flat:
            self.load_bursts_obsv()

            self.set_error()
            self.find_average_mdot()
            self.find_alpha()

            if self.auto_extra_b:
                self.fit_extra_b()

            if self.match:
                self.match_timing()

                if self.auto_truncate:
                    self.printv('Auto-fitting Eddington truncation')
                    temp_verbose = self.verbose
                    self.verbose = False
                    self.fit_truncation()
                    self.verbose = temp_verbose

                    idx = np.argmin(self.trunc['chi'])
                    xi_b = self.trunc['xi_b'][idx]
                    x = self.trunc['x'][idx]

                    self.printv('   Best truncation for x={x:.2f}, with xi_b={xi:.2f}'.format(x=x, xi=xi_b))

                    # ---redo truncation with best-fit---
                    self.ensure_observer_frame_is(False)
                    self.lum = np.array(self.lum_old)
                    self.truncate_eddington(x=x)
                    self.redo_fluence(xi_b)

                elif self.auto_xi:  # NOTE: only if not already auto-truncating
                    xi_b = self.fit_fluence()
                    self.redo_fluence(xi_b)
                    self.find_alpha()  # redo alpha

                self.match_params()
                self.find_chi()
                self.find_rmse()

            if self.do_plot:
                self.plot(title=True)

            if self.verbose:
                self.printx('all')
                title()
        else:
            if self.verbose:
                self.printx('flat')
                title()

        self.analysed = True

    # =====================================================================
    # LOADING, INITIALISING
    # =====================================================================
    def load(self, savelum=True, basename='run', re_load=False, load_analyser=False):
        """Load luminosity data from kepler simulation"""

        if load_analyser:
            filename = f'{self.run_str}.data'
            filepath = os.path.join(self.runs_home, filename)
            self.lum = np.loadtxt(filepath, skiprows=2)
        else:
            self.lum = b_utils.load(run=self.run, basename=basename, path=self.runs_home, save=savelum, re_load=re_load)

        if len(self.lum) == 1:
            sys.exit()

        self.loaded = True

        self.lum[:, 0] += - self.pre_t / self.red  # Zero time to start of PCA observations
        self.lumf = interpolate.interp1d(self.lum[:, 0], self.lum[:, 1])  # Callable function of luminosity
        self.lum_old = np.array(self.lum)  # copy to keep without alterations (e.g. truncating)
        #   (still has shocks/zeros until removed)

    def load_bursts_obsv(self):
        """Load observed burst properties"""
        # ---------------------------------------------
        # This data from MINBAR catalogue and analysis (Duncan Galloway)
        # ---------------------------------------------
        self.printv('Loading observed burst data')

        convert = 1e-9 * 4 * pi * D ** 2  # Convert from flux (1e-9) to luminsoty (erg)

        # ----- Times, peaks -----s
        # TODO: -fold these numbers into a file
        #      -remove distance dependence
        #
        # Values sourced from MINBAR (Duncan email 12/07/2016)
        self.bursts_obsv['t'] = np.array([52562.414, 52564.305, 52565.184,
                                          52566.427]) - 52562.07296  # time of bursts, zeroed to start of PCA observations (JD)
        self.bursts_obsv['t'] *= 3600 * 24  # days to sec
        self.bursts_obsv['dt'] = self.bursts_obsv['t'][1:] - self.bursts_obsv['t'][:-1]  # Recurrence times (s)
        self.bursts_obsv['peak'] = np.array(
            [215.11, 229.38, 232.42, 231.96]) * convert  # Peak flux (1e-9 erg cm^-2 --> erg)
        self.bursts_obsv['peak_error'] = np.array([3.95, 4.3, 3.77, 3.94]) * convert  # Flux error
        self.bursts_obsv['fluence'] = np.array([2.620, 2.649, 2.990, 3.460]) * convert * 1e3  # (1e-6) to erg
        self.bursts_obsv['fluence_error'] = np.array([0.021, 0.018, 0.017, 0.022]) * convert * 1e3
        self.bursts_obsv['alpha'] = np.array([106.9, 118.2, 128.2])
        self.bursts_obsv['alpha_error'] = np.array([1.7, 1.9, 2.1])

        # ----- Lightcurves -----
        # 'Time [s]' 'dt [s]' 'flux [10^-9 erg/cm^2/s]' 'flux error [10^-9 erg/cm^2/s]' 'blackbody temperature kT [keV]' 'kT error [keV]' 'blackbody normalisation K_bb [(km/d_10kpc)^2]' 'K_bb error [(km/d_10kpc)^2]' chi-sq
        self.obsv_lum = {}
        for i in range(1, 5):
            lcfile = 'burst{}_1808.txt'.format(i)
            lc_loc = '../obs_data/minbar_lightcurves/'
            obs_lumpath = os.path.join(LOC, lc_loc, lcfile)

            self.obsv_lum['b' + str(i)] = np.loadtxt(obs_lumpath, skiprows=21)

        # PCA observing windows  [time, +-dt]
        pca_file = '1808-369_outburst_2002-Oct.txt'
        obs_path = '../obs_data/'
        pca_filepath = os.path.join(LOC, obs_path, pca_file)

        self.acc_windows = np.loadtxt(pca_filepath, skiprows=30, usecols=[0, 1])
        self.acc_windows *= 8.64e4  # From days to seconds
        self.acc_windows[:, 0] -= self.acc_windows[0, 0]  # Zero time to PCA data

    def load_acc(self):
        """Load accretion rate data used in kepler simulation"""
        # acc == [time(s), mdot(g/s)]
        # asm == [time(s), dt, mdot(g/s), u(mdot)]
        # --- All in KEPLER reference frame (and mdots remain so; time points don't)---
        self.ensure_observer_frame_is(False)
        self.printv('Loading accretion rates used in KEPLER')

        asm_file = '../files/asm.txt'
        asm_path = os.path.join(LOC, asm_file)

        pca_error_file = '../obs_data/1808_detailed.txt'
        pca_error_path = os.path.join(LOC, pca_error_file)
        pca_error = np.loadtxt(pca_error_path, skiprows=51, usecols=[3, 4])
        self.acc_pca_error_frac = pca_error[:, 1] / pca_error[:, 0]  # fractional uncertainty

        self.acc_asm = np.loadtxt(asm_path, skiprows=6)
        self.acc = np.loadtxt(self.path + self.run_str + '.acc', skiprows=2)

        self.acc[:, 0] -= self.pre_t / self.red  # Zero time to PCA data
        self.acc_asm[:, 0] -= self.pre_t / self.red

        # NOTE: Multiplied by anisotropy factor because Kepler used accratef multiplier post-file-read
        self.acc[:, 1] *= self.xi_p
        self.acc_asm[:, 1] *= self.xi_p

        self.accf = interpolate.interp1d(self.acc[:, 0], self.acc[:, 1])  # Callable function of accrate (mdot)

    def load_abu(self):
        """ Reads in accretion composition from kepler dump file"""
        # Assumes there is a dump runxx#1000
        self.printv('Loading chemical abundances from KEPLER dump file')

        try:
            dumpfile = self.run_str + '#1000'
            dumppath = os.path.join(self.path, dumpfile)
            kdump = kepdump.load(dumppath)

            # [X,Y,Z], [H1,He4,N14]
            self.abu = kdump.compsurf[[1, 4, 6]]
        except:
            print('Dump {}#1000 not found'.format(self.run_str))
            # NOTE: Fails when not found. Needs fixing

    # ==========================================================================
    # EXTRACTING, ANALYSING
    # ==========================================================================
    def shift_frame(self):
        """Shifts relevant values between the local and observer frame, through GR + anisotropy correction"""
        Lshift = self.red * self.xi_b
        Tshift = self.red

        if self.observer_frame:
            self.printv('~SHIFTING TO LOCAL FRAME')
            l_factor = Lshift
            t_factor = 1 / Tshift

        else:
            self.printv(
                '~SHIFTING TO OBSERVER FRAME. (1+z)={red:.3f}, xi_b={xi:.3f}'.format(red=self.red, xi=self.xi_b))
            l_factor = 1 / Lshift
            t_factor = Tshift

        # ----------- Shift time ---------------
        self.lum[:, 0] *= t_factor
        self.lum_old[:, 0] *= t_factor

        if not self.flat:
            self.acc[:, 0] *= t_factor  # Leave mdot unchanged
            self.acc_asm[:, :2] *= t_factor

        self.bursts['tpre'] *= t_factor
        self.bursts['tstart'] *= t_factor
        self.bursts['t'] *= t_factor
        self.bursts['tend'] *= t_factor
        self.bursts['length'] *= t_factor
        self.bursts['dt'] *= t_factor

        # ----------- Shift luminosity ---------------
        self.lum[:, 1] *= l_factor
        self.lum_old[:, 1] *= l_factor
        self.bursts['peak'] *= l_factor

        if not self.flat:
            self.accf = interpolate.interp1d(self.acc[:, 0], self.acc[:, 1])  # Update acc function
            self.lumf = interpolate.interp1d(self.lum[:, 0], self.lum[:, 1])  # Update lum function

        self.observer_frame = not self.observer_frame

    def ensure_observer_frame_is(self, obs):
        """Ensures values are in observer (obs=True), or local frame (obs=False)"""
        if self.observer_frame != obs:
            self.shift_frame()

    def ensure_analysed_is(self, anal):
        """
        Checks that model has (or hasn't) been analysed
        """
        words = {True: "hasn't yet been",
                 False: 'has already been'}
        line2 = {True: 'run self.analyse_all() first',
                 False: 'reload model first'}

        if self.analysed != anal:
            print('ERROR: model {} analysed'.format(words[anal]))
            print(line2[anal])
            sys.exit()

    def remove_zeros(self):
        """
        During shocks, kepler can also give zero luminosity (for some reason...)
        """
        replace_with = 1e34
        zeros = np.where(self.lum[:, 1] == 0.0)
        Nz = len(zeros)
        self.printv('Removed {n} zeros from luminosity'.format(n=Nz))
        self.lum[zeros, 1] = replace_with

    def remove_shocks(self, maxima_idx):
        """
        Cut out convective shocks (extreme spikes in luminosity).
        Identifies spikes, and replaces them with interpolation from neighbours.
        """
        # ---------------------------
        # maxima_idx = []   : list of indices of local maxima to check
        # ---------------------------
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
                    self.printv('    {time:.2f}      {lum:.2e}   {fac:.2f}'.format(time=time, lum=l, fac=factor))

                # --- replace with mean of two neighbours ---
                self.lum[i, 1] = 0.5 * (left[-1] + right[0])

    def truncate_eddington(self, x=0.0):
        """
        Truncates luminosities at the Eddington limit
        """
        # ----------------------------------------------------------------
        # NOTE: - Determined by H fraction. Remaining composition assumed to be helium
        #       - Uses redshift (1+z) to account for diff between newtonian and GR-frames
        # ----------------------------------------------------------------
        # x = flt  : hydrogen fraction (for calculating Eddington limit)
        # ----------------------------------------------------------------
        self.ensure_observer_frame_is(False)
        self.printv('Truncating super-Eddington luminosities, with X={x:.2f}'.format(x=x))

        N = len(self.lum[:, 0])
        L_edd_H = 1.26e38 * M / Msun  # Eddington luminosity (pure H1)
        L_edd = L_edd_H * 2.0 / (x + 1)  # Edd for given hydrogen
        L_edd_red = L_edd / self.red  # redshift accounts for Newtonian frame

        trunc = self.lum[:, 1] > L_edd_red
        self.lum[trunc, 1] = L_edd_red

    def identify_bursts(self):
        """Extracts times, separations, and mean separation of bursts
        """
        self.printv('Identifying burst times and peaks in KEPLER model')

        bmax = 1000  # Max possible number of bursts expected
        btimes = np.ndarray(bmax)  # Detected burst peak times
        btimes[:] = np.nan
        candidates = np.ndarray(10000)  # Possible burst peaks
        candidates[:] = np.nan

        # ==============IDENTIFY BURSTS============
        cnum = 0  # candidate counter
        tol = 1  # No. of neighbour zones to consider for maxima
        maxl = 1e38  # Chosen typical max peak
        lum_thresh = 0.02  # cut-off luminosity for finding bursts, as fraction of maxl
        end_frac = 0.005  # end of burst defined when luminosity falls to this fraction of peak
        t_radius = 30  # minimum time (s) that each burst should be separated by

        # ===== cut out everything below threshold =====
        b_idx = np.where(self.lum[:, 1] > lum_thresh * maxl)[0]  # Threshold luminosity 2% of max
        n = len(self.lum[:, 0])

        # ===== pick out all local maxima (larger than +/- tol neighbours) =====
        for i in b_idx[:-1]:
            l = self.lum[i, 1]
            if l > self.lum[i - tol, 1] and l > self.lum[i + tol, 1]:  # If local maxima
                candidates[cnum] = i
                cnum += 1

        candidates = candidates[~np.isnan(candidates)]  # Trim nans

        self.remove_shocks(candidates.astype('int'))

        bnum = 0

        # ===== burst peak if maxima in 20 s radius =====
        for j, i in enumerate(candidates):
            t = self.lum[int(i), 0]
            t0 = np.searchsorted(self.lum[:, 0], t - t_radius)  # 20 s earlier
            t1 = np.searchsorted(self.lum[:, 0], t + t_radius)  # 20 s later
            l = self.lum[int(i), 1]

            if l == np.max(self.lum[t0:t1, 1]):  # If highest maxima in 20 s radius
                btimes[bnum] = t
                bnum += 1
            else:
                candidates[j] = np.nan

        candidates = candidates[~np.isnan(candidates)].astype(int)  # Trim nans
        btimes = btimes[~np.isnan(btimes)]

        # ====== Find burst start (25% of peak), and burst end (2% of peak) ======
        tpre_idx = np.searchsorted(self.lum[:, 0], btimes - 20)  # Pre-burst indexes
        t_start = np.ndarray(bnum)  # Times of burst starts
        t_start_idx = np.ndarray(bnum, dtype=int)  # Indexes of burst starts
        t_end = np.ndarray(bnum)  # Times of burst ends
        t_end_idx = np.ndarray(bnum, dtype=int)  # Indexes of burst ends

        start = 0.25  # Define burst start as given fraction of peak lum

        for i, s_idx in enumerate(tpre_idx):
            j = s_idx  # start looking from here
            peak = self.lum[candidates[i], 1]

            while self.lum[j, 1] < start * peak:
                j += 1

            t_start[i] = self.lum[j, 0]
            t_start_idx[i] = j

            # End must be at least 5 seconds after peak (accounts for sudden spikes)
            while self.lum[j, 1] > end_frac * peak or (self.lum[j, 0] - t_start[i]) < 5:
                if j + 1 == n:
                    if self.verbose:
                        warning()
                        print('WARNING: File ends during burst tail')
                        print('Length/Fluence of last burst may be invalid')
                        warning()
                    break
                j += 1

            t_end[i] = self.lum[j, 0]
            t_end_idx[i] = j

        if len(btimes) > 1:  # Check if only one burst
            dt = np.diff(btimes)
        else:
            dt = [np.nan]

        if np.min(dt) < 10 * 3600:  # Check if any recurrence are too small
            self.hydrogen = True  # Implies hydrogen-content bursts

        # ----------------------------------
        self.bursts['t'] = btimes  # Time of peaks (s)
        self.bursts['tpre'] = btimes - 20  # Pre-burst reference, 20s before peak (s)
        self.bursts['tstart'] = t_start  # (s)
        self.bursts['tend'] = t_end  # Time of burst end (2% of peak) (s)
        self.bursts['length'] = t_end - t_start  # Burst lengths (s)

        self.bursts['idx'] = candidates  # .lum indexes of peaks,
        self.bursts['tpre_idx'] = tpre_idx
        self.bursts['tstart_idx'] = t_start_idx  # Array indices
        self.bursts['tend_idx'] = t_end_idx

        self.bursts['peak'] = self.lum[candidates, 1]  # Peak luminosities (erg/s)
        self.bursts['num'] = len(btimes)  # Number of bursts
        self.bursts['dt'] = dt  # Recurrence times (s)

        if self.bursts['num'] == 0:
            print('\nWARNING: No bursts in this model!\n')

    def set_error(self):
        """Set kepler burst uncertainties"""
        # NOTE: Always in observer frame
        self.ensure_observer_frame_is(True)
        self.printv('Calculating uncertainties in KEPLER values')

        error = 0.03  # 1 standard deviation (%), informed by burst trains
        N = self.bursts['num']
        dt_error = self.bursts['dt'] * error
        t_error = np.ndarray(N)

        for i in range(N - 1):  # Propogating dt_error through the burst sequence (quadrature)
            sum_squares = np.sum(dt_error[:i + 1] ** 2)  # Sum of squares of u(dt) up to this burst
            t_error[i + 1] = np.sqrt(sum_squares)

        t_error[0] = t_error[1]  # First dt unknown, just equate to second burst

        self.bursts['t_error'] = t_error
        self.bursts['dt_error'] = dt_error
        self.bursts['peak_error'] = self.bursts['peak'] * error
        self.bursts['fluence_error'] = self.bursts['fluence'] * error

    def match_timing(self):
        """ Detects time difference between model bursts and observations"""
        self.ensure_observer_frame_is(True)  # Check in correct frame for comparison
        self.printv('Matching KEPLER burst times to observations')

        # Sim burst indexes that should match (Account for extra pre bursts, eg. from longer rise time)
        num_matchable = self.bursts['num'] - self.extra_b  # How many bursts can be compared?

        # only try firt N matchable bursts
        if num_matchable < 6:
            j = np.searchsorted(self.to_match, self.bursts['num'])
            self.to_match = self.to_match[:j]
            N = len(self.to_match)
            if self.verbose:
                warning()
                print('WARNING: Too few bursts to match')
                print('Matching {}/4 bursts'.format(N))
                warning()
        else:
            N = 4

        tmatch = np.ndarray(N)
        dtmatch = np.ndarray(N - 1)
        dterr = np.ndarray(N - 1)

        tmatch = self.bursts['t'][self.to_match] - self.bursts_obsv['t'][:N]

        # First gap spans several bursts:
        dtmatch[0] = np.diff(self.bursts['t'][self.to_match[:2]]) - self.bursts_obsv['dt'][0]
        dtmatch[1:] = self.bursts['dt'][self.to_match[1:N - 1]] - self.bursts_obsv['dt'][1:N - 1]

        # TODO Probably breaks when num_matchable = 1
        dtspan0 = self.bursts['dt_error'][self.to_match[0]: self.to_match[1]]
        dterr[0] = np.sqrt(np.sum(dtspan0 ** 2))
        dterr[1:] = self.bursts['dt_error'][self.to_match[1]: self.to_match[N - 1]]

        self.bursts['tmatch'] = tmatch  # Timing mismatch (s)
        self.bursts['dtmatch'] = dtmatch  # Recurrence time mismatch (s)

        # normalised to sigma (assumed zero uncertainty for observed, since ~1sec)
        self.bursts['tmatch_std'] = tmatch / self.bursts['t_error'][self.to_match]
        self.bursts['dtmatch_std'] = dtmatch / dterr

        self.matched = True

    def match_params(self):
        """Matches other burst properties such as peaks, fluence"""
        self.ensure_observer_frame_is(True)  # Check in correct frame for comparison
        self.printv('Matching peaks and fluences to observations')

        N = len(self.to_match)
        self.bursts['pmatch'] = self.bursts['peak'][self.to_match] - self.bursts_obsv['peak'][:N]
        self.bursts['fmatch'] = self.bursts['fluence'][self.to_match] - self.bursts_obsv['fluence'][:N]
        self.bursts['amatch'] = self.bursts['alpha'][self.to_match[1:] - 1] - self.bursts_obsv['alpha'][:N]

        # normalised to sigma
        p_err = np.sqrt(self.bursts['peak_error'][self.to_match] ** 2 + self.bursts_obsv['peak_error'] ** 2)
        f_err = np.sqrt(self.bursts['fluence_error'][self.to_match] ** 2 + self.bursts_obsv['peak_error'] ** 2)
        # a_err = np.sqrt()
        self.bursts['pmatch_std'] = self.bursts['pmatch'] / p_err
        self.bursts['fmatch_std'] = self.bursts['fmatch'] / f_err

    def find_fluence(self):
        """
        Calculates burst fluences by integrating over burst luminosity
        """
        self.ensure_observer_frame_is(True)
        self.printv('Integrating burst luminosities to find fluences')
        # NOTE: Fluence always in observer frame.
        #       Implicitly includes anisotropy(xi_b) factor in shifted self.lum

        n = self.bursts['num']
        fluences = np.ndarray(n)

        for i in range(n):
            t0 = self.bursts['tpre_idx'][i]
            t1 = self.bursts['tend_idx'][i]
            fluences[i] = integrate.trapz(y=self.lum[t0:t1 + 1, 1], x=self.lum[t0:t1 + 1, 0])

        self.bursts['fluence'] = fluences  # Burst fluence (ergs)

    def redo_fluence(self, xi_b):
        """Re-calculates fluence with new xi_b (eg. after auto_xi_b or auto_truncate)"""
        self.ensure_observer_frame_is(False)  # do this so xi_b can be applied to lum before fluence calculation
        self.printv('Recalculating fluences')
        self.xi_b = xi_b
        self.find_fluence()
        self.set_error()

    def find_alpha(self):
        """Calculates accretion fluences, and then alpha by dividing by burst fluences"""
        self.ensure_observer_frame_is(True)
        self.printv('Calculating accretion fluences, then alpha ratios')

        N = self.bursts['num']
        p_fluence = np.zeros(N - 1)  # persistent fluence from surface
        acc_fluence = self.bursts['mass'] * self.g  # Energy released by accretion between bursts

        for i in range(N - 1):
            t0 = self.bursts['tend_idx'][i]
            t1 = self.bursts['tpre_idx'][i + 1]
            p_fluence[i] = integrate.trapz(x=self.lum[t0:t1, 0], y=self.lum[t0:t1, 1])

        alpha = (p_fluence + acc_fluence) / self.bursts['fluence'][1:]  # Alpha unknown for very first burst

        self.bursts['acc_fluence'] = acc_fluence
        self.bursts['p_fluence'] = p_fluence
        self.bursts['alpha'] = alpha

    def find_average_mdot(self):
        """Finds average accretion rate (and total mass) between each pair of bursts"""
        self.ensure_observer_frame_is(False)  # Should be in local frame when calculating mdot
        self.printv('Finding average accrates between bursts')

        N = len(self.bursts['dt'])
        mdot_avg = np.zeros(N)  # Average accretion rate between each burst pair
        mass = np.zeros(N)  # Total mass accreted between each burst pair
        avg_slope = np.zeros(N)
        intercepts = np.zeros(N)

        # === Integrate mdot, using trapezoidal rule ===
        for i, dt in enumerate(self.bursts['dt']):
            t_a = self.bursts['t'][i]
            t_b = self.bursts['t'][i + 1]

            if t_b > self.acc[-1, 0]:
                warning()
                print('WARNING! Burst occurs after end of accretion file')
                warning()
                break

            # ==== Average mdot ====
            acc_a = self.accf(t_a)
            acc_b = self.accf(t_b)

            idx_a = np.searchsorted(self.acc[:, 0], t_a)  # Indexes to the right of a,b
            idx_b = np.searchsorted(self.acc[:, 0], t_b)
            # (This does account for a,b falling between self.acc points)
            mass[i] = integrate.trapz(y=np.concatenate(([acc_a], self.acc[idx_a:idx_b, 1], [acc_b])),
                                      x=np.concatenate(([t_a], self.acc[idx_a:idx_b, 0], [t_b])))

            mdot_avg[i] = mass[i] / dt
            avg_slope[i] = (acc_b - acc_a) / (t_b - t_a)

        self.bursts['mass'] = mass
        self.bursts['mdot_avg'] = mdot_avg
        self.bursts['avg_slope'] = avg_slope

    def find_chi(self):
        """Finds fitting-value chi(^2), to measure quality of fit to observations"""
        self.printv('Calculating fitting-parameter (chi^2)')
        N = len(self.to_match)

        # Handle 3 intervening dt's being combined
        dt_error = np.sum(self.bursts['dt_error'][self.to_match[0]:self.to_match[1]])  # Sum first 3 dt's
        dt_error = np.append(dt_error, self.bursts['dt_error'][self.to_match[1:N - 1]])  # Append remaining dt's

        # Mismatch terms for each fitting variable, scaled by uncertainty
        tmatch = self.bursts['tmatch'] / self.bursts['t_error'][self.to_match]  # Timings
        dtmatch = self.bursts['dtmatch'] / dt_error  # Recurrence times
        pmatch = self.bursts['pmatch'] / self.bursts['peak_error'][self.to_match]  # Burst peaks
        fmatch = self.bursts['fmatch'] / self.bursts['fluence_error'][self.to_match]  # Fluences

        self.chi['t'] = np.sum(tmatch ** 2) / len(tmatch)  # Add in quadrature
        self.chi['sign_t'] = np.sum(tmatch * np.absolute(tmatch)) / len(tmatch)  # Preserve sign.
        self.chi['dt'] = np.sum(dtmatch ** 2) / len(dtmatch)
        self.chi['sign_dt'] = np.sum(dtmatch * np.absolute(dtmatch)) / len(dtmatch)
        self.chi['fluence'] = np.sum(fmatch ** 2) / len(fmatch)
        self.chi['sign_fluence'] = np.sum(fmatch * np.absolute(fmatch)) / len(fmatch)
        self.chi['peak'] = np.sum(pmatch ** 2) / len(pmatch)
        self.chi['sign_peak'] = np.sum(pmatch * np.absolute(pmatch)) / len(pmatch)

        tot = ['t', 'dt', 'fluence', 'peak']
        total = 0
        for v in tot:
            total += self.chi[v]

        self.chi['total'] = total / len(tot)

    def find_rmse(self):
        """Calculates Root Mean Square Error (RMS) of each matched variable"""
        self.ensure_observer_frame_is(True)
        self.printv('Calculating RMSEs')

        def rms(var):
            return np.sqrt(np.mean(var ** 2))

        self.rms['t'] = rms(self.bursts['tmatch'])
        self.rms['dt'] = rms(self.bursts['dtmatch'])
        self.rms['peak'] = rms(self.bursts['pmatch'])
        self.rms['fluence'] = rms(self.bursts['fmatch'])

        self.rms['t_std'] = rms(self.bursts['tmatch_std'])
        self.rms['dt_std'] = rms(self.bursts['dtmatch_std'])
        self.rms['peak_std'] = rms(self.bursts['pmatch_std'])
        self.rms['fluence_std'] = rms(self.bursts['fmatch_std'])

    def fit_extra_b(self):
        """Finds No. of extra bursts that gives best chi - i.e. extra_b agnostic"""
        self.printv('Finding extra_b that gives best timings match')

        # check Nbursts >= 6
        ntries = self.bursts['num'] - 5  # Number of extra_b to try
        if ntries < 1:
            print('Not enough bursts to match!')
            self.best_extra_b = -1
            self.best_chi = -1
            return

        extra_bs = np.arange(ntries)  # Each extra_b to try
        chis = np.zeros(ntries)  # To hold each chi^2

        # ===== Try all valid extra_b =====
        for i in extra_bs:
            self.printv('   Trying {} extra bursts'.format(i))
            self.extra_b = i

            original_verbose = self.verbose
            self.verbose = False
            self.match_timing()
            self.match_params()
            self.find_chi()
            self.verbose = original_verbose

            chis[i] = self.chi['t']

        # ===== Choose best one =====
        idx = np.argmin(chis)

        self.printv('Best guess for extra_b: {}'.format(extra_bs[idx]))
        self.printv('Corresponding chi^2:    {:.2f}'.format(chis[idx]))

        self.best_extra_b = extra_bs[idx]
        self.best_chi = chis[idx]

        # ===== Re-update with final extra_b =====
        self.extra_b = self.best_extra_b
        self.match_timing()
        self.match_params()
        self.find_chi()

    def fit_fluence(self):
        """
        Returns the xi_b that gives best fit to fluence
        """
        self.ensure_observer_frame_is(True)
        self.printv('Fitting fluence by scaling xi_b')

        N = len(self.to_match)
        fluence = self.bursts['fluence'][self.to_match] / 1e39
        fluence_obs = self.bursts_obsv['fluence'][:N] / 1e39

        # --- Solution for xi_b when minimising chi^2 function ---
        best_xi_b = np.sum(fluence ** 2) / np.sum(fluence * fluence_obs)

        self.printv('    Suggested xi_b: {:.2f}'.format(best_xi_b))

        return best_xi_b

    def fit_truncation(self):
        """
        Determines the luminosity truncation that results in the best fluence fit
        """
        trunc = {}
        dx = 0.05  # hydrogen intervals to try
        Nx = int(2 / dx + 1)
        N = len(self.to_match)
        trunc['x'] = np.arange(0, 1 + dx, dx)
        trunc['xi_b'] = np.zeros(Nx)
        trunc['chi'] = np.zeros(Nx)

        for i, x in enumerate(trunc['x']):
            self.truncate_eddington(x=x)
            self.find_fluence()
            xi_b = self.fit_fluence()

            flu = self.bursts['fluence'][self.to_match] / 1e39
            fluObs = self.bursts_obsv['fluence'][:N] / 1e39

            trunc['chi'][i] = np.mean((flu / xi_b - fluObs) ** 2)  # RMS for given xi_b
            trunc['xi_b'][i] = xi_b

        self.trunc = trunc

    def save_bursts(self, path=None):
        """Saves burst lightcurves to txt files. Excludes 'pre' bursts """
        if path == None:  # default to model directory
            path = self.path

        self.ensure_observer_frame_is(False)
        self.printv('Saving burst lightcurves to: {path}'.format(path=path))

        N = self.bursts['num']
        exb = self.extra_b

        for i in range(exb, N):
            bnum = i - exb + 1  # burst label
            self.printv('   burst {bnum}'.format(bnum=bnum))

            i_start = self.bursts['tpre_idx'][i]
            i_zero = self.bursts['tstart_idx'][i]
            i_end = self.bursts['tend_idx'][i]

            time = self.lum[i_start:i_end, 0] - self.lum[i_zero, 0]
            lum = self.lum[i_start:i_end, 1]
            uncertainty = 0.02
            u_lum = lum * uncertainty

            lightcurve = np.array([time, lum, u_lum]).transpose()
            header = 'time luminosity u_luminosity'.format(path=self.path)
            b_file = 'b{n}.txt'.format(n=bnum)
            filepath = os.path.join(path, b_file)

            np.savetxt(filepath, lightcurve, header=header)

    # =====================================================================
    # PLOTTING
    # =====================================================================
    def plot(self, **kwargs):
        """Plots Kepler bursts against observed bursts, including the accretion rate curve"""
        plotting.plot(self, **kwargs)

    def plot_lightcurves(self, **kwargs):
        """Plot kepler burst light curves against observed bursts"""
        plotting.plot_lightcurves(self, **kwargs)

    def plot_truncated(self, **kwargs):
        plotting.plot_truncated(self, **kwargs)

    def plot_multicurve(self, **kwargs):
        """Plots all burst lightcurves on a single axis"""
        plotting.plot_multicurve(self, **kwargs)

    def plot_fluence(self, **kwargs):
        """Plots fluences of Model vs. Observations"""
        plotting.plot_fluence(self, **kwargs)

    def plot_accrate(self, **kwargs):
        """Plots accretion rate over 2002 outburst, includign PCA/ASM data"""
        plotting.plot_accrate(self, **kwargs)

    def save_plots(self):
        """Saves all plots for paper"""
        self.plot(save=True)
        self.plot_lightcurves(save=True)
        # self.plot_fluence(save=True)
        self.plot_accrate(save=True)
        self.plot_truncated(save=True)

    # =====================================================================
    # PRINTING
    # =====================================================================
    def printv(self, string, dash=False):
        """Only print if verbosity switched on"""
        # string  = str  : string to print
        # dash    = bool : also print dashes afterwards
        if self.verbose:
            print(string)
            if dash:
                dashes()

    def printx(self, var):
        """Prints given variable of BurstRun object"""
        printx(self, var)

    def print_chi(self):
        """Prints list of chi^2 values on one line, for copying into spreadsheet"""
        print_chi(self)

    def latex_table(self, path='/home/zac/projects/mdot/paper/'):
        """Saves table of burst properties formatted for latex environment"""
        table = np.arange(5)
        header = ['Burst', '$t$ (hr)', '$\Delta t$ (hr)', '$\lpeak$ (erg s$^{-1}$)', '$\eburst$ (erg)']
        hcol = ['B1', 'B2', 'B3', 'B4']
        # matrix2latex.matrix2latex(table, 'table', headerRow=header)

    def print_observables(self):
        """Prints observable burst properties with uncertainties, with corresponding observed values"""
        self.ensure_observer_frame_is(True)

        title()
        print('MODEL')
        title()

        dic = {'t': {'units': 'hr', 'fac': 3600, 'fmt': 'f', 'offset': 0},
               'dt': {'units': 'hr', 'fac': 3600, 'fmt': 'f', 'offset': 1},
               'peak': {'units': 'erg/s', 'fac': 1, 'fmt': 'e', 'offset': 0},
               'fluence': {'units': 'erg', 'fac': 1, 'fmt': 'e', 'offset': 0},
               }

        for var in dic:
            to_match = self.to_match - dic[var]['offset']
            units = dic[var]['units']
            fmt = dic[var]['fmt']
            fac = dic[var]['fac']

            for err in ['', '_error']:
                print('{0}{2} ({1}):  '.format(var, units, err), end='')
                # print_list(self.bursts[var+err][to_match]/fac, decimal=2, fmt=fmt)
                print_list(self.bursts_obsv[var + err] / fac, decimal=2, fmt=fmt)
            dashes()


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

            burstfit = burstfit_1808.BurstRun(run, flat_run=True, truncate=False,
                                              runs_home=load_path, extra_b=0,
                                              verbose=False, pre_t=0, load_analyser=True)
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
    # ax.set_ylim([0,1e5])

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
