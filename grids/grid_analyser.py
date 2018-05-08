# ========================
# Utilities for grids of kepler models
# Author: Zac Johnston (zac.johnston@monash.edu)
# ========================

# standard
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# kepler_grids
from . import grid_tools
from . import grid_strings

# concord
import con_versions
import ctools

GRIDS_PATH = os.environ['KEPLER_GRIDS']
MODELS_PATH = os.environ['KEPLER_MODELS']


# -----------------------------------
# TODO:
#       - plot mean lightcurve for given params
#       -
# -----------------------------------
params_exclude = {'gs1826': {'qb': [0.3, 0.5, 0.7, 0.9], 'z': [0.001, 0.003]},
                  'biggrid2': {'qb': [0.075], 'z': [0.001], 'x': [0.5]},
                  }


def default_plt_options():
    """Initialise default plot parameters"""
    params = {'mathtext.default': 'regular',
              'font.family': 'serif'}
    plt.rcParams.update(params)


default_plt_options()


class Kgrid:
    """
    An object for interracting with large model grids
    """

    def __init__(self, source, basename='xrb', con_ver=6,
                 load_lc=False, verbose=True,
                 powerfits=True, exclude_defaults=True,
                 load_concord_summ=True, burst_analyser=False, **kwargs):
        """
        source   =  str  : source object being modelled (e.g. gs1826)
        basename =  str  : basename of individual models (e.g. xrb)
        mass     =  flt  : mass of neutron star in M_sun (may become deprecated)
        (path    = str   : path to dir of grids)
        """
        source = grid_strings.source_shorthand(source=source)
        self.path = kwargs.get('path', GRIDS_PATH)
        self.models_path = kwargs.get('models_path', MODELS_PATH)
        self.source_path = grid_strings.get_source_path(source)
        self.source = source
        self.basename = basename
        self.con_ver = con_ver
        self.verbose = verbose
        self.burst_analyser = burst_analyser

        # ==== Load tables of models attributes ====
        self.printv('Loading kgrid')
        self.params = grid_tools.load_grid_table(tablename='params',
                                                 source=source, verbose=verbose)
        self.summ = grid_tools.load_grid_table(tablename='summ', source=source,
                                               verbose=verbose, burst_analyser=burst_analyser)

        # ===== extract the unique parameters =====
        self.unique_params = {}
        for v in self.params.columns[2:]:
            self.unique_params[v] = np.unique(self.params[v])

        if load_concord_summ:
            self.concord_summ = grid_tools.load_grid_table(tablename='concord_summ',
                                                           source=source, con_ver=con_ver, verbose=verbose)
        self.n_models = len(self.params)

        self.printv('=' * 40)
        self.printv('Loaded')
        self.printv(f'Source: {source}')
        self.printv(f'Models in grid: {self.n_models}')
        self.printv(f'Concord config: {con_ver}')

        self.powerfits = None
        if powerfits:
            self.get_powerfits()

        # ===== Load mean lightcurve data =====
        self.mean_lc = {'columns': ['Time', 'L', 'u(L)', 'R', 'u(R)']}
        if load_lc:
            self.load_all_mean_lightcurves()

        # ===== exclude misc params from plotting, etc. =====
        self.exclude_defaults = exclude_defaults
        self.params_exclude = params_exclude.get(source, {})

    def printv(self, string, **kwargs):
        """Prints string if self.verbose == True
        """
        if self.verbose:
            print(string, **kwargs)

    def get_nruns(self, batch):
        """Returns number of models in a batch
        """
        return len(self.get_params(batch=batch))

    def get_params(self, batch=None, run=None, params=None, exclude=None):
        """Returns models with given batch/run/params
        
        params  = {}      : params that must be satisfied
        exclude = {}      : params to exclude/blacklist completely
        
        Can be used multiple ways:
            - get_params(batch=2, run=1): returns run 1 of batch 2
            - get_params(batch=2): returns all models in batch 2
            - get_params(params={'z':0.01}): returns all models with z=0.01
        """
        if all(x is None for x in (batch, run, params, exclude)):
            raise ValueError('Must specify at least one argument')

        params_full = {}

        if batch is not None:
            params_full['batch'] = batch

        if run is not None:
            params_full['run'] = run

        if params is not None:
            params_full = {**params_full, **params}

        if self.exclude_defaults:
            if exclude is None:
                exclude = {}
            exclude = {**exclude, **self.params_exclude}

        models = grid_tools.reduce_table(table=self.params, params=params_full,
                                         exclude=exclude, verbose=self.verbose)
        return models

    def get_summ(self, batch=None, run=None, params=None, exclude=None):
        """Get summary of given batch/run/params
        """
        subset = self.get_params(batch=batch, run=run, params=params,
                                 exclude=exclude)
        idxs = subset.index.values
        return self.summ.iloc[idxs]

    def get_concord_sum(self, batch=None, run=None, params=None):
        """
        Get concord results of given triplet/run/params
        """
        # TODO: move to define_sources
        base_mdot = {'gs1826': 0.0796, '4u1820': 0.226}[self.source]
        if params is None:
            params = dict()
        params['accrate'] = base_mdot

        subset = self.get_params(batch=batch, run=run, params=params)
        idxs = np.array([])
        for i in subset.index:
            batch = subset.get_value(i, 'batch')
            run = subset.get_value(i, 'run')

            batch_idxs = np.where(self.concord_summ['triplet'] == batch)[0]
            run_idxs = np.where(self.concord_summ['run'] == run)[0]
            idx = np.intersect1d(batch_idxs, run_idxs)
            idxs = np.concatenate([idxs, idx])

        return self.concord_summ.loc[idxs]

    def get_lhood(self, triplet, run):
        """Returns lhood value of given triplet-run
        """
        # TODO: smarter way to do this intersection?
        if triplet not in self.concord_summ['triplet'].values:
            return np.nan

        idxs1 = np.where(self.concord_summ['triplet'] == triplet)[0]
        idx2 = np.where(self.concord_summ.iloc[idxs1]['run'] == run)[0][0]
        idx = idxs1[idx2]

        return self.concord_summ.iloc[idx]['lhood']

    def get_powerfits(self):
        """Calculate power-law fits to burst properties (only dt currently)
        """
        qb_list = self.unique_params['qb']
        z_list = self.unique_params['z']
        x_list = self.unique_params['x']
        mass_list = self.unique_params['mass']

        params = {'z': z_list, 'qb': qb_list, 'x': x_list, 'mass': mass_list}

        enum_params = grid_tools.enumerate_params(params)
        N = len(enum_params['qb'])
        powerfits = pd.DataFrame()

        for p in params:
            powerfits[p] = enum_params[p]

        for key in ['m', 'y0']:
            powerfits[key] = np.full(N, np.nan)

        for i in range(N):
            qb = powerfits.iloc[i]['qb']
            z = powerfits.iloc[i]['z']
            x = powerfits.iloc[i]['x']
            mass = powerfits.iloc[i]['mass']
            idxs = grid_tools.reduce_table_idx(table=self.params, verbose=False,
                                               params={'qb': qb, 'z': z, 'x': x, 'mass': mass})

            if len(idxs) < 2:
                continue
            else:
                sub_params = self.params.iloc[idxs]
                sub_summ = self.summ.iloc[idxs]

                mdot = sub_params['accrate'].values * sub_params['xi'].values
                tdel = sub_summ['tDel'].values
                u_tdel = sub_summ['uTDel'].values

                # noinspection PyTupleAssignmentBalance
                m, y0 = np.polyfit(x=np.log(mdot), y=np.log(tdel), w=np.log(1 / u_tdel), deg=1)
                powerfits['m'][i] = m
                powerfits['y0'][i] = y0

        nan_mask = np.array(np.isnan(powerfits['m']))  # Remove nans
        self.powerfits = powerfits.iloc[~nan_mask]

    def predict_recurrence(self, x, z, qb, mdot, mass):
        """Predict recurrence time for given params
        
        mdot    =  flt : accretion rate (Edd)
        """
        params = {'z': z, 'x': x, 'qb': qb, 'mass': mass}
        idx = grid_tools.reduce_table_idx(table=self.powerfits, params=params)

        # ===== check if params not in powerfits =====
        if (len(idx) == 0):
            self.printv(f'CAUTION: ' +
                        f'tDel not predicted for z={z}, x={x}, qb={qb}, ' +
                        f'mass={mass:.1f}; Using closest values:')
            sub_table = self.powerfits.copy()
            for param, val in params.items():
                closest_idx = (np.abs(sub_table[param].values - val)).argmin()
                closest_val = sub_table[param].values[closest_idx]
                sub_table = grid_tools.reduce_table(table=sub_table,
                                                    params={param: closest_val}, verbose=False)
                params[param] = closest_val
                self.printv(f'{param}={params[param]}')
        else:
            sub_table = grid_tools.reduce_table(table=self.powerfits,
                                                params=params)

        m = float(sub_table['m'])
        y0 = float(sub_table['y0'])
        dt = (np.exp(1) ** y0) * (mdot ** m)
        return dt

    def plot_mean_lc(self, batch, run, show=True):
        """Plots mean lightcurve for given batch model
        """
        # -------------------------
        # TODO: - option to plot individual model curves (adapt from analyser_tools)
        # -------------------------
        fig, ax = plt.subplots()
        ax = self.add_lc_plot(ax=ax, batch=batch, run=run)
        ax.set_ylabel(r'$L$ ($10^{38}$ ergs $s^{-1}$)')
        ax.set_xlabel(r'Time (s)')
        if show:
            plt.show(block=False)
        return ax

    def save_mean_lc(self, params, error=True, show=False):
        """Save a series of mean lightcurve plots for given params
        """
        models = self.get_params(params=params)

        for i, row in models.iterrows():
            title = ''
            for p in params:
                title += f'{p}={params[p]:.3f}_'

            batch = int(row['batch'])
            run = int(row['run'])
            mdot = row['accrate'] * row['xi']
            title += f'mdot={mdot:.4f}'

            ax = self.plot_mean_lc(batch=batch, run=run, show=show)
            ax.set_title(title)
            ax.set_xlim([-20, 100])
            ax.set_ylim([-0.1, 4.0])

            filename = f'{self.source}_{title}.png'
            filepath = os.path.join(self.source_path, 'plots',
                                    'mean_lightcurves', filename)
            plt.savefig(filepath)
            plt.close()

    def plot_mean_lc_param(self, params, show=True, legend=False,
                           skip=1, **kwargs):
        """Plots mean lightcurves for given param (vary by accrate)
        
        skip  =  int  : only plot every 'skip' LC
        """
        accrate_unique = self.unique_params['accrate']

        fig, ax = plt.subplots()
        ax.set_ylabel(r'Lum ($10^{38}$ erg)')
        ax.set_xlabel('Time (s)')
        ax.set_xlim([-10, 80])
        # ax.set_title(fr'z={z:.3f}, Qb={qb:.1f} (Legend: $\dot{{M}}_\mathrm{{Edd}}$)')

        for accrate in accrate_unique:
            params['accrate'] = accrate
            subset = self.get_params(params=params)
            n_models = len(subset)

            if n_models == 0:
                continue
            for i in range(1, n_models, skip):
                batch = int(subset.iloc[i]['batch'])
                run = int(subset.iloc[i]['run'])
                mdot = accrate * subset.iloc[i]['xi']
                label = f'{mdot:.4f}'

                ax = self.add_lc_plot(ax=ax, batch=batch, run=run, label=label, **kwargs)

        if legend:
            ax.legend()
        if show:
            plt.show(block=False)
        return ax

    def add_lc_plot(self, ax, batch, run, label='', error=True):
        """Add mean lightcurves to a provided axis
        
        param  =  str  : param to label curve with (none if empty)
        """
        yscale = 1e38
        # ===== Check if loaded =====
        if batch not in self.mean_lc:
            self.printv('Loading lightcurve')
            self.load_mean_lightcurves(batch)

        mean_lc = self.mean_lc[batch][run]
        x = mean_lc[:, 0]
        y = mean_lc[:, 1] / yscale

        if error:
            err = mean_lc[:, 2] / yscale
            y_lo = y - err
            y_hi = y + err
            ax.fill_between(x, y_lo, y_hi, color='0.8')

        ax.plot(x, y, label=label)
        return ax

    def load_mean_lightcurves(self, batch):
        """
        Loads mean lightcurve files for given batch
        --------------------------------------------------------------
        columns: [time (s), Lum (erg), u(Lum), Radius (cm), u(Radius)]
                  0         1          2       3            4
        """
        batch_str = f'{self.source}_{batch}'
        path = os.path.join(self.source_path, 'mean_lightcurves', batch_str)

        n_runs = self.get_nruns(batch)
        self.mean_lc[batch] = {}

        for run in range(1, n_runs + 1):
            run_str = grid_strings.get_run_string(run, basename=self.basename)
            filename = f'{batch_str}_{run_str}_mean.data'
            filepath = os.path.join(path, filename)

            self.mean_lc[batch][run] = np.loadtxt(filepath)

    def load_all_mean_lightcurves(self):
        """Loads all mean lightcurves
        """
        batches = np.unique(self.params['batch'])
        last = batches[-1]
        for batch in batches:
            self.printv(f'Loading mean lightcurves: batches {batch}/{last}', end='\r')
            self.load_mean_lightcurves(batch=batch)
        self.printv('')

    def plot_burst_property(self, bprop, var, fixed, save=False, show=True,
                            exclude_defaults=False, powerfits=False):
        """Plots given burst property against accretion rate (including xi factors)
        
        bprop   =  str   : property to plot on y-axis (e.g. 'tDel')
        var     =  str   : variable to iterate over (e.g. plot all available 'Qb')
        fixed   =  dict  : variables to hold fixed (e.g. 'z':0.01)
        """
        var, fixed = self.check_var_fixed(var=var, fixed=fixed)

        accrate_unique = self.unique_params['accrate']
        var_unique = self.unique_params[var]
        params = dict(fixed)

        uncertainty_keys = {False: {'tDel': 'uTDel', 'fluence': 'uFluence',
                                    'peakLum': 'uPeakLum'},
                            True: {'dt': 'u_dt', 'fluence': 'u_fluence',
                                   'peak': 'u_peak'},
                            }[self.burst_analyser]

        # ylims = {'fluence': [0.3e40, 1.e40],
        #          'tDel': [0, 20],
        #          'peakLum': [1e38, 5e38]}
        # ylim = ylims[bprop]
        u_prop = uncertainty_keys[bprop]

        unit_factors = {'tDel': 3600}  # unit factors
        if bprop in unit_factors:
            unit_f = unit_factors[bprop]
        else:
            unit_f = 1.0

        # ===== Axis properties =====
        fig, ax = plt.subplots()
        title = ''
        for p, pv in fixed.items():
            title += f'{p}={pv:.3f}, '

        # ax.set_ylim(ylim)
        ax.set_xlim([0.02, 0.25])
        ax.set_xlabel(r'$\dot{M} \; (\dot{M}_\mathrm{Edd})$ ')
        ax.set_ylabel(bprop)
        ax.set_title(title)
        plt.tight_layout()

        for v in var_unique:
            # ===== check if any models exist =====
            params[var] = v
            subset = self.get_params(params=params)

            if len(subset) == 0:
                continue
            # if exclude_defaults and (v in self.params_exclude[var]):
            #     continue

            mdot_x = []
            prop_y = []
            u_y = []

            for accrate in accrate_unique:
                params['accrate'] = accrate
                subset = self.get_params(params=params)
                idxs = subset.index

                mdot_tmp = np.array(accrate * subset['xi'])
                prop_tmp = np.array(self.summ.iloc[idxs][bprop] / unit_f)
                u_tmp = np.array(self.summ.iloc[idxs][u_prop] / unit_f)

                # === remove zero-uncertainties (if model has only 3 bursts)===
                if 0.0 in u_tmp:
                    frac_err = 0.13  # typical fractional error
                    idx = np.where(u_tmp == 0.0)[0][0]
                    u_tmp[idx] = frac_err * prop_tmp[idx]

                mdot_x = np.concatenate([mdot_x, mdot_tmp])
                prop_y = np.concatenate([prop_y, prop_tmp])
                u_y = np.concatenate([u_y, u_tmp])

            label = f'{var}={v:.4f}'
            ax.errorbar(x=mdot_x, y=prop_y, yerr=u_y, ls='', marker='o',
                        label=label, capsize=3)

            del (params['accrate'])

        # ===== plot power laws =====
        # return params, var_unique
        # TODO: only plot powerfits for plotted vars (e.g. X)
        if powerfits:

            ax.set_prop_cycle(None)  # reset color cycle
            x = np.linspace(0.01, 1.0, 100)
            n = len(self.powerfits)

            for i in range(n):
                m = self.powerfits.iloc[i]['m']
                y0 = self.powerfits.iloc[i]['y0']
                y = (np.exp(1) ** y0) * (x ** m) / 3600
                label = f'{m:.2f}'
                ax.plot(x, y, label=label)

        ax.legend()

        if show:
            plt.show(block=False)

        if save:
            fixed_str = ''
            for p, v in fixed.items():
                fixed_str += f'_{p}={v:.4f}'

            save_dir = os.path.join(self.source_path, 'plots', bprop)
            filename = f'bprop_{bprop}_{self.source}_C{self.con_ver:02}{fixed_str}.png'
            filepath = os.path.join(save_dir, filename)

            self.printv(f'Saving {filepath}')
            plt.savefig(filepath)

        return ax

    def plot_lhood(self, var, fixed, show=True,
                   save=False, exclude_defaults=True, **kwargs):
        """Plots likelihood of best concord against gamma (and iterates over chosen var)
        
        fixed   =  {}  : variables to hold fixed (e.g. 'z':0.01)
        
        Notes:  two ways to use:
                1. specify qb to plot over z, and vice versa (shortcut usage)
                2. specify var and fixed manually
        Caution: Has not been rigorously tested against all combinations of
                    these. Has some implicit assumptions about params given
        """
        var, fixed = self.check_var_fixed(var=var, fixed=fixed)

        accrate0 = np.unique(self.params['accrate'])[-1]  # just use one accrate to pick models
        var_unique = self.unique_params[var]
        params = dict(fixed)
        params['accrate'] = accrate0

        # ===== plotting stuff =====
        fig, ax = plt.subplots()
        ax.set_xlabel(r'$\gamma$  ($\dot{M}$ multiplier)')
        ax.set_ylabel(r'lhood')

        title = ''
        for p, pv in fixed.items():
            title += f'{p}={pv:.3f}, '

        ylims = con_versions.get_lhood_ylims(self.con_ver)
        ax.set_ylim(ylims)
        ax.set_xlim([0.75, 3.0])
        ax.set_title(title)
        plt.tight_layout()

        for v in var_unique:
            # ===== Check if any models exist =====
            params[var] = v
            subset = self.get_params(params=params)
            n_models = len(subset)
            if n_models == 0:
                continue

            # if exclude_defaults and (v in self.default_exclude[var]):
            #     continue

            xi_unique = np.unique(subset['xi'])
            x = []
            y = []

            for xi in xi_unique:
                params['xi'] = xi

                subset = self.get_params(params=params)
                triplet = subset['batch'].values[0]
                run = subset['run'].values[0]

                x_tmp = np.array([xi])
                lhood = np.array([self.get_lhood(triplet=triplet, run=run)])

                if not np.isnan(lhood[0]):
                    x = np.concatenate([x, x_tmp])
                    y = np.concatenate([y, lhood])

            label = f'{var}={v:.4f}'
            ax.plot(x, y, ls='-', marker='o', label=label)
            del (params['xi'])

        ax.legend()
        if show:
            plt.show(block=False)
        if save:
            not_var = self.get_not_vars(var)
            val = fixed[not_var]

            save_dir = os.path.join(self.source_path, 'plots', 'lhood')
            filename = f'lhood_{self.source}_C{self.con_ver:02}_{not_var}={val:.3f}.png'
            filepath = os.path.join(save_dir, filename)

            self.printv(f'Saving {filepath}')
            plt.savefig(filepath)

        return ax

    def plot_grid_params(self, var, fixed, show=True):
        """Visualises the models that exist in the grid by parameter
        
        var   = [2x str]  : list of the two parameters to plot on the axes
        fixed = {}        : specify the constant values of the remaining three
                                paramters
        """
        if len(var) != 2:
            raise ValueError("'var' must specify two parameters to plot on axes")

        if len(fixed) != 3:
            raise ValueError("'fixed' must specify three paramters to hold constant")

        fig, ax = plt.subplots()
        subset = self.get_params(params=fixed)

        # TODO: check only one unique value of all other params
        x = subset[var[0]]
        y = subset[var[1]]
        ax.plot(x, y, marker='o', ls='none')

        ax.set_xlabel(var[0])
        ax.set_ylabel(var[1])
        title = ''
        for f in fixed:
            title += f'{f}={fixed[f]}, '
        ax.set_title(title)

        plt.show(block=False)

    def check_var_fixed(self, var, fixed):
        if var in fixed:
            raise ValueError('var cant also be in fixed')
        return var, fixed

    def get_not_vars(self, var):
        p_list = ['x', 'z', 'qb', 'mass']
        return [p for p in p_list
                if (p != var)]

    def save_all_plots(self, fixed=None, bprops=('tDel', 'fluence', 'peakLum'),
                       do_bprops=True, **kwargs):
        """Saves all lhood and var plots for given z,qb
        """
        self.printv('Saving lhood and bprop plots:')
        if fixed is None:
            if self.source == 'gs1826':
                fixed = {'z': [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02],
                         'qb': [0.05, 0.1, 0.15, 0.2]}
            elif self.source == '4u1820':
                fixed = {'z': [0.005, 0.01, 0.015, 0.02],
                         'qb': [0.1, 0.2],
                         'x': [0.0, 0.05, 0.1]}
            else:
                print("Defaulting to unique params")
                fixed = {}
                for p in ('x', 'z', 'qb', 'mass'):
                    fixed[p] = self.unique_params[p]

        for var in fixed:
            self.printv(f'Saving plot var={var}')
            not_vars = self.get_not_vars(var)
            sub_fixed = {v: fixed[v] for v in not_vars}
            full_fixed = grid_tools.enumerate_params(sub_fixed)
            n_fixed = len(full_fixed[not_vars[0]])

            for i in range(n_fixed):
                fixed_input = {x: full_fixed[x][i]
                               for x in full_fixed}
                # if do_lhoods:
                #     self.plot_lhood(var=var, fixed={pfix:val}, save=True,
                #                         show=False, **kwargs)
                if do_bprops:
                    for bprop in bprops:
                        self.plot_burst_property(bprop=bprop, var=var,
                                                 powerfits=False, fixed=fixed_input,
                                                 save=True, show=False, **kwargs)
                plt.close('all')

    def plot_matched_lightcurves(self, triplet, run):
        """Plots matched lightcurves for given run of batch (triplet)
        """
        self.printv(f'Plotting matched lightcurves for batch={triplet}, run={run}')
        self.print_params(batch=triplet, run=run)
        self.printv(triplet)
        ctools.plot_lightcurves(run=run, batches=triplet, con_ver=self.con_ver,
                                source=self.source)

    def print_params(self, batch, run):
        """Prints essential params for given batch-run
        """
        cols = ['batch', 'run', 'z', 'x', 'qb', 'xi', 'mass']
        params = self.get_params(batch=batch, run=run)
        out_string = params.to_string(columns=cols, index=False)
        print(out_string)

    def best_concord_lhood(self, params={}, plot=False):
        """Returns model with highest likelihood (from given params)
        """
        if len(params) == 0:
            concord_summ = self.concord_summ
        else:
            concord_summ = self.get_concord_sum(params=params)

        lhoods = concord_summ['lhood'].values
        best_idx = np.nanargmax(lhoods)
        best_concord = concord_summ.iloc[best_idx]

        batch = int(best_concord['triplet'])
        run = int(best_concord['run'])
        model = self.get_params(batch=batch, run=run)

        if plot:
            self.plot_matched_lightcurves(triplet=batch, run=run)

        return model, best_concord


def get_unique_param(param, source):
    """Return unique values of given parameter
    """
    source = grid_strings.source_shorthand(source=source)
    params_filepath = grid_strings.get_params_filepath(source)
    param_table = pd.read_table(params_filepath, delim_whitespace=True)
    return np.unique(param_table[param])


def check_kgrid(kgrid, source):
    if kgrid is None:
        kgrid = Kgrid(source, load_concord_summ=False, exclude_test_batches=False,
                      powerfits=False)
    return kgrid


def printv(string, verbose):
    """Prints string if verbose == True
    """
    if verbose:
        print(string)
