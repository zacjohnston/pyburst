# ========================
# Utilities for grids of kepler models
# Author: Zac Johnston (zac.johnston@monash.edu)
# ========================

# standard
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
import sys

# kepler_grids
from . import grid_tools, grid_strings, grid_versions
from pyburst.burst_analyser import burst_tools

GRIDS_PATH = os.environ['KEPLER_GRIDS']
MODELS_PATH = os.environ['KEPLER_MODELS']

# -----------------------------------
# TODO:
#       - Check for grid "completeness" (missing parameter combinations)
# -----------------------------------


def default_plt_options():
    """Initialise default plot parameters"""
    params = {'mathtext.default': 'regular',
              'font.family': 'serif', 'text.usetex': False}
    plt.rcParams.update(params)


default_plt_options()


class Kgrid:
    """
    An object for interracting with large model grids
    """

    def __init__(self, source, basename='xrb', grid_version=0,
                 load_lc=False, verbose=True, linregress_burst_rate=False,
                 lampe_analyser=False, load_bursts=False, use_sub_cols=False,
                 **kwargs):
        """
        source   =  str  : source object being modelled (e.g. gs1826)
        basename =  str  : basename of individual models (e.g. xrb)
        mass     =  flt  : mass of neutron star in M_sun (may become deprecated)
        (path    = str   : path to dir of grids)
        """
        default_plt_options()
        source = grid_strings.source_shorthand(source=source)
        self.path = kwargs.get('path', GRIDS_PATH)
        self.models_path = kwargs.get('models_path', MODELS_PATH)
        self.source_path = grid_strings.get_source_path(source)
        self.source = source
        self.basename = basename
        self.verbose = verbose
        self.lampe_analyser = lampe_analyser
        self.grid_version = grid_versions.GridVersion(source, grid_version)
        self.printv(self.grid_version)
        self.config = grid_tools.load_config(source, select='plotting')

        # ==== Load tables of models attributes ====
        self.use_sub_cols = use_sub_cols
        self.sub_cols = ['batch', 'run', 'accrate', 'qb', 'x', 'z', 'mass', 'qnuc']
        self.params = None
        self.summ = None
        self.bursts = None

        self.tablenames = ['params', 'summ']
        if load_bursts:
            self.tablenames += ['bursts']

        self.load_tables()

        # ===== extract the unique parameters =====
        self.unique_params = {}
        for v in self.params.columns[2:]:
            self.unique_params[v] = np.unique(self.params[v])

        if self.verbose:
            grid_tools.check_complete(param_table=self.params, raise_error=False)

        self.n_models = len(self.params)
        self.printv('=' * 40)
        self.printv('Loaded')
        self.printv(f'Source: {source}')
        self.printv(f'Models in grid: {self.n_models}')

        self.linear_rates = None
        if linregress_burst_rate:
            self.linregress_burst_rate()

        # ===== Load mean lightcurve data =====
        self.mean_lc = {'columns': ['Time', 'L', 'u(L)', 'R', 'u(R)']}
        self.burst_lc = {'columns': ['Time', 'L']}
        if load_lc:
            self.load_all_mean_lightcurves()

        # ==== Keep only important columns (may break other things...) =====
        if self.use_sub_cols:
            self.printv('NOTE: Cutting out param columns according to sub_cols. '
                        'May break functions which rely on secondary parameters')
            self.params = self.params[self.sub_cols]

    def printv(self, string, **kwargs):
        """Prints string if self.verbose == True
        """
        if self.verbose:
            print(string, **kwargs)

    def get_nruns(self, batch):
        """Returns number of models in a batch
        """
        return len(self.get_params(batch=batch))

    def load_tables(self):
        """Loads grid tables of model inputs and outputs, excluding models as defined
            in grid_version
        """
        # TODO: apply exclusions to bursts table
        tables = {}
        for tablename in self.tablenames:
            tables[tablename] = grid_tools.load_grid_table(tablename=tablename,
                                                           source=self.source,
                                                           verbose=self.verbose)

        self.params = grid_tools.reduce_table(table=tables['params'], params={},
                                              exclude_any=self.grid_version.exclude_any,
                                              exclude_all=self.grid_version.exclude_all)
        idxs = self.params.index.values
        self.summ = tables['summ'].loc[idxs]
        self.bursts = tables.get('bursts')

    def get_params(self, batch=None, run=None, params=None, exclude_any=None,
                   exclude_all=None):
        """Returns models with given batch/run/params
        
        params  = {}      : params that must be satisfied
        exclude = {}      : params to exclude/blacklist completely
        
        Can be used multiple ways:
            - get_params(batch=2, run=1): returns run 1 of batch 2
            - get_params(batch=2): returns all models in batch 2
            - get_params(params={'z':0.01}): returns all models with z=0.01
        """
        def add_optional(parameter, empty):
            """Includes optional extra values if provided
            """
            return empty if (parameter is None) else parameter

        if all(x is None for x in (batch, run, params, exclude_any)):
            raise ValueError('Must specify at least one argument')

        params_full = {}

        if batch is not None:
            params_full['batch'] = batch
        if run is not None:
            params_full['run'] = run
        if params is not None:
            params_full = {**params_full, **params}

        exclude_any = add_optional(exclude_any, empty={})
        exclude_all = add_optional(exclude_all, empty=[])

        models = grid_tools.reduce_table(table=self.params, params=params_full,
                                         exclude_any=exclude_any,
                                         exclude_all=exclude_all)
        return models

    def get_summ(self, batch=None, run=None, params=None,
                 exclude_any=None, exclude_all=None):
        """Get summary of given batch/run/params
        """
        subset = self.get_params(batch=batch, run=run, params=params,
                                 exclude_any=exclude_any, exclude_all=exclude_all)
        idxs = subset.index.values
        return self.summ.loc[idxs]

    def linregress_burst_rate(self):
        """Calculate linear fits to burst rate versus accretion rate
        """
        param_list = ('x', 'z', 'mass', 'qb')
        params = {}
        for param in param_list:
            params[param] = self.unique_params[param]

        linear_rate = pd.DataFrame()
        enum_params = grid_tools.enumerate_params(params)
        n = len(enum_params[param_list[0]])

        for p in params:
            linear_rate[p] = enum_params[p]

        for key in ['m', 'y0']:
            linear_rate[key] = np.full(n, np.nan)

        for row in linear_rate.itertuples():
            i = row.Index
            sys.stdout.write(f'\rCalculating linear fits to burst rates: {i+1}/{n}')

            sub_params = {'qb': row.qb, 'z': row.z, 'x': row.x, 'mass': row.mass}
            table_params = self.get_params(params=sub_params)
            table_summ = self.get_summ(params=sub_params)

            nan_mask = np.array(np.isnan(table_summ['rate']))  # Remove nans
            table_summ = table_summ.iloc[~nan_mask]
            table_params = table_params.iloc[~nan_mask]

            if len(table_params) < 2:
                continue
            else:
                accrate = table_params['accrate'].values * table_params['acc_mult'].values
                rate = table_summ['rate'].values
                m, y0, _, _, _ = linregress(x=accrate, y=rate)
                linear_rate.loc[i, 'm'] = m
                linear_rate.loc[i, 'y0'] = y0

        sys.stdout.write('\n')
        nan_mask = np.array(np.isnan(linear_rate['m']))  # Remove nans
        self.linear_rates = linear_rate.iloc[~nan_mask]

    def predict_recurrence(self, accrate, params):
        """Predict recurrence time (s) for given params
        
        accrate : flt
            accretion rate (fraction of Eddington)
        params : dict
            specify model parameters (x, z, qb, mass)
        """
        idx = grid_tools.reduce_table_idx(table=self.linear_rates, params=params)

        if len(idx) == 0:
            self.printv(f'dt not predicted for {params}. Using closest values:')
            sub_table = self.linear_rates.copy()
            for param, val in params.items():
                closest_idx = (np.abs(sub_table[param].values - val)).argmin()
                closest_val = sub_table[param].values[closest_idx]
                sub_table = grid_tools.reduce_table(table=sub_table,
                                                    params={param: closest_val})
                params[param] = closest_val
                self.printv(f'{param}={params[param]}')
        else:
            sub_table = grid_tools.reduce_table(table=self.linear_rates, params=params)

        rate = accrate * float(sub_table['m']) + float(sub_table['y0'])
        day_sec = 24*3600
        return day_sec / rate

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

    def save_mean_lc(self, params, show=False):
        """Save a series of mean lightcurve plots for given params
        """
        models = self.get_params(params=params)

        for i, row in models.iterrows():
            title = ''
            for p in params:
                title += f'{p}={params[p]:.3f}_'

            batch = int(row['batch'])
            run = int(row['run'])
            mdot = row['accrate'] * row['acc_mult']
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
                mdot = accrate * subset.iloc[i]['acc_mult']
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

    def load_burst_lightcurves(self, batch, burst=None):
        """Loads individual burst lightcurves for given batch

        Note: save LCs with burst_analyser.BurstRun.save_burst_lightcurves()
        """
        self.burst_lc[batch] = {}
        batch_table = self.get_summ(batch=batch)
        self.printv(f'Loading burst lightcurves for batch: {batch}')

        for model in batch_table.itertuples():
            run = model.run
            n_bursts = model.num
            self.burst_lc[batch][run] = {}

            if burst is None:
                bursts = range(n_bursts)
            else:
                bursts = [burst]

            for burst_i in bursts:
                sys.stdout.write(f'\rLoading burst lc: run {run}, burst {burst_i+1}')
                lc = burst_tools.load_burst_lightcurve(burst_i, run=run, batch=batch,
                                                       source=self.source)
                self.burst_lc[batch][run][burst_i] = lc

    def load_mean_lightcurves(self, batch):
        """
        Loads mean lightcurve files for given batch
        --------------------------------------------------------------
        columns: [time (s), Lum (erg), u(Lum), Radius (cm), u(Radius)]
                  0         1          2       3            4
        """
        batch_str = f'{self.source}_{batch}'
        path = os.path.join(self.source_path, 'mean_lightcurves', batch_str)
        batch_table = self.get_params(batch=batch)
        self.mean_lc[batch] = {}

        for run in batch_table['run']:
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

    def plot_burst_property(self, bprops, var, fixed, xaxis='accrate', save=False,
                            show=True, linear_rates=False, interpolate=True,
                            shaded=True, exclude_stable=False, legend=True,
                            fix_ylims=True):
        """Plots given burst property against accretion rate
        
        bprop : [str]
            properties to plot on y-axis (e.g. 'tDel')
        var : str
            variable to iterate over (e.g. plot all available 'Qb')
        fixed : dict
            variables to hold fixed (e.g. 'z':0.01)
        """
        fontsize = 14
        precisions = {'z': 4, 'x': 2, 'qb': 3, 'mass': 1}
        var, fixed = check_var_fixed(var=var, fixed=fixed)
        xlabel = {'accrate': r'$\dot{M} / \dot{M}_\mathrm{Edd}$'}.get(xaxis, xaxis)
        x_unique = self.unique_params[xaxis]

        var_unique = self.unique_params[var]
        params = dict(fixed)
        uncertainty_keys = {'dt': 'u_dt', 'fluence': 'u_fluence',
                            'peak': 'u_peak', 'rate': 'u_rate',
                            'alpha': 'u_alpha'}

        y_labels = {'dt': r'$\Delta t$ (hr)',
                    'fluence': r'$E_b$ ($10^{39}$ erg)',
                    'peak': r'$L_{peak}$ ($10^{38}$ erg s$^{-1}$)',
                    'rate': 'Burst rate (day$^{-1}$)',
                    'alpha': r'$\alpha$',
                    'length': 'Burst length (min)',
                    'tail_index': 'Power Index',
                    }
        # TODO: Move these into config file [plotting]
        ylims = {'rate': {
                    'grid5': [0.0, 24],
                    'he2': [0.0, 55],
                    'ks1': [2.0, 20],
                  },
                 'dt': {
                     'grid5': [0.0, 10],
                     'he2': [0.0, 14],
                     'ks1': [0.0, 8],
                 },
                 'fluence': {
                     'grid5': [3.0, 14],
                     'he2': [0.5, 9],
                     'ks1': [2.0, 10],
                 },
                 'peak': {
                     'grid5': [0.0, 7.0],
                     'ks1': [1.0, 7],
                 },
                 'length': [4, 32],
                 }
        y_factors = {'tDel': 3600, 'dt': 3600, 'length': 60,
                     'fluence': 1e39, 'peak': 1e38}

        n_bprops = len(bprops)
        fig, ax = plt.subplots(n_bprops, 1, figsize=(6, 4*n_bprops))
        if not isinstance(ax, np.ndarray):
            ax = [ax]

        # === make title ===
        title = ''
        for p, pv in fixed.items():
            precision = precisions.get(p, 3)
            title += f'{p}={pv:.{precision}f}, '

        ax[0].set_title(title, fontsize=14)
        ax[-1].set_xlabel(xlabel, fontsize=fontsize)

        for i, bprop in enumerate(bprops):
            u_bprop = uncertainty_keys.get(bprop, f'u_{bprop}')
            y_factor = y_factors.get(bprop, 1.0)

            y_label = y_labels.get(bprop, bprop)
            ax[i].set_ylabel(y_label, fontsize=fontsize)

            for v in var_unique:
                # ===== check if any models exist =====
                params[var] = v
                subset = self.get_params(params=params)
                if len(subset) == 0:
                    continue

                mdot_x = []
                prop_y = []
                u_y = []

                for x_value in x_unique:
                    params[xaxis] = x_value
                    subset = self.get_params(params=params)
                    idxs = subset.index

                    if exclude_stable and self.summ.loc[idxs]['stable_burning'].bool():
                        continue
                    mdot_tmp = np.full(len(subset), x_value)
                    prop_tmp = np.array(self.summ.loc[idxs][bprop] / y_factor)
                    u_tmp = np.array(self.summ.loc[idxs][u_bprop] / y_factor)

                    mdot_x = np.concatenate([mdot_x, mdot_tmp])
                    prop_y = np.concatenate([prop_y, prop_tmp])
                    u_y = np.concatenate([u_y, u_tmp])

                precision = precisions.get(var, 3)
                if var == 'z':
                    label = r'$Z_{CNO}$' + f'={v:.{precision}f}'
                else:
                    label = f'{var}={v:.{precision}f}'

                if shaded:
                    ax[i].fill_between(mdot_x, prop_y+u_y, prop_y-u_y, alpha=0.3)

                ax[i].errorbar(x=mdot_x, y=prop_y, yerr=u_y, marker='o',
                               label=label, capsize=3, ls='-' if interpolate else 'none')
                del (params[xaxis])

            if linear_rates:
                xlims = (0.0, 1.0)
                ax[i].set_prop_cycle(None)  # reset color cycle
                linear = grid_tools.reduce_table(self.linear_rates, params=fixed)
                for row in linear.itertuples():
                    rate = row.m * np.array(xlims) + row.y0
                    ax[i].plot(xlims, rate)

            bprop_ylims = ylims.get(bprop)
            if fix_ylims and (bprop_ylims is not None):
                ylim = bprop_ylims.get(self.source)
                ax[i].set_ylim(ylim)

        if legend:
            loc = {'rate': 'upper left'}.get(bprops[0], 'upper right')
            ax[0].legend(fontsize=fontsize - 2, loc=loc)

        plt.tight_layout()

        # TODO: use generic save_display()
        if show:
            plt.show(block=False)
        if save:
            fixed_str = ''
            for p, v in fixed.items():
                precision = precisions.get(p, 3)
                fixed_str += f'_{p}={v:.{precision}f}'

            save_dir = os.path.join(self.source_path, 'plots', 'grid')
            filename = f'grid_{self.source}{fixed_str}.png'
            filepath = os.path.join(save_dir, filename)

            self.printv(f'Saving {filepath}')
            plt.savefig(filepath)
        return ax

    def plot_summ(self, var='num', batch=None, vlines=True, hline=None):
        """Plot any column from summ stable, versus batch/run

        Parameters
        ----------
        var : str
            variable from summ table to plot on y-axis
        batch : int (optional)
            if specified, plot only this batch, with its runs on the x-axis
        vlines : bool (optional)
            plot vertical lines between x-axis and points, if plotting single batch
        hline : int (optional)
            place to plot horizontal bar (if None, don't plot)
        """
        title = f'{self.source}_V{self.grid_version.version}'

        if batch is None:
            summ_table = self.summ
            x_axis = 'batch'
        else:
            summ_table = self.get_summ(batch=batch)
            x_axis = 'run'
            title = f'{title} Batch_{batch}'

        fig, ax = plt.subplots()

        if hline is not None:
            ax.plot([0, np.max(summ_table[x_axis])], [hline, hline], color='red')

        if vlines and (batch is not None):
            for row in summ_table.itertuples():
                ax.plot([row.run, row.run], [0, row.num], color='black')

        ax.plot(summ_table[x_axis], summ_table[var], marker='o', ls='none')
        ax.set_xlabel(x_axis)
        ax.set_ylabel(f'{var}')
        ax.set_title(title)
        plt.show(block=False)

    def save_all_plots(self, fixed=('x', 'mass', 'qb'), var='z', xaxis='accrate',
                       bprops=('rate', 'fluence', 'peak'), **kwargs):
        """Saves burst_property plots for various iterations of parameters
        """
        # TODO: docstring
        self.printv('Saving bprop plots:')
        unique = {}
        for p in fixed:
            unique[p] = self.unique_params[p]

        full_fixed = grid_tools.enumerate_params(unique)
        n_fixed = len(full_fixed[fixed[0]])

        for i in range(n_fixed):
            fixed_input = {x: full_fixed[x][i] for x in full_fixed}

            self.plot_burst_property(bprops=bprops, var=var, xaxis=xaxis, save=True,
                                     fixed=fixed_input, show=False, **kwargs)
            # TODO: grab fig from plot_burst_property and close explicitly
            plt.close('all')

    def print_params(self, batch, run):
        """Prints essential params for given batch-run
        """
        cols = ['batch', 'run', 'accrate', 'z', 'x', 'qb', 'mass', 'qnuc']
        params = self.get_params(batch=batch, run=run)
        out_string = params.to_string(columns=cols, index=False)
        print(out_string)

    def print_unique_params(self):
        grid_tools.print_params_summary(self.params)

    def print_batch_summary(self, batch, batch_n=None, show=None):
        """Pretty print a summary of params in a batch

        parameters
        ----------
        batch : int
        batch_n : int (optional)
            summarise all batches between batch and batch_n
        show : [str] (optional)
            specify parameters to show
        """
        if batch_n is None:
            batches = [batch]
        else:
            batches = np.arange(batch, batch_n+1)

        batch_params = self.get_combined_params(batches)
        grid_tools.print_params_summary(batch_params, show=show)

    def get_combined_params(self, batches):
        """Returns sub-table of self.params with specified batches

        parameters
        ----------
        batches : sequence(int)
        """
        return self.get_combined(batches, label='params')

    def get_combined_summ(self, batches):
        """Returns sub-table of self.summ with specified batches

        parameters
        ----------
        batches : sequence(int)
        """
        return self.get_combined(batches, label='summ')

    def get_combined(self, batches, label):
        """Returns sub-table of self.params or self.summ, with specified batches

        parameters
        ----------
        batches : sequence(int)
        label : str
            one of (params, summ)
        """
        if label == 'params':
            func = self.get_params
        elif label == 'summ':
            func = self.get_summ
        else:
            raise ValueError("'label' must be one of (params, summ)")

        table = pd.DataFrame()
        for batch in batches:
            batch_params = func(batch)
            table = table.append(batch_params, ignore_index=True)
        return table

def printv(string, verbose):
    """Prints string if verbose == True
    """
    if verbose:
        print(string)


def check_var_fixed(var, fixed):
    if var in fixed:
        raise ValueError('var cant also be in fixed')
    return var, fixed


def get_not_vars(var, var2='', var3=''):
    # TODO: more elegant way to do this
    p_list = ['x', 'z', 'qb', 'mass', 'accrate']
    return [p for p in p_list
            if (p != var) and (p != var2) and (p != var3)]
