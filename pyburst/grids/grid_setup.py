import os
import numpy as np
import pandas as pd
import subprocess

# pyburst
from . import grid_analyser
from . import grid_tools
from . import grid_strings
from pyburst.mcmc import mcmc_versions, mcmc_tools
from pyburst.kepler import kepler_jobs, kepler_files
from pyburst.misc.pyprint import print_title, print_dashes
from pyburst.qnuc import qnuc_tools
from pyburst.physics import gravity

# ====================================
# Kepler model grid generator
# --------------------
# Generates kepler model generator files, in addition to setting up model grids
# and jobscripts
# --------------------
# Note: This module is a bit of a patchwork and not so nice to look at. Things might
#           not work elegently or as expected
# --------------------
# Author: Zac Johnston
# Email: zac.johnston@monash.edu
# ====================================

flt2 = '{:.2f}'.format
flt4 = '{:.4f}'.format
flt8 = '{:.8f}'.format

FORMATTERS = {'z': flt8, 'y': flt8, 'x': flt8, 'accrate': flt8,
              'tshift': flt2, 'qb': flt8, 'acc_mult': flt2, 'qb_delay': flt2,
              'mass': flt8}

GRIDS_PATH = os.environ['KEPLER_GRIDS']
MODELS_PATH = os.environ['KEPLER_MODELS']

# TODO: This whole module is a mess. Needs lots of tidying up
# TODO: Rewrite docstrings
# TODO: Allow enumerating over multiple parameters, create_batches()

param_list = ['x', 'z', 'qb', 'accrate', 'accdepth', 'accmass', 'mass']


def create_batch(batch, dv, source,
                 params={'x': [0.6, 0.8], 'z': [0.01, 0.02],
                         'tshift': [0.0], 'accrate': [0.05],
                         'qb': [0.125], 'acc_mult': [1.0], 'qnuc': [5.0],
                         'qb_delay': [0.0], 'mass': [1.4],
                         'accmass': [1e16], 'accdepth': [1e20]},
                 lburn=1, t_end=1.3e5, exclude=None, basename='xrb', walltime=96,
                 nstop=10000000, nsdump=500, auto_t_end=True, notes='No notes given',
                 nbursts=20, kgrid=None, nuc_heat=True, setup_test=False,
                 predict_qnuc=False, grid_version=None, qnuc_source='heat', minzone=51,
                 zonermax=10, zonermin=-1, thickfac=0.001, substrate='fe54',
                 substrate_off=True, adapnet_filename=None, bdat_filename=None,
                 ibdatov=1, params_full=None):
    """Generates a grid of Kepler models, containing n models over the range x

    Parameters
    ---------
    batch : int
    params : {}
        specifiy model parameters. If variable: give range
    dv : {}
        stepsize in variables (if ==-1: keep param as is)
    exclude : {}
        specify any parameter values to exclude from grid
    params : {}
        mass of NS (in Msun). Only changes geemult (gravity multiplier)
    auto_t_end : bool
        auto-choose t_end based on predicted recurrence time
    kgrid : Kgrid
        pre-loaded Kgrid object, optional (avoids reloading)
    """
    # TODO: WRITE ALL PARAM DESCRIPTIONS
    # TODO: set default values for params
    # TODO: Overhaul/tidy up
    # TODO: use pd table instead of dicts of arrays
    source = grid_strings.source_shorthand(source=source)
    mass_ref = 1.4  # reference NS mass (Msun)
    radius_ref = 10  # default NS radius (km)

    print_batch(batch=batch)

    if params_full is None:
        params_expanded, var = expand_params(dv, params)

        # ===== Cut out any excluded values =====
        if exclude is None:
            exclude = {}
        cut_params(params=params_expanded, exclude=exclude)
        print_grid_params(params_expanded)

        params_full = grid_tools.enumerate_params(params_expanded)

    n_models = len(params_full['x'])

    if kgrid is None:
        print('No kgrid provided. Loading:')
        kgrid = grid_analyser.Kgrid(load_lc=False, source=source)

    params_full['y'] = 1 - params_full['x'] - params_full['z']  # helium-4 values
    params_full['geemult'] = params_full['mass'] / mass_ref  # Gravity multiplier

    gravities = gravity.get_acceleration_newtonian(r=radius_ref,
                                                   m=params_full['mass']).value
    params_full['radius'] = np.full(n_models, radius_ref)
    params_full['gravity'] = gravities

    # TODO: rewrite properly (use tables)
    if predict_qnuc:
        if len(params['qnuc']) > 1:
            raise ValueError('Cannot provide multiple "qnuc" in params if predict_qnuc=True')

        linr_qnuc = qnuc_tools.linregress_qnuc(qnuc_source, grid_version=grid_version)
        for i in range(n_models):
            params_qnuc = {}
            for param in param_list:
                params_qnuc[param] = params_full[param][i]
            params_full['qnuc'][i] = qnuc_tools.predict_qnuc(params=params_qnuc,
                                                             source=qnuc_source,
                                                             linr_table=linr_qnuc)

    # ===== Create top grid folder =====
    batch_model_path = grid_strings.get_batch_models_path(batch, source)
    grid_tools.try_mkdir(batch_model_path)

    # Directory to keep MonARCH logs and sbatch files
    logpath = grid_strings.get_source_subdir(batch_model_path, 'logs')
    grid_tools.try_mkdir(logpath)

    # ===== Write parameter table MODELS.txt and NOTES.txt=====
    write_model_table(n=n_models, params=params_full, lburn=lburn, path=batch_model_path)
    filepath = os.path.join(batch_model_path, 'NOTES.txt')
    with open(filepath, 'w') as f:
        f.write(notes)

    print_dashes()
    kepler_jobs.write_both_jobscripts(run0=1, run1=n_models, batch=batch,
                                      source=source, basename=basename,
                                      path=logpath, walltime=walltime,
                                      adapnet_filename=adapnet_filename,
                                      bdat_filename=bdat_filename)

    # ===== Directories and templates for each model =====
    for i in range(n_models):
        # ==== Create directory tree ====
        print_dashes()
        model = i + 1
        run_path = grid_strings.get_model_path(model, batch, source, basename=basename)

        # ==== Create task directory ====
        grid_tools.try_mkdir(run_path)

        # ==== Write burn file, set initial composition ====
        x0 = params_full['x'][i]
        z0 = params_full['z'][i]
        kepler_files.write_rpabg(x0, z0, run_path, substrate=substrate)

        # ==== Create model generator file ====
        accrate0 = params_full['accrate'][i]

        if auto_t_end:
            mdot = params_full['accrate'][i] * params_full['acc_mult'][i]
            rate_params = {}
            for param in ('x', 'z', 'qb', 'mass'):
                rate_params[param] = params_full[param][i]
            fudge = 0.5  # extra time to ensure complete final burst
            tdel = kgrid.predict_recurrence(accrate=mdot, params=rate_params)
            t_end = (nbursts + fudge) * tdel
            print(f'Using predicted dt={tdel/3600:.1f} hr')
            if t_end < 0:
                print('WARN! negative dt predicted. Defaulting n * 1.5hr')
                t_end = nbursts * 1.5 * 3600

        run = i + 1
        print(f'Writing genfile for xrb{run}')
        header = f'This generator belongs to model: {source}_{batch}/{basename}{run}'

        accdepth = params_full['accdepth'][i]
        if (params_full['x'][i] > 0.0) and (accdepth > 1e20):
            print(f"!!!WARNING!!!: accdepth of {accdepth:.0e} may be too deep for" +
                  " models accreting hydrogen")
        print(f'Using accdepth = {accdepth:.1e}')

        kepler_files.write_genfile(h1=params_full['x'][i], he4=params_full['y'][i],
                                   n14=params_full['z'][i], qb=params_full['qb'][i],
                                   acc_mult=params_full['acc_mult'][i], qnuc=params_full['qnuc'][i],
                                   lburn=lburn, geemult=params_full['geemult'][i],
                                   path=run_path, t_end=t_end, header=header,
                                   accrate0=accrate0, accdepth=accdepth,
                                   accmass=params_full['accmass'][i],
                                   nsdump=nsdump, nstop=nstop,
                                   nuc_heat=nuc_heat, setup_test=setup_test, cnv=0,
                                   minzone=minzone, zonermax=zonermax, zonermin=zonermin,
                                   thickfac=thickfac, substrate_off=substrate_off,
                                   ibdatov=ibdatov)


def random_models(batch0, source, n_models, n_epochs, ref_source, kgrid, ref_mcmc_version,
                  constant=None, epoch_independent=('x', 'z', 'mass'),
                  epoch_dependent=('accrate', 'qb'), epoch_chosen=None):
    """Creates random sample of model parameters
    """
    aliases = {'mass': 'm_nw', 'accrate': 'mdot'}
    if constant is None:
        constant = {'tshift': 0.0, 'acc_mult': 1.0, 'qnuc': 5.0, 'qb_delay': 0.0,
                    'accmass': 1e16, 'accdepth': 1e19}
    if epoch_chosen is None:
        epoch_chosen = {}

    mv = mcmc_versions.McmcVersion(source=ref_source, version=ref_mcmc_version)
    params_full = {}

    # ===== fill model params =====
    for key in epoch_independent:
        mv_key = aliases.get(key, key)
        params_full[key] = mcmc_tools.get_random_params(mv_key, n_models=n_models, mv=mv)

    # ===== fill constant params =====
    for key, val in constant.items():
        params_full[key] = np.full(n_models, val)

    for i in range(n_epochs):
        for key in epoch_dependent:
            if key in epoch_chosen:
                val = epoch_chosen[key][i]
                params_full[key] = np.full(n_models, val)
            else:
                mv_key = aliases.get(key, key)
                mv_key = f'{mv_key}{i+1}'
                params_full[key] = mcmc_tools.get_random_params(mv_key,
                                                                n_models=n_models, mv=mv)

        create_batch(batch0+i, dv={}, params={}, source=source, nbursts=30, kgrid=kgrid,
                     walltime=96, setup_test=False, nsdump=500, nuc_heat=True,
                     predict_qnuc=False, grid_version=0, substrate_off=True, ibdatov=1,
                     params_full=params_full)


def setup_mcmc_sample(batch0, sample_source, chain, n_models_epoch, n_epochs, ref_source,
                      ref_mcmc_version, kgrid, constant=None,
                      epoch_independent=('x', 'z', 'mass'),
                      epoch_dependent=('accrate', 'qb'), discard=1000, cap=None):
    """Creates batches of models, with random sample of params drawn from MCMC chain
    """
    aliases = {'mass': 'm_nw', 'accrate': 'mdot'}
    if constant is None:
        constant = {'tshift': 0.0, 'acc_mult': 1.0, 'qnuc': 5.0, 'qb_delay': 0.0,
                    'accmass': 1e16, 'accdepth': 1e20}

    mv = mcmc_versions.McmcVersion(source=ref_source, version=ref_mcmc_version)
    params_full = {}
    param_sample, idxs = mcmc_tools.get_random_sample(chain, n=n_models_epoch,
                                                      discard=discard, cap=cap)
    batch1 = batch0 + n_epochs - 1
    save_sample_array(param_sample, source=sample_source, batch0=batch0, batch1=batch1)
    idx_string = get_index_str(idxs, discard=discard, cap=cap)

    # ===== fill model params =====
    for key in epoch_independent:
        mv_key = aliases.get(key, key)
        params_full[key] = get_mcmc_params(mv_key, param_sample=param_sample, mv=mv)

    # ===== fill constant params =====
    for key, val in constant.items():
        params_full[key] = np.full(n_models_epoch, val)

    for i in range(n_epochs):
        for key in epoch_dependent:
            mv_key = aliases.get(key, key)
            mv_key = f'{mv_key}{i+1}'
            params_full[key] = get_mcmc_params(mv_key, param_sample=param_sample, mv=mv)

        create_batch(batch0+i, dv={}, params={}, source=sample_source, nbursts=35,
                     kgrid=kgrid, walltime=96, setup_test=False, nsdump=500,
                     nuc_heat=True, predict_qnuc=False, substrate_off=True, ibdatov=1,
                     params_full=params_full, notes=idx_string)


def save_sample_array(param_sample, source, batch0, batch1):
    """Saves original parameter sample to file for safe keeping
    """
    filename = f'param_sample_{source}_{batch0}-{batch1}.txt'
    path = grid_strings.get_source_path(source)
    filepath = os.path.join(path, filename)
    print(f'Saving parameter sample: {filepath}')
    np.savetxt(filepath, param_sample)


def get_mcmc_params(key, param_sample, mv):
    """Returns model param from mcmc sample point params
    """
    idx = mv.param_keys.index(key)
    return param_sample[:, idx]


def get_index_str(idxs, discard, cap, header=None):
    """Returns str of indexes to save to file
    """
    if header is None:
        header = 'Indexes of samples from mcmc chain ' \
                 f'(after slicing: discard={discard}, cap={cap})'
    string = f'{header}\n'

    for i in idxs:
        string += f'{i}\n'
    return string


def print_grid_params(params):
    """Takes dict of unique params and prints them and total number of models
    """
    tot = 1
    for key, p in params.items():
        tot *= len(p)
        print(f'{key}: {p}')

    print(f'\nTotal models: {tot}\n')
    print('=' * 40)


def cut_params(params, exclude):
    """Removes specified value combinations from the given params
    """
    for ex_var, ex_list in exclude.items():
        for ex in ex_list:
            if ex in params[ex_var]:
                print(f'Excluding {ex_var}={ex:.3f} from grid')
                ex_idx = np.searchsorted(params[ex_var], ex)
                params[ex_var] = np.delete(params[ex_var], [ex_idx])


def expand_params(dv={'x': 0.05},
                  params={'x': [0.4, 0.5], 'z': [0.02],
                          'tshift': [20.0], 'accrate': [-1],
                          'qb': [0.3], 'acc_mult': [1.05],
                          'qb_delay': [0.0], 'mass': [1.4]}):
    """Expand variable parameters to fill their ranges, given specified stepsizes
    """
    params_full = dict(params)
    nv = len(dv.keys())  # number of variables
    var = find_varying(params, nv)

    # ===== Create full lists of model parameters =====
    for key in var:
        if key not in dv:
            raise ValueError(f'no stepsize (dv) given for: {key}')
        if dv[key] != -1:  # otherwise leave as is
            p0 = params[key][0]
            p1 = params[key][1]
            nstep = int(round((np.diff(params[key])[0] / dv[key])))  # number of steps
            params_full[key] = np.linspace(p0, p1, nstep + 1)

    return params_full, var


def find_varying(params, nvmax):
    """Returns list of keys with varying params (i.e. params with ranges).

    params = {}  : dictionary of params, each having array of length 1 or 2 (constant or varying)
    nvmax   = int : max number of varying params expected
    """
    print('Finding variable parameters')
    if nvmax < 0:
        raise ValueError(f'nvmax ({nvmax}) must be positive')

    var = []
    cnt = 0
    for p in params:
        if len(params[p]) == 2:
            if (params[p][1] - params[p][0]) < 0:
                raise ValueError(f'range is inverted for param: {p}')
            elif cnt >= nvmax:
                raise ValueError(f'too many param ranges were given. Expected {nvmax}')
            else:
                var.append(p)
                cnt += 1
    return var


def check_grid_params(params_full, source, precision=6, kgrid=None):
    """Check if any param combinations already exist in grid

    returns True if any model already exists

    params_full = dict  : dict of params for each model
    precision   = int   : number of decimal places to compare
    """
    source = grid_strings.source_shorthand(source=source)
    n_models = len(params_full['x'])
    any_matches = None

    if kgrid is None:
        print('No kgrid provided. Loading:')
        kgrid = grid_analyser.Kgrid(source=source, load_lc=False,
                                    powerfits=False, verbose=False)
    for i in range(n_models):
        model_param = {}

        for key, vals in params_full.items():
            val_rounded = float(f'{vals[i]:.{precision}f}')
            model_param[key] = val_rounded

        model = kgrid.get_params(params=model_param)

        if len(model) == 0:
            any_matches = False
        else:
            print('WARNING: a model with the following params already exists:')
            for var, v in model_param.items():
                print(f'{var} = {v:.3f}')
            any_matches = True
            break

    return any_matches


def write_model_table(n, params, lburn, path, filename='MODELS.txt'):
    """Writes table of model parameters to file

    Parameters
    ----------
    n : int
        number of models
    params : {}
        dictionary of parameters
    lburn : int
        lburn switch (0,1)
    path : str
    filename : str
    """
    print('Writing MODEL.txt table')
    runlist = np.arange(1, n + 1, dtype='int')
    lburn_list = np.full(n, lburn, dtype='int')

    p = dict(params)
    p['run'] = runlist
    p['lburn'] = lburn_list

    cols = ['run', 'z', 'y', 'x', 'accrate', 'qb', 'qnuc',
            'tshift', 'acc_mult', 'qb_delay', 'mass', 'radius', 'gravity',
            'lburn', 'accmass', 'accdepth']
    ptable = pd.DataFrame(p)
    ptable = ptable[cols]  # Fix column order

    table_str = ptable.to_string(index=False, formatters=FORMATTERS)

    filepath = os.path.join(path, filename)
    with open(filepath, 'w') as f:
        f.write(table_str)


def extend_runs(summ_table, source, nbursts=None, t_end=None,
                basename='xrb', nstop=9999999, nsdump=500, walltime=96,
                do_cmd_files=True, do_jobscripts=True, adapnet_filename=None):
    """Modifies existing models (in summ_table) for resuming, to simulate more bursts
    """
    source = grid_strings.source_shorthand(source)

    if t_end is None:
        if nbursts is None:
            raise ValueError('Must supply one of nbursts, t_end')
        mask = summ_table['num'] < nbursts
        short_table = summ_table[mask]
    else:
        short_table = summ_table

    if do_cmd_files:
        for model in short_table.itertuples():
            if nbursts is not None:
                t_end = (nbursts + 0.75) * model.dt
            lines = [f'p nstop {nstop}', f'p nsdump {nsdump}',
                     f'@time>{t_end:.3e}', 'end']
            overwrite_cmd(model.run, model.batch, source=source, lines=lines, basename=basename)

    if do_jobscripts:
        batches = np.unique(short_table['batch'])
        for batch in batches:
            batch_table = grid_tools.reduce_table(short_table, params={'batch': batch})
            runs = np.array(batch_table['run'])
            kepler_jobs.write_jobscripts(batch, run0=runs[0], run1=runs[-1],
                                         runs=runs, source=source,
                                         walltime=walltime, restart=True,
                                         adapnet_filename=adapnet_filename)

    return short_table


def add_to_cmd(run, batch, source, add_line, basename='xrb'):
    """Prepends command to model .cmd file, for changing params on restart
    """
    filepath = grid_strings.cmd_filepath(run, batch, source=source, basename=basename)
    print(f'Writing: {filepath}')
    with open(filepath) as f:
        lines = f.readlines()

    lines = [f'{add_line}\n'] + lines
    with open(filepath, 'w') as f:
        f.writelines(lines)


def overwrite_cmd(run, batch, source, lines, basename='xrb'):
    """Overwrites model .cmd file with given lines

    lines : [str]
        list of lines to write. Will automatically include endline characters
    """
    for i, line in enumerate(lines):
        if '\n' not in line:
            lines[i] += '\n'

    filepath = grid_strings.cmd_filepath(run, batch, source=source, basename=basename)
    print(f'Writing: {filepath}')
    with open(filepath, 'w') as f:
        f.writelines(lines)


def get_short_models(summ_table, n_bursts):
    """Returns table of models with less than n_bursts
    """
    idxs = np.where(summ_table['num'] < n_bursts)[0]
    short_table = summ_table.iloc[idxs]
    return short_table


def get_table_subset(table, batches):
    """returns subset of table with given batches
    """
    idxs = np.array([])
    for batch in batches:
        idxs = np.append(idxs, np.where(table['batch'] == batch)[0])

    idxs = idxs.astype(int)
    return table.iloc[idxs]


def sync_model_restarts(source, target, basename='xrb', verbose=True,
                        batches=None, runs=None, short_model_table=None,
                        sync_model_files=True, sync_jobscripts=True, sync_model_tables=True,
                        dry_run=False, modelfiles=('.cmd', '.lc', 'z1')):
    """Sync kepler models to cluster for resuming extended runs

    Parameters
    ----------
    source : str
    target : str
    basename : str (optional)
    verbose : bool (optional)
    batches : arraylike (optional)
    runs : arraylike (optional)
    short_model_table : pd.DataFrame (optional)
        table containing all batches/runs of models with too few n_bursts
    sync_model_files : bool (optional)
        sync model output files (.lc, .cmd, z1, rpabg)
    sync_jobscripts : bool (optional)
        sync jobscript submission files (.qsub)
    sync_model_tables : bool (optional)
        sync MODELS.txt files
    dry_run : bool (optional)
        do everything except actually send the files (for sanity checking)
    modelfiles : list (optional)
        the model files (by extension) which will be synced
    """
    if short_model_table is None:
        if batches is None or runs is None:
            raise ValueError('Must provide either short_model_table or both batches and runs')
    else:
        batches = np.unique(short_model_table['batch'])

    targets = {
        'icer': f'isync:~/kepler/runs/',
        'oz': 'oz:/fred/oz011/zac/kepler/runs/',
        'monarch': 'm2:/home/zacpetej/id43/kepler/runs/',
        'carbon': f'zac@carbon.sci.monash.edu:/home/zac/{source}'}
    target_path = targets[target]
    sync_paths = []

    for batch in batches:
        batch_str = grid_strings.get_batch_string(batch, source)
        batch_path = os.path.join(MODELS_PATH, '.', source, batch_str)

        if short_model_table is not None:
            batch_table = grid_tools.reduce_table(short_model_table, params={'batch': batch})
            runs = np.array(batch_table['run'])

        if sync_jobscripts:
            # TODO: make universal jobfile string function (for here and kepler_jobs.py)
            span_str = kepler_jobs.get_span_string(runs[0], runs[-1])
            jobscript = f'icer_restart_{source}_{batch}_{span_str}.sh'
            jobscript_path = os.path.join(batch_path, 'logs', jobscript)
            sync_paths += [jobscript_path]

        if sync_model_tables:
            model_table = os.path.join(batch_path, 'MODELS.txt')
            sync_paths += [model_table]

        if sync_model_files:
            for run in runs:
                run_str = grid_strings.get_run_string(run, basename)
                run_path = os.path.join(batch_path, run_str)

                for filetype in modelfiles:
                    if filetype == 'rpabg':
                        filename = 'rpabg'
                    else:
                        filename = f'{run_str}{filetype}'

                    filepath = os.path.join(run_path, filename)
                    sync_paths += [filepath]

    command = ['rsync', '-avR'] + sync_paths + [target_path]
    if verbose:
        for l in command:
            print(l)

    if not dry_run:
        subprocess.run(command)


def print_batch(batch):
    print_title()
    print_title()
    print(f'Batch: {batch}')
    print_title()
    print_title()
