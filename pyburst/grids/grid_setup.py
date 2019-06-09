import os
import numpy as np
import pandas as pd
import subprocess
import configparser
import ast

# pyburst
from . import grid_analyser
from . import grid_tools
from . import grid_strings
from pyburst.mcmc import mcmc_versions, mcmc_tools
from pyburst.kepler import kepler_jobs, kepler_files
from pyburst.misc.pyprint import print_title, print_dashes
from pyburst.qnuc import qnuc_tools
from pyburst.physics import gravity

"""
======================================
Grid Setup
--------------------
Tools for creating kepler model grids
  including: jobscripts, model files, parameter tables, etc.
--------------------
NOTE: This module is pretty spaghettified.
      Things might not work elegently or as expected.
      USE WITH CAUTION!
--------------------
Author: Zac Johnston
Email: zac.johnston@monash.edu
======================================
"""

flt2 = '{:.2f}'.format
flt4 = '{:.4f}'.format
flt8 = '{:.8f}'.format

FORMATTERS = {'z': flt8, 'y': flt8, 'x': flt8, 'accrate': flt8,
              'tshift': flt2, 'qb': flt8, 'acc_mult': flt2, 'qb_delay': flt2,
              'mass': flt8}

GRIDS_PATH = os.environ['KEPLER_GRIDS']
MODELS_PATH = os.environ['KEPLER_MODELS']

# TODO: This whole module is a mess. Needs lots of tidying up
# TODO: Allow enumerating over multiple parameters, create_batches()

param_list = ['x', 'z', 'qb', 'accrate', 'accdepth', 'accmass', 'mass']


def create_batch(batch, source, params, dv,
                 t_end=1.3e5, exclude=None, basename='xrb', walltime=96,
                 auto_t_end=True, notes='No notes given', nbursts=20, kgrid=None,
                 nuc_heat=True, setup_test=False,
                 auto_qnuc=False, grid_version=None, qnuc_source='heat',
                 substrate='fe54', substrate_off=True, adapnet_filename=None,
                 bdat_filename=None, params_full=None,
                 numerical_params=None, scratch_file_sys=False):
    """Generates a grid of Kepler models, containing n models over the range x

    Parameters
    ---------
    batch : int
    source : str
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
    t_end : float (optional)
    basename : str (optional)
    walltime : int (optional)
    notes : str (optional)
    nbursts : int (optional)
    auto_qnuc : bool (optional)
    nuc_heat : bool (optional)
    setup_test : bool (optional)
    grid_version : int (optional)
    qnuc_source : str (optional)
    substrate : str (optional)
    substrate_off : bool (optional)
    adapnet_filename : str (optional)
    bdat_filename : str (optional)
    params_full : {} (optional)
    numerical_params : {} (optional)
        Overwrite default numerical kepler parameters (e.g. nsdump, zonermax, lburn,
        For all parameters: see 'numerical_params' in config/default.ini
    scratch_file_sys : bool (optional)
        whether to use the scratch file system on ICER cluster
    """
    # TODO:
    #   - WRITE ALL PARAM DESCRIPTIONS
    #   - Overhaul/tidy up
    #   - use pd table instead of dicts of arrays

    print_batch(batch=batch)
    source = grid_strings.source_shorthand(source=source)
    mass_ref = 1.4  # reference NS mass (Msun)
    radius_ref = 10  # default NS radius (km)

    supplied_config = {'params': params,
                       'dv': dv,
                       'numerical_params': numerical_params}

    config = setup_config(supplied_config=supplied_config, source=source)
    # TODO: print numerical_params being used

    if params_full is None:
        params_expanded, var = expand_params(params=config['params'], dv=config['dv'])
        params_full = exclude_params(params_expanded=params_expanded, exclude=exclude)

    n_models = len(params_full['x'])

    if kgrid is None:
        print('No kgrid provided. Loading default:')
        kgrid = grid_analyser.Kgrid(load_lc=False, linregress_burst_rate=True,
                                    source=source)

    params_full['y'] = 1 - params_full['x'] - params_full['z']  # helium-4 values
    params_full['geemult'] = params_full['mass'] / mass_ref  # Gravity multiplier

    gravities = gravity.get_acceleration_newtonian(r=radius_ref,
                                                   m=params_full['mass']).value
    params_full['radius'] = np.full(n_models, radius_ref)
    params_full['gravity'] = gravities

    if auto_qnuc:
        predict_qnuc(params_full=params_full, qnuc_source=qnuc_source,
                     grid_version=grid_version)

    # ===== Create top grid folder =====
    batch_model_path = grid_strings.get_batch_models_path(batch, source)
    grid_tools.try_mkdir(batch_model_path)

    # Directory to keep MonARCH logs and sbatch files
    logpath = grid_strings.get_source_subdir(batch_model_path, 'logs')
    grid_tools.try_mkdir(logpath)

    # ===== Write parameter table MODELS.txt and NOTES.txt=====
    write_model_table(n=n_models, params=params_full, path=batch_model_path)
    filepath = os.path.join(batch_model_path, 'NOTES.txt')
    with open(filepath, 'w') as f:
        f.write(notes)

    # ===== Write jobscripts for submission on clusters =====
    print_dashes()
    kepler_jobs.write_both_jobscripts(run0=1, run1=n_models, batch=batch,
                                      source=source, basename=basename,
                                      path=logpath, walltime=walltime,
                                      adapnet_filename=adapnet_filename,
                                      bdat_filename=bdat_filename,
                                      scratch_file_sys=scratch_file_sys)

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
                                   acc_mult=params_full['acc_mult'][i],
                                   qnuc=params_full['qnuc'][i],
                                   geemult=params_full['geemult'][i],
                                   accrate0=params_full['accrate'][i],
                                   accmass=params_full['accmass'][i],
                                   accdepth=params_full['accdepth'][i],
                                   path=run_path, t_end=t_end, header=header,
                                   nuc_heat=nuc_heat, setup_test=setup_test,
                                   substrate_off=substrate_off,
                                   numerical_params=config['numerical_params'])


def setup_config(supplied_config, source):
    """Returns combined dict of params from default, source, and supplied
    """
    def overwrite(old_dict, new_dict):
        for key, val in new_dict.items():
            old_dict[key] = val

    if supplied_config['numerical_params'] is None:
        supplied_config['numerical_params'] = {}

    default_config = load_config(config_source='default')
    source_config = load_config(config_source=source)
    combined_config = dict(default_config)

    for category, contents in combined_config.items():
        print(f'Overwriting default {category} with source-specific and '
              f'user-supplied {category}')
        overwrite(old_dict=contents, new_dict=source_config[category])
        overwrite(old_dict=contents, new_dict=supplied_config[category])

    return combined_config


def load_config(config_source):
    """Loads config parameters from file
    """
    config_filepath = grid_strings.config_filepath(source=config_source,
                                                   module_dir='grids')
    print(f'Loading config: {config_filepath}')

    if not os.path.exists(config_filepath):
        raise FileNotFoundError(f'Config file not found: {config_filepath}.'
                                "\nTry making one from the template 'default.ini'")

    ini = configparser.ConfigParser()
    ini.read(config_filepath)

    config = {}
    for section in ini.sections():
        config[section] = {}
        for option in ini.options(section):
            config[section][option] = ast.literal_eval(ini.get(section, option))

    return config


def exclude_params(params_expanded, exclude):
    """Cut out specified params from expanded_params
    """
    if exclude is None:
        exclude = {}
    cut_params(params=params_expanded, exclude=exclude)
    print_grid_params(params_expanded)

    params_full = grid_tools.enumerate_params(params_expanded)
    return params_full


def predict_qnuc(params_full, qnuc_source, grid_version):
    linr_qnuc = qnuc_tools.linregress_qnuc(qnuc_source, grid_version=grid_version)
    n_models = len(params_full['x'])

    for i in range(n_models):
        params_qnuc = {}
        for param in param_list:
            params_qnuc[param] = params_full[param][i]
        params_full['qnuc'][i] = qnuc_tools.predict_qnuc(params=params_qnuc,
                                                         source=qnuc_source,
                                                         linr_table=linr_qnuc)


def random_models(batch0, source, n_models, n_epochs, ref_source, kgrid, ref_mcmc_version,
                  constant=None, epoch_independent=('x', 'z', 'mass'),
                  epoch_dependent=('accrate', 'qb'), epoch_chosen=None,
                  scratch_file_sys=False):
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
                     walltime=96, setup_test=False, nuc_heat=True,
                     auto_qnuc=False, grid_version=0, substrate_off=True,
                     params_full=params_full, scratch_file_sys=scratch_file_sys)


def setup_mcmc_sample(batch0, sample_source, chain, discard, n_models_epoch, n_epochs,
                      ref_source, ref_mcmc_version, kgrid, nbursts, constant=None,
                      epoch_independent=('x', 'z', 'mass'), walltime=96,
                      epoch_dependent=('accrate', 'qb'), cap=None,
                      scratch_file_sys=False):
    """Creates batches of models, with random sample of params drawn from MCMC chain
    """
    aliases = {'mass': 'm_nw', 'accrate': 'mdot'}
    # TODO: use config defaults for constants
    if constant is None:
        constant = {'tshift': 0.0, 'acc_mult': 1.0, 'qnuc': 5.0, 'qb_delay': 0.0,
                    'accmass': 1e16, 'accdepth': 1e20}#, 'x':0.0, 'z':0.015}

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

        create_batch(batch0+i, dv={}, params={}, source=sample_source, nbursts=nbursts,
                     kgrid=kgrid, walltime=walltime, setup_test=False,
                     nuc_heat=True, auto_qnuc=False, substrate_off=True,
                     params_full=params_full, notes=idx_string,
                     scratch_file_sys=scratch_file_sys)


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


def expand_params(params, dv):
    """Expand variable parameters to fill their ranges, given specified stepsizes
    """
    params_full = dict(params)
    n_var = len(dv.keys())  # number of variables
    var = find_varying(params, n_var)

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


def write_model_table(n, params, path, filename='MODELS.txt'):
    """Writes table of model parameters to file

    Parameters
    ----------
    n : int
        number of models
    params : {}
        dictionary of parameters
    path : str
    filename : str
    """
    print('Writing MODEL.txt table')
    runlist = np.arange(1, n + 1, dtype='int')

    p = dict(params)
    p['run'] = runlist

    cols = ['run', 'z', 'y', 'x', 'accrate', 'qb', 'qnuc',
            'tshift', 'acc_mult', 'qb_delay', 'mass', 'radius', 'gravity',
            'accmass', 'accdepth']
    ptable = pd.DataFrame(p)
    ptable = ptable[cols]  # Fix column order

    table_str = ptable.to_string(index=False, formatters=FORMATTERS)

    filepath = os.path.join(path, filename)
    with open(filepath, 'w') as f:
        f.write(table_str)


def extend_runs(summ_table, source, nbursts=None, t_end=None,
                basename='xrb', nstop=9999999, nsdump=500, walltime=96,
                do_cmd_files=True, do_jobscripts=True, adapnet_filename=None,
                scratch_file_sys=False):
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
                                         adapnet_filename=adapnet_filename,
                                         scratch_file_sys=scratch_file_sys)

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

    jobscript_aliases = {'oz': 'monarch'}
    targets = {
        'icer': f'isync:~/kepler/runs/',
        'oz': 'oz:/fred/oz011/zac/kepler/runs/',
        'monarch': 'm2:/home/zacpetej/id43/kepler/runs/',
        'carbon': f'zac@carbon.sci.monash.edu:/home/zac/{source}'}

    target_path = targets[target]
    jobscript_prefix = jobscript_aliases.get(target, target)
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
            jobscript = f'{jobscript_prefix}_restart_{source}_{batch}_{span_str}.sh'
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
