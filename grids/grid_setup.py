import numpy as np
import pandas as pd
import subprocess, sys, os

# kepler_grids
from . import grid_analyser
from . import grid_tools
from . import grid_strings
from pygrids.kepler import kepler_jobscripts, kepler_files
from pygrids.misc.pyprint import print_title, print_dashes
from pygrids.physics import burning
from pygrids.qnuc import qnuc_tools

# Concord
import define_sources

# ====================================
# Kepler batch Generator
# --------------------
# Generates kepler model generator files, in addition to setting up model grids,
# in particular for running models on the Monarch cluster.
# --------------------
# Author: Zac Johnston (2016)
# Email: zac.johnston@monash.edu
# ====================================

flt2 = '{:.2f}'.format
flt4 = '{:.4f}'.format
FORMATTERS = {'z': flt4, 'y': flt4, 'x': flt4, 'accrate': flt4,
              'tshift': flt2, 'qb': flt4, 'xi': flt2, 'qb_delay': flt2,
              'mass': flt2}

GRIDS_PATH = os.environ['KEPLER_GRIDS']
MODELS_PATH = os.environ['KEPLER_MODELS']

# TODO: Rewrite docstrings
# TODO: Allow enumerating over multiple parameters, create_batches()

param_list = ['x', 'z', 'qb', 'accrate', 'accdepth', 'accmass', 'mass']

def print_batch(batch, source):
    print_title()
    print_title()
    print(f'Batch: {batch}')
    print_title()
    print_title()


def create_epoch_grid(batch0, dv, params, source, kgrid=None,
                      split_qos=True, qos='medium', **kwargs):
    """Generate a set of batches differing only by accrate,
    corresponding to different source epochs
    """
    batches = grid_tools.expand_batches(batches=batch0, source=source)
    source_object = define_sources.Source(source=source)
    mdots = source_object.mdots
    qos_list = []
    n_epochs = source_object.n_epochs

    if kgrid is None:
        print('No kgrid provided. Loading:')
        kgrid = grid_analyser.Kgrid(load_lc=False, source=source)

    # ===== split qos approximately in half =====
    if split_qos:
        split = int(n_epochs / 2)
        for i in range(split):
            qos_list += ['general']
        for i in range(n_epochs - split):
            qos_list += ['medium']
    else:
        for i in range(n_epochs):
            qos_list += [qos]

    for i, batch in enumerate(batches):
        params['accrate'] = [mdots[i]]
        create_batch(batch0=batch, dv=dv, params=params, source=source,
                     qos=qos_list[i], kgrid=kgrid, **kwargs)


def create_batch(batch, dv, source,
                 params={'x': [0.6, 0.8], 'z': [0.01, 0.02],
                         'tshift': [0.0], 'accrate': [0.05],
                         'qb': [0.125], 'xi': [1.0], 'qnuc': 5.0,
                         'qb_delay': [0.0], 'mass': [1.4],
                         'accmass': [1e16], 'accdepth': [1e20]},
                 lburn=1, t_end=1.3e5, exclude={}, basename='xrb',
                 walltime=96, qos='general', nstop=10000000,
                 check_params=False, nsdump=500,
                 auto_t_end=True, notes='No notes given', debug=False,
                 nbursts=20, parallel=False, ntasks=8, kgrid=None,
                 nuc_heat=False, setup_test=False, predict_qnuc=True,
                 grid_version=None, qnuc_source='heat', **kwargs):
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
    qos : str
        quality of service (slurm), one of ['general', 'medium', 'short']
    auto_t_end : bool
        auto-choose t_end based on predicted recurrence time
    parallel : bool
        utilise parallel independent kepler tasks
    ntasks : int
        no. of tasks in each parallel job (split up by this)
    kgrid : Kgrid
        pre-loaded Kgrid object, optional (avoids reloading)
    """
    # TODO: WRITE ALL PARAM DESCRIPTIONS
    # TODO: set default values for params
    # TODO: Overhaul/tidy up
    # TODO: use pd table instead of dicts of arrays
    source = grid_strings.source_shorthand(source=source)
    mass_ref = 1.4  # reference NS mass (in Msun)
    print_batch(batch=batch, source=source)

    params = dict(params)
    params_expanded, var = expand_params(dv, params)

    # ===== Cut out any excluded values =====
    cut_params(params=params_expanded, exclude=exclude)
    print_grid_params(params_expanded)

    params_full = grid_tools.enumerate_params(params_expanded)
    n = len(params_full['x'])

    if parallel and (n % ntasks != 0):
        raise ValueError(f'n_models ({n}) not divisible by ntasks ({ntasks})')

    if kgrid is None:
        print('No kgrid provided. Loading:')
        kgrid = grid_analyser.Kgrid(load_lc=False, source=source)

    if check_params:
        print('Checking existing grid models for params')
        temp_params_full = dict(params_full)
        del (temp_params_full['accmass'])
        params_exist = check_grid_params(params_full=temp_params_full, source=source,
                                         kgrid=kgrid)
        if params_exist:
            cont = input('XXX Continue anyway? XXX [y/n]: ')
            if cont == 'n' or cont == 'N':
                sys.exit()

    params_full['y'] = 1 - params_full['x'] - params_full['z']  # helium-4 values
    params_full['geemult'] = params_full['mass'] / mass_ref  # Gravity multiplier

    # TODO: rewrite properly (use tables)
    if predict_qnuc:
        if len(params['qnuc']) > 1:
            raise ValueError('Cannot provide multiple "qnuc" in params if predict_qnuc=True')

        linr_qnuc = qnuc_tools.linregress_qnuc(qnuc_source, grid_version=grid_version)
        for i in range(n):
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
    write_model_table(n=n, params=params_full, lburn=lburn, path=batch_model_path)
    filepath = os.path.join(batch_model_path, 'NOTES.txt')
    with open(filepath, 'w') as f:
        f.write(notes)

    job_runs = []
    if parallel:
        n_jobs = int(n / ntasks)
        for i in range(n_jobs):
            start = i * ntasks
            job_runs += [[start + 1, start + ntasks]]
    else:
        job_runs += [[1, n]]

    print_dashes()
    for runs in job_runs:
        for restart in [True, False]:
            kepler_jobscripts.write_submission_script(run0=runs[0], run1=runs[1],
                                                      restart=restart, batch=batch,
                                                      source=source, basename=basename,
                                                      path=logpath, qos=qos,
                                                      walltime=walltime,
                                                      parallel=parallel, debug=debug)

    # ===== Directories and templates for each model =====
    for i in range(n):
        # ==== Create directory tree ====
        print_dashes()
        model = i + 1
        run_path = grid_strings.get_model_path(model, batch, source, basename=basename)

        # ==== Create task directory ====
        grid_tools.try_mkdir(run_path)

        # ==== Write burn file, set initial composition ====
        x0 = params_full['x'][i]
        z0 = params_full['z'][i]
        kepler_files.write_rpabg(x0, z0, run_path)

        # ==== Create model generator file ====
        accrate0 = params_full['accrate'][i]

        if auto_t_end:
            mdot = params_full['accrate'][i] * params_full['xi'][i]
            rate_params = {}
            for param in ('x', 'z', 'qb', 'mass'):
                rate_params[param] = params_full[param][i]
            fudge = 0.5  # extra time to ensure complete final burst
            tdel = kgrid.predict_recurrence(accrate=mdot, params=rate_params)
            t_end = (nbursts + fudge) * tdel
            print(f'Using predicted dt={tdel/3600:.1f} hr')
            if t_end < 0:
                print('WARN! negative dt predicted. Defaulting n * 1.5hr')
                t_end =nbursts * 1.5 * 3600

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
                                   xi=params_full['xi'][i], qnuc=params_full['qnuc'][i],
                                   lburn=lburn, geemult=params_full['geemult'][i],
                                   path=run_path, t_end=t_end, header=header,
                                   accrate0=accrate0, accdepth=accdepth,
                                   accmass=params_full['accmass'][i],
                                   nsdump=nsdump, nstop=nstop,
                                   nuc_heat=nuc_heat, setup_test=setup_test, cnv=0)


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
                          'qb': [0.3], 'xi': [1.05],
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
            'tshift', 'xi', 'qb_delay', 'mass', 'lburn',
            'accmass', 'accdepth']
    ptable = pd.DataFrame(p)
    ptable = ptable[cols]  # Fix column order

    table_str = ptable.to_string(index=False, formatters=FORMATTERS)

    filepath = os.path.join(path, filename)
    with open(filepath, 'w') as f:
        f.write(table_str)


def extend_runs(summ_table, source, nbursts=40, basename='xrb',
                nsdump=1000, walltime=96, do_cmd_files=True, do_jobscripts=True):
    """Modifies existing models (in summ_table) for resuming, to simulate more bursts
    """
    source = grid_strings.source_shorthand(source)
    mask = summ_table['num'] < nbursts
    short_table = summ_table[mask]

    if do_cmd_files:
        for model in short_table.itertuples():
            t_end = (nbursts + 0.75) * model.dt
            lines = [f'p nsdump {nsdump}', f'@time>{t_end:.3e}', 'end']
            overwrite_cmd(model.run, model.batch, source=source, lines=lines, basename=basename)

    if do_jobscripts:
        batches = np.unique(short_table['batch'])
        for batch in batches:
            batch_table = grid_tools.reduce_table(short_table, params={'batch': batch})
            runs = np.array(batch_table['run'])
            kepler_jobscripts.write_submission_script(batch, run0=runs[0], run1=runs[-1],
                                                      runs=runs, source=source,
                                                      walltime=walltime, restart=True)


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


def sync_model_restarts(short_model_table, source, basename='xrb', verbose=False,
                        sync_model_files=True, sync_jobscripts=True, sync_model_tables=True,
                        dry_run=False, modelfiles=('.cmd', '.lc', 'z1')):
    """Sync kepler models to cluster for resuming extended runs

    Parameters
    ----------
    short_model_table : pd.DataFrame
        table containing all batches/runs of models with too few n_bursts
    source : str
    basename : str
    verbose : bool
    sync_model_files : bool
        sync model output files (.lc, .cmd, z1, rpabg)
    sync_jobscripts : bool
        sync jobscript submission files (.qsub)
    sync_model_tables : bool
        sync MODELS.txt files
    dry_run : bool
        do everything but actually send the files (for sanity checking)
    modelfiles : list
        the model files (by extension) which will be synced
    """
    batches = np.unique(short_model_table['batch'])
    target_path = 'isync:~/kepler/runs/'
    sync_paths = []

    for batch in batches:
        batch_str = grid_strings.get_batch_string(batch, source)
        batch_path = os.path.join(MODELS_PATH, '.', batch_str)

        batch_table = grid_tools.reduce_table(short_model_table, params={'batch': batch})
        runs = np.array(batch_table['run'])

        if sync_jobscripts:
            span_str = kepler_jobscripts.get_span_string(runs[0], runs[-1])
            jobscript = f'icer_restart_{source}_{batch}_{span_str}.qsub'
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
