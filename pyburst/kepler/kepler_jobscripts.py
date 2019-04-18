# ========================================================
# Functions for writing job submission scripts on cluster (e.g. monarch, ICER)
# ========================================================
import os

# pyburst
from pyburst.grids import grid_strings


def write_both_submission_scripts(batch, source, walltime, path=None, run0=None,
                                  run1=None, runs=None, qos='normal', basename='xrb',
                                  debug=False, adapnet_filename=None, bdat_filename=None):
    """Iterates write_submission_script() for both restart=True/False
    """
    for restart in [True, False]:
        write_submission_script(run0=run0, run1=run1, runs=runs, restart=restart,
                                batch=batch, source=source, basename=basename,
                                path=path, qos=qos, walltime=walltime, debug=debug,
                                adapnet_filename=adapnet_filename,
                                bdat_filename=bdat_filename)


def write_submission_script(batch, source, walltime, path=None,
                            run0=None, run1=None, runs=None,
                            qos='normal', basename='xrb',
                            restart=False, debug=False,
                            adapnet_filename=None, bdat_filename=None):
    """Writes jobscripts to execute on MONARCH/ICER cluster

    Parameter:
    ----------
    runs : list (optional)
        specify an arbitrary list of runs, instead of a span from run0-run1
    path : str
        target path for slurm script
    """
    source = grid_strings.source_shorthand(source=source)
    run0, run1, runs = check_runs(run0, run1, runs)

    if path is None:
        batch_path = grid_strings.get_batch_models_path(batch=batch, source=source)
        path = os.path.join(batch_path, 'logs')

    extensions = {'monarch': '.sh', 'icer': '.qsub'}

    job_str = get_jobstring(batch=batch, run0=run0, run1=run1, source=source)
    time_str = f'{walltime:02}:00:00'

    for cluster in ['monarch', 'icer']:
        print('Writing submission script for cluster:', cluster)
        ext = extensions[cluster]

        script_str = get_submission_str(run0=run0, run1=run1, runs=runs, source=source,
                                        batch=batch, basename=basename, qos=qos,
                                        time_str=time_str, job_str=job_str,
                                        cluster=cluster, debug=debug, restart=restart,
                                        adapnet_filename=adapnet_filename,
                                        bdat_filename=bdat_filename)

        span = get_span_string(run0, run1)
        prepend_str = {True: 'restart_'}.get(restart, '')

        filename = f'{cluster}_{prepend_str}{source}_{batch}_{span}{ext}'
        filepath = os.path.join(path, filename)

        with open(filepath, 'w') as f:
            f.write(script_str)


def get_submission_str(run0, run1, source, runs, batch, basename, cluster,
                       qos, time_str, job_str, debug, restart,
                       adapnet_filename=None, bdat_filename=None):
    source = grid_strings.source_shorthand(source=source)
    span_str = get_span_string(run0, run1, runs=runs)

    # TODO: check if adapnet/bdat exists
    if adapnet_filename is None:
        adapnet_filename = 'adapnet_alex_email_dec.5.2016.cfg'
    if bdat_filename is None:
        bdat_filename = '20161114Reaclib.bdat5.fixed'

    # ===== restart parameters =====
    cmd_str = {True: 'z1', False: 'xrb_g'}[restart]
    debug_str = {True: 'x', False: ''}[debug]

    if cluster == 'monarch':
        return f"""#!/bin/bash
###################################
#SBATCH --job-name={job_str}
#SBATCH --output=arrayJob_%A_%a.out
#SBATCH --error=arrayJob_%A_%a.err
#SBATCH --array={span_str}
#SBATCH --time={time_str}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --qos={qos}
#SBATCH --mem-per-cpu=1024
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zac.johnston@monash.edu
###################################

N=$SLURM_ARRAY_TASK_ID
EXE_PATH=$KEPLER_PATH/gfortran/keplery
ADAPNET_PATH=$PYBURST/files/{adapnet_filename}
BDAT_PATH=$PYBURST/files/{bdat_filename}

cd $KEPLER_MODELS/{source}/{source}_{batch}/{basename}$N/
ln -sf $ADAPNET_PATH ./adapnet.cfg
ln -sf $BDAT_PATH ./bdat
$EXE_PATH {basename}$N {cmd_str} {debug_str}"""

    elif cluster == 'icer':
        return f"""#!/bin/bash --login
###################################
#SBATCH --job-name {job_str}
#SBATCH --array={span_str}
#SBATCH --time={time_str}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=intel18
#SBATCH --mem-per-cpu=1024
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zac.johnston@monash.edu
###################################
N=$SLURM_ARRAY_TASK_ID
EXE_PATH=$KEPLER_PATH/gfortran/keplery
ADAPNET_PATH=$PYBURST/files/{adapnet_filename}
BDAT_PATH=$PYBURST/files/{bdat_filename}

cd $KEPLER_MODELS/{source}/{source}_{batch}/xrb$N/
ln -sf $ADAPNET_PATH ./adapnet.cfg
ln -sf $BDAT_PATH ./bdat
$EXE_PATH {basename}$N {cmd_str}
"""
    else:
        raise ValueError('invalid cluster. Must be one of [monarch, icer]')


def check_runs(run0, run1, runs):
    """Checks run parameters, and returns necessary values

    Behaviour:
        if runs is None: assume full span from run0-run1
        if runs is not None: use runs specified

    Returns:
        run0, run1, runs
    """
    if (run0 is None
            and run1 is None
            and runs is None):
        raise ValueError('Must provide both run0 and run1, or runs')

    if runs is None:
        return run0, run1, runs
    else:
        return runs[0], runs[-1], runs


def get_span_string(run0, run1, runs=None):
    """Returns string of run0-run1, (or run0 if run0 == run1)
    """
    if runs is not None:
        string = ''
        for run in runs:
            string += f'{run},'
        return string

    elif run0 == run1:
        return f'{run0}'
    else:
        return f'{run0}-{run1}'


def get_jobstring(batch, run0, run1, source, include_source=True):
    source = grid_strings.source_shorthand(source=source)
    span = get_span_string(run0, run1)
    source_str = ''

    if include_source:
        source_str = f'{source[:2]}_'

    return f'{source_str}{batch}_{span}'
