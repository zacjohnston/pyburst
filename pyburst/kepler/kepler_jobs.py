# ========================================================
# Functions for writing job submission scripts on cluster (e.g. monarch, ICER)
# ========================================================
import os

# pyburst
from pyburst.grids import grid_strings


def write_both_jobscripts(batch, source, walltime, path=None, run0=None,
                          run1=None, runs=None, basename='xrb',
                          adapnet_filename=None, bdat_filename=None,
                          scratch_file_sys=False):
    """Iterates write_jobscripts() for both restart=True/False
    """
    for restart in [True, False]:
        write_jobscripts(run0=run0, run1=run1, runs=runs, restart=restart,
                         batch=batch, source=source, basename=basename, path=path,
                         walltime=walltime, adapnet_filename=adapnet_filename,
                         bdat_filename=bdat_filename, scratch_file_sys=scratch_file_sys)


def write_jobscripts(batch, source, walltime, path=None, run0=None, run1=None,
                     runs=None, basename='xrb', restart=False,
                     adapnet_filename=None, bdat_filename=None,
                     scratch_file_sys=False):
    """Writes jobscripts to execute on cluster (eg, Monarch, ICER...)

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

    job_str = get_jobstring(batch=batch, run0=run0, run1=run1, source=source)
    time_str = f'{walltime:02}:00:00'

    # TODO: combine clusters into single 'slurm' script (how to handle constraints?)
    for cluster in ['monarch', 'icer']:
        span = get_span_string(run0, run1)
        prepend_str = {True: 'restart_'}.get(restart, '')

        filename = f'{cluster}_{prepend_str}{source}_{batch}_{span}.sh'
        filepath = os.path.join(path, filename)
        print(f'Writing: {filepath}')

        script_str = get_submission_str(run0=run0, run1=run1, runs=runs, source=source,
                                        batch=batch, basename=basename, time_str=time_str,
                                        job_str=job_str, cluster=cluster, restart=restart,
                                        adapnet_filename=adapnet_filename,
                                        bdat_filename=bdat_filename,
                                        scratch_file_sys=scratch_file_sys)
        with open(filepath, 'w') as f:
            f.write(script_str)


def get_submission_str(run0, run1, source, runs, batch, basename, cluster, time_str,
                       job_str, restart, adapnet_filename=None, bdat_filename=None,
                       scratch_file_sys=False):
    """Returns string of submission script contents
    """
    source = grid_strings.source_shorthand(source=source)
    span_str = get_span_string(run0, run1, runs=runs)

    if adapnet_filename is None:
        adapnet_filename = 'adapnet_alex_email_dec.5.2016.cfg'
    if bdat_filename is None:
        bdat_filename = '20161114Reaclib.bdat5.fixed'

    if scratch_file_sys:
        model_dir = 'SCRATCH'
    else:
        model_dir = 'KEPLER_MODELS'

    cmd_str = {True: 'z1', False: 'xrb_g'}[restart]
    constraint_str = {'icer': 'SBATCH --constraint=intel18'}.get(cluster, '')

    return f"""#!/bin/bash --login
###################################
#SBATCH --job-name {job_str}
#SBATCH --array={span_str}
#SBATCH --time={time_str}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#{constraint_str}
#SBATCH --mem-per-cpu=1024
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zac.johnston@monash.edu
###################################
N=$SLURM_ARRAY_TASK_ID
EXE_PATH=$KEPLER_PATH/gfortran/keplery
ADAPNET_PATH=$PYBURST/files/{adapnet_filename}
BDAT_PATH=$PYBURST/files/{bdat_filename}

for FILE in ${{ADAPNET_PATH}} ${{BDAT_PATH}}; do
    if [ ! -f ${{FILE}} ]; then
        echo "File not found: ${{FILE}}"
        exit 1
    fi
done

cd ${model_dir}/{source}/{source}_{batch}/{basename}$N/
ln -sf $ADAPNET_PATH ./adapnet.cfg
ln -sf $BDAT_PATH ./bdat
$EXE_PATH {basename}$N {cmd_str}
"""


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
