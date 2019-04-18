# ========================================================
# Functions for writing job submission scripts on cluster (e.g. monarch, ICER)
# ========================================================
import os
import subprocess

# kepler_grids
from pyburst.grids import grid_strings


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


def write_submission_script(batch, source, walltime, path=None,
                            run0=None, run1=None, runs=None,
                            parallel=False, qos='normal', basename='xrb',
                            restart=False, max_tasks=16, debug=False,
                            adapnet_filename=None, bdat_filename=None,
                            dependency=False):
    """Writes jobscripts to execute on MONARCH/ICER cluster

    Parameter:
    ----------
    runs : list (optional)
        specify an arbitrary list of runs, instead of a span from run0-run1
    parallel : bool
        launch parallel independent kepler tasks
    path : str
        target path for slurm script
    max_tasks : int
        max number of tasks allowed on one node
    """
    source = grid_strings.source_shorthand(source=source)
    run0, run1, runs, n_runs = check_runs(run0, run1, runs)

    if path is None:
        batch_path = grid_strings.get_batch_models_path(batch=batch, source=source)
        path = os.path.join(batch_path, 'logs')

    if parallel:
        if n_runs > max_tasks:
            raise ValueError(f'ntasks ({n_runs}) larger than max_tasks ({max_tasks})')

    extensions = {'monarch': '.sh', 'icer': '.qsub'}

    job_str = get_jobstring(batch=batch, run0=run0, run1=run1, source=source)
    time_str = f'{walltime:02}:00:00'

    for cluster in ['monarch', 'icer']:
        print('Writing submission script for cluster:', cluster)
        ext = extensions[cluster]

        script_str = get_submission_str(run0=run0, run1=run1, runs=runs, source=source,
                                        batch=batch, basename=basename, qos=qos,
                                        time_str=time_str, job_str=job_str,
                                        cluster=cluster, parallel=parallel,
                                        debug=debug, restart=restart,
                                        adapnet_filename=adapnet_filename,
                                        bdat_filename=bdat_filename,
                                        dependency=dependency)

        span = get_span_string(run0, run1)
        prepend_str = {True: 'restart_'}.get(restart, '')

        filename = f'{cluster}_{prepend_str}{source}_{batch}_{span}{ext}'
        filepath = os.path.join(path, filename)

        with open(filepath, 'w') as f:
            f.write(script_str)

    if parallel:
        write_parallel_script(run0=run0, run1=run1, batch=batch, path=path,
                              restart=restart, basename=basename, source=source,
                              debug=debug, adapnet_filename=adapnet_filename,
                              bdat_filename=bdat_filename)


def get_submission_str(run0, run1, source, runs, batch, basename, cluster,
                       qos, time_str, parallel, job_str, debug, restart,
                       adapnet_filename=None, bdat_filename=None, dependency=False):
    source = grid_strings.source_shorthand(source=source)
    span_str = get_span_string(run0, run1, runs=runs)
    batch_str = get_jobstring(batch=batch, run0=run0, run1=run1, source=source,
                              include_source=False)
    # TODO: check if adapnet/bdat exists
    if adapnet_filename is None:
        adapnet_filename = 'adapnet_alex_email_dec.5.2016.cfg'
    if bdat_filename is None:
        bdat_filename = '20161114Reaclib.bdat5.fixed'

    # ===== restart parameters =====
    cmd_str = {True: 'z1', False: 'xrb_g'}[restart]
    restart_str = {True: 'restart_', False: ''}[restart]
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


def write_parallel_script(run0, run1, batch, path, source, restart, debug=False,
                          basename='xrb', gen_file='xrb_g', adapnet_filename=None,
                          bdat_filename=None):
    """========================================================
    Writes a bash script to launch parallel kepler tasks
    ========================================================"""
    source = grid_strings.source_shorthand(source=source)
    print('Writing MPI script')
    if adapnet_filename is None:
        adapnet_filename = 'adapnet_alex_email_dec.5.2016.cfg'
    if bdat_filename is None:
        bdat_filename = '20161114Reaclib.bdat5.fixed'

    # ===== restart things =====
    debug_str = {True: 'x', False: ''}[debug]
    restart_str = {True: 'restart_', False: ''}[restart]
    start_str = {True: 'Restarting', False: 'Starting'}[restart]
    execute_str = {True: f'./k $run_str z1 {debug_str}',
                   False: f'./k $run_str {gen_file}'}[restart]

    filename = f'parallel_{restart_str}{source}_{batch}_{run0}-{run1}.sh'
    filepath = os.path.join(path, filename)

    with open(filepath, 'w') as f:
        f.write(f"""#!/bin/bash

exe_path=$KEPLER_PATH/gfortran/keplery
batch_dir=$KEPLER_MODELS/{source}_{batch}
ADAPNET_PATH=$PYBURST/files/{adapnet_filename}
BDAT_PATH=$PYBURST/files/{bdat_filename}

for run in $(seq {run0} {run1}); do
   run_str="{basename}${{run}}"
   echo "{start_str}"
   cd $batch_dir/$run_str
   ln -sf $exe_path ./k
   ln -sf $ADAPNET_PATH ./adapnet.cfg
   ln -sf $BDAT_PATH ./bdat
   {execute_str} > ${{run_str}}_std.out &
done

echo 'Waiting for jobs to finish'
wait
echo 'All jobs finished'""")

    # ===== make executable =====
    subprocess.run(['chmod', '+x', filepath])


def check_runs(run0, run1, runs):
    """Checks run parameters, and returns necessary values

    Behaviour:
        if runs is None: assume full span from run0-run1
        if runs is not None: use runs specified

    Returns:
        run0, run1, runs, n_runs
    """
    if (run0 is None
            and run1 is None
            and runs is None):
        raise ValueError('Must provide both run0 and run1, or runs')

    if runs is None:
        return run0, run1, runs, (run1 - run0 + 1)
    else:
        return runs[0], runs[-1], runs, len(runs)
