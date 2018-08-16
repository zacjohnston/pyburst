import os, sys
import subprocess

# kepler_grids
from . import grid_strings

# ========================================================
# Functions for writing job submission scripts on cluster (e.g. monarch, ICER)
# ========================================================
MODELS_PATH = os.environ['KEPLER_MODELS']


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


def write_individual_scripts(batches, runs, source, walltime, **kwargs):
    """Writes multiple jobscripts for individual models

    Created for the purpose of resubmitting particular jobs that
        failed to start.

    e.g. batches=[1,2,2,3], runs=[3,4,5,3] will write scripts for the models:
        batch_1_3, batch_2_4, batch_2_5, batch_3_3

    Parameters
    ----------
    batches : 1darray
        array of batches to write scripts for
    runs : 1darray
        array of runs corresponding to each batch in 'batches'
    """
    for i, batch in enumerate(batches):
        run = runs[i]
        batch_str = grid_strings.get_batch_string(batch, source)
        path = os.path.join(MODELS_PATH, batch_str, 'logs')

        write_submission_script(batch, run0=run, run1=run, source=source,
                                walltime=walltime, path=path, **kwargs)


def write_submission_script(batch, source, walltime, path=None,
                            run0=None, run1=None, runs=None,
                            parallel=False, qos='medium', basename='xrb',
                            restart=False, max_tasks=16, debug=False):
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

    batch_str = grid_strings.get_batch_string(batch, source)
    if path is None:
        path = os.path.join(MODELS_PATH, batch_str, 'logs')

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
                                        debug=debug, restart=restart)

        span = get_span_string(run0, run1)
        prepend_str = {True: 'restart_'}.get(restart, '')

        filename = f'{cluster}_{prepend_str}{source}_{batch}_{span}{ext}'
        filepath = os.path.join(path, filename)

        with open(filepath, 'w') as f:
            f.write(script_str)

    if parallel:
        write_parallel_script(run0=run0, run1=run1, batch=batch, path=path,
                              restart=restart, basename=basename, source=source,
                              debug=debug)


def get_submission_str(run0, run1, source, runs, batch, basename, cluster,
                       qos, time_str, parallel, job_str, debug, restart):
    source = grid_strings.source_shorthand(source=source)
    span_str = get_span_string(run0, run1, runs=runs)
    batch_str = get_jobstring(batch=batch, run0=run0, run1=run1, source=source,
                              include_source=False)

    # ===== restart parameters =====
    cmd_str = {True: 'z1', False: 'xrb_g'}[restart]
    restart_str = {True: 'restart_', False: ''}[restart]
    debug_str = {True: 'x', False: ''}[debug]

    parallel_file = f'parallel_{restart_str}{source}_{batch_str}.sh'

    if cluster == 'monarch':
        if parallel:
            ntasks = (run1 + 1) - run0
            return f"""#!/bin/bash
#SBATCH --job-name={job_str}
#SBATCH --output=job_{batch}.out
#SBATCH --error=job_{batch}.err
#SBATCH --time={time_str}
#SBATCH --nodes=1
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task=1
#SBATCH --qos={qos}_qos
#SBATCH --partition=batch,medium
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zac.johnston@monash.edu
######################
cd /home/zacpetej/id43/kepler/runs/{source}_{batch}/logs
./{parallel_file}
"""

        else:
            return f"""#!/bin/bash
#SBATCH --job-name={job_str}
#SBATCH --output=arrayJob_%A_%a.out
#SBATCH --error=arrayJob_%A_%a.err
#SBATCH --array={span_str}
#SBATCH --time={time_str}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --qos={qos}_qos
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zac.johnston@monash.edu

######################
N=$SLURM_ARRAY_TASK_ID
EXE_PATH=/home/zacpetej/id43/kepler/gfortran/keplery
cd /home/zacpetej/id43/kepler/runs/{source}_{batch}/{basename}$N/
ln -sf $EXE_PATH ./k
./k {basename}$N {cmd_str} {debug_str}"""

    elif cluster == 'icer':
        if parallel:
            ntasks = (run1 + 1) - run0
            mem = 2000 * ntasks
            disksize = 2 * ntasks
            return f"""#!/bin/bash --login
#PBS -N {job_str}
#PBS -l walltime={time_str}
#PBS -l mem={mem}mb
#PBS -l file={disksize}gb
#PBS -l nodes=1:ppn={ntasks}
#PBS -j oe
#PBS -m abe
#PBS -M zac.johnston@monash.edu
###################################
module load GNU/6.2
cd /mnt/home/f0003004/kepler/runs/{source}_{batch}/logs
./{parallel_file}
qstat -f $PBS_JOBID     # Print statistics """

        else:
            return f"""#!/bin/bash --login
#PBS -N {job_str}
#PBS -l walltime={time_str}
#PBS -l mem=2000mb
#PBS -l file=4gb
#PBS -l nodes=1:ppn=1
#PBS -j oe
#PBS -m abe
#PBS -M zac.johnston@monash.edu
#PBS -t {span_str}
###################################
N=$PBS_ARRAYID
EXE_PATH=$KEPLER_PATH/gfortran/keplery
module load GNU/6.2
cd /mnt/home/f0003004/kepler/runs/{source}_{batch}/{basename}$N/
ln -sf $EXE_PATH ./k

./k {basename}$N {cmd_str} {debug_str}
qstat -f $PBS_JOBID     # Print statistics """


    else:
        raise ValueError('invalid cluster. Must be one of [monarch, icer]')


def write_parallel_script(run0, run1, batch, path, source, restart, debug=False,
                          basename='xrb', gen_file='xrb_g'):
    """========================================================
    Writes a bash script to launch parallel kepler tasks
    ========================================================"""
    source = grid_strings.source_shorthand(source=source)
    print('Writing MPI script')

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

exe_path=$KEPLER_MODELS/../gfortran/keplery
batch_dir=$KEPLER_MODELS/{source}_{batch}

for run in $(seq {run0} {run1}); do
   run_str="{basename}${{run}}"
   echo "{start_str}"
   cd $batch_dir/$run_str
   ln -sf $exe_path ./k
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
