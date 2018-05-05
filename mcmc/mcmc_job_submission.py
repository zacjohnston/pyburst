import os

GRIDS_PATH = os.environ['KEPLER_GRIDS']


def write_submission_script(source, version, n_walkers, n_steps, dump_step,
                            walltime=20, n_threads=8, cluster='icer'):
    """Writes a script for submitting a job on a cluster

    Parameter:
    ----------
    path : str
        target path for slurm script
    walltime: int
        job time limit (hr)
    """
    extensions = {'monarch': '.sh', 'icer': '.qsub'}

    job_str = get_jobstring(source=source, version=version, n_walkers=n_walkers,
                            n_threads=n_threads, n_steps=n_steps)
    time_str = f'{walltime:02}:00:00'

    print(f'Writing submission script for cluster: {cluster}')
    ext = extensions[cluster]
    script_str = get_submission_str(source=source, version=version,
                                    n_walkers=n_walkers, n_threads=n_threads,
                                    n_steps=n_steps, time_str=time_str,
                                    job_str=job_str, cluster=cluster,
                                    dump_step=dump_step)

    # prepend_str = {True:'restart_', False:''}[restart]
    path = os.path.join(GRIDS_PATH, 'sources', source, 'logs')
    filename = f'{cluster}_{job_str}{ext}'
    filepath = os.path.join(path, filename)

    with open(filepath, 'w') as f:
        f.write(script_str)


def get_jobstring(source, version, n_walkers, n_steps, n_threads):
    return f'{source}_V{version}_T{n_threads}_W{n_walkers}_S{n_steps}'


def get_submission_str(source, version, n_walkers, n_threads, n_steps,
                       cluster, time_str, job_str, dump_step,
                       disksize=8000, memory=16000, ):
    """Returns a script string for submitting a job on a cluster

    Parameter:
    ----------
    time_str : str
        walltime required for job (format HH:MM:SS)
    disksize : int
        disk storage required for job (MB)
    memory: int
        memory required for job (MB)
    dump_step : int
        save mcmc state every "dump_step" steps
    """
    if cluster == 'icer':
        return f"""#!/bin/bash --login
#PBS -N {job_str}
#PBS -l walltime={time_str}
#PBS -l mem={memory}mb
#PBS -l file={disksize}mb
#PBS -l nodes=1:ppn={n_threads}
#PBS -j oe
#PBS -m abe
#PBS -M zac.johnston@monash.edu
###################################
source /mnt/home/f0003004/python/mypy3.6.5/bin/activate
cd /mnt/home/f0003004/kepler_grids/scripts
python3 run_mcmc.py {version} {source} {n_walkers} {n_steps} {n_threads} {dump_step}
qstat -f $PBS_JOBID     # Print statistics """

    elif cluster == 'monarch':
        return f"""#!/bin/bash
#SBATCH --job-name={job_str}
#SBATCH --output={job_str}.out
#SBATCH --error={job_str}.err
#SBATCH --time={time_str}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={n_threads}
#SBATCH --qos={qos}_qos
#SBATCH --partition=batch,medium
#SBATCH --mem-per-cpu={memory}
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zac.johnston@monash.edu
######################
cd /mnt/home/f0003004/kepler_grids/scripts
python3 run_mcmc.py {version} {source} {n_walkers} {n_steps} {n_threads} {dump_step}
"""
