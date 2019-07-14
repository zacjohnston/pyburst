import numpy as np
import sys
import os
import time

# pyburst
from pyburst.mcmc import mcmc, mcmc_tools
from pyburst.misc import pyprint

# =============================================================================
# Usage:
# python run_concord.py [version] [source] [n_walkers] [n_steps] [n_threads] [dump_step]
# =============================================================================

def main(source, version, n_steps, dump_step=None, n_walkers=1000, n_threads=8,
         restart_step=None):
    """Performs an MCMC simulation using the given source grid
    """
    pyprint.print_title(f'{source}  V{version}')
    mcmc_path = mcmc_tools.get_mcmc_path(source)
    chain0 = None

    if dump_step is None:
        dump_step = n_steps
    dump_step = int(dump_step)
    n_threads = int(n_threads)
    n_walkers = int(n_walkers)

    if (n_steps % dump_step) != 0:
        raise ValueError(f'n_steps={n_steps} is not divisible by dump_step={dump_step}')

    if restart_step is None:
        restart = False
        start = 0
        pos = mcmc.setup_positions(source=source, version=version, n_walkers=n_walkers)
    else:
        restart = True
        start = int(restart_step)
        chain0 = mcmc_tools.load_chain(source=source, version=version, n_walkers=n_walkers,
                                       n_steps=start)
        pos = chain0[:, -1, :]

    sampler = mcmc.setup_sampler(source=source, version=version, pos=pos,
                                 n_threads=n_threads)
    iterations = round(n_steps / dump_step)
    t0 = time.time()

    # ===== do 'dump_step' steps at a time =====
    for i in range(iterations):
        step0 = start + (i * dump_step)
        step1 = start + ((i + 1) * dump_step)

        print('-' * 30)
        print(f'Doing steps: {step0} - {step1}')
        pos, lnprob, rstate = mcmc.run_sampler(sampler, pos=pos, n_steps=dump_step)
        # pos, lnprob, rstate, blob = mcmc.run_sampler(sampler, pos=pos, n_steps=dump_step)

        # ===== concatenate loaded chain to current chain =====
        if restart:
            save_chain = np.concatenate([chain0, sampler.chain], 1)
        else:
            save_chain = sampler.chain

        # === save chain state ===
        filename = mcmc_tools.get_mcmc_string(source=source, version=version, prefix='chain',
                                              n_steps=step1, n_walkers=n_walkers, extension='.npy')
        filepath = os.path.join(mcmc_path, filename)
        print(f'Saving: {filepath}')
        np.save(filepath, save_chain)

        # ===== save sampler state =====
        #  TODO: delete previous checkpoint after saving
        mcmc_tools.save_sampler_state(sampler, source=source, version=version,
                                      n_steps=step1, n_walkers=n_walkers)

    print('=' * 30)
    print('Done!')

    t1 = time.time()
    dt = t1 - t0
    time_per_step = dt / n_steps
    time_per_sample = dt / (n_walkers * n_steps)

    print(f'Total compute time: {dt:.0f} s ({dt/3600:.2f} hr)')
    print(f'Average time per step: {time_per_step:.1f} s')
    print(f'Average time per sample: {time_per_sample:.4f} s')


if __name__ == "__main__":
    min_args = 3
    n_args = len(sys.argv)

    if n_args < min_args:
        print(f"""Must provide at least {min_args} parameters:
                    1. source    : source object (e.g., gs1826)
                    2. version   : mcmc version ID
                    3. n_steps   : number of mcmc steps to take
                    (4. dump_step    : steps to do between savedumps)
                    (5. n_walkers    : number of mcmc walkers)
                    (6. n_threads    : number of threads/cores to use)
                    (7. restart_step : step to restart from)""")
        sys.exit(0)

    if n_args == min_args:
        main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    else:
        main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
             **dict(arg.split('=') for arg in sys.argv[4:]))



