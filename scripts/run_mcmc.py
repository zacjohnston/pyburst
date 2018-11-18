# kepler_grids
from pyburst.mcmc import mcmc
from pyburst.mcmc import mcmc_tools

import numpy as np
import sys
import os
import time


# =============================================================================
# Usage:
# python run_concord.py [version] [source] [n_walkers] [n_steps] [n_threads] [dumpstep]
# =============================================================================
print('=' * 30)
GRIDS_PATH = os.environ['KEPLER_GRIDS']
nparams = 6
nargs = len(sys.argv)

if (nargs != nparams + 1) and (nargs != nparams + 2):
    print(f"""Must provide {nparams} parameters:
                1. version  : mcmc version ID
                2. source   : source object (e.g., gs1826)
                3. n_walkers : number of mcmc walkers
                4. n_steps   : number of mcmc steps to take
                5. n_threads  : number of threads/cores to use
                6. dumpstep : steps to do between savedumps
                7. (step0   : step to restart from. Optional)""")
    sys.exit()

version = int(sys.argv[1])
source = sys.argv[2]
n_walkers = int(sys.argv[3])
n_steps = int(sys.argv[4])
n_threads = int(sys.argv[5])
dumpstep = int(sys.argv[6])
mcmc_path = mcmc_tools.get_mcmc_path(source)

# ===== if restart =====
if nargs == (nparams + 2):
    restart = True
    start = int(sys.argv[7])
    chain0 = mcmc_tools.load_chain(source=source, version=version, n_walkers=n_walkers,
                                   n_steps=start)
    pos = chain0[:, -1, :]
else:
    restart = False
    start = 0
    pos = mcmc.setup_positions(source=source, version=version, n_walkers=n_walkers)

sampler = mcmc.setup_sampler(source=source, version=version,
                             pos=pos, n_threads=n_threads)
iterations = round(n_steps / dumpstep)
t0 = time.time()

# ===== do 'dumpstep' steps at a time =====
for i in range(iterations):
    step0 = start + (i * dumpstep)
    step1 = start + ((i + 1) * dumpstep)

    print('-' * 30)
    print(f'Doing steps: {step0} - {step1}')
    pos, lnprob, rstate = mcmc.run_sampler(sampler, pos=pos, n_steps=dumpstep)

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
print('=' * 30)
