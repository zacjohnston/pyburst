import pandas as pd

# pyburst
from pyburst.mcmc import burstfit, mcmc_versions, mcmc_tools
from pyburst.grids import grid_analyser
from pyburst.observations import obs_tools

"""
quick n dirty module for synthetic data in MCMC paper (2019)
"""
param_used = {'mdot1': 0.0987,
              'mdot2': 0.1376,
              'mdot3': 0.1567,
              'x': 0.719,
              'z': 0.0032,
              'qb1': 0.059,
              'qb2': 0.0231,
              'qb3': 0.133,
              'm_nw': 2.14,
              'm_gr': 1.96420,
              'd_b': 7.036123,
              'xi_ratio': 0.73909}

def generate_synth_data(source, batches, run, mc_source, mc_version,
                        free_params=('m_gr', 'd_b', 'xi_ratio'),
                        params=None):
    """"""
    if params is None:
        params = generate_params(source, batches=batches, run=run,
                                 mc_source=mc_source, mc_version=mc_version,
                                 free_params=free_params)

    bfit = burstfit.BurstFit(mc_source, version=mc_version, debug=False)
    mv = mcmc_versions.McmcVersion(mc_source, version=mc_version)

    bprops = bfit.bprop_sample(x=None, params=params)
    table = pd.DataFrame()
    pd.set_option("display.precision", 5)
    for i, key in enumerate(mv.bprops):
        bp_i = 2 * i
        u_i = bp_i + 1
        u_key = f'u_{key}'

        table[key] = bprops[:, bp_i]
        table[u_key] = bprops[:, u_i]

    return table


def generate_params(source, batches, run, mc_source, mc_version,
                    free_params=('m_gr', 'd_b', 'xi_ratio')):
    """"""
    synth_grid = grid_analyser.Kgrid(source, use_sub_cols=True)
    mv = mcmc_versions.McmcVersion(mc_source, version=mc_version)

    n_epochs = len(batches)
    pkeys = mv.param_keys
    params = dict.fromkeys(pkeys)

    # ===== Pull model params from kepler grid =====
    for key in mv.interp_keys:
        key = mv.param_aliases.get(key, key)
        if key in mv.epoch_unique:
            for i in range(n_epochs):
                grid_params = synth_grid.get_params(batches[i], run)
                e_key = f'{key}{i+1}'

                grid_key = {'mdot': 'accrate'}.get(key, key)
                params[e_key] = float(grid_params[grid_key])
        else:
            grid_key = {'m_nw': 'mass'}.get(key, key)
            grid_params = synth_grid.get_params(batches[0], run)
            params[key] = float(grid_params[grid_key])

    # ===== Randomly generate free params =====
    for key in free_params:
        params[key] = mcmc_tools.get_random_params(key, n_models=1, mv=mv)[0]

    return params


