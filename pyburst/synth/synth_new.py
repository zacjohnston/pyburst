# pyburst
from pyburst.mcmc import burstfit, mcmc_versions, mcmc_tools
from pyburst.grids import grid_analyser

# quick n dirty module for synthetic data in MCMC paper (2019)


def generate_params(source, batches, run, mc_source, mc_version,
                    free_params=('m_gr', 'd_b', 'xi_ratio')):

    synth_grid = grid_analyser.Kgrid(source, use_sub_cols=True)
    bfit = burstfit.BurstFit(mc_source, version=mc_version, debug=False)
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


