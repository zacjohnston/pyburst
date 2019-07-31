import numpy as np
import pandas as pd

# pyburst
from pyburst.mcmc import burstfit, mcmc_versions, mcmc_tools, mcmc_plot
from pyburst.grids import grid_analyser
from pyburst.observations import obs_tools
from pyburst.plotting import plot_tools
"""
quick n dirty module for synthetic data in MCMC paper (2019)
"""
# ==========================
# from: synth5_7-9, run=2
# ==========================
param_used = {'mdot1': 0.102,
              'mdot2': 0.1371,
              'mdot3': 0.1497,
              'x': 0.6971,
              'z': 0.0061,
              'qb1': 0.1551,
              'qb2': 0.1549,
              'qb3': 0.1774,
              'm_nw': 2.02,
              'm_gr': 1.6918,
              'd_b': 7.05839,
              'xi_ratio': 1.0190}


def plot_posteriors(chain, source, version, discard, cap=None):
    """Plots posteriors against true values
    """
    truth = get_truth_values(source, version)
    mcmc_plot.plot_posteriors(chain, source=source, version=version,
                              discard=discard, cap=cap, truth_values=truth)


def get_truth_values(source, version):
    """Returns truth values of original params, with formatted labels
    """
    pkeys = mcmc_versions.get_parameter(source, version, 'param_keys')
    pkey_labels = plot_tools.convert_mcmc_labels(param_keys=pkeys)

    truth = dict()
    for i, key in enumerate(pkeys):
        key_formatted = pkey_labels[i]
        truth[key_formatted] = param_used[key]

    return truth


def generate_synth_data(source, batches, run, mc_source, mc_version,
                        reproduce=True, free_params=('m_gr', 'd_b', 'xi_ratio'),
                        u_fedd_frac=0.08, u_fper_frac=0.01, noise_mag=0.01,
                        introduce_noise=True):
    if reproduce:
        print('Reusing same params')
        params = param_used
    else:
        print('Generating new random params!')
        params = generate_params(source, batches=batches, run=run,
                                 mc_source=mc_source, mc_version=mc_version,
                                 free_params=free_params)

    table = setup_synth_table(source, batches=batches, run=run, mc_source=mc_source,
                              mc_version=mc_version, free_params=free_params,
                              params=params, u_fedd_frac=u_fedd_frac,
                              u_fper_frac=u_fper_frac)

    if introduce_noise:
        add_noise(table, magnitude=noise_mag)

    # add epoch column
    epochs = np.arange(1, len(batches) + 1)
    table['epoch'] = epochs
    table['cbol'] = 1.0
    table['u_cbol'] = 0.0

    obs_tools.save_summary(table, source=source)


def add_noise(table, magnitude=0.01):
    print(f'adding noise: sigma={magnitude}')
    n_rows = len(table)
    for col in table:
        noise = 1 + magnitude * np.random.normal(size=n_rows)
        table[col] *= noise


def setup_synth_table(source, batches, run, mc_source, mc_version,
                      free_params=('m_gr', 'd_b', 'xi_ratio'),
                      params=None, u_fedd_frac=0.08, u_fper_frac=0.01):
    """"""
    if params is None:
        params = generate_params(source, batches=batches, run=run,
                                 mc_source=mc_source, mc_version=mc_version,
                                 free_params=free_params)

    bfit = burstfit.BurstFit(mc_source, version=mc_version, debug=False,
                             u_fper_frac=u_fper_frac, u_fedd_frac=u_fedd_frac)
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


