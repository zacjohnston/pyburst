import numpy as np
import matplotlib.pyplot as plt
import chainconsumer

import ctools
from pygrids.grids import grid_analyser
from pygrids.mcmc import mcmc_tools
from pygrids.physics import gparams


def compare_lc(burst, point, batches=[19, 1, 1], runs=[10, 20, 12]):
    kg2 = grid_analyser.Kgrid('biggrid2', load_concord_summ=False, exclude_defaults=True,
                              powerfits=False, burst_analyser=True)
    obs = ctools.load_obs('gs1826')
    # models = ctools.load_models(batches=batches, runs=runs, source='biggrid2')

    idx = burst - 1
    ob = obs[idx]
    # mod = models[idx]

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim((-10, 120))
    kg2.load_mean_lightcurves(19)
    kg2.load_mean_lightcurves(1)
    kg2.load_mean_lightcurves(37)

    batch = batches[idx]
    run = runs[idx]
    mlc = kg2.mean_lc[batch][run]

    # MODEL
    tshift = [9, 9, 9][idx]
    redshift = point[-3]
    f_b = point[-2] * 1e45
    flux_factor = (redshift * 4 * np.pi * f_b)

    m_time = mlc[:, 0]*redshift + tshift
    m_flux = mlc[:, 1] / flux_factor
    u_mflux = mlc[:, 2] / flux_factor

    ax.plot(m_time, m_flux)
    ax.fill_between(m_time, m_flux-u_mflux, m_flux+u_mflux, color='0.8')

    # OBSERVATION
    ax.errorbar(ob.time.value + 0.5*ob.dt.value, ob.flux.value, yerr=ob.flux_err.value,
                ls='none', marker='o', capsize=3)

    plt.show(block=False)


def plot_posteriors(chain=None, discard=10000):
    if chain is None:
        chain = mcmc_tools.load_chain('sim_test', n_walkers=960, n_steps=20000, version=5)
    params = [r'Accretion rate ($\dot{M} / \dot{M}_\text{Edd}$)', 'Hydrogen',
              r'$Z_{\text{CNO}}$', r'$Q_\text{b}$ (MeV nucleon$^{-1}$)',
              'gravity ($10^{14}$ cm s$^{-2}$)', 'redshift (1+z)',
              'distance (kpc)', 'inclination (degrees)']

    g = gparams.get_acceleration_newtonian(10, 1.4).value / 1e14
    chain[:, :, 4] *= g

    cc = chainconsumer.ChainConsumer()
    cc.add_chain(chain[:, discard:, :].reshape((-1, 8)))
    cc.configure(kde=False, smooth=0)

    fig = cc.plotter.plot_distributions(display=True)

    for i, p in enumerate(params):
        fig.axes[i].set_title('')
        fig.axes[i].set_xlabel(p)#, fontsize=10)

    plt.tight_layout()
    return fig
