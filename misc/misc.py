import numpy as np
import matplotlib.pyplot as plt

import ctools
from pygrids.grids import grid_analyser


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
