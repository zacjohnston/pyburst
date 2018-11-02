import numpy as np
import pandas as pd
import astropy.constants as const
import astropy.units as units
import os
import sys
import matplotlib.pyplot as plt
import chainconsumer

# concord
import ctools
import anisotropy

# kepler_grids
from pygrids.grids import grid_analyser
from pygrids.mcmc import mcmc_tools, burstfit
from pygrids.physics import gparams
from pygrids.burst_analyser import burst_analyser

GRIDS_PATH = os.environ['KEPLER_GRIDS']
MODELS_PATH = os.environ['KEPLER_MODELS']


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


def compare_bprops(source, version, n_walkers, n_steps, epoch):
    """Plot model/obs comparison of single epoch, as one plot"""
    bfit = burstfit.BurstFit(source, version)
    max_p = mcmc_tools.get_max_lhood_params(source, version=version, n_walkers=n_walkers,
                                            n_steps=n_steps)
    interp_params = np.zeros(4)
    interp_params[0] = max_p[epoch-1]
    interp_params[1:] = max_p[epoch:epoch:3]

    interp = bfit.interpolate(interp_params)

    # 2. shift each bprop to observer
    # 3. get obs_data values
    # 4. give to bfit.compare()


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


def check_n_bursts(batches, source, kgrid):
    """Compares n_bursts detected with kepler_analyser against burstfit_1808
    """
    mismatch = np.zeros(4)
    filename = f'mismatch_{source}_{batches[0]}-{batches[-1]}.txt'
    filepath = os.path.join(GRIDS_PATH, filename)

    for batch in batches:
        summ = kgrid.get_summ(batch)
        n_runs = len(summ)

        for i in range(n_runs):
            run = i + 1
            n_bursts1 = summ.iloc[i]['num']
            sys.stdout.write(f'\r{source}_{batch} xrb{run:02}')

            model = burst_analyser.BurstRun(run, batch, source, verbose=False)
            model.analyse()
            n_bursts2 = model.n_bursts

            if n_bursts1 != n_bursts2:
                m_new = np.array((batch, run, n_bursts1, n_bursts2))
                mismatch = np.vstack((mismatch, m_new))

        np.savetxt(filepath, mismatch)
    return mismatch


def convert_adelle_table(filename='allruns_attribs.txt',
                         path='/c/zac/backups/kepler/adelle_1',
                         savename='MODELS.txt'):
    """Loads model table of Adelle's grid, and converts to kepler_grids format
    """
    col_order = ['run', 'z', 'y', 'x', 'qb', 'accrate', 'tshift', 'acc_mult',
                 'qb_delay', 'mass', 'lburn']

    col_rename = {'Qbvalue': 'qb',
                  'X': 'x',
                  'massvalue': 'mass',
                  'numbursts': 'num',
                  'runid': 'run'}

    col_missing = {'tshift': 0.0,
                   'acc_mult': 1.0,
                   'qb_delay': 0.0,
                   'lburn': 1}

    col_convert = {'run': int}

    filepath = os.path.join(path, filename)
    savepath = os.path.join(path, savename)
    table = pd.read_table(filepath, delim_whitespace=True)
    # ===== rename columns =====
    table = table.rename(index=str, columns=col_rename)

    # ===== add missing columns =====
    for col, value in col_missing.items():
        table[col] = value

    # ===== convert some cols to int =====
    for col, dtype in col_convert.items():
        table[col] = table[col].astype(dtype)

    table['y'] = 1 - table['x'] - table['z']

    table_str = table.to_string(index=False)

    with open(savepath, 'w') as f:
        f.write(table_str)

    return table[col_order]

def get_fper(accrate, redshift, distance, inc):
    c = const.c.to(units.cm / units.s)
    mdot_edd = 1.75e-8 * (units.Msun / units.year).to(units.g / units.s)
    phi = (redshift - 1) * c.value**2 / redshift
    _, xi_p = anisotropy.anisotropy(inc*units.deg)
    d = distance * units.kpc.to(units.cm)

    return accrate * mdot_edd * phi / (redshift * 4*np.pi * xi_p * d**2)
