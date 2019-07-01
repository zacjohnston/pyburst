import numpy as np
import os
import matplotlib.pyplot as plt
from astropy import units

# pyburst
from pyburst.mcmc import burstfit
from pyburst.misc import anisotropy_tools
from pyburst.physics import gravity

path = '/home/zac/projects/codes/pyburst/files/temp'
sim_filename = 'sim_gs1826_2_xrb9_mean.data'
model_filename = 'gs1826_2_xrb9_mean.data'

sim_filepath = os.path.join(path, sim_filename)
model_filepath = os.path.join(path, model_filename)

filepaths = {
    'model': os.path.join(path, model_filename),
    'sim': os.path.join(path, sim_filename),
}
delims = {'sim': ','}

duncan_params = {
    'd': 6.1,  # kpc
    'inclination': 60,  # deg
    'redshift': 1.26,
    'mass': 1.76,  # Msun
    't_offset': -10,  # sec
}


def load_lc(label):
    """Loads and returns lightcurve data from Duncan
    """
    filepath = filepaths.get(label)

    print(f'Loading {filepath}')
    print('Columns: t, dt, flux, u_flux')
    table = np.loadtxt(filepath, delimiter=delims.get(label))

    return table


def plot_sim():
    sim = load_lc('sim')
    fig, ax = plt.subplots()
    ax.errorbar(sim[:, 0] + 0.5*sim[:, 1], sim[:, 2], yerr=sim[:, 3],
                ls='none', marker='o', capsize=3, label='sim')
    ax.legend()
    plt.show(block=False)
    return fig, ax


def plot_model():
    model = load_lc('model')
    fig, ax = plt.subplots()
    ax.plot(model[:, 0], model[:, 1]/1e38, label='model')
    ax.legend()
    plt.show(block=False)
    return fig, ax


def shift_model(model, params=duncan_params, source='grid5', version=1):
    """Returns model lightcurve shifted into observer frame (s, 1e-9 erg/cm^2/s)
    """
    lc = np.zeros_like(model[:, :2])
    bfit = burstfit.BurstFit(source, version=version)
    bfit_params = get_burstfit_params(params)

    lc[:, 0] = bfit.shift_to_observer(model[:, 0], bprop='dt', params=bfit_params)
    lc[:, 0] *= 3600  # bfit outputs in hrs
    lc[:, 0] -= params['t_offset']

    lc[:, 1] = bfit.shift_to_observer(model[:, 1], bprop='peak', params=bfit_params)
    lc[:, 1] *= 1e9  # flux units

    return lc


def get_burstfit_params(params, r_nw=10):
    """Converts Duncan's params to those for input to BurstFit
    """
    out_params = dict()

    xi_b, xi_p = anisotropy_tools.anisotropy.anisotropy(params['inclination']*units.deg)
    g = gravity.get_acceleration_newtonian(r=r_nw, m=params['mass'])
    m_gr, r_gr = gravity.get_mass_radius(g, redshift=params['redshift'])

    out_params['d_b'] = params['d'] * np.sqrt(xi_b)
    out_params['m_nw'] = params['mass']
    out_params['m_gr'] = m_gr.value
    out_params['xi_ratio'] = xi_p / xi_b

    return out_params
