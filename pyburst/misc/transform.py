import numpy as np
import os
import matplotlib.pyplot as plt

# pyburst
from pyburst.mcmc import burstfit

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


def shift_model(model, params):
    """Shifts given model lightcurve to observer frame
    """
    pass


def get_burstfit_params(params):
    """Converts Duncan's params to those for input to BurstFit
    """
    out_params = dict()

    out_params['d_b'] = params['d']
    out_params['m_nw'] = 1.4
    out_params['m_gr'] = 1.4
    out_params['xi_ratio'] = 1.0

    return out_params
