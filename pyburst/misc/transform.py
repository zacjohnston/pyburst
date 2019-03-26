import numpy as np
import os

path = '/home/zac/projects/kepler_grids/misc/'
filename = 'sim_gs1826_2_xrb9_mean.data'
filepath = os.path.join(path, filename)

modelfile = 'gs1826_2_xrb9_mean.data'

def load_lc():
    """Loads and returns lightcurve data from Duncan
    """
    print(f'Loading {filepath}')
    print('Columns: t, dt, flux, u_flux')
    table = np.loadtxt(filepath, delimiter=',')
    return table


