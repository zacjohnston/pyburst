import numpy as np
import os

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

def load_lc(label):
    """Loads and returns lightcurve data from Duncan
    """
    filepath = filepaths.get(label)

    print(f'Loading {filepath}')
    print('Columns: t, dt, flux, u_flux')
    table = np.loadtxt(filepath, delimiter=delims.get(label))

    return table


