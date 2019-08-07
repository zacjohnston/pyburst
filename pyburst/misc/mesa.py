import numpy as np
import os

# pyburst
from pyburst.burst_analyser import burst_analyser


def model_filepath(run):
    """Returns str filepath to mesa model lightcurve
    """
    path = '/home/zac/projects/kepler_grids/obs_data/mesa/model_lc'
    filename = f'LCTrainVar{run}.txt'
    return os.path.join(path, filename)


def load_model_lc(run):
    """Loads and returns np.array of mesa model lightcurve
    """
    filepath = model_filepath(run)
    return np.loadtxt(filepath, skiprows=2)


def setup_analyser(run):
    """Sets up burst_analyser with haxx
    """
    lc = load_model_lc(run)
    lc[:, 0] *= 3600
    bg = burst_analyser.BurstRun(run=run, batch=1, source='meisel',
                                 load_lum=False, load_config=False,
                                 analyse=False, truncate_edd=False)

    # inject lightcurve
    bg.lum = lc
    bg.setup_lum_interpolator()

    bg.flags['lum_loaded'] = True
    bg.options['load_model_params'] = False

    bg.analyse()
    return bg
