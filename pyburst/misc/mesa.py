import numpy as np
import pandas as pd
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
    model = burst_analyser.BurstRun(run=run, batch=1, source='meisel',
                                 load_lum=False, load_config=False,
                                 analyse=False, truncate_edd=False)

    # inject lightcurve
    model.lum = lc
    model.setup_lum_interpolator()

    model.flags['lum_loaded'] = True
    model.options['load_model_params'] = False

    model.analyse()
    return model

def extract_bprops(runs, mdots,
                   bprops=('dt', 'rate', 'fluence', 'peak')):
    """Quick and dirty extraction of burst properties from Mesa models
    """
    n_models = len(runs)
    models = pd.DataFrame()
    models['accrate'] = mdots

    summ = dict.fromkeys(bprops)
    for key in bprops:
        u_key = f'u_{key}'
        summ[key] = np.full(n_models, np.nan)
        summ[u_key] = np.full(n_models, np.nan)

    for i, run in enumerate(runs):
        model = setup_analyser(run)

        for bprop in bprops:
            u_bprop = f'u_{bprop}'
            summ[bprop][i] = model.summary[bprop]
            summ[u_bprop][i] = model.summary[u_bprop]

    for key in summ:
        models[key] = summ[key]

    return models
