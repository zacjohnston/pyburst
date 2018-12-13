import os

PYBURST_PATH = os.environ['PYBURST']
OBS_DATA_PATH = os.path.join(PYBURST_PATH, 'files', 'obs_data')


def summary_filepath(source):
    filename = f'{source}.dat'
    return os.path.join(OBS_DATA_PATH, source, filename)
