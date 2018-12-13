import os

PYBURST_PATH = os.environ['PYBURST']
OBS_DATA_PATH = os.path.join(PYBURST_PATH, 'files', 'obs_data')

full_source_labels = {'gs1826': 'gs1826-24',
                      '4u1820': '4u1820-303'}

def summary_filepath(source):
    """Returns filepath to source summary table file

    parameters
    ----------
    source : str
    """
    filename = f'{source}.dat'
    return os.path.join(OBS_DATA_PATH, source, filename)


def epoch_filepath(epoch, source):
    """Returns filepath to epoch file

    parameters
    ----------
    epoch : int
    source : str
    """
    full_source = full_source_labels[source]
    filename = f'{full_source}_{epoch}.dat'
    return os.path.join(OBS_DATA_PATH, source, filename)
