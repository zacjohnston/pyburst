import numpy as np
import os

# kepler
import kepdump

# kepler_grids
from pygrids.grids import grid_strings
from pygrids.misc.pyprint import printv

GRIDS_PATH = os.environ['KEPLER_GRIDS']
MODELS_PATH = os.environ['KEPLER_MODELS']

def load_dump(cycle, run, batch, source, basename='xrb',
              prefix='', verbose=False):
    batch_str = grid_strings.get_batch_string(batch, source)
    run_str = grid_strings.get_run_string(run, basename)
    filename = get_dump_filename(cycle, run, basename, prefix=prefix)

    filepath = os.path.join(MODELS_PATH, batch_str, run_str, filename)
    printv(f'Loading: {filepath}', verbose=verbose)
    return kepdump.load(filepath, graphical=False, silent=True)


def get_dump_filename(cycle, run, basename, prefix=''):
    return f'{prefix}{basename}{run}#{cycle}'


def get_cycles(run, batch, source):
    """Returns list of dump cycles available for given model
    """
    path = grid_strings.get_model_path(run, batch, source=source)
    file_list = os.listdir(path)

    cycles = []
    for file in file_list:
        if '#' in file:
            idx = file.find('#')
            cyc = file[idx+1:]
            if cyc == 'nstop':
                continue
            else:
                cycles += [int(cyc)]
    return np.sort(cycles)


def extract_temps(run, batch, source, cycles=None, basename='xrb',
                  temp_zone=20):
    """Extracts base temperature versus time from mode dumps. Returns as [t (s), T (K)]

    cycles : [int] (optional)
        specifiy which dump cycles to load. If None, load all available
    zone : int
        zone (index) to extract temperature from
    """
    # TODO: option to extract from approximate depth
    if cycles is None:
        cycles = get_cycles(run, batch, source)

    temps = np.zeros((len(cycles), 2))
    for i, cycle in enumerate(cycles):
        dump = load_dump(cycle, run=run, batch=batch, source=source, basename=basename)
        temps[i] = np.array((dump.time, dump.tn[temp_zone]))
    return temps
