import os

# kepler
import kepdump

# kepler_grids
from pygrids.grids import grid_strings
from pygrids.grids.grid_strings import printv

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
