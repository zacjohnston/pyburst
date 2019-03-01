import numpy as np
import pandas as pd
import os
import sys
from scipy.interpolate import interp1d

# kepler
try:
    import kepdump
except ModuleNotFoundError:
    print('Kepler python module "kepdump" not found. Some functionality disabled.')

# kepler_grids
from pyburst.grids import grid_strings
from pyburst.misc.pyprint import printv

GRIDS_PATH = os.environ['KEPLER_GRIDS']
MODELS_PATH = os.environ['KEPLER_MODELS']


def load_dumps(run, batch, source, cycles=None, basename='xrb'):
    """Returns dict of dumpfiles, in form {cycle: dump_object}
    """
    dumpfiles = {}
    cycles = check_cycles(cycles=cycles, run=run, batch=batch, source=source)
    for i, cycle in enumerate(cycles):
        print_cycle_progress(cycle, cycles=cycles, i=i, prefix=f'Loading dumpfiles: ')
        dumpfiles[cycle] = load_dump(cycle, run=run, batch=batch, source=source,
                                     basename=basename, verbose=False)
    return dumpfiles

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


def extract_dump_table(run, batch, source, cycles=None, dumps=None, basename='xrb'):
    """Returns pandas table of summary dump values

    cycles : [int] (optional)
        list of cycles to extract. If None, uses all available
    dumps : {cycle: dump_object} (optional)
        Pre-loaded dumpfiles
    """
    cycles = check_cycles(cycles=cycles, run=run, batch=batch, source=source)
    if dumps is None:
        dumps = load_dumps(run, batch=batch, source=source, cycles=cycles,
                           basename=basename)
    table = pd.DataFrame()
    table['cycle'] = cycles

    for row in table.itertuples():
        table.loc[row.Index, 'time'] = dumps[row.cycle].time
    return table


def dump_dict(dump):
    """Returns dict of common profiles (radial quantities)
    """
    return {'y': dump.y,
            'tn': dump.tn,
            'xkn': dump.xkn,
            'abar': dump.abar,
            'zbar': dump.zbar,
            }


def check_cycles(cycles, run, batch, source):
    """Get available cycles if none provided
    """
    if cycles is None:
        return get_cycles(run=run, batch=batch, source=source)
    else:
        return cycles


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


def get_cycle_times(cycles, run, batch, source, basename='xrb', prefix=''):
    """Returns array of timestep values (s) for given cycles
    """
    times = np.zeros(len(cycles))
    for i, cycle in enumerate(cycles):
        print_cycle_progress(cycle=cycle, cycles=cycles,
                             i=i, prefix='Getting cycle times: ')
        dump = load_dump(cycle, run=run, batch=batch, source=source,
                         basename=basename, prefix=prefix)
        times[i] = dump.time
    return times


def extract_temps(run, batch, source, depths, cycles=None, basename='xrb'):
    """Extracts temperature versus time from mode dumps (at given depth)
        Returns as [t (s), T_1 .. T_n (K)], where n=len(depths)

    cycles : [int] (optional)
        specifiy which dump cycles to load. If None, load all available
    depths : array
        column depth(s) (g/cm^2) at which to extract temperatures
    """
    if cycles is None:
        cycles = get_cycles(run, batch, source)

    temps = np.zeros((len(cycles), 1+len(depths)))
    for i, cycle in enumerate(cycles):
        print_cycle_progress(cycle, cycles, i, prefix='Extracting temperature: ')
        dump = load_dump(cycle, run=run, batch=batch, source=source, basename=basename)
        depth_temps = get_depth_temps(dump=dump, depths=depths)
        temps[i] = np.concatenate(([dump.time], depth_temps))
    return temps


def get_depth_temps(dump, depths):
    """Returns temperature at given depth(s) (g/cm^2)
    """
    linear = interp_temp(dump)
    return linear(depths)


def get_substrate_zone(dump):
    """Returns column depth (g/cm^2) of transition to substrate
    """
    mass_coord = dump.ymb
    substrate_mass = dump.parm('bmasslow')
    idx = np.searchsorted(np.sort(mass_coord), substrate_mass)
    return len(mass_coord) - idx

def interp_temp(dump, i0=1, i2=-2):
    """Returns a linear interpolation function for given temperature profile
    """
    return interp1d(dump.y[i0:i2], dump.tn[i0:i2])


def print_cycle_progress(cycle, cycles, i, prefix=''):
    sys.stdout.write(f'\r{prefix}cycle {cycle}/{cycles[-1]} '
                     f'({(i+1) / len(cycles) * 100:.1f}%)')
    if cycle == cycles[-1]:
        sys.stdout.write('\n')
