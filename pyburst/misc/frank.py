import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pyburst
from pyburst.kepler import kepler_tools
from pyburst.grids import grid_strings, grid_tools
from pyburst.burst_analyser import burst_tools

# ==========================================
#          Requested data:
# ==========================================
# --------------------------------
#   model info (per model)
# --------------------------------
#   - mdot
#   - surface gravity
#   - Qb
# --------------------------------
#   LC data (one table per model)
# --------------------------------
#   - time
#   - cycle
# --------------------------------
#   profile data columns (one table per dump; one row per zone)
# --------------------------------
#   - zone_number
#   - radius
#   - pressure
#   - density
#   - temperature
#   - heat flux
#   - velocity
#   - conductivity or opacity
#   - energy generation rate
#   - {composition information}
# --------------------------------
# composition information:
#   - X_H, X_He, X_N, X_Fe (or whatever substrate is used)
#   - <Z>, <A>
# ==========================================

# TODO: Verify zone centre/interface values


def extract_batch_profiles(batch, source='frank', basename='xrb'):
    """Extracts and saves all profiles of all runs from a batch
    """
    nruns = grid_tools.get_nruns(batch=batch, source=source, basename=basename)

    for run in range(1, nruns + 1):
        extract_run_profiles(run=run, batch=batch, source=source, basename=basename)


def extract_run_profiles(run, batch, source='frank', basename='xrb'):
    """Extracts and saves tables for all cycle profiles of a run
    """
    cycles = kepler_tools.get_cycles(run=run, batch=batch, source=source)
    n_cycles = len(cycles)

    for i, cycle in enumerate(cycles):
        percent = (i+1) / n_cycles * 100
        sys.stdout.write(f'\rExtracting profile {source}_{batch}_{run}_{cycle}  ({percent:.2f}%)')
        table = extract_profile(cycle=cycle, run=run, batch=batch, source=source,
                                basename=basename)
        save_profile(table=table, cycle=cycle, run=run, batch=batch, source=source,
                     basename=basename, verbose=False)
    sys.stdout.write('\n')

def profile_filepath(cycle, run, batch, source='frank', basename='xrb'):
    """Returns string for path to profile table file
    """
    path = profile_path(run=run, batch=batch, source=source, basename=basename)
    filename = profile_filename(cycle=cycle, run=run, batch=batch, source=source,
                                basename=basename)
    return os.path.join(path, filename)


def profile_path(run, batch, source='frank', basename='xrb'):
    """Return path to directory containing profile data
    """
    path = batch_path()
    run_str = grid_strings.get_run_string(run=run, basename=basename)
    return os.path.join(path, run_str)


def batch_path(batch, source='frank'):
    """Returns string of path to batch dir
    """
    path = grid_strings.get_source_subdir(source, 'profiles')
    batch_str = grid_strings.get_batch_string(batch=batch, source=source)
    return os.path.join(path, batch_str)


def lum_filepath(run, batch, source='frank', basename='xrb'):
    """Returns string of path to lum table file
    """
    path = batch_path(batch=batch, source=source)
    filename = lum_filename(run=run, batch=batch, source=source, basename=basename)
    return os.path.join(path, filename)


def profile_filename(cycle, run, batch, source='frank', basename='xrb'):
    """Returns string for profile table filename
    """
    return f'profile_{source}_{batch}_{basename}{run}_{cycle}.txt'


def lum_filename(run, batch, source='frank', basename='xrb'):
    """Returns string for lum table filename
    """
    return f'lum_{source}_{batch}_{basename}{run}.txt'


def save_profile(table, cycle, run, batch, source='frank', basename='xrb', verbose=True):
    """Saves profile table to file
    """
    filepath = profile_filepath(cycle=cycle, run=run, batch=batch, source=source,
                                basename=basename)
    grid_tools.write_pandas_table(table=table, filepath=filepath, verbose=verbose)


def save_lum(table, run, batch, source='frank', basename='xrb', verbose=True):
    """Saves table of luminosity with time to file
    """
    filepath = lum_filepath(run=run, batch=batch, source=source, basename=basename)
    grid_tools.write_pandas_table(table=table, filepath=filepath, verbose=verbose)


def extract_lum(run, batch, source='frank', basename='xrb'):
    """Returns nparray for whole model of time, lum, cycle
    """
    table = pd.DataFrame()
    cycles = kepler_tools.get_cycles(run=run, batch=batch, source=source)
    lum = burst_tools.load_lum(run=run, batch=batch, source=source, basename=basename)
    lum = lum[1:]  # no dump for zeroth step

    table['cycle'] = cycles
    table['time'] = lum[:, 0]
    table['lum'] = lum[:, 1]

    return table


def extract_profile(cycle, run, batch, source='frank', basename='xrb',
                    endpoints=(1, -1)):
    """Returns DataFrame table of profile information for given dump cycle
    """
    table = pd.DataFrame()

    dump = kepler_tools.load_dump(cycle=cycle, run=run, batch=batch,
                                  source=source, basename=basename)
    _slice = slice(endpoints[0], endpoints[1])
    n_zones = len(dump.y[_slice])

    # --- Thermodynamics ---
    table['zone'] = np.arange(n_zones)
    table['radius'] = dump.rn[_slice]
    table['column'] = dump.y[_slice]
    table['pressure'] = dump.pn[_slice]
    table['density'] = dump.dn[_slice]
    table['temp'] = dump.tn[_slice]
    # table['heat_flux'] = dump.    # xln?
    table['velocity'] = dump.un[_slice]
    table['opacity'] = dump.xkn[_slice]
    table['energy_rate'] = dump.sburn[_slice]  # snn/sburn?
    table['gamma'] = dump.gamma[_slice]

    # --- Composition ---
    # table['h1'] = dump.abub['h1'][_slice]
    # table['he4'] = dump.abub['he4'][_slice]
    # table['n14'] = dump.abub['n14'][_slice]
    # table['fe54'] = dump.abub['fe54'][_slice]
    table['zbar'] = dump.zbar[_slice]
    table['abar'] = dump.abar[_slice]

    return table
