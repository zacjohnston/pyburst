import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pyburst
from pyburst.kepler import kepler_tools
from pyburst.grids import grid_strings

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

def profile_path(run, batch, source='frank', basename='xrb'):
    """Return path to directory containing profile data
    """
    path = grid_strings.get_source_subdir(source, 'profiles')
    run_str = grid_strings.get_run_string(run=run, basename=basename)
    batch_str = grid_strings.get_batch_string(batch=batch, source=source)
    return os.path.join(path, batch_str, run_str)


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
    table['pressure'] = dump.pn[_slice]
    table['density'] = dump.dn[_slice]
    table['temp'] = dump.tn[_slice]
    # table['head_flux'] = dump.
    # table['velocity'] = dump.
    table['opacity'] = dump.xkn[_slice]
    table['energy_rate'] = dump.sburn[_slice]  # snn/sburn?

    # --- Composition ---
    table['zbar'] = dump.zbar[_slice]
    table['abar'] = dump.abar[_slice]

    return table
