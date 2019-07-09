import numpy as np
import os
import sys

# pyburst
from pyburst.misc import pyprint
from pyburst.grids import grid_strings, grid_tools
from pyburst.burst_analyser import burst_analyser


def save_all_lightcurves(batches, source, path=None, align='t_start', reload=False):
    """Saves all lightcurve files for each model in given batches
    """
    for batch in batches:
        n_runs = grid_tools.get_nruns(batch, source=source)
        for run in range(1, n_runs+1):
            model = burst_analyser.BurstRun(run, batch=batch, source=source,
                                            reload=reload)
            model.save_burst_lightcurves(path=path, align=align)
