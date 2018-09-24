import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pygrids.grids import grid_analyser
from pygrids.burst_analyser import burst_analyser

def compare(batch, source, ref_source):
    """Compares models with differe bdats/adapnets"""
    kgrid = grid_analyser.Kgrid(source)
    kgrid_ref = grid_analyser.Kgrid(ref_source)
    sub = kgrid.get_params(batch)


def extract_ref_subset(param_table, kgrid_ref):
    """Returns subset of reference grid that matches comparison subset
    """
    table_out = pd.DataFrame()

    for row in param_table.itertuples():
        params = {'z': row.z, 'x': row.x, 'accrate': row.accrate,
                  'qb': row.qb, 'mass': row.mass}

        sub = kgrid_ref.get_params(params=params)
        if len(sub) is 0:
            raise RuntimeError(f'No corresponding model for {params}')
        if len(sub) > 1:
            raise RuntimeError(f'Multiple models match {params}')

        table_out = pd.concat((table_out, sub), ignore_index=True)

    return table_out
