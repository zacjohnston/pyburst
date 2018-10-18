import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# kepler_grids
from pygrids.grids import grid_analyser


def plot_batch(source, batch, error=False):
    kgrid = grid_analyser.Kgrid(source=source, linregress_burst_rate=False,
                                load_lc=True)

    table = kgrid.get_params(batch)

    fig, ax = plt.subplots()
    for row in table.itertuples():
        kgrid.add_lc_plot(ax, batch=batch, run=row.run, label=f'{row.run}', error=error)

    plt.tight_layout()
    plt.show(block=False)
