import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# pyburst
from pyburst.grids import grid_tools

"""
Uses minbar table from:
https://burst.sci.monash.edu/wiki/uploads/MINBAR/minbar.txt 
"""

pyburst_path = os.environ['PYBURST']
filename = 'minbar.txt'
filepath = os.path.join(pyburst_path, 'files', filename)

y_scales = {}

def load_table():
    """Loads minbar table from URL
    """
    return pd.read_table(filepath, delim_whitespace=True, header=9)


class Minbar:
    """
    Class for exploring the MINBAR catalogue
    """
    def __init__(self):
        self.table = load_table()
        self.sources = np.unique(self.table['name'])

    def get_source(self, source):
        """Returns subset of minbar table for given source
        """
        if source not in self.sources:
            raise ValueError(f'source "{source}" not in minbar table. '
                             'Look in self.sources for full list')

        return grid_tools.reduce_table(self.table, params={'name': source})

    def plot_time(self, var, source):
        """Plots given burst var versus time (MJD)
        """
        if var not in self.table.columns:
            raise ValueError(f'var "{var}" not in table. '
                             'Check self.table.columns for full list')

        source_table = self.get_source(source)
        yscale = y_scales.get(var, 1.0)

        fig, ax = plt.subplots()
        ax.set_xlabel('Time (MJD)')
        ax.set_ylabel(var)

        ax.plot(source_table['time'], source_table[var]/yscale, marker='o', ls='')

        plt.show(block=False)

