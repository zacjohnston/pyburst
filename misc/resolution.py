import numpy as np
import matplotlib.pyplot as plt

from pygrids.grids import grid_analyser

# resolution tests

def plot(params, res_param, source, source2):
    kgrid = grid_analyser.Kgrid(source)
    p_tab = kgrid.get_params(params=params)
    s_tab = kgrid.get_summ(params=params)

