import numpy as np
import matplotlib.pyplot as plt

from ..grids import grid_analyser


class Best:
    """Testing comparisons of LC from 'best' MCMC sample,
    against observed LC
    """
    def __init__(self, source='test_bg2', version=48):
        self.grid = grid_analyser.Kgrid('test_bg2', load_lc=True)
