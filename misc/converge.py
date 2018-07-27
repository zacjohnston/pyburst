import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from functools import reduce

from ..burst_analyser import burst_analyser
from ..grids import grid_analyser


class BurstRun:
    def __init__(self, run, batch=1, source='test_bg2', plot=True, plot_conv=True,
                 min_points=10):
        self.min_points = min_points  # Note: dt will use one less point
        self.bg = burst_analyser.BurstRun(run, batch, source=source)
        self.n = self.bg.n_bursts + 1 - min_points
        self.bprops = ['dt', 'fluence', 'peak']

        self.slope = {}
        self.slope_err = {}
        self.residual = {}

        for bprop in self.bprops:
            self.slope[bprop], self.slope_err[bprop] = self.linregress(bprop)
            self.residual[bprop] = np.abs(self.slope[bprop] / self.slope_err[bprop])

        self.discard = self.get_discard()

        if plot:
            self.plot()
        if plot_conv:
            self.bg.plot_convergence()

    def linregress(self, bprop):
        y = self.bg.bursts[bprop]
        x = np.arange(len(y))
        slope = np.full(self.n, np.nan)
        slope_err = np.full(self.n, np.nan)

        for i in range(self.n):
            lin = linregress(x[i:], y[i:])
            slope[i] = lin[0]
            slope_err[i] = lin[-1]

        return slope, slope_err

    def get_discard(self):
        """Returns min no. of bursts to discard to achieve zero slope
        """
        zero_slope_idxs = []
        for bprop in self.bprops:
            zero_slope_idxs += np.where(self.residual[bprop] < 1)

        min_discard = reduce(np.intersect1d, zero_slope_idxs)
        if len(min_discard) == 0:
            return np.nan
        else:
            return min_discard[0]

    def plot(self):
        fig, ax = plt.subplots(3, 1, figsize=(10, 12))
        x = np.arange(self.n)
        fontsize = 14

        for i, bprop in enumerate(self.bprops):
            y = self.slope[bprop]
            y_err = self.slope_err[bprop]
            ax[i].set_ylabel(bprop, fontsize=fontsize)
            ax[i].errorbar(x, y, yerr=y_err, ls='none', marker='o', capsize=3)
            ax[i].plot([0, self.n-1], [0, 0], ls='--')

        ax[-1].set_xlabel('Discarded bursts', fontsize=fontsize)
        plt.tight_layout()
        plt.show(block=False)


class restart:
    def __init__(self):
        self.grid = grid_analyser.Kgrid('biggrid2', load_concord_summ=False,
                                        exclude_defaults=True, powerfits=False,
                                        burst_analyser=True)
