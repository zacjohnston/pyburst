import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from ..burst_analyser import burst_analyser


class BurstRun:
    def __init__(self, run, batch=1, source='test_bg2', bprop='dt', plot_conv=True):
        min_points = 4
        self.bg = burst_analyser.BurstRun(run, batch, source=source)
        self.n = self.bg.n_bursts + 1 - min_points
        self.bprop = bprop

        self.slope, self.slope_err = self.linregress()
        self.res = np.abs(self.slope / self.slope_err)

        self.plot()
        self.plot2()
        if plot_conv:
            self.bg.plot_convergence()

    def linregress(self):
        y = self.bg.bursts[self.bprop]
        x = np.arange(len(y))
        slope = np.full(self.n, np.nan)
        slope_err = np.full(self.n, np.nan)

        for i in range(self.n):
            lin = linregress(x[i:], y[i:])
            slope[i] = lin[0]
            slope_err[i] = lin[-1]

        return slope, slope_err

    def plot(self):
        fig, ax = plt.subplots()
        x = np.arange(self.n)

        ax.plot(x, self.res, ls='none', marker='o')
        ax.plot([0, self.n], [1, 1], color='C1')
        ax.plot([0, self.n], [2, 2], ls='--', color='C1')
        plt.show(block=False)

    def plot2(self):
        fig, ax = plt.subplots()
        x = np.arange(self.n)
        ax.errorbar(x, self.slope, yerr=self.slope_err, ls='none', marker='o', capsize=3, color='C0')
        ax.plot([0, self.n], [0, 0], ls='--', color='C1')
        plt.show(block=False)