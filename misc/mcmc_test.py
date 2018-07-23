import numpy as np
import matplotlib.pyplot as plt

from ..mcmc import burstfit


class BfitTester:

    def __init__(self, source, version, priors_only=True):
        self.bfit = burstfit.BurstFit(source, version, re_interp=True,
                                      priors_only=priors_only)

    def plot_lhood(self, param='mdot1', n_points=500, scale=None):
        """Plots lhood along a slice
        """
        p_idx = self.bfit.param_idxs[param]
        bounds = self.bfit.mcmc_version.prior_bounds[p_idx]

        params = np.array(self.bfit.mcmc_version.initial_position)
        x = np.linspace(bounds[0], bounds[1], n_points)
        y = np.full_like(x, np.nan)

        for i in range(n_points):
            params[p_idx] = x[i]
            y[i] = self.bfit.lhood(params)

        if scale is None:
            finite = np.isfinite(y)
            max = np.max(y[finite])
            print(max)
            # scale = np.abs(np.mean(y[finite]))/10
            scale = max / 5
            print(f'Using scale: {scale:.1f}')

        plot_lnpdf(x, y, scale=scale)


def plot_lnpdf(x, y, scale=1000):
    fig, ax = plt.subplots()
    ax.plot(x, np.exp(y/scale))
    plt.show(block=False)