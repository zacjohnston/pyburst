import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pygrids.synth import synth
from pygrids.mcmc import mcmc_plot, mcmc_versions

def plot_posteriors(chain, source, mc_version, discard, **kwargs):
    """Plots mcmc posteriors for synthetic data
    """
    mcv = mcmc_versions.McmcVersion(source, mc_version)

    truth = synth.get_true_values(source, version=mcv.synth_version,
                                  group=mcv.synth_group)
    mcmc_plot.plot_posteriors(chain, source=source, version=mcv.synth_version,
                              discard=discard, truth_values=truth, **kwargs)
