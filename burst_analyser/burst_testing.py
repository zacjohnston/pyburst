import numpy as np

from pygrids.burst_analyser import burst_analyser


def test_bursts(n_bursts, run=1, batch=281, source='biggrid2', plot=True):
    """Test burst_analyser when model has zero bursts
    """
    model_ref = burst_analyser.BurstRun(run, batch=batch, source=source, analyse=False,
                                        plot=False, reload=False, load_bursts=True,
                                        load_summary=True, load_lum=False)
    model_test = burst_analyser.BurstRun(run, batch=batch, source=source, analyse=False,
                                         plot=False, reload=False, load_bursts=False,
                                         load_summary=False, load_lum=True)

    pre_idx = model_ref.bursts['t_pre_i'][n_bursts]
    model_test.lum = model_test.lum[:pre_idx]
    model_test.analyse()

    if plot:
        model_test.plot()
        model_test.plot_convergence()
        model_test.plot_linregress()
