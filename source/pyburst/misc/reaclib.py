import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyburst.grids import grid_analyser

def compare(batch, source, ref_source, bprops=('rate', 'fluence', 'peak')):
    """Compares models with differe bdats/adapnets"""
    kgrid = grid_analyser.Kgrid(source, linregress_burst_rate=False)
    kgrid_ref = grid_analyser.Kgrid(ref_source, linregress_burst_rate=False)
    sub_params = kgrid.get_params(batch).reset_index()
    sub_summ = kgrid.get_summ(batch).reset_index()
    params_ref, summ_ref = extract_ref_subset(param_table=sub_params, kgrid_ref=kgrid_ref)

    fig, ax = plt.subplots(len(bprops), 1, figsize=(10, 12))

    for i, bprop in enumerate(bprops):
        u_bprop = f'u_{bprop}'
        ratio = sub_summ[bprop] / summ_ref[bprop]
        u_frac = sub_summ[u_bprop]/sub_summ[bprop] + summ_ref[u_bprop]/summ_ref[bprop]
        u_ratio = ratio * u_frac
        n = len(ratio)
        ax[i].errorbar(np.arange(n), ratio, yerr=u_ratio, ls='none', marker='o', capsize=3)
        ax[i].plot([0, n], [1, 1], color='black')
        ax[i].set_ylabel(bprop)

    plt.tight_layout()
    plt.show(block=False)


def extract_ref_subset(param_table, kgrid_ref):
    """Returns subset of reference grid that matches comparison subset
    """
    params_out = pd.DataFrame()
    summ_out = pd.DataFrame()

    for row in param_table.itertuples():
        params = {'z': row.z, 'x': row.x, 'accrate': row.accrate,
                  'qb': row.qb, 'mass': row.mass}

        sub_params = kgrid_ref.get_params(params=params)
        sub_summ = kgrid_ref.get_summ(params=params)
        if len(sub_params) is 0:
            raise RuntimeError(f'No corresponding model for {params}')
        if len(sub_params) > 1:
            raise RuntimeError(f'Multiple models match {params}')

        params_out = pd.concat((params_out, sub_params), ignore_index=True)
        summ_out = pd.concat((summ_out, sub_summ), ignore_index=True)
    return params_out, summ_out
