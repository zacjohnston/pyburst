import numpy as np
import matplotlib.pyplot as plt
import anisotropy


def load_models(models=None):
    """Loads in anisotropy tables from He & Keek 2016
    """
    if models is None:
        models = ('a', 'b', 'c', 'd')

    tables = {}
    for model in models:
        tables[model] = anisotropy.load_he16(f'he16_{model}')

    return tables


def plot_ratio(tables=None, models=None, fontsize=18):
    """Plot xi_p/xi_b ratio for different models"""
    if tables is None:
        tables = load_models(models)

    fig, ax = plt.subplots()
    ax.set_xlabel('i (deg)', fontsize=fontsize)
    ax.set_ylabel(r'$\xi_p / \xi_b$', fontsize=fontsize)
    ax.set_xlim([0, 90])
    ax.set_ylim([-0.5, 4])

    for model, table in tables.items():
        inc = table['col1']
        inv_xi_b = table['col2'] + table['col3']
        inv_xi_p = table['col4']
        xi_ratio = inv_xi_b / inv_xi_p
        ax.plot(inc, xi_ratio, label=model)

    ax.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.show(block=False)
