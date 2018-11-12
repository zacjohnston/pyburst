import numpy as np
import matplotlib.pyplot as plt

def set_axes(ax, title='', xlabel='', ylabel='', yscale='linear', xscale='linear',
             fontsize=14, yticks=True, xticks=True):
    """Standardised formatting of labels etc.
    """
    if not yticks:
        ax.axes.tick_params(axis='both', left='off', labelleft='off')
    if not xticks:
        ax.axes.tick_params(axis='both', bottom='off', labelbottom='off')

    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)


def units_scale(quantity):
    """Returns typical scale factor for given quantity
    """
    scales = {
        'rate': 1.0,
        'dt': 1.0,
        'fluence': 1e39,
        'peak': 1e38,
    }
    return scales.get(quantity, 1.0)
