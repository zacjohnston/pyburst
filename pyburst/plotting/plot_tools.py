import matplotlib.pyplot as plt


def default_plt_options():
    """Initialise default plot parameters"""
    params = {'mathtext.default': 'regular',
              'font.family': 'serif',
              'text.usetex': False}
    plt.rcParams.update(params)


default_plt_options()


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


def unit_scale(quantity):
    """Returns typical scale factor for given quantity
    """
    scales = {
        'rate': 1.0,
        'dt': 1.0,
        'fluence': 1e39,
        'peak': 1e38,
    }
    return scales.get(quantity, 1.0)


def unit_label(quantity):
    """Returns units as a string, for given quantity
    """
    labels = {
        'rate': 'day$^{-1}$',
        'dt': 'h',
        'fluence': '$10^{39}$ erg',
        'peak': '$10^{38}$ erg s$^{-1}$',
    }
    return labels.get(quantity, '')


def label(quantity):
    """Returns string of MCMC parameter label
    """
    labels = {
        'x': r'$X_0$',
        'z': r'$Z_\mathrm{CNO}$',
        'redshift': r'$(1+z)$',
        'd_b': r'$d \sqrt{\xi_\mathrm{b}}$',
        'xi_ratio': r'$\xi_\mathrm{p} / \xi_\mathrm{b}$',
        'm_gr': r'$M_\mathrm{GR}$',
        'm_nw': r'$M_\mathrm{NW}$',
        'xedd_ratio': r'$X_\mathrm{Edd} / X_0$',
        'xedd': r'$X_\mathrm{Edd}$',
    }

    # assumes last character is a single integer specifying epoch
    if 'qb' in quantity:
        return r'$Q_\mathrm{b' + f'{quantity[-1]}' + '}$'
    elif 'mdot' in quantity:
        return rf'$\dot{{M}}_{quantity[-1]}$'
    else:
        return labels.get(quantity, quantity)


def convert_mcmc_labels(param_keys):
    """Returns sequence of formatted parameter labels
    """
    keys = list(param_keys)
    for i, key in enumerate(keys):
        keys[i] = label(key)
    return keys
