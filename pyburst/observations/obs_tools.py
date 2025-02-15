import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


# pyburst
from . import obs_strings
from pyburst.grids.grid_tools import write_pandas_table

# TODO:
#   - load epoch
#   - add rate column
#   - add length column


def load_summary(source):
    """Loads summary of observed data

    parameters
    ----------
    source : str
    """
    filepath = obs_strings.summary_filepath(source)
    return pd.read_table(filepath, delim_whitespace=True)


# TODO: WARNING
#   LOSES PRECISION ON FLUENCE DUE TO STRING REPRESENTATION
# def save_summary(table, source):
#     """Saves summary table to file
#
#     parameters
#     ----------
#     table : pandas.DataFrame
#     source : str
#     """
#     filepath = obs_strings.summary_filepath(source)
#     write_pandas_table(table, filepath=filepath)


def load_epoch_lightcurve(epoch, source):
    """Returns table of epoch lightcurve data

    parameters
    ----------
    epoch : int
    source : str
    """
    columns = ('time', 'dt', 'flux', 'u_flux', 'kt', 'u_kt', 'normal', 'u_normal', 'chi2')
    factors = {'flux': 1e-9, 'u_flux': 1e-9}

    filepath = obs_strings.epoch_filepath(epoch=epoch, source=source)
    table = pd.read_csv(filepath, delim_whitespace=True, comment='#', header=None,
                        names=columns)

    for key, item in factors.items():
        table[key] *= item
    return table


def add_power_laws(source):
    """Calculates power law fits to lightcurve tails, and saves to summary table
    """
    x_idxs = {
        '4u1820': {1997: 17, 2009: 9},   # lightcurve point to start fitting from
        'gs1826': {1998: 60, 2000: 32, 2007: 33}
    }.get(source)
    summ_table = load_summary(source)

    for epoch in summ_table.itertuples():
        lc_table = load_epoch_lightcurve(epoch=epoch.epoch, source=source)
        x_idx = x_idxs.get(epoch.epoch)
        power_fit, u_power_fit = fit_power_law(lc_table, x_idx=x_idx)
        print(power_fit, u_power_fit)
        summ_table.loc[epoch.Index, 'tail_index'] = power_fit[0]
        summ_table.loc[epoch.Index, f'u_tail_index'] = u_power_fit[0]

    save_summary(summ_table, source=source)
    return summ_table

def fit_power_law(lc_table, x_idx, yscale=1e-8, p0=(-1, 5, 0)):
    """Fits power law to lightcurve tail

    x_idx : int
        index of lightcurve point to begin fitting from
    """

    time = lc_table.time + 0.5 * lc_table.dt
    flux = lc_table.flux / yscale
    u_flux = lc_table.u_flux / yscale

    pfit, pcov = curve_fit(func_powerlaw, time[x_idx:], flux[x_idx:],
                           p0=p0, sigma=u_flux[x_idx:])
    u_pfit = np.sqrt(np.diag(pcov))
    return pfit, u_pfit


def func_powerlaw(x, a, b, c):
    return c + b * x ** a


def get_peak_length(lc_table, peak_frac=0.75):
    """Returns peak length for given lightcurve table
    """
    lc_filled = fill_lightcurve(lc_table)
    peak = np.max(lc_table.flux)
    mask = lc_filled[:, 1] > peak_frac*peak
    time_slice = lc_filled[mask, 0]

    return time_slice[-1] - time_slice[0]


def fill_lightcurve(lc_table, n_x=1000):
    """Returns lightcurve [time, flux] with higher time-sampling (interpolated)
    """
    lc_array = extract_lightcurve_array(lc_table)
    interp = interpolate_lightcurve(lc_array=lc_array)

    t0 = lc_array[0, 0]
    t1 = lc_array[-1, 0]
    x = np.linspace(t0, t1, n_x)
    y = interp(x)

    return np.stack([x, y], axis=1)


def extract_lightcurve_array(lc_table):
    """Returns simple LC array with [time, flux, u_flux] columns
    """
    x = np.array(lc_table['time'] + 0.5 * lc_table['dt'])
    y = np.array(lc_table['flux'])
    u_y = np.array(lc_table['u_flux'])
    return np.stack([x, y, u_y], axis=1)


def interpolate_lightcurve(lc_table=None, lc_array=None):
    """Returns linear interpolator of epoch lightcurve
    """
    if lc_array is None:
        if lc_table is None:
            raise ValueError('Must provide one of [lc_table, lc_array]')
        else:
            lc_array = extract_lightcurve_array(lc_table)

    return interp1d(x=lc_array[:, 0], y=lc_array[:, 1])
