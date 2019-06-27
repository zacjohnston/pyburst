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
    return pd.read_csv(filepath, delim_whitespace=True)


def save_summary(table, source):
    """Saves summary table to file

    parameters
    ----------
    table : pandas.DataFrame
    source : str
    """
    filepath = obs_strings.summary_filepath(source)
    write_pandas_table(table, filepath=filepath)


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


def add_peak_length(source):
    """Calculcates peak length from lightcurve and adds to summary file
    """
    # TODO: finish
    # table = load_summary(source)
    #
    # for epoch in table['epoch']:
    #     lcurve = load_epoch_lightcurve(epoch=epoch, source=source)
    #     x = np.linspace()
    pass


def add_tail_timescales(source, percent=50):
    """Calculates decay tail timescales from observed lightcurves, and adds to summary
    """
    # TODO: needs refining, get correct uncertainty statistics
    table = load_summary(source)

    for epoch in table.itertuples():
        lcurve = load_epoch_lightcurve(epoch=epoch.epoch, source=source)
        timescale, u_timescale = get_tail_timescale(lcurve, epoch_row=epoch,
                                                    frac=percent/100)
        table.loc[epoch.Index, f'tail_{percent}'] = timescale
        table.loc[epoch.Index, f'u_tail_{percent}'] = u_timescale

    save_summary(table, source=source)
    return table


def get_tail_timescale(lc_table, epoch_row, frac=0.5):
    """Returns tail timescales for given lightcurve table
    """
    # TODO: Very rough at the moment
    lc = extract_lightcurve_array(lc_table)
    frac_lum = epoch_row.peak * frac
    peak_idx = np.argmax(lc[:, 1])

    lc = lc[peak_idx:]
    time_since_peak = lc[:, 0] - lc[0, 0]

    mask = lc[:, 1] < frac_lum
    mask_left = (lc[:, 1] - lc[:, 2]) < frac_lum
    mask_right = (lc[:, 1] + lc[:, 2]) < frac_lum

    time_diff = time_since_peak[mask][0]
    u_time_diff = (time_since_peak[mask_right][0] - time_since_peak[mask_left][0])/2

    return time_diff, u_time_diff


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
