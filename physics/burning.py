import numpy as np
import astropy.units as units


def predict_qnuc(accrate, x0, z, dt, radius=10):
    xbar = get_xbar(accrate, x0=x0, z=z, dt=dt, radius=radius)
    return adelle_qnuc(xbar)


def adelle_qnuc(xbar):
    """Returns Qnuc (MeV/nucleon) according to Adelle et al. (2018)
    """
    return 1.305 + (6.9511 * xbar) - (1.9218 * xbar**2)


def get_xbar(accrate, x0, z, dt, radius=10):
    """Returns average hydrogen mass fraction (Xbar) at burst ignition
    """
    x_ignition = get_x_ignition(accrate, x0=x0, z=z, dt=dt, radius=radius)
    return 0.5 * (x0 + x_ignition)


def get_x_ignition(accrate, x0, z, dt, radius=10):
    """Returns hydrogen fraction at ignition depth
    """
    y_ignition = get_y_ignition(dt, accrate, radius=radius)
    y_depletion = get_y_depletion(accrate, x0, z)
    return x0 * (1 - y_ignition/y_depletion)


def get_y_depletion(accrate, x0, z):
    """Returns column density (y, g/cm^2) of hydrogen depletion

    Equation: Cumming & Bildsten (2000)
    """
    return 6.8e8 * (accrate/0.1) * (0.01/z) * (x0/0.71)


def get_y_ignition(dt, accrate, radius=10):
    """Returns column density (y, g/cm^2) at ignition depth

    dt : float
        recurrence time (s)
    accrate : float
        accretion rate (fraction of Eddington rate)
    radius : float
        radius of neutron star (km)
    """
    r_cm = radius * units.km.to(units.cm)
    accrate_gram_sec = convert_accrate(accrate)
    return (dt * accrate_gram_sec) / (4 * np.pi * r_cm**2)


def convert_accrate(accrate):
    """Returns accrate in g/s, when given as Eddington fraction
    """
    accrate_edd = 1.75e-8 * units.M_sun.to(units.g) / units.year.to(units.s)
    return accrate * accrate_edd
