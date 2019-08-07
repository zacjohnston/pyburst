import numpy as np
from astropy import units
import astropy.constants as const

# pyburst
from pyburst.physics import gravity

def eddington_lum_newtonian(mass, x):
    """Returns the spherical Eddington luminosity for a given solar mass and composition
       (erg / s)

    Parameters
    ----------
    mass : flt
        solar mass (M_sun)
    x : flt
        hydrogen composition (mass fraction)
    """
    # Starts with Eddington limit for pure hydrogen
    l_edd = 4*np.pi * const.G * (mass*units.M_sun) * const.m_p * const.c / const.sigma_T
    l_edd = l_edd.to(units.erg / units.s).value   # cgs units
    l_edd = l_edd * 2 / (x + 1)  # correct for hydrogen/helium ratio

    return l_edd


def eddington_lum_gr(mass, radius, x):
    """Returns the spherical Eddington luminosity for a given solar mass and composition
       (erg / s)

    Parameters
    ----------
    mass : flt
        GR neutron star mass (M_sun)
    radius : flt
        GR neutron star radius (km)
    x : flt
        hydrogen composition (mass fraction)
    """
    redshift = gravity.get_redshift(r=radius, m=mass)

    # Start with Eddington limit for pure hydrogen
    l_edd = 4*np.pi * const.G * const.m_p * const.c * redshift * (mass*units.M_sun) / const.sigma_T
    l_edd = l_edd.to(units.erg / units.s).value   # cgs units
    l_edd = l_edd * 2 / (x + 1)  # correct for hydrogen/helium ratio

    return l_edd
