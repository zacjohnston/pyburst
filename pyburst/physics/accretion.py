import numpy as np
from astropy import units
import astropy.constants as const

def eddington_lum(mass, x):
    """Returns the spherical Eddington luminosity for a given solar mass and composition
       (erg / s)

    Parameters
    ----------
    mass : flt
        solar mass (M_sun)
    x : flt
        hydrogen composition (mass fraction)
    """
    # TODO: This is Newtonian. Need GR version
    l_edd = 4*np.pi * const.G * (mass*units.M_sun) * const.m_p * const.c / const.sigma_T
    l_edd = l_edd.to(units.erg / units.s).value   # cgs units
    l_edd = l_edd * 2 / (x + 1)  # correct for hydrogen/helium ratio

    return l_edd
