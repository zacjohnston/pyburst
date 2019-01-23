import numpy as np

from pyburst.grids import grid_analyser
from pyburst.physics import gravity


def add_alpha(kgrid):
    """Adds alpha column to given Kgrid

    kgrid : grid_analyser.Kgrid
        grid object containing model data
    """
    pass


def add_redshift(kgrid, m_ratio=1.0, radius=None):
    """Adds redshift (1+z) column to given Kgrid

    kgrid : grid_analyser.Kgrid
        grid object containing model data
    m_ratio : flt
        mass ratio, M_gr / M_newton
    radius : flt | array
        newtonian radius of models
    """
    if radius is None:
        print('Using default radius=10km')
        radius = 10

    masses = np.array(kgrid.params.mass)
    r_ratios, redshifts = gravity.gr_corrections(r=radius, m=masses, phi=m_ratio)
    kgrid.params['redshift'] = redshifts


def add_phi(kgrid, m_ratio=1.0, radius=None):
    """Adds phi (gravitational potential) column to given Kgrid

    kgrid : grid_analyser.Kgrid
        grid object containing model data
    m_ratio : flt
        mass ratio, M_gr / M_newton
    """

