import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import astropy.constants as const
from scipy.optimize import brentq

# kepler_grids
from pyburst.misc.pyprint import print_title, print_dashes

# Constants in cgs units
G = const.G.to(u.cm**3/(u.g*u.s**2))
c = const.c.to(u.cm/u.s)
Msun = const.M_sun.to(u.g)

# TODO: allow flexibility with parsing units, e.g. check_units()
# TODO: inverse redshift


def apply_units(r, m):
    """Return radius and mass with applied units

    Assumes radius in km, mass in Msun
    """
    return r*1e5*u.cm, m*Msun


def get_redshift(r, m):
    """Returns redshift (1+z) for given radius and mass (assuming GR)
    """
    zeta = get_zeta(r=r, m=m)
    return 1 / np.sqrt(1 - 2*zeta)


def get_zeta(r, m):
    """Returns zeta factor (GM/Rc^2) for given radius and mass
    """
    R, M = apply_units(r=r, m=m)
    zeta = (G * M) / (R * c**2)

    if True in zeta >= 0.5:
        raise ValueError(f'R, M ({r:.2f}, {m:.2f}) returns zeta >= 0.5')

    return np.array(zeta)


def get_mass_radius(g, redshift):
    """Return GR mass and radius for given gravity and redshift

    g : gravitational acceleration
    redshift : (1+z) redshift factor
    """
    red2 = redshift**2
    red2m1 = red2 - 1

    R = (c**2 / (2 * g)) * (red2m1 / redshift)
    M = (g * R**2) / (G * redshift)

    return M.to(u.M_sun), R.to(u.km)


def get_accelerations(r, m):
    """Returns both gravitational accelerations (Newtonian, GR), given R and M
    """
    g_newton = get_acceleration_newtonian(r=r, m=m)
    g_gr = get_acceleration_gr(r=r, m=m)
    return g_newton, g_gr


def get_acceleration_newtonian(r, m):
    """Returns gravitational accelerations (Newtonian), given R and M
    """
    R, M = apply_units(r=r, m=m)
    g_newton = G*M/R**2
    return g_newton


def get_acceleration_gr(r, m):
    """Returns gravitational accelerations (GR), given R and M
    """
    redshift = get_redshift(r=r, m=m)
    g_newton = get_acceleration_newtonian(r=r, m=m)
    g_gr = g_newton * redshift
    return g_gr


def inverse_acceleration(g, m=None, r=None):
    """Returns R or M, given g and one of R or M
    """
    def root(r, m, g):
        return get_acceleration_gr(r=r, m=m).value - g.value

    if (m == None) and (r == None):
        print('ERROR: need to specify one of m or r')
    if (m != None) and (r != None):
        print('Error: can only specify one of m or r')

    g *= 1e14 * u.cm/u.s/u.s

    if r == None:
        r = brentq(root, 6, 50, args=(m, g))
        return r

def plot_g():
    """Plots g=constant curves against R, M
    """
    # g_li3.2/st = [1.0, 1.5, 2.0, 3.0]
    g_list = [1.06, 1.33, 21.858, 2.66, 3.45, 4.25]
    m_list = np.linspace(1, 2, 50)
    r_list = np.zeros(50)

    fig, ax = plt.subplots()

    for g in g_list:
        for i, m in enumerate(m_list):
            r_list[i] = inverse_acceleration(g=g, m=m)

        ax.plot(m_list, r_list, label=f'{g:.2f}')

    ax.set_xlabel('Mass (Msun)')
    ax.set_ylabel('Radius (km)')
    ax.legend()
    plt.show(block=False)


def gr_corrections(r, m, phi=1.0, verbose=False):
    """Returns GR correction factors given R, M (Eq. B5, Keek & Heger 2007)

    parameters
    ----------
    m : flt
        Newtonian mass (Msol) (i.e. Kepler frame)
    r   : flt
        Newtonian radius (km)
    phi : flt
        Ratio of GR mass to Newtonian mass (NOTE: unrelated to grav potential phi)
    """
    zeta = get_zeta(r=r, m=m)

    b = (9*zeta**2*phi**4 + np.sqrt(3)*phi**3 * np.sqrt(16 + 27*zeta**4 * phi**2))**(1/3)
    a = (2/9)**(1/3) * (b**2 / phi**2 - 2 * 6**(1/3)) / (b * zeta**2)
    xi = (zeta * phi/2) * (1 + np.sqrt(1 - a) + np.sqrt(2 + a + 2 / np.sqrt(1 - a)))

    redshift = xi**2/phi    # NOTE: xi unrelated to anisotropy factor

    if verbose:
        print_title(f'Using R={r:.3f} and M={m}:')
        print_dashes()
        print(f'    R_GR = {r*xi:.2f} km')
        print(f'(1+z)_GR = {redshift:.3f}')
        print_dashes()

    return xi, redshift


def get_potential_newtonian(r, m):
    """Returns gravitational potentials (phi) given R and M (Newton)
    """
    R, M = apply_units(r=r, m=m)
    phi_newton = -G*M/R
    return phi_newton


def get_potential_gr(r, m):
    """Returns gravitational potentials (phi) given R and M (GR)
    """
    redshift = get_redshift(r=r, m=m)
    phi_gr = -(redshift-1)*c**2 / redshift
    return phi_gr


def get_potentials(r, m):
    """Returns both gravitational potentials (phi) given R and M (Newtonian, GR)
    """
    phi_newton = get_potential_newtonian(r=r, m=m)
    phi_gr = get_potential_gr(r=r, m=m)
    return phi_newton, phi_gr


def gravity_summary(r, m):
    """Prints summary gravitational properties given R, M
    """
    redshift = get_redshift(r=r, m=m)
    zeta = get_zeta(r=r, m=m)
    phi_newton, phi_gr = get_potentials(r=r, m=m)
    g_newton, g_gr = get_accelerations(r=r, m=m)

    print_dashes()
    print('R (km),  M (Msun)')
    print(f'{r:.2f},   {m:.2f}')

    print_dashes()
    print('g (Newtonian)')
    print(f'{g_newton:.3e}')

    print_dashes()
    print('g (GR)')
    print(f'{g_gr:.3e}')

    print_dashes()
    print('(1+z) (GR)')
    print(f'{redshift:.3f}')

    print_dashes()
    print('potential (Newtonian, erg/g)')
    print(f'{phi_newton:.3e}')

    print_dashes()
    print('potential (GR, erg/g)')
    print(f'{phi_gr:.3e}')


    return g_newton, g_gr, phi_newton, phi_gr
