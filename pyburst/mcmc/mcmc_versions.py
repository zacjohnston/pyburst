import numpy as np
from scipy.stats import norm

# kepler_grids
from pyburst.grids import grid_strings

# -----------------------------------
# Define different grid boundaries.
# Must be in same order as 'params' in BurstFit
# '***' signifies values that changed over the previous version
# First layer identifies param_keys
# -----------------------------------
# NOTE: in self.bprops, interp_bprops ALWAYS come before analytic_bprops
# -----------------------------------

# ===== Define order/number of params provided to BurstFit =====
param_keys = {
    1: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'qb1', 'qb2', 'qb3', 'm_nw', 'm_gr', 'd_b', 'xi_ratio'],
    2: ['mdot1', 'x', 'z', 'qb1', 'm_nw', 'm_gr', 'd_b', 'xi_ratio'],
    3: ['mdot1', 'mdot2', 'qb1', 'qb2', 'm_nw', 'm_gr', 'd_b', 'xi_ratio'],
    4: ['mdot1', 'qb1', 'm_nw', 'm_gr', 'd_b', 'xi_ratio'],
}

# ===== Define order/number of params for a single interpolated point =====
interp_keys = {
    1: ['mdot', 'x', 'z', 'qb', 'mass'],
    2: ['mdot', 'x', 'z', 'mass'],
    3: ['mdot', 'x', 'qb', 'mass'],
    4: ['mdot', 'qb', 'mass'],
}

# ===== Define params that are unique for each epoch =====
epoch_unique = {
    1: ['mdot'],
    2: ['mdot', 'qb'],
}

# ===== Define alias from interp-->param keys =====
param_aliases = {
    1: {'mass': 'm_nw'},
}

grid_bounds = {
    1: {
        1: ((0.07, 0.18),  # mdot1
            (0.07, 0.18),  # mdot2
            (0.07, 0.18),  # mdot3
            (0.64, 0.73),  # x
            (0.001, 0.015),  # z
            (0.0, 0.6),  # qb1
            (0.0, 0.6),  # qb2
            (0.0, 0.6),  # qb3
            (1.4, 2.3),  # m_nw
            (1.0, 3.0),  # m_gr
            (1., 15.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
    },

    2: {
        1: ((0.07, 0.18),  # mdot1
            (0.64, 0.73),  # x
            (0.001, 0.015),  # z
            (0.0, 0.6),  # qb1
            (1.4, 2.3),  # m_nw
            (1.0, 3.0),  # m_gr
            (1., 15.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
    },

    3: {
        1: ((0.2, 0.5),  # mdot1
            (0.2, 0.5),  # mdot2
            (0.0, 0.35),  # qb1
            (0.0, 0.35),  # qb2
            (1.2, 2.0),  # m_nw
            (1.0, 2.1),  # m_gr
            (1., 20.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
        2: ((0.175, 0.4),  # mdot1
            (0.175, 0.4),  # mdot2
            (0.05, 0.4),  # qb1
            (0.05, 0.4),  # qb2
            (1.2, 2.0),  # m_nw
            (1.0, 2.1),  # m_gr
            (1., 20.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
        3: ((0.2, 0.5),  # mdot1
            (0.2, 0.5),  # mdot2
            (0.0, 0.4),  # qb1
            (0.0, 0.4),  # qb2
            (1.2, 2.0),  # m_nw
            (1.0, 3.0),  # m_gr
            (1., 20.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
    },

    4: {
        1: ((0.2, 0.5),  # mdot1
            (0.0, 0.35),  # qb1
            (1.2, 2.0),  # m_nw
            (1.0, 2.1),  # m_gr
            (1., 20.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
        2: ((0.175, 0.4),  # mdot1
            (0.05, 0.4),  # qb1
            (1.2, 2.0),  # m_nw
            (1.0, 2.1),  # m_gr
            (1., 20.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
    },
}

# ===== Define prior pdfs for parameters =====
def flat_prior(x):
    return 1


log_norm = norm(loc=-0.5, scale=0.25).pdf
log_norm2 = norm(loc=-0.1, scale=0.5).pdf

def log_z(z, z_sun=0.01):
    """PDF of log10(z/z_solar)"""
    logz = np.log10(z / z_sun)
    return log_norm(logz)

def log_z2(z, z_sun=0.01):
    """PDF of log10(z/z_solar)"""
    logz = np.log10(z / z_sun)
    return log_norm2(logz)

def gaussian(mean, std):
    """Returns function for Gaussian distribution
    """
    return norm(loc=mean, scale=std).pdf


priors = {
    'z': {
        1: log_z,  # log10-space [z/solar]
    },
    'd_b': {},
    'm_gr': {}
}


# ===== initial position of walkers =====

initial_position = {
    1: {
        1: (0.082, 0.11, 0.13, 0.69, 0.007, 0.15, 0.05, 0.05, 1.7, 2.1, 7.2, 1.6),
    },
    2: {
        1: (0.085, 0.69, 0.006, 0.15, 1.7, 2.0, 7.3, 1.6),
        2: (0.11,  0.69, 0.006, 0.05, 1.7, 2.0, 7.3, 1.6),
        3: (0.13,  0.69, 0.006, 0.05, 1.7, 2.0, 7.3, 1.6),
    },
    3: {
        1: (0.3, 0.42, 0.11, 0.06, 1.6, 2.0, 7.7, 1.4),
    },
    4: {
        1: (0.32, 0.07, 1.76, 2.0, 7.7, 1.6),
        2: (0.42, 0.055, 1.6, 2.0, 7.7, 1.4),
    },
}
# To add a new version definition, add an entry to each of the parameters
#   in version_definitions

# Structure:
#   -parameter
#     ---source
#         ---version
#               ---parameter value

# TODO: reform into tables (saved as files), and a function to add versions (rows)
source_defaults = {
    'interpolator': {
        'grid5': 0,
        'synth5': 1,
        'he2': 0,
    },

    'param_keys': {
        'grid5': param_keys[1],
        'synth5': param_keys[1],
        'he2': param_keys[3],
    },

    'interp_keys': {
        'grid5': interp_keys[1],
        'synth5': interp_keys[1],
        'he2': interp_keys[4],
    },

    'epoch_unique': {
        'grid5': epoch_unique[2],
        'synth5': epoch_unique[2],
        'he2': epoch_unique[2],
    },

    'epoch': {
        'grid5': None,
        'synth5': None,
        'he2': None,
    },

    'param_aliases': {
        'grid5': param_aliases[1],
        'synth5': param_aliases[1],
        'he2': param_aliases[1],
    },
    # TODO: rename interp_bprop_keys, etc.
    'interp_bprops': {
        'grid5': ('rate', 'fluence', 'peak'),
        'synth5': ('rate', 'fluence', 'peak'),
        'he2': ('rate', 'fluence'),
    },

    'analytic_bprops': {
        'grid5': ('fper', 'fedd'),
        'synth5': ('fper',),
        'he2': ('fper', 'fedd'),
    },

    'weights': {
        'grid5': {'rate': 1.0, 'fluence': 1.0, 'peak': 1.0, 'fper': 1.0, 'fedd': 1.0},
        'synth5': {'rate': 1.0, 'fluence': 1.0, 'peak': 1.0, 'fper': 1.0},
        'he2': {'rate': 1.0, 'fluence': 1.0, 'fper': 1.0, 'fedd': 1.0},
    },

    'disc_model': {},

    'grid_bounds': {
        'grid5': grid_bounds[1][1],
        'synth5': grid_bounds[1][1],
        'he2': grid_bounds[3][1],
    },

    'priors': {  # if not defined here, the default/fallback will be flat_prior()
        'grid5': {
            'z': log_z2,
        },
        'synth5': {},
        'he2': {
            'd_b': gaussian(mean=7.846, std=0.333),
        },
    },

    'initial_position': {
        'grid5': initial_position[1][1],
        'synth5': initial_position[1][1],
        'he2': initial_position[3][1],
    },

    'x_edd_option': {
      'grid5': 'x_0',
      'synth5': None,
      'he2': 0.0,
    },

    'synthetic': {  # whether the data being matches is synthetic
        'grid5': False,
        'synth5': True,
        'he2': False,
    },

    # ===== Special definitions for synthetic data sources =====
    'interp_source': {  # Source interpolator to use
        'synth5': 'grid5',
    },

    'synth_version': {  # Version ID of synthetic data table
        'synth5': None,
    },

    'synth_group': {
        'synth5': None,
    },
}

# Summary
# ------------------------------
# grid5:
#   12 : as 2, with bprop fedd
#   13 : as 12, epoch 1998
#   14 : as 12, epoch 2000
#   15 : as 12, epoch 2007

#   23 : as 12, z-prior (-0.1 +/- 0.5)
#   24 : as 23, epoch 1998
#   25 : as 23, epoch 2000
#   26 : as 23, epoch 2007

# ------------------------------
# he2
#   1 : default grid
#   2 : as 1, epoch 1997
#   3 : as 1, epoch 2009

#   4 : as 1, qnuc=1.0
#   5 : as 4, epoch 1997
#   6 : as 4, epoch 2009

#   7 : as 1, m_gr=3.0


version_definitions = {
    'interpolator': {
        'grid5': {},
        'synth5': {},
        'he2': {
            4: 1,
            5: 1,
            6: 1,
        },
    },

    'interp_bprops': {
        'grid5': {},
        'synth5': {},
        'he2': {},
    },

    'analytic_bprops': {
        'grid5': {},
        'synth5': {},
        'he2': {},
    },

    'weights': {
        'grid5': {},
        'synth5': {},
        'he2': {},

    },

    'param_keys': {
        'grid5': {
            2: param_keys[2],
            3: 2,
            4: 2,
            24: 2,
            25: 2,
            26: 2,
        },
        'synth5': {},
        'he2': {
            2: param_keys[4],
            3: 2,
            5: 2,
            6: 2,
        },
    },

    'interp_keys': {
        'grid5': {},
        'synth5': {},
        'he2': {},
    },

    'epoch_unique': {
        'grid5': {},
        'synth5': {},
        'he2': {},
    },

    'epoch': {
        'grid5': {
            2: 1998,
            3: 2000,
            4: 2007,
            24: 1998,
            25: 2000,
            26: 2007,
        },
        'synth5': {},
        'he2': {
            2: 1997,
            3: 2009,
            5: 1997,
            6: 2009,
        },
    },

    'param_aliases': {
        'grid5': {},
        'synth5': {},
        'he2': {},
    },

    'grid_bounds': {
        'grid5': {
            2: grid_bounds[2][1],
            3: 2,
            4: 2,
            24: 2,
            25: 2,
            26: 2,
        },
        'synth5': {},
        'he2': {
            2: grid_bounds[4][1],
            3: 2,
            4: grid_bounds[3][2],
            5: grid_bounds[4][2],
            6: 5,
            7: grid_bounds[3][3],
        },
    },

    'priors': {
         'grid5': {},
         'synth5': {},
         'he2': {},
    },

    'initial_position': {
        'grid5': {
            2: initial_position[2][1],
            3: initial_position[2][2],
            4: initial_position[2][3],
            24: 2,
            25: 3,
            26: 4,
        },
        'synth5': {},
        'he2': {
            2: initial_position[4][1],
            3: initial_position[4][2],
            5: 2,
            6: 3,
        },
    },

    'x_edd_option': {
        'grid5': {},
        'synth5': {},
        'he2': {},
    },

    'disc_model': {},

    # ===== Special definitions for synthetic data sources =====
    'interp_source': {
        'synth5': {},
    },

    'synth_version': {
        'synth5': {},
    },

    'synth_group': {
        'synth5': {},
    },
}


class McmcVersion:
    """Class for holding different mcmc versions
    """

    def __init__(self, source, version, verbose=False):
        source = grid_strings.check_synth_source(source)

        if source not in source_defaults['param_keys']:
            raise ValueError(f'source ({source}) not defined in mcmc_versions')

        self.source = source
        self.version = version
        self.verbose = verbose
        self.param_keys = self.get_parameter('param_keys')
        self.interp_keys = self.get_parameter('interp_keys')
        self.epoch_unique = self.get_parameter('epoch_unique')
        self.epoch = self.get_parameter('epoch')
        self.param_aliases = self.get_parameter('param_aliases')
        self.interp_bprops = self.get_parameter('interp_bprops')
        self.analytic_bprops = self.get_parameter('analytic_bprops')
        self.bprops = self.interp_bprops + self.analytic_bprops
        self.weights = self.get_parameter('weights')
        self.interpolator = self.get_parameter('interpolator')
        self.grid_bounds = np.array(self.get_parameter('grid_bounds'))
        self.initial_position = self.get_parameter('initial_position')
        self.x_edd_option = self.get_parameter('x_edd_option')
        self.priors = self.get_priors()
        self.synthetic = source_defaults['synthetic'][source]
        self.disc_model = None

        if self.synthetic:
            self.setup_synthetic()
        else:
            self.interp_source = None
            self.synth_version = None
            self.synth_group = None

    def __repr__(self):
        if self.synthetic:
            synth_str = (f'\ninterp source    : {self.interp_source}'
                         + f'\nsynth version    : {self.synth_version}'
                         + f'\nsynth group      : {self.synth_group}')
        else:
            synth_str = ''

        grid_bound_str = '\ngrid bounds      :'
        for i, param in enumerate(self.param_keys):
            bounds = self.grid_bounds[i]
            grid_bound_str += f'\n  {param}\t({bounds[0]:.3f}, {bounds[1]:.3f})'

        priors_str = '\nprior pdfs       :'
        for param, prior in self.priors.items():
            priors_str += f'\n  {param}   \t{prior}'

        return (f'MCMC version definitions for {self.source} V{self.version}'
                + f'\nparam keys       : {self.param_keys}'
                + f'\ninterp keys      : {self.interp_keys}'
                + f'\nepoch            : {self.epoch}'
                + f'\nepoch unique     : {self.epoch_unique}'
                + f'\nparam aliases    : {self.param_aliases}'
                + f'\nbprops           : {self.bprops}'
                + f'\ninterp_bprops    : {self.interp_bprops}'
                + f'\nanalytic_bprops  : {self.analytic_bprops}'
                + f'\nweights          : {self.weights}'
                + f'\ninitial position : {self.initial_position}'
                + f'\ndisc model       : {self.disc_model}'
                + f'\ninterpolator     : {self.interpolator}'
                + f'\nx_edd option     : {self.x_edd_option}'
                + f'{grid_bound_str}'
                + f'{priors_str}'
                + f'\nsynthetic        : {self.synthetic}'
                + synth_str
                )

    def setup_synthetic(self):
        self.interp_source = self.get_parameter('interp_source')
        self.synth_version = self.get_parameter('synth_version')
        self.synth_group = self.get_parameter('synth_group')

        # Attempts to get default synth_version/synth_group from version ID
        #   1. Uses last digit for synth_group (note: 0 interpreted as 10)
        #   2. Uses leading digits +1 for synth_version
        #   e.g., version=24 becomes synth_version=2+1=2, synth_group=4
        #   note: this means group 10 of synth_version 2 corresponds to self.version 20
        v_str = f'{self.version:03d}'

        if self.synth_version is None:
            default_version = 1 + int(v_str[:2])    # use leading digits
            print(f'synth_version undefined fro {self.source} V{self.version}, '
                  f'defaulting to {default_version}')
            self.synth_version = default_version

        if self.synth_group is None:
            default_group = int(v_str[-1])

            if default_group is 0:
                default_group = 10

            print(f'synth_group undefined for {self.source} V{self.version}, '
                  f'defaulting to {default_group}')
            self.synth_group = default_group

    def get_parameter(self, parameter):
        return get_parameter(self.source, self.version, parameter, verbose=self.verbose)

    def get_priors(self):
        return get_priors(self.source, self.version)


def get_parameter(source, version, parameter, verbose=False):
    source = grid_strings.check_synth_source(source)
    default = source_defaults[parameter][source]
    output = version_definitions[parameter][source].get(version, default)

    if verbose and output is default:
        print(f"mcmc_versions: '{parameter}' not specified. Using default values")

    if parameter not in ('interpolator', 'synth_version', 'synth_group', 'epoch') \
            and type(output) is int:
        return version_definitions[parameter][source][output]
    else:
        return output


def get_priors(source, version):
    params = get_parameter(source=source, version=version, parameter='param_keys')
    pdfs = {}
    for param in params:
        default = source_defaults['priors'][source].get(param, flat_prior)
        v_definition = version_definitions['priors'][source].get(version)

        if v_definition is None:
            value = default
        else:
            value = v_definition.get(param, default)

        if type(value) is int:  # allow pointing to previous versions
            pdfs[param] = version_definitions['priors'][source].get(value, default)
        else:
            pdfs[param] = value

    return pdfs
