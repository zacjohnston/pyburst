import numpy as np
from scipy.stats import norm, beta

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
    3: ['mdot1', 'mdot2', 'qb1', 'qb2', 'm_gr', 'd_b', 'xi_ratio'],
    4: ['mdot1', 'qb1', 'm_gr', 'd_b', 'xi_ratio'],
    5: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'qb1', 'qb2', 'qb3', 'm_gr', 'd_b', 'xi_ratio'],
    6: ['mdot1', 'x', 'z', 'qb1', 'm_gr', 'd_b', 'xi_ratio'],
    9: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'qb1', 'qb2', 'qb3', 'm_gr', 'd_b', 'xi_ratio', 'xedd_ratio'],
    10: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'qb1', 'qb2', 'qb3', 'm_nw', 'm_gr', 'd_b', 'xi_ratio', 'xedd_ratio'],
}

# ===== Define order/number of params for a single interpolated point =====
interp_keys = {
    1: ['mdot', 'x', 'z', 'qb', 'mass'],
    2: ['mdot', 'x', 'z', 'mass'],
    3: ['mdot', 'x', 'qb', 'mass'],
    4: ['mdot', 'qb', 'mass'],
    5: ['mdot', 'x', 'z', 'qb'],
    6: ['mdot', 'qb'],
}

# ===== Define params that are unique for each epoch =====
epoch_unique = {
    1: ['mdot', 'qb'],
}

# ===== Define alias from interp-->param keys =====
param_aliases = {
    1: {'mass': 'm_nw'},
}

grid_bounds = {
    1: {
        1: ((0.06, 0.18),  # mdot1
            (0.06, 0.18),  # mdot2
            (0.06, 0.18),  # mdot3
            (0.64, 0.76),  # x
            (0.0025, 0.03),  # z
            (0.0, 0.6),  # qb1
            (0.0, 0.6),  # qb2
            (0.0, 0.6),  # qb3
            (1.4, 2.6),  # m_nw
            (1.0, 2.2),  # m_gr
            (1., 15.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
        2: ((0.06, 0.18),  # mdot1
            (0.06, 0.18),  # mdot2
            (0.06, 0.18),  # mdot3
            (0.64, 0.76),  # x
            (0.0025, 0.03),  # z
            (0.0, 0.6),  # qb1
            (0.0, 0.6),  # qb2
            (0.0, 0.6),  # qb3
            (1.4, 2.6),  # m_nw
            (1.0, 10.),  # m_gr
            (1., 15.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
        3: ((0.07, 0.18),  # mdot1
            (0.07, 0.18),  # mdot2
            (0.07, 0.18),  # mdot3
            (0.64, 0.73),  # x
            (0.0025, 0.03),  # z
            (0.0, 0.6),  # qb1
            (0.0, 0.6),  # qb2
            (0.0, 0.6),  # qb3
            (1.7, 2.6),  # m_nw
            (1.0, 10.0),  # m_gr
            (1., 15.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
        4: ((0.07, 0.18),  # mdot1
            (0.07, 0.18),  # mdot2
            (0.07, 0.18),  # mdot3
            (0.64, 0.73),  # x
            (0.0025, 0.02),  # z
            (0.0, 0.6),  # qb1
            (0.0, 0.6),  # qb2
            (0.0, 0.6),  # qb3
            (1.7, 2.3),  # m_nw
            (1.0, 2.2),  # m_gr
            (1., 15.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
    },

    2: {
        1: ((0.06, 0.18),  # mdot1
            (0.64, 0.76),  # x
            (0.0025, 0.03),  # z
            (0.0, 0.6),  # qb1
            (1.4, 2.6),  # m_nw
            (1.0, 2.2),  # m_gr
            (1., 15.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
        2: ((0.07, 0.18),  # mdot1
            (0.64, 0.73),  # x
            (0.0025, 0.02),  # z
            (0.0, 0.6),  # qb1
            (1.4, 2.3),  # m_nw
            (1.0, 10.),  # m_gr
            (1., 15.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
    },

    # fixed mass (he2)
    3: {
        1: ((0.175, 0.475),  # mdot1
            (0.175, 0.475),  # mdot2
            (0.01, 0.5),  # qb1
            (0.01, 0.5),  # qb2
            (1.0, 3.5),  # m_gr
            (1., 20.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
        2: ((0.175, 0.475),  # mdot1
            (0.175, 0.475),  # mdot2
            (0.01, 0.5),  # qb1
            (0.01, 0.5),  # qb2
            (1.0, 2.2),  # m_gr
            (1., 20.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
    },

    # [epochs] fixed mass (he2)
    4: {
        1: ((0.175, 0.475),  # mdot1
            (0.01, 0.5),  # qb1
            (1.0, 3.5),  # m_gr
            (1., 20.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
        2: ((0.175, 0.475),  # mdot1
            (0.01, 0.5),  # qb1
            (1.0, 2.2),  # m_gr
            (1., 20.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
    },

    # Fixed mass (grid5)
    5: {
        1: ((0.07, 0.18),  # mdot1
            (0.07, 0.18),  # mdot2
            (0.07, 0.18),  # mdot3
            (0.64, 0.76),  # x
            (0.0025, 0.03),  # z
            (0.0, 0.6),  # qb1
            (0.0, 0.6),  # qb2
            (0.0, 0.6),  # qb3
            (1.0, 2.2),  # m_gr
            (1., 15.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
        2: ((0.07, 0.18),  # mdot1
            (0.07, 0.18),  # mdot2
            (0.07, 0.18),  # mdot3
            (0.64, 0.76),  # x
            (0.0025, 0.03),  # z
            (0.0, 0.6),  # qb1
            (0.0, 0.6),  # qb2
            (0.0, 0.6),  # qb3
            (1.0, 10.0),  # m_gr
            (1., 15.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
    },

    # Epoch fixed mass (grid5)
    6: {
        1: ((0.07, 0.18),  # mdot1
            (0.64, 0.76),  # x
            (0.0025, 0.03),  # z
            (0.0, 0.6),  # qb1
            (1.0, 2.2),  # m_gr
            (1., 15.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
    },

    # xedd_ratio (Fixed mass, grid5)
    9: {
        1: ((0.07, 0.18),  # mdot1
            (0.07, 0.18),  # mdot2
            (0.07, 0.18),  # mdot3
            (0.64, 0.76),  # x
            (0.0025, 0.03),  # z
            (0.0, 0.6),  # qb1
            (0.0, 0.6),  # qb2
            (0.0, 0.6),  # qb3
            (1.0, 2.2),  # m_gr
            (1., 15.),  # d_b
            (0.1, 10.),  # xi_ratio
            (0.0, 1.0),  # xedd_ratio
            ),
    },

    # xedd_ratio (varying mass, grid5)
    10: {
        1: ((0.07, 0.18),  # mdot1
            (0.07, 0.18),  # mdot2
            (0.07, 0.18),  # mdot3
            (0.64, 0.76),  # x
            (0.0025, 0.02),  # z
            (0.0, 0.6),  # qb1
            (0.0, 0.6),  # qb2
            (0.0, 0.6),  # qb3
            (1.4, 2.6),  # m_nw
            (1.0, 2.2),  # m_gr
            (1., 15.),  # d_b
            (0.1, 10.),  # xi_ratio
            (0.0, 1.0),  # xedd_ratio
            ),
    },
}

# ===== Define prior pdfs for parameters =====
def flat_prior(x):
    return 1


log_norm = norm(loc=-0.5, scale=0.25).pdf
log_norm2 = norm(loc=-0.1, scale=0.5).pdf
log_beta = beta(a=10.1, b=3.5, loc=-3.5, scale=4.5).pdf

def log_z(z, z_sun=0.01):
    """PDF of log10(z/z_solar)"""
    logz = np.log10(z / z_sun)
    return log_norm(logz)

def log_z2(z, z_sun=0.01):
    """PDF of log10(z/z_solar)"""
    logz = np.log10(z / z_sun)
    return log_norm2(logz)

def log_z_beta(z, z_sun=0.01):
    logz = np.log10(z / z_sun)
    return log_beta(logz)


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
        1: (0.086, 0.116, 0.133, 0.705, 0.011, 0.41, 0.20, 0.20, 2.45, 2.1, 6.56, 1.45),
        2: (0.086, 0.116, 0.133, 0.705, 0.011, 0.41, 0.20, 0.20, 2.45, 1.6, 6.56, 1.45),
        3: (0.086, 0.116, 0.133, 0.705, 0.011, 0.41, 0.20, 0.20, 2.45, 3.5, 6.56, 1.45),
        4: (0.085, 0.115, 0.135, 0.72, 0.009, 0.43, 0.22, 0.2, 2.3, 1.8, 6.3, 1.4),
    },
    2: {
        1: (0.083, 0.73, 0.011, 0.22, 2.05, 2.1, 6.5, 1.5),
        2: (0.10,  0.72, 0.012, 0.22, 1.95, 2.1, 6.5, 1.35),
        3: (0.11,  0.71, 0.014, 0.22, 1.8, 2.1, 6.4, 1.35),
    },
    3: {
        1: (0.20, 0.33, 0.4, 0.09, 2.32, 7.95, 1.68),
        2: (0.20, 0.33, 0.4, 0.09, 2.1, 7.95, 1.68),
        3: (0.20, 0.33, 0.4, 0.09, 1.6, 7.95, 1.68),
    },
    4: {
        1: (0.20, 0.10, 2.48, 7.9, 2.04),
        2: (0.25, 0.04, 2.24, 7.9, 1.31),
        3: (0.20, 0.10, 2.1, 7.9, 2.04),
        4: (0.25, 0.04, 2.1, 7.9, 1.31),
    },
    5: {
        1: (0.088, 0.11, 0.13, 0.71, 0.01, 0.5, 0.3, 0.27, 2.0, 6.7, 1.4),
        2: (0.073, 0.095, 0.11, 0.65, 0.013, 0.42, 0.3, 0.3, 2.4, 7.5, 1.4),
    },
    6: {
        1: (0.08, 0.71, 0.01, 0.5, 2.0, 6.7, 1.4),
        2: (0.11, 0.71, 0.01, 0.3, 2.0, 6.7, 1.4),
        3: (0.13, 0.71, 0.01, 0.3, 2.0, 6.7, 1.4),
    },
    9: {
        1: (0.072, 0.095, 0.11, 0.66, 0.012, 0.15, 0.08, 0.12, 2.1, 7.6, 1.3, 0.5),
        2: (0.073, 0.10, 0.114, 0.67, 0.011, 0.13, 0.1, 0.13, 2.1, 7.1, 1.35, 0.78),
        3: (0.077, 0.104, 0.12, 0.72, 0.014, 0.5, 0.2, 0.2, 2.1, 6.6, 1.4, 0.8),
    },
    10: {
        1: (0.072, 0.095, 0.11, 0.66, 0.012, 0.15, 0.08, 0.12, 2.0, 2.1, 7.6, 1.3, 0.5),
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
        'he2': 0,
    },

    'grid_bounds': {
        'grid5': grid_bounds[1][1],
        'he2': grid_bounds[3][1],
    },

    'initial_position': {
        'grid5': initial_position[1][1],
        'he2': initial_position[3][1],
    },

    'priors': {  # if not defined here, the default/fallback will be flat_prior()
        'grid5': {
            'z': log_z_beta,
        },
        'he2': {
            'd_b': gaussian(mean=7.846, std=0.333),
        },
    },

    'param_keys': {
        'grid5': param_keys[1],
        'he2': param_keys[3],
    },

    'interp_keys': {
        'grid5': interp_keys[1],
        'he2': interp_keys[6],
    },

    'epoch_unique': {
        'grid5': epoch_unique[1],
        'he2': epoch_unique[1],
    },

    'epoch': {
        'grid5': None,
        'he2': None,
    },

    'constants': {
        'grid5': {},
        'he2': {'m_nw': 1.4},
    },

    'param_aliases': {
        'grid5': param_aliases[1],
        'he2': param_aliases[1],
    },
    # TODO: rename interp_bprop_keys, etc.
    'interp_bprops': {
        'grid5': ('rate', 'fluence', 'peak'),
        'he2': ('rate',),
    },

    'analytic_bprops': {
        'grid5': ('fper', 'fedd'),
        'he2': ('fper', 'fedd'),
    },

    'weights': {
        'grid5': {'rate': 1.0, 'fluence': 1.0, 'peak': 1.0, 'fper': 1.0, 'fedd': 1.0},
        'he2': {'rate': 1.0, 'fluence': 1.0, 'fper': 1.0, 'fedd': 1.0},
    },

    'x_edd_option': {
      'grid5': 'x_0',
      'he2': 0.0,
    },

    'synthetic': {  # whether the data being matches is synthetic
        'grid5': False,
        'he2': False,
    },

    'system': {
        'grid5': 'gs1826',
        'he2': '4u1820'
    },
}

# Summary
# ------------------------------
# grid5:
#   1  : base grid
#   2 : as 1, epoch 1998
#   3 : as 1, epoch 2000
#   4 : as 1, epoch 2007

#   5 : as 1, (prior: m_gr = 1.6 +/- 0.1)

#   6 : as 1, (with x_edd)
#   7 : as 6, epoch 1998
#   8 : as 6, epoch 2000
#   9 : as 6, epoch 2007

#   11 : as 1, (m_gr < 10)

# ------------------------------
# he2
#   1 : default grid (fixed mass=1.4)
#   2 : as 1, epoch 1997
#   3 : as 1, epoch 2009

#   4 : as 1, (m_gr < 2.2)
#   5 : as 4, epoch 1997
#   6 : as 4, epoch 2009

#   7 : as 1, m_gr prior (1.6 +/- 0.1)


version_definitions = {
    'interpolator': {
        'grid5': {
            7: 1,
            8: 1,
            9: 1,
        },
        'he2': {},
    },

    'grid_bounds': {
        'grid5': {
            2: grid_bounds[2][1],
            3: 2,
            4: 2,
            6: grid_bounds[10][1],
            7: grid_bounds[2][2],
            8: 7,
            9: 7,
            11: grid_bounds[1][2],
        },
        'he2': {
            2: grid_bounds[4][1],
            3: 2,
            4: grid_bounds[3][2],
            5: grid_bounds[4][2],
            6: 5,
        },
    },

    'initial_position': {
        'grid5': {
            2: initial_position[2][1],
            3: initial_position[2][2],
            4: initial_position[2][3],
            5: initial_position[1][2],
            6: initial_position[10][1],
            7: 2,
            8: 3,
            9: 4,
            11: initial_position[1][3],
        },
        'he2': {
            2: initial_position[4][1],
            3: initial_position[4][2],
            4: initial_position[3][2],
            5: initial_position[4][3],
            6: initial_position[4][4],
            7: initial_position[3][3],
        },
    },

    'priors': {
        'grid5': {
            5: {'m_gr': gaussian(mean=1.6, std=0.1)},
        },
        'he2': {
            7: {'m_gr': gaussian(mean=1.6, std=0.1)},
        },
    },

    'param_keys': {
        'grid5': {
            2: param_keys[2],
            3: 2,
            4: 2,
            6: param_keys[10],
            7: 2,
            8: 2,
            9: 2,
        },
        'he2': {
            2: param_keys[4],
            3: 2,
            5: 2,
            6: 2,
        },
    },

    'epoch': {
        'grid5': {
            2: 1998,
            3: 2000,
            4: 2007,
            7: 1998,
            8: 2000,
            9: 2007,
        },
        'he2': {
            2: 1997,
            3: 2009,
            5: 1997,
            6: 2009,
        },
    },

    'interp_bprops': {
        'grid5': {},
        'he2': {},
    },

    'weights': {
        'grid5': {},
        'he2': {},
    },

    'constants': {
        'grid5': {},
        'he2': {},
    },

    'analytic_bprops': {
        'grid5': {},
        'he2': {},
    },

    'interp_keys': {
        'grid5': {},
        'he2': {},
    },

    'epoch_unique': {
        'grid5': {},
        'he2': {},
    },

    'param_aliases': {
        'grid5': {},
        'he2': {},
    },

    'x_edd_option': {
        'grid5': {},
        'he2': {},
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
        self.constants = self.get_parameter_dict('constants')
        self.priors = self.get_priors()
        self.synthetic = source_defaults['synthetic'][source]
        self.system = source_defaults['system'].get(source)

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
                + f'\nsystem           : {self.system}'
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
                + f'\ninterpolator     : {self.interpolator}'
                + f'\nx_edd option     : {self.x_edd_option}'
                + f'\nconstants        : {self.constants}'
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

    def get_parameter_dict(self, parameter):
        return get_parameter_dict(self.source, self.version, parameter=parameter)


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
    """Retrieves prior pdfs for each parameter
    """
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


def get_parameter_dict(source, version, parameter):
    """Retrieve a parameter that's defined on an item-by-item basis
    """
    valid_parameters = ['constants']
    if parameter not in valid_parameters:
        raise ValueError(f'parameter={parameter}, must be one of {valid_parameters}')

    default = source_defaults[parameter][source]
    v_definition = version_definitions[parameter][source].get(version, {})
    return {**default, **v_definition}
