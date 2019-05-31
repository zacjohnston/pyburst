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
# TODO: reset indexes
param_keys = {
    7: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'qb1', 'qb2', 'qb3', 'm_nw', 'm_gr', 'd_b', 'xi_ratio'],
    8: ['mdot1', 'mdot2', 'x', 'qb1', 'qb2', 'm_nw', 'm_gr', 'd_b', 'xi_ratio'],
    9: ['mdot1', 'x', 'z', 'qb1', 'm_nw', 'm_gr', 'd_b', 'xi_ratio'],
    10: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'qb1', 'qb2', 'qb3', 'm_nw', 'm_gr', 'd_b', 'xi_ratio', 'xedd_ratio'],
    11: ['mdot1', 'x', 'z', 'qb1', 'm_nw', 'm_gr', 'd_b', 'xi_ratio', 'xedd_ratio'],
    12: ['mdot1', 'mdot2', 'qb1', 'qb2', 'm_nw', 'm_gr', 'd_b', 'xi_ratio'],
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
    7: {
        1: ((0.08, 0.18),  # mdot1
            (0.08, 0.18),  # mdot2
            (0.08, 0.18),  # mdot3
            (0.67, 0.76),  # x
            (0.001, 0.0125),  # z
            (0.0, 0.8),  # qb1
            (0.0, 0.8),  # qb2
            (0.0, 0.8),  # qb3
            (1.7, 2.9),  # m_nw
            (1.0, 2.3),  # m_gr
            (1., 15.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
        2: ((0.08, 0.18),  # mdot1
            (0.08, 0.18),  # mdot2
            (0.08, 0.18),  # mdot3
            (0.67, 0.76),  # x
            (0.0025, 0.0125),  # z
            (0.0, 0.8),  # qb1
            (0.0, 0.8),  # qb2
            (0.0, 0.8),  # qb3
            (1.7, 2.9),  # m_nw
            (1.0, 2.3),  # m_gr
            (1., 15.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
        3: ((0.08, 0.18),  # mdot1
            (0.08, 0.18),  # mdot2
            (0.08, 0.18),  # mdot3
            (0.67, 0.76),  # x
            (0.001, 0.0125),  # z
            (0.0, 0.8),  # qb1
            (0.0, 0.8),  # qb2
            (0.0, 0.8),  # qb3
            (1.7, 2.9),  # m_nw
            (1.0, 2.3),  # m_gr
            (1., 15.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
    },

    8: {
        1: ((0.15, 0.4),  # mdot1
            (0.15, 0.4),  # mdot2
            (0.0, 0.05),  # x
            (0.05, 0.4),  # qb1
            (0.05, 0.4),  # qb2
            (1.1, 2.0),  # m_nw
            (1.0, 2.2),  # m_gr
            (1., 20.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
        2: ((0.15, 0.35),  # mdot1
            (0.15, 0.35),  # mdot2
            (0.0, 0.1),  # x
            (0.05, 0.4),  # qb1
            (0.05, 0.4),  # qb2
            (1.1, 2.0),  # m_nw
            (1.0, 2.2),  # m_gr
            (1., 20.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
        3: ((0.2, 0.4),  # mdot1
            (0.2, 0.4),  # mdot2
            (0.0, 0.05),  # x
            (0.05, 0.3),  # qb1
            (0.05, 0.3),  # qb2
            (1.1, 2.0),  # m_nw
            (1.0, 2.2),  # m_gr
            (1., 20.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
    },
    9: {
        1: ((0.08, 0.18),  # mdot1
            (0.67, 0.76),  # x
            (0.001, 0.0125),  # z
            (0.0, 0.8),  # qb1
            (1.7, 2.9),  # m_nw
            (1.0, 2.3),  # m_gr
            (1., 15.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
    },

    10: {
        1: ((0.08, 0.18),  # mdot1
            (0.08, 0.18),  # mdot2
            (0.08, 0.18),  # mdot3
            (0.67, 0.76),  # x
            (0.001, 0.0125),  # z
            (0.0, 0.8),  # qb1
            (0.0, 0.8),  # qb2
            (0.0, 0.8),  # qb3
            (1.7, 2.9),  # m_nw
            (1.0, 2.2),  # m_gr
            (1., 15.),  # d_b
            (0.1, 10.),  # xi_ratio
            (0.0, 1.0),  # xedd_ratio
            ),
    },

    11: {
        1: ((0.08, 0.18),  # mdot1
            (0.67, 0.76),  # x
            (0.001, 0.0125),  # z
            (0.0, 0.8),  # qb1
            (1.7, 2.9),  # m_nw
            (1.0, 2.2),  # m_gr
            (1., 15.),  # d_b
            (0.1, 10.),  # xi_ratio
            (0.0, 1.0),  # xedd_ratio
            ),
    },

    12: {
        1: ((0.175, 0.4),  # mdot1
            (0.175, 0.4),  # mdot2
            (0.05, 0.5),  # qb1
            (0.05, 0.5),  # qb2
            (1.1, 2.0),  # m_nw
            (1.0, 2.2),  # m_gr
            (1., 20.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
    },
}

# ===== Define prior pdfs for parameters =====
def flat_prior(x):
    return 1


log_norm = norm(loc=-0.5, scale=0.25).pdf
def log_z(z, z_sun=0.01):
    """PDF of log10(z/z_solar)"""
    logz = np.log10(z / z_sun)
    return log_norm(logz)


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
    7: {
        1: (0.09, 0.12, 0.14, 0.72, 0.005, 0.4, 0.2, 0.2, 2.4, 2.0, 6.0, 1.6),
        2: (0.09, 0.12, 0.14, 0.73, 0.005, 0.4, 0.3, 0.2, 2.5, 1.76, 5.7, 1.6),
        3: (0.09, 0.12, 0.136, 0.73, 0.007, 0.4, 0.2, 0.2, 2.5, 1.9, 5.74, 1.6),
    },
    8: {
        1: (0.2, 0.27, 0.02, 0.35, 0.25, 1.3, 1.5, 7.5, 1.4),
        2: (0.21, 0.29, 0.02, 0.35, 0.16, 1.35, 2.1, 7.4, 1.4),
        3: (0.26, 0.36, 0.02, 0.18, 0.1, 1.8, 2.1, 7.4, 1.5),
    },
    9: {
        1: (0.095, 0.7, 0.0035, 0.4, 2.3, 2.0, 6.5, 1.7),
        2: (0.12,  0.7, 0.0035, 0.2, 2.3, 2.0, 6.5, 1.7),
        3: (0.14,  0.7, 0.0035, 0.2, 2.3, 2.0, 6.5, 1.7),
    },
    10: {
        1: (0.10, 0.13, 0.15, 0.7, 0.004, 0.2, 0.15, 0.15, 2.1, 2.0, 6.7, 1.6, 0.9),
    },
    11: {
        1: (0.09, 0.7, 0.004, 0.15, 2.1, 2.0, 6.7, 1.6, 0.9),
        2: (0.13, 0.7, 0.004, 0.15, 2.1, 2.0, 6.7, 1.6, 0.9),
        3: (0.15, 0.7, 0.004, 0.15, 2.1, 2.0, 6.7, 1.6, 0.9),
    },
    12: {
        1: (0.26, 0.36, 0.18, 0.1, 1.8, 2.1, 7.4, 1.5),
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
    'param_keys': {
        'grid5': param_keys[7],
        'synth5': param_keys[7],
        'he2': param_keys[8],
    },

    'interp_keys': {
        'grid5': interp_keys[1],
        'synth5': interp_keys[1],
        'he2': interp_keys[3],
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
        'grid5': ('fper',),
        'synth5': ('fper',),
        'he2': ('fper', 'fedd'),
    },

    'weights': {
        'grid5': {'rate': 1.0, 'fluence': 1.0, 'peak': 1.0, 'fper': 1.0},
        'synth5': {'rate': 1.0, 'fluence': 1.0, 'peak': 1.0, 'fper': 1.0},
        'he2': {'rate': 1.0, 'fluence': 1.0, 'fper': 1.0, 'fedd': 1.0},
    },

    'disc_model': {},

    'interpolator': {
        'grid5': 3,
        'synth5': 1,
        'he2': 0,
    },

    'grid_bounds': {
        'grid5': grid_bounds[7][1],
        'synth5': grid_bounds[7][1],
        'he2': grid_bounds[8][1],
    },

    'priors': {  # if not defined here, the default/fallback will be flat_prior()
        'grid5': {
          'z': priors['z'][1],
        },
        'synth5': {},
        'he2': {
            'd_b': gaussian(mean=7.846, std=0.333),
        },
    },

    'initial_position': {
        'grid5': initial_position[7][1],
        'synth5': initial_position[7][1],
        'he2': initial_position[8][1],
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
#   2  : priors: z
#   3  : priors: z, d_b
#   4  : priors: z, d_b, m_gr
#   5  : priors: d_b

#   6  : priors: d_b  (up to z=0.015)
#   7  : as 3, with tail_50

#   8  : as 3, fitting epoch 1998
#   9  : as 3, fitting epoch 2000
#   10 : as 3, fitting epoch 2007

#   11 : as 3, 15x weight on brate

#   12 : as 2, with bprop fedd
#   13 : as 12, fitting epoch 1998
#   14 : as 12, fitting epoch 2000
#   15 : as 12, fitting epoch 2007

#   16 : as 12, with no priors
#   17 : as 12, with param x_edd
#   18 : as 17, epoch 1998
#   19 : as 17, epoch 2000
#   20 : as 17, epoch 2007
# ------------------------------
# he2
#   1 : flat priors
#   2 : single d_b prior (7.6)
#   3 : using combined d_b prior
#   4 : as 3, x=0.10 (upper accrate = 0.35)
#   5 : as 4, excluding mass=1.4
#   6 : sparse grid
#   7 : fixed x=0.0

version_definitions = {
    'interpolator': {
        'grid5': {
            6: 2,
            7: 4,
        },
        'synth5': {},
        'he2': {
            4: 1,
            5: 2,
            6: 3,
            7: 5,
        },
    },

    'interp_bprops': {
        'grid5': {
            7: ('rate', 'fluence', 'peak', 'tail_50'),
        },
        'synth5': {},
        'he2': {},
    },

    'analytic_bprops': {
        'grid5': {
            12: ('fper', 'fedd'),
            13: ('fper', 'fedd'),
            14: ('fper', 'fedd'),
            15: ('fper', 'fedd'),
            16: ('fper', 'fedd'),
            17: ('fper', 'fedd'),
            18: ('fper', 'fedd'),
            19: ('fper', 'fedd'),
            20: ('fper', 'fedd'),
        },
        'synth5': {},
        'he2': {},
    },

    'weights': {
        'grid5': {
            7: {'rate': 1.0, 'fluence': 1.0, 'peak': 1.0, 'fper': 1.0, 'tail_50': 1.0},
            11: {'rate': 15.0, 'fluence': 1.0, 'peak': 1.0, 'fper': 1.0},
            12: {'rate': 1.0, 'fluence': 1.0, 'peak': 1.0, 'fper': 1.0, 'fedd': 1.0},
            13: {'rate': 1.0, 'fluence': 1.0, 'peak': 1.0, 'fper': 1.0, 'fedd': 1.0},
            14: {'rate': 1.0, 'fluence': 1.0, 'peak': 1.0, 'fper': 1.0, 'fedd': 1.0},
            15: {'rate': 1.0, 'fluence': 1.0, 'peak': 1.0, 'fper': 1.0, 'fedd': 1.0},
            16: {'rate': 1.0, 'fluence': 1.0, 'peak': 1.0, 'fper': 1.0, 'fedd': 1.0},
            17: {'rate': 1.0, 'fluence': 1.0, 'peak': 1.0, 'fper': 1.0, 'fedd': 1.0},
            18: {'rate': 1.0, 'fluence': 1.0, 'peak': 1.0, 'fper': 1.0, 'fedd': 1.0},
            19: {'rate': 1.0, 'fluence': 1.0, 'peak': 1.0, 'fper': 1.0, 'fedd': 1.0},
            20: {'rate': 1.0, 'fluence': 1.0, 'peak': 1.0, 'fper': 1.0, 'fedd': 1.0},
        },
        'synth5': {},
        'he2': {},

    },

    'param_keys': {
        'grid5': {
            -1: param_keys[7],  # dummy version for synth reference
            8: param_keys[9],
            9: param_keys[9],
            10: param_keys[9],
            13: param_keys[9],
            14: param_keys[9],
            15: param_keys[9],
            17: param_keys[10],
            18: param_keys[11],
            19: param_keys[11],
            20: param_keys[11],
        },
        'synth5': {},
        'he2': {
            7: param_keys[12],
        },
    },

    'interp_keys': {
        'grid5': {},
        'synth5': {},
        'he2': {
            7: interp_keys[4],
        },
    },

    'epoch_unique': {
        'grid5': {},
        'synth5': {},
        'he2': {},
    },

    'epoch': {
        'grid5': {
            8: 1998,
            9: 2000,
            10: 2007,
            13: 1998,
            14: 2000,
            15: 2007,
            18: 1998,
            19: 2000,
            20: 2007,
        },
        'synth5': {},
        'he2': {},
    },

    'param_aliases': {
        'grid5': {},
        'synth5': {},
        'he2': {},
    },

    'grid_bounds': {
        'grid5': {
            -1: 2,
            2: grid_bounds[7][3],
            3: 2,
            4: 2,
            5: 2,
            7: 2,
            8: grid_bounds[9][1],
            9: grid_bounds[9][1],
            10: grid_bounds[9][1],
            11: 2,
            12: 2,
            13: 8,
            14: 8,
            15: 8,
            16: 2,
            17: grid_bounds[10][1],
            18: grid_bounds[11][1],
            19: grid_bounds[11][1],
            20: grid_bounds[11][1],
        },
        'synth5': {},
        'he2': {
            4: grid_bounds[8][2],
            5: 4,
            6: grid_bounds[8][3],
            7: grid_bounds[12][1],
        },
    },

    'priors': {
         'grid5': {
             3: {'d_b': gaussian(mean=5.7, std=0.2)},
             4: {'d_b': gaussian(mean=5.7, std=0.2),
                 'm_gr': gaussian(mean=1.0, std=0.5)},
             5: {'d_b': gaussian(mean=5.7, std=0.2),
                 'z': flat_prior},
             6: {'d_b': gaussian(mean=5.7, std=0.2),
                 'z': flat_prior},
             7: {'d_b': gaussian(mean=5.7, std=0.2)},
             8: {'d_b': gaussian(mean=5.7, std=0.2)},
             9: {'d_b': gaussian(mean=5.7, std=0.2)},
             10: {'d_b': gaussian(mean=5.7, std=0.2)},
             11: {'d_b': gaussian(mean=5.7, std=0.2)},
             16: {'z': flat_prior},
         },
         'synth5': {},
         'he2': {
             1: {'d_b': flat_prior},
             2: {'d_b': gaussian(mean=7.6, std=0.4)},
         },
    },

    'initial_position': {
        'grid5': {
            3: initial_position[7][2],
            4: 3,
            5: initial_position[7][3],
            6: 5,
            7: 3,
            8: initial_position[9][1],
            9: initial_position[9][2],
            10: initial_position[9][3],
            11: 3,
            13: initial_position[9][1],
            14: initial_position[9][2],
            15: initial_position[9][3],
            17: initial_position[10][1],
            18: initial_position[11][1],
            19: initial_position[11][2],
            20: initial_position[11][3],
        },
        'synth5': {},
        'he2': {
            4: initial_position[8][2],
            5: 4,
            6: initial_position[8][3],
            7: initial_position[12][1],
        },
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
