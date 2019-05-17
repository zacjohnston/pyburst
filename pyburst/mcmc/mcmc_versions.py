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

# ===== Define order/number of params provided to BurstFit =====
param_keys = {
    7: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'qb1', 'qb2', 'qb3', 'm_nw', 'm_gr', 'd_b', 'xi_ratio'],
    8: ['mdot1', 'mdot2', 'x', 'qb1', 'qb2', 'm_nw', 'm_gr', 'd_b', 'xi_ratio'],
    9: ['mdot1', 'x', 'z', 'qb1', 'm_nw', 'm_gr', 'd_b', 'xi_ratio'],
}

# ===== Define order/number of params for a single interpolated point =====
interp_keys = {
    1: ['mdot', 'x', 'z', 'qb', 'mass'],
    2: ['mdot', 'x', 'z', 'mass'],
    3: ['mdot', 'x', 'qb', 'mass'],
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
            (0.001, 0.015),  # z
            (0.0, 0.8),  # qb1
            (0.0, 0.8),  # qb2
            (0.0, 0.8),  # qb3
            (1.7, 2.9),  # m_nw
            (1.0, 2.1),  # m_gr
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
            (1.0, 2.1),  # m_gr
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
            (1.0, 2.1),  # m_gr
            (1., 15.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
    },

    8: {
        1: ((0.15, 0.4),  # mdot1
            (0.15, 0.4),  # mdot2
            (0.001, 0.05),  # x
            (0.05, 0.4),  # qb1
            (0.05, 0.4),  # qb2
            (1.1, 2.6),  # m_nw
            (0.9, 2.1),  # m_gr
            (1., 15.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
        2: ((0.15, 0.4),  # mdot1
            (0.15, 0.4),  # mdot2
            (0.001, 0.05),  # x
            (0.1, 0.4),  # qb1
            (0.1, 0.4),  # qb2
            (1.4, 2.6),  # m_nw
            (1.0, 2.1),  # m_gr
            (1., 15.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
        3: ((0.15, 0.4),  # mdot1
            (0.15, 0.4),  # mdot2
            (0.001, 0.05),  # x
            (0.05, 0.4),  # qb1
            (0.05, 0.4),  # qb2
            (1.1, 2.0),  # m_nw
            (1.0, 2.1),  # m_gr
            (1., 15.),  # d_b
            (0.1, 10.),  # xi_ratio
            ),
    },
    9: {
        1: ((0.08, 0.18),  # mdot1
            (0.67, 0.76),  # x
            (0.001, 0.0125),  # z
            (0.0, 0.8),  # qb1
            (1.7, 2.9),  # m_nw
            (1.0, 2.1),  # m_gr
            (1., 15.),  # d_b
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
        1: (0.09, 0.12, 0.136, 0.72, 0.005, 0.4, 0.2, 0.2, 2.3, 2.0, 6.0, 0.9),
        2: (0.09, 0.12, 0.136, 0.72, 0.005, 0.4, 0.3, 0.2, 2.5, 1.76, 5.7, 0.9),
        3: (0.09, 0.12, 0.136, 0.73, 0.007, 0.4, 0.2, 0.2, 2.5, 1.9, 5.74, 0.94),
    },
    8: {
        1: (0.21, 0.30, 0.02, 0.35, 0.15, 1.7, 1.9, 7.3, 0.93),
    },
    9: {
        1: (0.095, 0.72, 0.0035, 0.4, 2.6, 1.6, 5.7, 1.6),
        2: (0.12, 0.72, 0.0035, 0.2, 2.6, 1.6, 5.7, 1.6),
        3: (0.14, 0.72, 0.0035, 0.2, 2.6, 1.6, 5.7, 1.6),
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

    'bprops': {
        'grid5': ('rate', 'fluence', 'peak'),
        'synth5': ('rate', 'fluence', 'peak'),
        'he2': ('rate', 'fluence'),
    },

    'weights': {
        'grid5': {'rate': 1.0, 'fluence': 1.0, 'peak': 1.0, 'fper': 1.0},
        'synth5': {'rate': 1.0, 'fluence': 1.0, 'peak': 1.0, 'fper': 1.0},
        'he2': {'rate': 1.0, 'fluence': 1.0, 'peak': 1.0, 'fper': 1.0},
    },

    'disc_model': {},

    'interpolator': {
        'grid5': 1,
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
        'he2': {},
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
# -------
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

version_definitions = {
    'interpolator': {
        'grid5': {
            2: 3,
            3: 3,
            4: 3,
            5: 3,
            6: 2,
            7: 4,
            8: 3,
            9: 3,
            10: 3,
        },
        'synth5': {},
        'he2': {
            1: 1,
            2: 2,
            3: 3,
            4: 3,
        },
    },

    'bprops': {
        'grid5': {
            7: ('rate', 'fluence', 'peak', 'tail_50'),
        },
        'synth5': {},
        'he2': {
            3: ('rate', 'fluence', 'peak'),
            4: ('rate', 'fluence', 'peak'),
        },
    },

    'weights': {
        'grid5': {
            7: {'rate': 1.0, 'fluence': 1.0, 'peak': 1.0, 'fper': 1.0, 'tail_50': 1.0}
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
        },
        'synth5': {},
        'he2': {},
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
            8: 1998,
            9: 2000,
            10: 2007,
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
        },
        'synth5': {},
        'he2': {
            2: grid_bounds[8][2],
            3: grid_bounds[8][3],
            4: grid_bounds[8][3],
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
         },
         'synth5': {},
         'he2': {
             3: {'d_b': gaussian(mean=7.6, std=0.4)}
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
        },
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
        self.bprops = self.get_parameter('bprops')
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
