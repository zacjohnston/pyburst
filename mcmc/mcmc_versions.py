import numpy as np
from scipy.stats import norm

# kepler_grids
from pygrids.grids import grid_strings

# -----------------------------------
# Define different prior boundaries.
# Must be in same order as 'params' in BurstFit
# '***' signifies values that changed over the previous version
# First layer identifies param_keys
# -----------------------------------
z_sun = 0.01


# ===== Define order/number of params provided to BurstFit =====
param_keys = {
    1: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'qb', 'g', 'redshift', 'f_b', 'f_p'],
    2: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'g', 'redshift', 'f_b', 'f_p'],
    3: ['mdot1', 'mdot2', 'mdot3', 'x', 'logz', 'qb', 'g', 'redshift', 'f_b', 'f_p'],
    4: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'qb1', 'qb2', 'qb3', 'g', 'redshift', 'f_b', 'f_p'],
    5: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'qb1', 'qb2', 'qb3', 'g', 'redshift', 'd_b', 'xi_ratio'],
}

# ===== Define order/number of params for a single interpolated point =====
interp_keys = {
    1: ['mdot', 'x', 'z', 'qb', 'mass'],
    2: ['mdot', 'x', 'z', 'mass'],
}

# ===== Define params that are unique for each epoch =====
epoch_unique = {
    1: ['mdot'],
    2: ['mdot', 'qb'],
}

# ===== Define alias from interp-->param keys =====
param_aliases = {
    1: {'mass': 'g'},
    2: {'mass': 'g', 'z': 'logz'},
}

prior_bounds = {
    1: {
        1: ((0.08, 0.18),  # mdot1
            (0.08, 0.18),  # mdot2
            (0.08, 0.18),  # mdot3
            (0.67, 0.73),  # x
            (0.0025, 0.0075),  # z
            (0.0, 0.2),  # qb
            (1.4 / 1.4, 2.3 / 1.4),  # g
            (1.2, 1.4),  # redshift
            (0.01, 10),  # f_b
            (0.01, 10),  # f_p
            ),
    },

    2: {
        1: ((0.08, 0.22),  # mdot1
            (0.08, 0.22),  # mdot2
            (0.08, 0.22),  # mdot3
            (0.7, 0.74),  # x
            (0.0025, 0.0075),  # z
            (1.7 / 1.4, 2.3 / 1.4),  # g
            (1.2, 1.4),  # redshift
            (0.01, 10),  # f_b
            (0.01, 10),  # f_p
            ),
    },

    3: {
        1: ((0.1, 0.18),  # mdot1
            (0.1, 0.18),  # mdot2
            (0.1, 0.18),  # mdot3
            (0.7, 0.73),  # x
            (np.log10(0.0025/z_sun), np.log10(0.0075/z_sun)),  # logz
            (0.05, 0.15),  # qb
            (1.7 / 1.4, 2.3 / 1.4),  # g
            (1.2, 1.4),  # redshift
            (0.01, 10),  # f_b
            (0.01, 10),  # f_p
            ),
    },

    4: {
        1: ((0.08, 0.18),  # mdot1
            (0.08, 0.18),  # mdot2
            (0.08, 0.18),  # mdot3
            (0.67, 0.73),  # x
            (0.0025, 0.0075),  # z
            (0.0, 0.2),  # qb1
            (0.0, 0.2),  # qb2
            (0.0, 0.2),  # qb3
            (1.4 / 1.4, 2.3 / 1.4),  # g
            (1.2, 1.4),  # redshift
            (0.01, 10),  # f_b
            (0.01, 10),  # f_p
            ),
    },

    5: {
        1: ((0.08, 0.18),  # mdot1
            (0.08, 0.18),  # mdot2
            (0.08, 0.18),  # mdot3
            (0.67, 0.73),  # x
            (0.0025, 0.0075),  # z
            (0.0, 0.2),  # qb1
            (0.0, 0.2),  # qb2
            (0.0, 0.2),  # qb3
            (1.4 / 1.4, 2.3 / 1.4),  # g
            (1.2, 1.5),  # redshift
            (1, 15),  # f_b
            (0.1, 10),  # xi_ratio
            ),
        2: ((0.08, 0.18),  # mdot1
            (0.08, 0.18),  # mdot2
            (0.08, 0.18),  # mdot3
            (0.7, 0.74),  # x
            (0.005, 0.0125),  # z
            (0.0, 0.15),  # qb1
            (0.0, 0.15),  # qb2
            (0.0, 0.15),  # qb3
            (1.7 / 1.4, 2.5 / 1.4),  # g
            (1.2, 1.5),  # redshift
            (1, 15),  # d_b
            (0.1, 10),  # xi_ratio
            ),
        3: ((0.08, 0.20),  # mdot1
            (0.08, 0.20),  # mdot2
            (0.08, 0.20),  # mdot3
            (0.66, 0.74),  # x
            (0.0025, 0.0125),  # z
            (0.0, 0.2),  # qb1
            (0.0, 0.2),  # qb2
            (0.0, 0.2),  # qb3
            (1.5 / 1.4, 2.5 / 1.4),  # g
            (1.2, 1.7),  # redshift
            (1, 20),  # d_b
            (0.1, 10),  # xi_ratio
            ),
    },
}

# ===== Define prior pdfs for parameters =====
def flat_prior(x):
    return 1


prior_pdfs = {
    'z': {
        1: norm(loc=-0.5, scale=0.25).pdf,  # log10-space [z/solar]
    },

    'd_b': {
        1: norm(loc=5.7, scale=0.2).pdf,  # f_p/f_b (i.e. xi_p/xi_b)
    },

    'xi_ratio': {
        1: norm(loc=2.3, scale=0.4).pdf,  # f_p/f_b (i.e. xi_p/xi_b)
        2: norm(loc=1.5, scale=0.3).pdf,  # f_p/f_b (i.e. xi_p/xi_b)
    },

    'inc': {
        1: np.sin,
    },
}


# ===== initial position of walkers =====
initial_position = {
    1: {
        1: (0.162, 0.138, 0.102,
            0.71, 0.005, 0.075, 1.5, 1.35, 0.55, 1.3),
    },
    
    2: {
        1: (0.15, 0.13, 0.105,
            0.72, 0.005, 1.3, 1.35, 0.51, 1.2),      
    },
    
    3: {
        1: (0.15, 0.13, 0.105,
            0.72, -0.5, 0.075, 1.3, 1.35, 0.51, 1.2),
    },

    4: {
        1: (0.165, 0.14, 0.102,
            0.7, 0.004, 0.058, 0.062, 0.14, 1.5, 1.33, 0.55, 1.3),
        2: (0.12, 0.12, 0.12,  # (priors only test)
            0.7, 0.005, 0.1, 0.1, 0.1, 1.3, 1.3, 1., 2.3),
        3: (0.165, 0.14, 0.102,
            0.728, 0.007, 0.01, 0.01, 0.09, 1.6, 1.33, 0.55, 1.3),
    },

    5: {
        1: (0.165, 0.14, 0.102,
            0.725, 0.004, 0.068, 0.062, 0.14, 1.6, 1.3, 7.2, 1.8),
        2: (0.16, 0.138, 0.103,
            0.735, 0.012, 0.01, 0.007, 0.08, 1.75, 1.46, 7.1, 1.8),
        3: (0.165, 0.143, 0.107,
            0.7, 0.006, 0.01, 0.01, 0.08, 1.45, 1.45, 7.3, 1.7),
        4: (0.16, 0.14, 0.10,
            0.73, 0.0065, 0.01, 0.01, 0.08, 1.44, 1.42, 7., 1.6),
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
        'grid5': param_keys[1],
        'grid6': param_keys[5],
        'synth5': param_keys[5],
    },

    'interp_keys': {
        'grid5': interp_keys[1],
        'grid6': interp_keys[1],
        'synth5': interp_keys[1],
    },

    'epoch_unique': {
        'grid5': epoch_unique[1],
        'grid6': epoch_unique[2],
        'synth5': epoch_unique[2],
    },

    'param_aliases': {
        'grid5': param_aliases[1],
        'grid6': param_aliases[1],
        'synth5': param_aliases[1],
    },

    'bprops': {
        'grid5': ('rate', 'fluence', 'peak'),
        'grid6': ('rate', 'fluence', 'peak'),
        'synth5': ('rate', 'fluence', 'peak'),
    },

    'weights': {
        'grid5': {'rate': 1.0, 'fluence': 1.0, 'peak': 1.0, 'fper': 1.0},
        'grid6': {'rate': 1.0, 'fluence': 1.0, 'peak': 1.0, 'fper': 1.0},
        'synth5': {'rate': 1.0, 'fluence': 1.0, 'peak': 1.0, 'fper': 1.0},
    },

    'disc_model': {},

    'interpolator': {
        'grid5': 1,
        'grid6': 1,
        'synth5': 1,
    },

    'prior_bounds': {
        'grid5': prior_bounds[1][1],
        'grid6': prior_bounds[5][3],
        'synth5': prior_bounds[5][3],
    },

    'prior_pdfs': {
        'grid5': {
          'z': prior_pdfs['z'][1],
          'd_b': flat_prior,
          'xi_ratio': prior_pdfs['xi_ratio'][1],
        },

        'grid6': {
          'z': prior_pdfs['z'][1],
          'xi_ratio': flat_prior,
          'd_b': flat_prior,
        },

        'synth5': {
            'z': flat_prior,
            'xi_ratio': flat_prior,
            'd_b': flat_prior,
        },
    },

    'initial_position': {
        'grid5': initial_position[1][1],
        'grid6': initial_position[5][3],
        'synth5': initial_position[5][3],
    },

    'synthetic': {  # whether the data being matches is synthetic
        'grid5': False,
        'grid6': False,
        'synth5': True,
    },

    # ===== Special definitions for synthetic data sources =====
    'interp_source': {  # Source interpolator to use
        'synth5': 'grid5',
    },

    'synth_version': {  # Version ID of synthetic data table
        'synth5': 1,
    },

    'synth_group': {
        'synth5': None,
    },
}

# Summary
# -------
# grid5:
#   8  : as 5, but xi_ratio instead of f_p
#   7  : as 5, with 10x weight on burst rate
#   9  : as 8, with 10x weight on burst rate
#   10 : as 8, with flat xi_ratio prior
#   11 : as 9, with flat xi_ratio prior

version_definitions = {
    'interpolator': {
        'grid5': {
            9: 2,
            11: 2,
        },
        'grid6': {},
        'synth5': {},
    },

    'bprops': {
        'grid5': {},
        'grid6': {},
        'synth5': {},
    },

    'weights': {
        'grid5': {
            7: {'rate': 10.0, 'fluence': 1.0, 'peak': 1.0, 'fper': 1.0},
            9: {'rate': 10.0, 'fluence': 1.0, 'peak': 1.0, 'fper': 1.0},
            11: {'rate': 10.0, 'fluence': 1.0, 'peak': 1.0, 'fper': 1.0},
        },
        'grid6': {
            4: {'rate': 5.0, 'fluence': 1.0, 'peak': 1.0, 'fper': 1.0},
        },
        'synth5': {},

    },

    'param_keys': {
        'grid5': {
            4: param_keys[4],
            5: 4,
            6: 4,
            7: 4,
            8: param_keys[5],
            9: 8,
            10: 8,
            11: 8,
        },
        'grid6': {},
        'synth5': {},
    },

    'interp_keys': {
        'grid5': {},
        'grid6': {},
        'synth5': {},
    },

    'epoch_unique': {
        'grid5': {
            4: epoch_unique[2],
            5: 4,
            6: 4,
            7: 4,
            8: 4,
            9: 4,
            10: 4,
            11: 4,
        },
        'grid6': {},
        'synth5': {},
    },

    'param_aliases': {
        'grid5': {},
        'grid6': {},
        'synth5': {},
    },

    'prior_bounds': {
        'grid5': {
            4: prior_bounds[4][1],
            5: 4,
            6: 4,
            7: 4,
            8: prior_bounds[5][1],
            9: prior_bounds[5][2],
            10: 8,
            11: 9,
        },
        'grid6': {},
        'synth5': {},
    },

    'prior_pdfs': {
         'grid5': {
             2: {'z': flat_prior},
             3: {'xi_ratio': flat_prior},
             4: {'z': flat_prior},
             10: {'xi_ratio': flat_prior},
             11: {'xi_ratio': flat_prior},
         },
         'grid6': {
             2: {'xi_ratio': prior_pdfs['xi_ratio'][2]},
             3: {'d_b': prior_pdfs['d_b'][1]},
         },
         'synth5': {},
    },

    'initial_position': {
        'grid5': {
            4: initial_position[4][1],
            5: 4,
            6: initial_position[4][2],
            7: initial_position[4][3],
            8: initial_position[5][1],
            9: initial_position[5][2],
            10: 8,
            11: 9,
        },
        'grid6': {
            4: initial_position[5][4],
        },
        'synth5': {},
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
        'synth5': {
            1: 1,
        },
    },
}


class McmcVersion:
    """Class for holding different mcmc versions
    """

    def __init__(self, source, version):
        source = grid_strings.check_synth_source(source)

        if source not in source_defaults['param_keys']:
            raise ValueError(f'source ({source}) not defined in mcmc_versions')

        self.source = source
        self.version = version
        self.param_keys = self.get_parameter('param_keys')
        self.interp_keys = self.get_parameter('interp_keys')
        self.epoch_unique = self.get_parameter('epoch_unique')
        self.param_aliases = self.get_parameter('param_aliases')
        self.bprops = self.get_parameter('bprops')
        self.weights = self.get_parameter('weights')
        self.interpolator = self.get_parameter('interpolator')
        self.prior_bounds = np.array(self.get_parameter('prior_bounds'))
        self.initial_position = self.get_parameter('initial_position')
        self.prior_pdfs = self.get_prior_pdfs()
        self.synthetic = source_defaults['synthetic'][source]
        self.disc_model = None

        if self.synthetic:
            self.interp_source = self.get_parameter('interp_source')
            self.synth_version = self.get_parameter('synth_version')
            self.synth_group = self.get_parameter('synth_group')

            if self.synth_group is None:
                raise ValueError(f'synth_group undefined for {source} V{version}')
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

        return (f'MCMC version definitions for {self.source} V{self.version}'
                + f'\nparam keys       : {self.param_keys}'
                + f'\ninterp keys      : {self.interp_keys}'
                + f'\nepoch unique     : {self.epoch_unique}'
                + f'\nparam aliases    : {self.param_aliases}'
                + f'\nbprops           : {self.bprops}'
                + f'\nweights          : {self.weights}'
                + f'\ninitial position : {self.initial_position}'
                + f'\ndisc model       : {self.disc_model}'
                + f'\ninterpolator     : {self.interpolator}'
                + f'\nprior bounds     : {self.prior_bounds}'
                + f'\nprior pdfs       : {self.prior_pdfs}'
                + f'\nsynthetic        : {self.synthetic}'
                + synth_str
                )

    def get_parameter(self, parameter):
        return get_parameter(self.source, self.version, parameter)

    def get_prior_pdfs(self):
        return get_prior_pdfs(self.source, self.version)


def get_parameter(source, version, parameter):
    source = grid_strings.check_synth_source(source)
    default = source_defaults[parameter][source]
    output = version_definitions[parameter][source].get(version, default)

    if output is default:
        print(f"mcmc_versions: '{parameter}' not specified. Using default values")

    if parameter not in ('interpolator', 'synth_version', 'synth_group') \
            and type(output) is int:
        return version_definitions[parameter][source][output]
    else:
        return output


def get_prior_pdfs(source, version):
    pdfs = {}
    for var in source_defaults['prior_pdfs'][source]:
        default = source_defaults['prior_pdfs'][source][var]
        v_definition = version_definitions['prior_pdfs'][source].get(version)

        if v_definition is None:
            value = default
        else:
            value = v_definition.get(var, default)

        if type(value) is int:  # allow pointing to previous versions
            pdfs[var] = version_definitions['prior_pdfs'][source].get(value, default)
        else:
            pdfs[var] = value

    return pdfs
