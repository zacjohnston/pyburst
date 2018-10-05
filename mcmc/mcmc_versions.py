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
# TODO: clean the fuck up
z_sun = 0.01


# ===== Define order/number of params provided to BurstFit =====
param_keys = {
    1: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'qb', 'g', 'redshift', 'f_b', 'f_p'],
    2: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'g', 'redshift', 'f_b', 'f_p'],
    3: ['mdot1', 'mdot2', 'mdot3', 'x', 'logz', 'qb', 'g', 'redshift', 'f_b', 'f_p'],
    4: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'qb1', 'qb2', 'qb3', 'g', 'redshift', 'f_b', 'f_p'],
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
        1: ((0.1, 0.18),  # mdot1
            (0.1, 0.18),  # mdot2
            (0.1, 0.18),  # mdot3
            (0.7, 0.73),  # x
            (0.0025, 0.0075),  # z
            (0.05, 0.1),  # qb
            (1.7 / 1.4, 2.3 / 1.4),  # g
            (1.2, 1.4),  # redshift
            (0.01, 10),  # f_b
            (0.01, 10),  # f_p
            ),
        2: ((0.1, 0.18),  # mdot1
            (0.1, 0.18),  # mdot2
            (0.1, 0.18),  # mdot3
            (0.7, 0.73),  # x
            (0.0025, 0.0075),  # z
            (0.05, 0.15),  # qb
            (1.7 / 1.4, 2.3 / 1.4),  # g
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
        2: ((0.1, 0.2),  # mdot1
            (0.1, 0.2),  # mdot2
            (0.1, 0.2),  # mdot3
            (0.7, 0.73),  # x
            (0.0025, 0.0075),  # z
            (1.7 / 1.4, 2.3 / 1.4),  # g
            (1.2, 1.4),  # redshift
            (0.01, 10),  # f_b
            (0.01, 10),  # f_p
            ),
        3: ((0.1, 0.18),  # mdot1
            (0.1, 0.18),  # mdot2
            (0.1, 0.18),  # mdot3
            (0.7, 0.73),  # x
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
            (0.7, 0.73),  # x
            (0.0025, 0.0075),  # z
            (0.05, 0.15),  # qb1
            (0.05, 0.15),  # qb2
            (0.05, 0.15),  # qb3
            (1.7 / 1.4, 2.3 / 1.4),  # g
            (1.2, 1.4),  # redshift
            (0.01, 10),  # f_b
            (0.01, 10),  # f_p
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

    'f_ratio': {
        1: norm(loc=2.3, scale=0.2).pdf,  # f_p/f_b (i.e. xi_p/xi_b)
        2: flat_prior,                   # flat prior
    },

    'inc': {
        1: np.sin,
    },
}


# ===== initial position of walkers =====
# TODO: organise by param_keys
initial_position = {
    1: (0.15, 0.13, 0.105,
        0.72, 0.005, 1.3, 1.35, 0.51, 1.2),
    2: (0.162, 0.138, 0.102,
        0.71, 0.005, 0.075, 1.5, 1.35, 0.55, 1.3),
    3: (0.15, 0.13, 0.105,
        0.72, -0.5, 0.075, 1.3, 1.35, 0.51, 1.2),
    4: (0.162, 0.138, 0.102,
        0.71, 0.005, 0.053, 0.065, 0.14, 1.5, 1.35, 0.55, 1.3),
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
        'grid4': param_keys[2],
        'grid5': param_keys[1],
        'heat': param_keys[2],
    },

    'interp_keys': {
        'grid4': interp_keys[2],
        'grid5': interp_keys[1],
    },

    'epoch_unique': {
        'grid4': epoch_unique[1],
        'grid5': epoch_unique[1],
    },

    'param_aliases': {
        'grid4': param_aliases[1],
        'grid5': param_aliases[1],
    },

    'bprops': {
        'grid4': ('rate', 'fluence', 'peak'),
        'grid5': ('rate', 'fluence', 'peak'),
        'heat': ('rate', 'fluence', 'peak'),
    },

    'disc_model': {
        'grid4': 'he16_a',
        'grid5': 'he16_a',
        'heat': 'he16_a',
    },

    'interpolator': {
        'grid4': 1,
        'grid5': 1,
        'heat': 1,
    },

    'prior_bounds': {
        'grid4': prior_bounds[2][1],
        'grid5': prior_bounds[1][2],
        'heat': prior_bounds[2][2],
    },

    'prior_pdfs': {
        'grid4': {
          'z': prior_pdfs['z'][1],
          'f_ratio': prior_pdfs['f_ratio'][1],
          'inc': prior_pdfs['inc'][1],
        },

        'grid5': {
          'z': prior_pdfs['z'][1],
          'f_ratio': prior_pdfs['f_ratio'][1],
          'inc': prior_pdfs['inc'][1],
        },

        'heat': {
          'z': prior_pdfs['z'][1],
          'f_ratio': prior_pdfs['f_ratio'][1],
          'inc': prior_pdfs['inc'][1],
        },
    },

    'initial_position': {
        'grid4': initial_position[1],
        'grid5': initial_position[2],
        'heat': initial_position[1],
    },
}

version_definitions = {
    'interpolator': {
        'grid4': {
            1: 1,
            3: 2,
            4: 2,
            5: 3,
            6: 4,
            7: 3,
        },

        'grid5': {
            1: 2,
        },

        'heat': {},
    },

    'bprops': {
        'grid4': {},
        'grid5': {},
        'heat': {
        },
    },

    'param_keys': {
        'grid4': {
            5: param_keys[1],
            7: 5,
        },

        'grid5': {
            1: param_keys[2],
            4: param_keys[3],
            5: param_keys[4],
        },

        'heat': {},
    },

    'interp_keys': {
        'grid4': {},
        'grid5': {
            1: interp_keys[2],
        },
    },

    'epoch_unique': {
        'grid4': {},
        'grid5': {
            5: epoch_unique[2],
        },
    },

    'param_aliases': {
        'grid4': {},
        'grid5': {
            4: param_aliases[2],
        },
    },

    'prior_bounds': {
        'grid4': {
            5: prior_bounds[1][1],
            6: prior_bounds[2][3],
            7: prior_bounds[1][2],
        },

        'grid5': {
            1: prior_bounds[2][3],
            4: prior_bounds[3][1],
            5: prior_bounds[4][1],
        },

        'heat': {},
    },

    'prior_pdfs': {
         'grid4': {
            2: prior_pdfs['f_ratio'][2],
            4: prior_pdfs['f_ratio'][2],
         },

         'grid5': {
             3: prior_pdfs['f_ratio'][2],
         },

         'heat': {
            2: prior_pdfs['f_ratio'][2],
            4: prior_pdfs['f_ratio'][2],
         },
    },

    'initial_position': {
        'grid4': {
            5: initial_position[2],
            7: initial_position[2],
        },

        'grid5': {
            1: initial_position[1],
            4: initial_position[3],
            5: initial_position[4],
        },

        'heat': {},
    },

    'disc_model': {
        'grid4': {},
        'grid5': {},
        'heat': {},
    },
}


class McmcVersion:
    """Class for holding different mcmc versions
    """

    def __init__(self, source, version):
        source = grid_strings.check_synth_source(source)

        if source not in source_defaults['param_keys']:
            raise ValueError(f'source ({source}) not defined in mcmc_versions')
        elif version not in version_definitions['interpolator'][source]:
            print(f'version {version} of source {source} ' +
                  'not defined in mcmc_versions. Using default values')

        self.source = source
        self.version = version
        self.param_keys = get_parameter(source, version, 'param_keys')
        self.interp_keys = get_parameter(source, version, 'interp_keys')
        self.epoch_unique = get_parameter(source, version, 'epoch_unique')
        self.param_aliases = get_parameter(source, version, 'param_aliases')
        self.bprops = get_parameter(source, version, 'bprops')
        self.interpolator = get_parameter(source, version, 'interpolator')
        self.prior_bounds = np.array(get_parameter(source, version, 'prior_bounds'))
        self.initial_position = get_parameter(source, version, 'initial_position')
        self.prior_pdfs = get_prior_pdfs(source, version)

        if 'inc' in self.param_keys:
            self.disc_model = get_parameter(source, version, 'disc_model')
        else:
            self.disc_model = None

    def __repr__(self):
        return (f'MCMC version definitions for {self.source} V{self.version}'
                + f'\nparam keys       : {self.param_keys}'
                + f'\ninterp keys      : {self.interp_keys}'
                + f'\nepoch unique     : {self.epoch_unique}'
                + f'\nparam aliases    : {self.param_aliases}'
                + f'\nbprops           : {self.bprops}'
                + f'\ninitial position : {self.initial_position}'
                + f'\ndisc model       : {self.disc_model}'
                + f'\ninterpolator     : {self.interpolator}'
                + f'\nprior bounds     : {self.prior_bounds}'
                )


def get_parameter(source, version, parameter):
    source = grid_strings.check_synth_source(source)
    default = source_defaults[parameter][source]
    output = version_definitions[parameter][source].get(version, default)

    if (parameter != 'interpolator') and type(output) is int:
        return version_definitions[parameter][source][output]
    else:
        return output


def get_prior_pdfs(source, version):
    pdfs = {}
    for var in prior_pdfs:
        default = source_defaults['prior_pdfs'][source][var]
        value = version_definitions['prior_pdfs'][source].get(version, default)

        if type(value) is int:  # allow pointing to previous versions
            pdfs[var] = version_definitions['prior_pdfs'][source].get(value, default)
        else:
            pdfs[var] = value

    return pdfs
