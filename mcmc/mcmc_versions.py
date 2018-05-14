import numpy as np

# Define different prior boundaries.
# Must be in same order as 'params' in BurstFit
# '***' signifies values that changed over the previous version
# TODO: split these into sub-dicts according to param_keys
prior_bounds = {
    1: ((0.09552, 0.2388),  # mdot1
        (0.68, 0.74),  # x
        (0.005, 0.025),  # z
        (0.05, 0.3),  # qb
        (1.4, 1.76),  # mass
        (1.0, 2.0),  # redshift
        (0.0, np.inf),  # d
        (0, 90)  # inc
        ),
    2: ((0.09, 0.19),  # mdot1     ***
        (0.65, 0.75),  # x         ***
        (0.005, 0.015),  # z       ***
        (0.05, 0.15),  # qb        ***
        (1.4, 2.0),  # mass        ***
        (1.0, 2.0),  # redshift
        (0.0, np.inf),  # d
        (0, 90)  # inc
        ),
    3: ((0.09, 0.19),  # mdot1
        (0.65, 0.75),  # x
        (0.005, 0.015),  # z
        (0.05, 0.15),  # qb
        (1.4, 2.0),  # mass
        (1.0, 2.0),  # redshift
        (0.0, np.inf),  # d
        (0, 75)  # inc             ***
        ),
    4: ((0.11, 0.17),  # mdot1     ***
        (0.7, 0.8),  # x           ***
        (0.0025, 0.0125),  # z     ***
        (0.025, 0.125),  # qb      ***
        (1.4, 2.0),  # mass
        (1.0, 2.0),  # redshift
        (0.0, np.inf),  # d
        (0, 75)  # inc
        ),
    5: ((0.11, 0.17),  # mdot1
        (0.7, 0.8),  # x
        (0.0025, 0.0125),  # z
        (0.025, 0.125),  # qb
        (1.4, 2.6),  # mass        ***
        (1.0, 2.0),  # redshift
        (0.0, np.inf),  # d
        (0, 75)  # inc
        ),
    6: ((0.08, 0.24),  # mdot1     ***
        (0.7, 0.8),  # x
        (0.0025, 0.0125),  # z
        (0.025, 0.125),  # qb
        (1.4, 2.0),  # mass        ***
        (1.0, 2.0),  # redshift
        (0.0, np.inf),  # d
        (0, 75)  # inc
        ),
    7: ((0.08, 0.24),  # mdot1
        (0.7, 0.8),  # x
        (0.0025, 0.0175),  # z     ***
        (0.025, 0.125),  # qb
        (1.0, 2.),  # mass        ***
        (6.0, 20.0),  # radius
        (0.0, np.inf),  # d
        (0, 75)  # inc
        ),
    8: ((0.08, 0.24),  # mdot1
        (0.6, 0.8),  # x
        (0.0025, 0.0175),  # z    ***
        (0.025, 0.125),  # qb
        (1.0, 2.),  # mass        ***
        (6.0, 20.0),  # radius
        (0.0, np.inf),  # d
        (0, 75)  # inc
        ),
    9: ((0.08, 0.24),  # mdot1
        (0.6, 0.8),  # x
        (0.0025, 0.0175),  # z
        (0.025, 0.125),  # qb
        (0.8 / 1.4, 2.6 / 1.4),  # g    ***
        (1.0, 2.0),  # redshift     ***
        (0.0, np.inf),  # d
        (0, 75)  # inc
        ),
    10: ((0.08, 0.24),  # mdot1
         (0.6, 0.8),  # x
         (0.0025, 0.0175),  # z
         (0.025, 0.125),  # qb
         (0.8 / 1.4, 3.2 / 1.4),  # g    ***
         (1.0, 2.0),  # redshift
         (0.0, np.inf),  # d
         (0, 75)  # inc
         ),
    11: ((0.05, 0.24),  # mdot1
         (0.6, 0.8),  # x
         (0.0025, 0.0175),  # z
         (0.025, 0.125),  # qb
         (0.8 / 1.4, 3.2 / 1.4),  # g
         (1.0, 2.0),  # redshift
         (0.0, np.inf),  # d
         (0, 90)  # inc             ***
         ),
    12: ((0.05, 0.24),  # mdot1
         (0.05, 0.24),  # mdot2
         (0.05, 0.24),  # mdot3
         (0.6, 0.8),  # x
         (0.0025, 0.0175),  # z
         (0.025, 0.125),  # qb
         (0.8 / 1.4, 3.2 / 1.4),  # g    ***
         (1.0, 2.0),  # redshift
         (0.0, np.inf),  # d
         (0, 75)  # inc              ***
         ),
    13: ((0.05, 0.24),  # mdot1
         (0.05, 0.24),  # mdot2
         (0.05, 0.24),  # mdot3
         (0.6, 0.8),  # x
         (0.0025, 0.0175),  # z
         (0.025, 0.125),  # qb
         (0.8 / 1.4, 3.2 / 1.4),  # g    ***
         (1.0, 2.0),  # redshift
         (0.0, np.inf),  # d
         (0, 80)  # inc              ***
         ),
    14: ((0.05, 0.24),  # mdot1
         (0.05, 0.24),  # mdot2
         (0.05, 0.24),  # mdot3
         (0.5, 0.8),  # x
         (0.0025, 0.0175),  # z
         (0.025, 0.125),  # qb
         (0.8 / 1.4, 3.2 / 1.4),  # g    ***
         (1.0, 2.0),  # redshift
         (0.0, np.inf),  # d
         (0, 80)  # inc              ***
         ),
    15: ((0.06, 0.24),  # mdot1      ***
         (0.06, 0.24),  # mdot2      ***
         (0.06, 0.24),  # mdot3      ***
         (0.6, 0.8),  # x
         (0.0025, 0.0175),  # z
         (0.025, 0.125),  # qb
         (0.8 / 1.4, 3.2 / 1.4),  # g
         (1.0, 2.0),  # redshift
         (0.0, np.inf),  # d
         (0, 80)  # inc
         ),
    16: ((0.06, 0.24),  # mdot1
         (0.6, 0.8),  # x
         (0.0025, 0.0175),  # z
         (0.025, 0.125),  # qb
         (0.8 / 1.4, 3.2 / 1.4),  # g
         (1.0, 2.0),  # redshift
         (0.0, np.inf),  # d
         (0, 90)  # inc             ***
         ),
    17: ((0.06, 0.24),  # mdot1      ***
         (0.06, 0.24),  # mdot2      ***
         (0.06, 0.24),  # mdot3      ***
         (0.6, 0.8),  # x
         (0.0025, 0.0175),  # z
         (0.025, 0.125),  # qb
         (0.8 / 1.4, 3.2 / 1.4),  # g
         (1.0, 2.0),  # redshift
         (0.0, np.inf),  # d
         (0.4, np.inf),  # xi_b
         (0.4, np.inf),  # xi_p
         ),
    18: ((0.06, 0.24),  # mdot1      ***
         (0.06, 0.24),  # mdot2      ***
         (0.06, 0.24),  # mdot3      ***
         (0.6, 0.8),  # x
         (0.0015, 0.0175),  # z
         (0.025, 0.125),  # qb
         (0.8 / 1.4, 3.2 / 1.4),  # g
         (1.0, 2.0),  # redshift
         (0.0, np.inf),  # d
         (0, 80)  # inc
         ),
    19: ((0.06, 0.24),  # mdot1      ***
         (0.06, 0.24),  # mdot2      ***
         (0.06, 0.24),  # mdot3      ***
         (0.6, 0.8),  # x
         (0.0015, 0.0175),  # z
         (0.025, 0.125),  # qb
         (0.8 / 1.4, 3.2 / 1.4),  # g
         (1.0, 2.0),  # redshift
         (0.0, np.inf),  # d
         (0.4, np.inf),  # xi_b
         (0.4, np.inf),  # xi_p
         ),
    20: ((0.08, 0.24),  # mdot1
         (0.08, 0.24),  # mdot2
         (0.08, 0.24),  # mdot3
         (0.0015, 0.0175),  # z
         (0.025, 0.125),  # qb
         (1., 2.6/1.4),  # g      ***
         (1.0, 2.0),  # redshift
         (0.0, np.inf),  # d
         (0, 80)  # inc
         ),
    21: ((0.08, 0.24),  # mdot1
         (0.08, 0.24),  # mdot2
         (0.08, 0.24),  # mdot3
         (0.0015, 0.0175),  # z
         (0.025, 0.125),  # qb
         (1., 2.6 / 1.4),  # g
         (1.2, 1.5),  # redshift      ***
         (0.0, np.inf),  # d
         (0, 80)  # inc
         ),
    22: ((0.08, 0.24),  # mdot1
         (0.08, 0.24),  # mdot2
         (0.08, 0.24),  # mdot3
         (0.0015, 0.0175),  # z
         (0.025, 0.125),  # qb
         (1., 2.6 / 1.4),  # g
         (1.2, 1.5),  # redshift
         (0.4, np.inf),  # f1      ***
         (0.4, np.inf),  # f2
         ),
    23: ((0.08, 0.24),  # mdot1
         (0.08, 0.24),  # mdot2
         (0.08, 0.24),  # mdot3
         (0.0015, 0.0175),  # z
         (0.025, 0.125),  # qb
         (1., 2.6 / 1.4),  # g
         (1.2, 1.5),  # redshift
         (0.01, np.inf),  # f1      ***
         (0.01, np.inf),  # f2      ***
         ),
    24: ((0.08, 0.24),  # mdot1
         (0.08, 0.24),  # mdot2
         (0.08, 0.24),  # mdot3
         (0.65, 0.77),  # x         ***
         (0.0015, 0.0075),  # z     ***
         (0.025, 0.125),  # qb
         (1., 2.6 / 1.4),  # g
         (1.2, 1.5),  # redshift
         (0.01, np.inf),  # f1
         (0.01, np.inf),  # f2
         ),
    25: ((0.08, 0.24),  # mdot1
         (0.08, 0.24),  # mdot2
         (0.08, 0.24),  # mdot3
         (0.65, 0.77),  # x
         (0.0015, 0.0075),  # z
         (0.025, 0.125),  # qb
         (1., 2.6 / 1.4),  # g
         (1.2, 1.6),  # redshift     ***
         (0.01, np.inf),  # f1
         (0.01, np.inf),  # f2
         ),
    26: ((0.08, 0.24),  # mdot1
         (0.08, 0.24),  # mdot2
         (0.08, 0.24),  # mdot3
         (0.65, 0.77),  # x
         (0.0015, 0.0175),  # z     ***
         (0.025, 0.125),  # qb
         (1., 2.6 / 1.4),  # g
         (1.2, 1.5),  # redshift    ***
         (0.01, np.inf),  # f1
         (0.01, np.inf),  # f2
         ),
    27: ((0.1, 0.24),  # mdot1
         (0.1, 0.24),  # mdot2
         (0.1, 0.24),  # mdot3
         (0.65, 0.77),  # x
         (0.0015, 0.0175),  # z     ***
         (0.025, 0.125),  # qb
         (1., 2.6 / 1.4),  # g
         (1.2, 1.5),  # redshift    ***
         (0.01, np.inf),  # f1
         (0.01, np.inf),  # f2
         ),
}

# ===== Defines order/number of params provided to BurstFit =====
param_keys = {
    1: ['mdot1', 'x', 'z', 'qb', 'g', 'redshift', 'd', 'inc'],
    2: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'qb', 'g', 'redshift', 'd', 'inc'],
    3: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'qb', 'g', 'redshift', 'd', 'xi_b', 'xi_p'],
    4: ['mdot1', 'mdot2', 'mdot3', 'z', 'qb', 'g', 'redshift', 'd', 'inc'],
    5: ['mdot1', 'mdot2', 'mdot3', 'z', 'qb', 'g', 'redshift', 'f_b', 'f_p'],
    6: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'qb', 'g', 'redshift', 'f_b', 'f_p'],
}

# ===== initial position of walkers =====
initial_position = {
    1: (0.19,
        0.75, 0.005, 0.035, 1.5, 1.3, 4.5, 63.8),
    2: (0.19, 0.162, 0.119,
        0.75, 0.005, 0.035, 1.5, 1.3, 4.5, 63.8),
    3: (0.14,
        0.72, 0.005, 0.045, 1.5, 1.3, 8.4, 71.),
    4: (0.22, 0.187, 0.138,
        0.78, 0.004, 0.026, 1.9, 1.26, 5.0, 63.8),
    5: (0.22, 0.187, 0.138,
        0.78, 0.004, 0.026, 1.9, 1.1, 7.0, 77.),
    6: (0.17, 0.146, 0.106,
        0.77, 0.0026, 0.026, 1.45, 1.18, 8.0, 63.),
    7: (0.204, 0.174, 0.128,
        0.72, 0.003, 0.026, 1.45, 1.25, 5.0, 1., 1.),
    8: (0.204, 0.174, 0.128,
        0.72, 0.003, 0.026, 1.45, 1.25, 5.0, 63.),
    9: (0.204, 0.174, 0.128,
        0.003, 0.026, 1.45, 1.25, 6.0, 63.),
    10: (0.204, 0.174, 0.128,
         0.003, 0.026, 1.45, 1.25, 5e44, 5e44),
    11: (0.23, 0.197, 0.145,
         0.004, 0.026, 1.45, 1.45, 0.7, 1.75),
    12: (0.193, 0.164, 0.120,
         0.7, 0.004, 0.026, 1.4, 1.45, 0.64, 1.7),
}

# To add a new version definition, add an entry to each of the parameters
#   in version_definitions

# Structure:
#   -parameter
#     ---source
#         ---version
#               ---parameter value

version_definitions = {
    'disc_model_default':
        {
            'biggrid1': 'he16_d',
            'biggrid2': 'he16_d',
            'sim_test': 'he16_a',
        },
    'disc_model':
        {
            'biggrid1': {
                    2: 'he16_a',
                    5: 'he16_a',
                    6: 'he16_a',
                        },
            'biggrid2': {
                    2: 'he16_a',
                    3: 'he16_b',
                    10: 'he16_a',
                    11: 'he16_b',
                    12: 'he16_c',
                    14: 'he16_a',
                    15: 'he16_b',
                    16: 'he16_c',
                    18: 'he16_a',
                    19: 'he16_b',
                    20: 'he16_c',
                    22: 'he16_a',
                    23: 'he16_b',
                    24: 'he16_c',
                    25: 'he16_b',
                    26: 'he16_a',
                    27: 'he16_b',
                    38: 'he16_a',
                        },
            'sim_test': {},
        },
    'interpolator':
        {
            'biggrid1': {
                 1: 1,
                 2: 1,
                 3: 2,
                 4: 2,
                 5: 2,
                 6: 2,
                        },
            'biggrid2': {
                 1: 1,
                 2: 1,
                 3: 1,
                 4: 2,
                 5: 1,
                 6: 3,
                 7: 4,
                 8: 4,
                 9: 5,
                 10: 5,
                 11: 5,
                 12: 5,
                 13: 5,
                 14: 5,
                 15: 5,
                 16: 5,
                 17: 6,
                 18: 6,
                 19: 6,
                 20: 6,
                 21: 6,
                 22: 6,
                 23: 6,
                 24: 6,
                 25: 7,
                 26: 8,
                 27: 8,
                 28: 8,
                 29: 8,
                 30: 9,
                 31: 9,
                 32: 10,
                 33: 11,
                 34: 12,
                 35: 13,
                 36: 13,
                 37: 12,
                 38: 13,
                 39: 14,
                 40: 14,
                 41: 14,
                 42: 14,
                 43: 15,
                 44: 15,
                 45: 15,
                 46: 15,
                        },
            'sim_test':
                {1: 5,
                 2: 5,
                 3: 5,
                 4: 5,
                 5: 6,
                 6: 6,
                 7: 8,
                 },
            'sim': {
                1: 15,
                        },
        },
    'prior_bounds':
        {
            'biggrid1': {
                    1: prior_bounds[1],
                    2: prior_bounds[1],
                    3: prior_bounds[2],
                    4: prior_bounds[2],
                    5: prior_bounds[2],
                    6: prior_bounds[3],
                        },
            'biggrid2': {
                    1: prior_bounds[4],
                    2: prior_bounds[4],
                    3: prior_bounds[4],
                    4: prior_bounds[5],
                    5: prior_bounds[6],
                    6: prior_bounds[7],
                    7: prior_bounds[8],
                    8: prior_bounds[9],
                    9: prior_bounds[10],
                    10: prior_bounds[10],
                    11: prior_bounds[10],
                    12: prior_bounds[10],
                    13: prior_bounds[12],
                    14: prior_bounds[13],
                    15: prior_bounds[13],
                    16: prior_bounds[13],
                    17: prior_bounds[13],
                    18: prior_bounds[13],
                    19: prior_bounds[13],
                    20: prior_bounds[13],
                    21: prior_bounds[13],
                    22: prior_bounds[13],
                    23: prior_bounds[13],
                    24: prior_bounds[13],
                    25: prior_bounds[14],
                    26: prior_bounds[15],
                    27: prior_bounds[15],
                    28: prior_bounds[15],
                    29: prior_bounds[15],
                    30: prior_bounds[15],
                    31: prior_bounds[17],
                    32: prior_bounds[15],
                    33: prior_bounds[15],
                    34: prior_bounds[15],
                    35: prior_bounds[18],
                    36: prior_bounds[19],
                    37: prior_bounds[17],
                    38: prior_bounds[18],
                    39: prior_bounds[20],
                    40: prior_bounds[21],
                    41: prior_bounds[22],
                    42: prior_bounds[23],
                    43: prior_bounds[24],
                    44: prior_bounds[25],
                    45: prior_bounds[26],
                    46: prior_bounds[27],
                        },
            'sim_test': {
                    1: prior_bounds[9],
                    2: prior_bounds[10],
                    3: prior_bounds[11],
                    4: prior_bounds[11],
                    5: prior_bounds[11],
                    6: prior_bounds[11],
                    7: prior_bounds[16],
                        },
            'sim': {
                    1: prior_bounds[26],
                        },
        },
    'initial_position_default':
        {
            'biggrid1': initial_position[1],
            'biggrid2': initial_position[4],
            'sim_test': initial_position[3],
            'sim': initial_position[12],
        },
    'initial_position':
        {
            'biggrid1': {},
            'biggrid2': {
                21: initial_position[4],
                22: initial_position[5],
                23: initial_position[6],
                24: initial_position[6],
                25: initial_position[6],
                28: initial_position[6],
                29: initial_position[6],
                31: initial_position[7],
                36: initial_position[7],
                37: initial_position[7],
                38: initial_position[8],
                39: initial_position[9],
                40: initial_position[9],
                41: initial_position[10],
                42: initial_position[11],
                43: initial_position[12],
                44: initial_position[12],
                45: initial_position[12],
                46: initial_position[12],
                        },
            'sim_test': {},
            'sim': {},

        },
    'param_keys_default':
        {
            'biggrid1': param_keys[1],
            'biggrid2': param_keys[2],
            'sim_test': param_keys[1],
            'sim': param_keys[6],
        },
    'param_keys':
        {
            'biggrid1': {},
            'biggrid2':
                {
                    1: param_keys[1],
                    2: param_keys[1],
                    3: param_keys[1],
                    4: param_keys[1],
                    5: param_keys[1],
                    6: param_keys[1],
                    7: param_keys[1],
                    8: param_keys[1],
                    9: param_keys[1],
                    10: param_keys[1],
                    11: param_keys[1],
                    12: param_keys[1],
                    31: param_keys[3],
                    36: param_keys[3],
                    37: param_keys[3],
                    39: param_keys[4],
                    40: param_keys[4],
                    41: param_keys[5],
                    42: param_keys[5],
                    43: param_keys[6],
                    44: param_keys[6],
                    45: param_keys[6],
                    46: param_keys[6],
                },
            'sim_test': {},
            'sim': {},
        },
}


class McmcVersion:
    """Class for holding different mcmc versions
    """

    def __init__(self, source, version):
        if ('sim' in source) and (source != 'sim_test'):
            source = 'sim'

        if source not in version_definitions['param_keys_default']:
            raise ValueError(f'source ({source}) not defined in mcmc_versions')
        elif version not in version_definitions['interpolator'][source]:
            raise ValueError(f'version {version} of source {source} ' +
                             'is not defined in mcmc_versions')

        self.source = source
        self.version = version
        self.param_keys = get_param_keys(source, version)
        self.interpolator = get_interpolator(source, version)
        self.prior_bounds = get_prior_bounds(source, version)
        self.initial_position = get_initial_position(source, version)

        if 'inc' in self.param_keys:
            self.disc_model = get_disc_model(source, version)
        else:
            self.disc_model = None

    def __repr__(self):
        return (f'MCMC version definitions for {self.source} V{self.version}'
                + f'\nparam keys: {self.param_keys}'
                + f'\ninitial position: {self.initial_position}'
                + f'\ndisc model: {self.disc_model}'
                + f'\ninterpolator: {self.interpolator}'
                + f'\nprior bounds: \n{self.prior_bounds}'
                )


# ===== Convenience functions =====
def get_disc_model(source, version):
    default = version_definitions['disc_model_default'][source]
    return version_definitions['disc_model'][source].get(version, default)


def get_interpolator(source, version):
    return version_definitions['interpolator'][source][version]


def get_prior_bounds(source, version):
    return np.array(version_definitions['prior_bounds'][source][version])


def get_initial_position(source, version):
    default = version_definitions['initial_position_default'][source]
    return version_definitions['initial_position'][source].get(version, default)


def get_param_keys(source, version):
    default = version_definitions['param_keys_default'][source]
    return version_definitions['param_keys'][source].get(version, default)
