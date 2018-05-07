import numpy as np

# Define different prior boundaries.
# Must be in same order as 'params' in BurstFit
# '***' signifies values that changed over the previous version
prior_versions = {
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
        0.72, 0.003, 0.026, 1.45, 1.25, 5.0, 1., 1.)
}

# ===== Defines order/number of params provided to BurstFit =====
param_keys = {
    1: ['mdot1', 'x', 'z', 'qb', 'g', 'redshift', 'd', 'inc'],
    2: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'qb', 'g', 'redshift', 'd', 'inc'],
    3: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'qb', 'g', 'redshift', 'd', 'xi_b', 'xi_p'],
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
            'biggrid1':
                {
                    2: 'he16_a',
                    5: 'he16_a',
                    6: 'he16_a',
                },
            'biggrid2':
                {
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
                },
            'sim_test':
                {

                },
        },
    'interpolator':
        {
            'biggrid1':
                {1: 1,
                 2: 1,
                 3: 2,
                 4: 2,
                 5: 2,
                 6: 2,
                 },
            'biggrid2':
                {1: 1,
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
        },
    'prior_bounds':
        {
            'biggrid1':
                {
                    1: prior_versions[1],
                    2: prior_versions[1],
                    3: prior_versions[2],
                    4: prior_versions[2],
                    5: prior_versions[2],
                    6: prior_versions[3],
                },
            'biggrid2':
                {
                    1: prior_versions[4],
                    2: prior_versions[4],
                    3: prior_versions[4],
                    4: prior_versions[5],
                    5: prior_versions[6],
                    6: prior_versions[7],
                    7: prior_versions[8],
                    8: prior_versions[9],
                    9: prior_versions[10],
                    10: prior_versions[10],
                    11: prior_versions[10],
                    12: prior_versions[10],
                    13: prior_versions[12],
                    14: prior_versions[13],
                    15: prior_versions[13],
                    16: prior_versions[13],
                    17: prior_versions[13],
                    18: prior_versions[13],
                    19: prior_versions[13],
                    20: prior_versions[13],
                    21: prior_versions[13],
                    22: prior_versions[13],
                    23: prior_versions[13],
                    24: prior_versions[13],
                    25: prior_versions[14],
                    26: prior_versions[15],
                    27: prior_versions[15],
                    28: prior_versions[15],
                    29: prior_versions[15],
                    30: prior_versions[15],
                    31: prior_versions[17],
                    32: prior_versions[15],
                    33: prior_versions[15],
                    34: prior_versions[15],
                    35: prior_versions[18],
                    36: prior_versions[19],
                    37: prior_versions[17],
                },
            'sim_test':
                {
                    1: prior_versions[9],
                    2: prior_versions[10],
                    3: prior_versions[11],
                    4: prior_versions[11],
                    5: prior_versions[11],
                    6: prior_versions[11],
                    7: prior_versions[16],
                },
        },
    'initial_position_default':
        {
            'biggrid1': initial_position[1],
            'biggrid2': initial_position[4],
            'sim_test': initial_position[3],
        },
    'initial_position':
        {
            'biggrid1': {

            },
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
            },
            'sim_test': {

            },
        },
    'param_keys_default':
        {
            'biggrid1': param_keys[1],
            'biggrid2': param_keys[2],
            'sim_test': param_keys[1],
        },
    'param_keys':
        {
            'biggrid1':
                {

                },
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
                },
            'sim_test':
                {

                },
        },
}


class McmcVersion:
    """Class for holding different mcmc versions
    """

    def __init__(self, source, version):
        if source not in version_definitions['disc_model_default']:
            raise ValueError(f'source ({source}) not defined in mcmc_versions')
        elif version not in version_definitions['interpolator'][source]:
            raise ValueError(f'version {version} of source {source} ' +
                             'is not defined in mcmc_versions')
        self.source = source
        self.version = version
        self.disc_model = get_disc_model(source, version)
        self.interpolator = get_interpolator(source, version)
        self.prior_bounds = get_prior_bounds(source, version)
        self.initial_position = get_initial_position(source, version)
        self.param_keys = get_param_keys(source, version)

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
