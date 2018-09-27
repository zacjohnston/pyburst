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

prior_bounds = {
    1: {
        1: ((0.08, 0.24),  # mdot1
            (0.6, 0.8),  # x
            (0.0025, 0.0175),  # z
            (0.025, 0.125),  # qb
            (0.8 / 1.4, 2.6 / 1.4),  # g    ***
            (1.0, 2.0),  # redshift     ***
            (0.0, np.inf),  # d
            (0, 75)  # inc
            ),
        2: ((0.08, 0.24),  # mdot1
            (0.6, 0.8),  # x
            (0.0025, 0.0175),  # z
            (0.025, 0.125),  # qb
            (0.8 / 1.4, 3.2 / 1.4),  # g    ***
            (1.0, 2.0),  # redshift
            (0.0, np.inf),  # d
            (0, 75)  # inc
            ),
        3: ((0.05, 0.24),  # mdot1
            (0.6, 0.8),  # x
            (0.0025, 0.0175),  # z
            (0.025, 0.125),  # qb
            (0.8 / 1.4, 3.2 / 1.4),  # g
            (1.0, 2.0),  # redshift
            (0.0, np.inf),  # d
            (0, 90)  # inc             ***
            ),
        4: ((0.06, 0.24),  # mdot1
            (0.6, 0.8),  # x
            (0.0025, 0.0175),  # z
            (0.025, 0.125),  # qb
            (0.8 / 1.4, 3.2 / 1.4),  # g
            (1.0, 2.0),  # redshift
            (0.0, np.inf),  # d
            (0, 90)  # inc             ***
            ),
    },

    2: {
        1: ((0.05, 0.24),  # mdot1
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
        2: ((0.05, 0.24),  # mdot1
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
        3: ((0.05, 0.24),  # mdot1
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
        4: ((0.06, 0.24),  # mdot1      ***
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
        5: ((0.06, 0.24),  # mdot1      ***
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
        6: ((0.08, 0.24),  # mdot1      ***
            (0.08, 0.24),  # mdot2      ***
            (0.08, 0.24),  # mdot3      ***
            (0.65, 0.77),  # x
            (0.0015, 0.0175),  # z
            (0.025, 0.125),  # qb
            (1.0, 2.6 / 1.4),  # g
            (1.0, 1.5),  # redshift
            (0.0, np.inf),  # d
            (0, 90)  # inc
            ),
        7: ((0.08, 0.24),  # mdot1      ***
            (0.08, 0.24),  # mdot2      ***
            (0.08, 0.24),  # mdot3      ***
            (0.65, 0.77),  # x
            (0.0015, 0.0175),  # z
            (0.025, 0.125),  # qb
            (1.0, 2.6 / 1.4),  # g
            (1.0, 2.0),  # redshift
            (0.0, np.inf),  # d
            (0, 90)  # inc
            ),

    },

    3: {
        1: ((0.06, 0.24),  # mdot1      ***
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
        2: ((0.06, 0.24),  # mdot1      ***
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
    },

    4: {
        1: ((0.08, 0.24),  # mdot1
            (0.08, 0.24),  # mdot2
            (0.08, 0.24),  # mdot3
            (0.0015, 0.0175),  # z
            (0.025, 0.125),  # qb
            (1., 2.6 / 1.4),  # g      ***
            (1.0, 2.0),  # redshift
            (0.0, np.inf),  # d
            (0, 80)  # inc
            ),
        2: ((0.08, 0.24),  # mdot1
            (0.08, 0.24),  # mdot2
            (0.08, 0.24),  # mdot3
            (0.0015, 0.0175),  # z
            (0.025, 0.125),  # qb
            (1., 2.6 / 1.4),  # g
            (1.2, 1.5),  # redshift      ***
            (0.0, np.inf),  # d
            (0, 80)  # inc
            ),
    },

    5: {
        1: ((0.08, 0.24),  # mdot1
            (0.08, 0.24),  # mdot2
            (0.08, 0.24),  # mdot3
            (0.0015, 0.0175),  # z
            (0.025, 0.125),  # qb
            (1., 2.6 / 1.4),  # g
            (1.2, 1.5),  # redshift
            (0.4, np.inf),  # f_b      ***
            (0.4, np.inf),  # f_b
            ),
        2: ((0.08, 0.24),  # mdot1
            (0.08, 0.24),  # mdot2
            (0.08, 0.24),  # mdot3
            (0.0015, 0.0175),  # z
            (0.025, 0.125),  # qb
            (1., 2.6 / 1.4),  # g
            (1.2, 1.5),  # redshift
            (0.01, np.inf),  # f_b      ***
            (0.01, np.inf),  # f_p      ***
            ),
    },

    6: {
        1: ((0.08, 0.24),  # mdot1
            (0.08, 0.24),  # mdot2
            (0.08, 0.24),  # mdot3
            (0.65, 0.77),  # x         ***
            (0.0015, 0.0075),  # z     ***
            (0.025, 0.125),  # qb
            (1., 2.6 / 1.4),  # g
            (1.2, 1.5),  # redshift
            (0.01, np.inf),  # f_b
            (0.01, np.inf),  # f_p
            ),
        2: ((0.08, 0.24),  # mdot1
            (0.08, 0.24),  # mdot2
            (0.08, 0.24),  # mdot3
            (0.65, 0.77),  # x
            (0.0015, 0.0075),  # z
            (0.025, 0.125),  # qb
            (1., 2.6 / 1.4),  # g
            (1.2, 1.6),  # redshift     ***
            (0.01, np.inf),  # f_b
            (0.01, np.inf),  # f_p
            ),
        3: ((0.08, 0.24),  # mdot1
            (0.08, 0.24),  # mdot2
            (0.08, 0.24),  # mdot3
            (0.65, 0.77),  # x
            (0.0015, 0.0175),  # z     ***
            (0.025, 0.125),  # qb
            (1., 2.6 / 1.4),  # g
            (1.2, 1.5),  # redshift    ***
            (0.01, np.inf),  # f_b
            (0.01, np.inf),  # f_p
            ),
        4: ((0.1, 0.24),  # mdot1    ***
            (0.1, 0.24),  # mdot2    ***
            (0.1, 0.24),  # mdot3    ***
            (0.65, 0.77),  # x
            (0.0015, 0.0175),  # z
            (0.025, 0.125),  # qb
            (1., 2.6 / 1.4),  # g
            (1.2, 1.5),  # redshift
            (0.01, np.inf),  # f_b
            (0.01, np.inf),  # f_p
            ),
        5: ((0.1, 0.24),  # mdot1    ***
            (0.1, 0.24),  # mdot2    ***
            (0.1, 0.24),  # mdot3    ***
            (0.65, 0.77),  # x
            (0.0015, 0.0175),  # z
            (0.075, 0.2),  # qb
            (1., 2.6 / 1.4),  # g
            (1.2, 1.5),  # redshift
            (0.01, 10),  # f_b
            (0.01, 10),  # f_p
            ),
        6: ((0.08, 0.24),  # mdot1    ***
            (0.08, 0.24),  # mdot2    ***
            (0.08, 0.24),  # mdot3    ***
            (0.6, 0.77),  # x
            (0.0015, 0.0175),  # z
            (0.025, 0.2),  # qb
            (1., 2.6 / 1.4),  # g
            (1.2, 1.4),  # redshift
            (0.01, 10),  # f_b
            (0.01, 10),  # f_p
            ),
        7: ((0.1, 0.18),  # mdot1
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
    },

    7: {
        1: ((0.1, 0.24),  # mdot1
            (0.1, 0.24),  # mdot2
            (0.1, 0.24),  # mdot3
            (0.65, 0.77),  # x
            (0.0015, 0.0175),  # z
            (1., 2.6 / 1.4),  # g
            (1.2, 1.5),  # redshift
            (0.01, np.inf),  # f_b
            (0.01, np.inf),  # f_p
            ),
        2: ((0.1, 0.24),  # mdot1
            (0.1, 0.24),  # mdot2
            (0.1, 0.24),  # mdot3
            (0.65, 0.77),  # x
            (0.0015, 0.0175),  # z
            (1., 2.6 / 1.4),  # g
            (1.2, 1.7),  # redshift
            (0.01, np.inf),  # f_b
            (0.01, np.inf),  # f_p
            ),
        3: ((0.1, 0.24),  # mdot1
            (0.1, 0.24),  # mdot2
            (0.1, 0.24),  # mdot3
            (0.65, 0.77),  # x
            (0.0015, 0.0175),  # z
            (1., 2.6 / 1.4),  # g
            (1.2, 1.4),  # redshift
            (0.01, np.inf),  # f_b
            (0.01, np.inf),  # f_p
            ),
        4: ((0.1, 0.24),  # mdot1
            (0.1, 0.24),  # mdot2
            (0.1, 0.24),  # mdot3
            (0.65, 0.75),  # x
            (0.0015, 0.0175),  # z
            (1., 2.6 / 1.4),  # g
            (1.2, 1.4),  # redshift
            (0.01, 10),  # f_b
            (0.01, 10),  # f_p
            ),
        5: ((0.1, 0.24),  # mdot1
            (0.1, 0.24),  # mdot2
            (0.1, 0.24),  # mdot3
            (0.65, 0.75),  # x
            (0.0025, 0.0125),  # z
            (1., 2.6 / 1.4),  # g
            (1.2, 1.4),  # redshift
            (0.01, 10),  # f_b
            (0.01, 10),  # f_p
            ),
        6: ((0.1, 0.24),  # mdot1
            (0.1, 0.24),  # mdot2
            (0.1, 0.24),  # mdot3
            (0.65, 0.75),  # x
            (0.0015, 0.0125),  # z
            (1., 2.6 / 1.4),  # g
            (1.2, 1.4),  # redshift
            (0.01, 10),  # f_b
            (0.01, 10),  # f_p
            ),
        7: ((0.1, 0.24),  # mdot1
            (0.1, 0.24),  # mdot2
            (0.1, 0.24),  # mdot3
            (0.65, 0.75),  # x
            (0.0015, 0.0125),  # z
            (1., 2.3 / 1.4),  # g
            (1.2, 1.4),  # redshift
            (0.01, 10),  # f_b
            (0.01, 10),  # f_p
            ),
        8: ((0.1, 0.22),  # mdot1
            (0.1, 0.22),  # mdot2
            (0.1, 0.22),  # mdot3
            (0.65, 0.73),  # x
            (0.0025, 0.0075),  # z
            (1.7 / 1.4, 2.3 / 1.4),  # g
            (1.2, 1.4),  # redshift
            (0.01, 10),  # f_b
            (0.01, 10),  # f_p
            ),
        9: ((0.1, 0.22),  # mdot1
            (0.1, 0.22),  # mdot2
            (0.1, 0.22),  # mdot3
            (0.65, 0.73),  # x
            (0.0025, 0.0075),  # z
            (1.7 / 1.4, 2.3 / 1.4),  # g
            (1.2, 1.6),  # redshift
            (0.01, 10),  # f_b
            (0.01, 10),  # f_p
            ),
        10: ((0.08, 0.22),  # mdot1
             (0.08, 0.22),  # mdot2
             (0.08, 0.22),  # mdot3
             (0.7, 0.74),  # x
             (0.0025, 0.0075),  # z
             (1.7 / 1.4, 2.3 / 1.4),  # g
             (1.2, 1.4),  # redshift
             (0.01, 10),  # f_b
             (0.01, 10),  # f_p
             ),
        11: ((0.1, 0.2),  # mdot1
             (0.1, 0.2),  # mdot2
             (0.1, 0.2),  # mdot3
             (0.7, 0.73),  # x
             (0.0025, 0.0075),  # z
             (1.7 / 1.4, 2.3 / 1.4),  # g
             (1.2, 1.4),  # redshift
             (0.01, 10),  # f_b
             (0.01, 10),  # f_p
             ),
    },

    8: {
        1: ((0.1, 0.24),  # mdot1
            (0.1, 0.24),  # mdot2
            (0.1, 0.24),  # mdot3
            ),
    },

    9: {
        1: ((0.1, 0.22),  # mdot1
            (0.1, 0.22),  # mdot2
            (0.1, 0.22),  # mdot3
            (0.65, 0.72),  # x
            (0.0025, 0.0075),  # z
            (1.7 / 1.4, 2.3 / 1.4),  # g
            (1.2, 1.4),  # redshift
            (0.01, 10),  # f
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

# ===== Defines order/number of params provided to BurstFit =====
# TODO: Ensure correspond to prior_bounds
param_keys = {
    1: ['mdot1', 'x', 'z', 'qb', 'g', 'redshift', 'd', 'inc'],
    2: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'qb', 'g', 'redshift', 'd', 'inc'],
    3: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'qb', 'g', 'redshift', 'd', 'xi_b', 'xi_p'],
    4: ['mdot1', 'mdot2', 'mdot3', 'z', 'qb', 'g', 'redshift', 'd', 'inc'],
    5: ['mdot1', 'mdot2', 'mdot3', 'z', 'qb', 'g', 'redshift', 'f_b', 'f_p'],
    6: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'qb', 'g', 'redshift', 'f_b', 'f_p'],
    7: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'g', 'redshift', 'f_b', 'f_p'],
    8: ['mdot1', 'mdot2', 'mdot3'],
    9: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'g', 'redshift', 'f'],
}

# ===== initial position of walkers =====
# TODO: organise by param_keys
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
    13: (0.239, 0.206, 0.153,
         0.768, 0.002, 0.08, 1.02, 1.48, 0.28, 0.75),
    14: (0.218, 0.179, 0.133,
         0.7, 0.003, 0.08, 1.02, 1.48, 0.25, 2.6),
    15: (0.214, 0.182, 0.133,
         0.71, 0.005, 1.5, 1.45, 0.64, 1.75),
    16: (0.18, 0.15, 0.11,
         0.73, 0.004, 1.54, 1.47, 0.6, 1.6),
    17: (0.18, 0.15, 0.11,
         0.73, 0.004, 1.54, 1.37, 0.55, 1.55),
    18: (0.17, 0.146, 0.106,
         0.7, 0.005, 0.1, 1.5, 1.4, 8.0, 63.),
    19: (0.17, 0.146, 0.106,
         0.76, 0.0025, 0.08, 1.05, 1.1, 6.5, 40.),
    20: (0.17, 0.146, 0.106,
         0.76, 0.0025, 0.03, 1.05, 1.3, 10, 75.),
    21: (0.235, 0.207, 0.152,
         0.7, 0.012, 0.05, 1.25, 1.55, 7, 55.),
    22: (0.21, 0.1789, 0.1311),
    23: (0.18, 0.16, 0.12),
    24: (0.193, 0.164, 0.120,
         0.7, 0.004, 0.026, 1.4, 1.3, 0.64, 1.5),
    25: (0.20, 0.16, 0.12,
         0.68, 0.004, 1.4, 1.3, 0.65, 1.6),
    26: (0.18, 0.15, 0.11,
         0.71, 0.004, 1.4, 1.35, 1.),
    27: (0.15, 0.13, 0.105,
         0.72, 0.005, 1.3, 1.35, 0.51, 1.2),
    28: (0.15, 0.13, 0.105,
         0.72, 0.005, 0.075, 1.3, 1.35, 0.51, 1.2),
}

# To add a new version definition, add an entry to each of the parameters
#   in version_definitions

# Structure:
#   -parameter
#     ---source
#         ---version
#               ---parameter value

version_defaults = {
    'param_keys':
        {
            'biggrid1': param_keys[1],
            'biggrid2': param_keys[2],
            'sim_test': param_keys[1],
            'sim10': param_keys[6],
            'grid4': param_keys[7],
            'heat': param_keys[7],
        },

    'bprops':
        {
            'biggrid1': ('dt', 'fluence', 'peak'),
            'biggrid2': ('dt', 'fluence', 'peak'),
            'grid4': ('rate', 'fluence', 'peak'),
            'heat': ('rate', 'fluence', 'peak'),
        },
    'disc_model':
        {
            'biggrid1': 'he16_d',
            'biggrid2': 'he16_d',
            'sim_test': 'he16_a',
            'sim10': 'he16_a',
            'grid4': 'he16_a',
            'heat': 'he16_a',
        },

    'interpolator':
        {
            'biggrid1': 1,
            'biggrid2': 1,
            'grid4': 1,
            'heat': 1,
        },

    'prior_bounds':
        {
            'biggrid1': {},
            'biggrid2': prior_bounds[2][2],
            'grid4': prior_bounds[7][10],
            'heat': prior_bounds[7][11],
        },

    'prior_pdfs':
        {
          'biggrid2': {
              'z': prior_pdfs['z'][1],
              'f_ratio': prior_pdfs['f_ratio'][1],
              'inc': prior_pdfs['inc'][1],
          },
          'grid4': {
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

    'initial_position':
        {
            'biggrid1': initial_position[1],
            'biggrid2': initial_position[4],
            'sim_test': initial_position[3],
            'sim10': initial_position[12],
            'grid4': initial_position[27],
            'heat': initial_position[27],
        },
}

version_definitions = {
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
                47: 16,
                48: 16,
                49: 16,
                50: 16,
                51: 17,
                52: 17,
                53: 17,
                54: 16,
                55: 18,
                56: 18,
                57: 19,
                58: 20,
                59: 21,
                60: 22,
                61: 23,
                62: 24,
                63: 25,
                64: 25,
                65: 25,
                66: 25,
                67: 26,
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
            'sim10': {
                1: 15,
                2: 15,
                3: 15,
                4: 15,
                5: 15,
                6: 15,
                7: 15,
            },
            'grid4': {
                1: 1,
                3: 2,
                4: 2,
                5: 3,
            },
            'heat': {
            },
        },

    'bprops':
        {
            'biggrid1': {},
            'biggrid2':
                {
                    63: ('rate', 'fluence', 'peak'),
                    64: 63,
                    65: 63,
                    66: 63,
                    67: 63,
                },
            'grid4': {},
            'heat': {
            },
        },
    'param_keys':
        {
            'biggrid1': {},
            'biggrid2':
                {
                    1: param_keys[1],
                    2: 1,
                    3: 1,
                    4: 1,
                    5: 1,
                    6: 1,
                    7: 1,
                    8: 1,
                    9: 1,
                    10: 1,
                    11: 1,
                    12: 1,
                    31: param_keys[3],
                    36: 36,
                    37: 36,
                    39: param_keys[4],
                    40: 39,
                    41: param_keys[5],
                    42: 41,
                    43: param_keys[6],
                    44: 43,
                    45: 43,
                    46: 43,
                    47: param_keys[7],
                    48: 47,
                    49: 47,
                    50: 47,
                    51: 43,
                    52: param_keys[8],
                    53: 52,
                    54: 47,
                    55: 47,
                    56: 47,
                    57: 47,
                    58: 47,
                    59: 43,
                    60: 43,
                    61: 47,
                    62: 47,
                    63: 47,
                    64: param_keys[9],
                    65: 47,
                    66: 47,
                    67: 47,
                },
            'sim_test': {},
            'sim10': {
                4: param_keys[2],
                5: 4,
                6: 4,
                7: 4,
            },
            'grid4': {
                5: param_keys[6],
            },
            'heat': {
            },
        },

    'prior_bounds':
        {
            'biggrid1': {},
            'biggrid2': {
                8: prior_bounds[1][1],
                9: prior_bounds[1][2],
                10: 9,
                11: 9,
                12: 9,
                13: prior_bounds[2][1],
                14: prior_bounds[2][2],
                15: 14,
                16: 14,
                17: 14,
                18: 14,
                19: 14,
                20: 14,
                21: 14,
                22: 14,
                23: 14,
                24: 14,
                25: prior_bounds[2][3],
                26: prior_bounds[2][4],
                27: 26,
                28: 26,
                29: 26,
                30: 26,
                31: prior_bounds[3][1],
                32: 26,
                33: 26,
                34: 26,
                35: prior_bounds[2][5],
                36: prior_bounds[3][2],
                37: 31,
                38: 35,
                39: prior_bounds[4][1],
                40: prior_bounds[4][2],
                41: prior_bounds[5][1],
                42: prior_bounds[5][2],
                43: prior_bounds[6][1],
                44: prior_bounds[6][2],
                45: prior_bounds[6][3],
                46: prior_bounds[6][4],
                47: prior_bounds[7][1],
                48: 47,
                49: prior_bounds[7][2],
                50: prior_bounds[7][3],
                51: prior_bounds[6][5],
                52: prior_bounds[8][1],
                53: 52,
                54: prior_bounds[7][4],
                55: prior_bounds[7][5],
                56: prior_bounds[7][6],
                57: prior_bounds[7][7],
                58: 57,
                59: prior_bounds[6][6],
                60: 59,
                61: prior_bounds[7][8],
                62: 61,
                63: 61,
                64: prior_bounds[9][1],
                65: 61,
                66: prior_bounds[7][9],
                67: 61,
            },
            'sim_test': {
                1: prior_bounds[1][1],
                2: prior_bounds[1][2],
                3: prior_bounds[1][3],
                4: 3,
                5: 3,
                6: 3,
                7: prior_bounds[1][4],
            },
            'sim10': {
                1: prior_bounds[6][3],
                2: 1,
                3: 1,
                4: prior_bounds[2][6],
                5: prior_bounds[2][7],
                6: 5,
                7: 5
            },
            'grid4': {
                5: prior_bounds[6][7],
            },
            'heat': {
            },
        },

    'prior_pdfs':
        {
         'biggrid2': {},
         'grid4': {
            2: prior_pdfs['f_ratio'][2],
            4: prior_pdfs['f_ratio'][2],
                 },
         'heat': {
            2: prior_pdfs['f_ratio'][2],
            4: prior_pdfs['f_ratio'][2],
                 },
        },

    'initial_position':
        {
            'biggrid1': {},
            'biggrid2': {
                21: initial_position[4],
                22: initial_position[5],
                23: initial_position[6],
                24: 23,
                25: 23,
                28: 23,
                29: 23,
                31: initial_position[7],
                36: 31,
                37: 31,
                38: initial_position[8],
                39: initial_position[9],
                40: 39,
                41: initial_position[10],
                42: initial_position[11],
                43: initial_position[12],
                44: 43,
                45: 43,
                46: 43,
                47: initial_position[15],
                48: initial_position[16],
                49: 48,
                50: initial_position[17],
                51: initial_position[14],
                52: initial_position[22],
                53: initial_position[23],
                54: 50,
                55: 50,
                56: 50,
                57: 50,
                58: 50,
                59: initial_position[24],
                60: 59,
                61: initial_position[25],
                62: 61,
                63: 61,
                64: initial_position[26],
                65: 61,
                66: 61,
                67: 61,
            },
            'sim_test': {},
            'sim10': {
                2: initial_position[13],
                3: initial_position[14],
                4: initial_position[18],
                5: initial_position[19],
                6: initial_position[20],
                7: initial_position[21],
            },
            'grid4': {
                5: initial_position[28],
            },
            'heat': {
            },
        },

    'disc_model':
        {
            'biggrid1': {
                2: 'he16_a',
                5: 2,
                6: 2,
            },
            'biggrid2': {
                2: 'he16_a',
                3: 'he16_b',
                10: 2,
                11: 3,
                12: 'he16_c',
                14: 2,
                15: 3,
                16: 12,
                18: 2,
                19: 3,
                20: 12,
                22: 2,
                23: 3,
                24: 12,
                25: 3,
                26: 2,
                27: 3,
                38: 2,
            },
            'sim_test': {},
            'sim10': {},
            'grid4': {
            },
            'heat': {
            },
        },
}


class McmcVersion:
    """Class for holding different mcmc versions
    """

    def __init__(self, source, version):
        source = grid_strings.check_synth_source(source)

        if source not in version_defaults['param_keys']:
            raise ValueError(f'source ({source}) not defined in mcmc_versions')
        elif version not in version_definitions['interpolator'][source]:
            print(f'version {version} of source {source} ' +
                  'not defined in mcmc_versions. Using default values')

        self.source = source
        self.version = version
        self.param_keys = get_parameter(source, version, 'param_keys')
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
                + f'\nbprops           : {self.bprops}'
                + f'\ninitial position : {self.initial_position}'
                + f'\ndisc model       : {self.disc_model}'
                + f'\ninterpolator     : {self.interpolator}'
                + f'\nprior bounds     : {self.prior_bounds}'
                )


def get_parameter(source, version, parameter):
    source = grid_strings.check_synth_source(source)
    default = version_defaults[parameter][source]
    output = version_definitions[parameter][source].get(version, default)

    if (parameter != 'interpolator') and type(output) is int:
        return version_definitions[parameter][source][output]
    else:
        return output


def get_prior_pdfs(source, version):
    pdfs = {}
    for var in prior_pdfs:
        default = version_defaults['prior_pdfs'][source][var]
        value = version_definitions['prior_pdfs'][source].get(version, default)

        if type(value) is int:  # allow pointing to previous versions
            pdfs[var] = version_definitions['prior_pdfs'][source].get(value, default)
        else:
            pdfs[var] = value

    return pdfs
