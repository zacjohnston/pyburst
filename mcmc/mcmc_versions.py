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
    1: ['mdot1', 'x', 'z', 'qb', 'g', 'redshift', 'd', 'inc'],
    2: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'qb', 'g', 'redshift', 'd', 'inc'],
    3: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'qb', 'g', 'redshift', 'd', 'xi_b', 'xi_p'],
    4: ['mdot1', 'mdot2', 'mdot3', 'z', 'qb', 'g', 'redshift', 'd', 'inc'],
    5: ['mdot1', 'mdot2', 'mdot3', 'z', 'qb', 'g', 'redshift', 'f_b', 'f_p'],
    6: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'qb', 'g', 'redshift', 'f_b', 'f_p'],
    7: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'g', 'redshift', 'f_b', 'f_p'],
    8: ['mdot1', 'mdot2', 'mdot3'],
    9: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'g', 'redshift', 'f'],
    10: ['mdot1', 'mdot2', 'mdot3', 'x', 'logz', 'qb', 'g', 'redshift', 'f_b', 'f_p'],
    11: ['mdot1', 'mdot2', 'mdot3', 'x', 'z', 'qb1', 'qb2', 'qb3', 'g', 'redshift', 'f_b', 'f_p'],
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
        8: ((0.1, 0.18),  # mdot1
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
        12: ((0.1, 0.18),  # mdot1
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

    10: {
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

    11: {
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
    28: (0.162, 0.138, 0.102,
         0.71, 0.005, 0.075, 1.5, 1.35, 0.55, 1.3),
    29: (0.15, 0.13, 0.105,
         0.72, -0.5, 0.075, 1.3, 1.35, 0.51, 1.2),
    30: (0.162, 0.138, 0.102,
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
        'grid4': param_keys[7],
        'grid5': param_keys[6],
        'heat': param_keys[7],
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
        'grid4': prior_bounds[7][10],
        'grid5': prior_bounds[6][8],
        'heat': prior_bounds[7][11],
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
        'grid4': initial_position[27],
        'grid5': initial_position[28],
        'heat': initial_position[27],
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
            5: param_keys[6],
            7: 5,
        },

        'grid5': {
            1: param_keys[7],
            4: param_keys[10],
            5: param_keys[11],
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
            5: prior_bounds[6][7],
            6: prior_bounds[7][12],
            7: prior_bounds[6][8],
        },

        'grid5': {
            1: prior_bounds[7][12],
            4: prior_bounds[10][1],
            5: prior_bounds[11][1],
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
            5: initial_position[28],
            7: initial_position[28],
        },

        'grid5': {
            1: initial_position[27],
            4: initial_position[29],
            5: initial_position[30],
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
