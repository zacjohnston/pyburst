import numpy as np

version_defaults = {
    'param_keys': {
        'gs1826': ['accrate', 'x', 'z', 'qb', 'mass'],
        'biggrid1': ['accrate', 'x', 'z', 'qb', 'mass'],
        'biggrid2': ['accrate', 'x', 'z', 'qb', 'mass'],
                  },
    'bprops': {
        'gs1826': ('dt', 'u_dt', 'fluence', 'u_fluence', 'peak', 'u_peak'),
        'biggrid1': ('dt', 'u_dt', 'fluence', 'u_fluence', 'peak', 'u_peak'),
        'biggrid2': ('dt', 'u_dt', 'fluence', 'u_fluence', 'peak', 'u_peak'),
              },
    'batches_exclude': {
        'gs1826': {},
        'biggrid1': {'batch': [255, 256, 257, 258, 259, 260, 471, 472, 473, 418, 419, 420]},
        'biggrid2': {},
                       },

    'params_exclude': {
        'gs1826': {
                'qb': [0.5, 0.7, 0.9],
                'x': [0.6],
                'xi': [0.8, 0.9, 1.0, 1.1, 3.2],
                'z': [0.001, 0.003],
                  },
        'biggrid1': {},
        'biggrid2': {
                'accrate': np.append(np.arange(5, 10) / 100, np.arange(11, 24, 2) / 100),
                'x': [0.5, 0.6, 0.8],
                'z': [0.001],
                'qb': [.075],
                'mass': [0.8, 3.2],
        },
    },
}


version_definitions = {
    'param_keys': {  # This will set the order of params when calling interpolator
        'gs1826': {},
        'biggrid1': {},
        'biggrid2': {
            14: ['accrate', 'z', 'qb', 'mass'],
            15: ['accrate', 'x', 'z', 'qb', 'mass'],
            16: ['accrate', 'x', 'z', 'mass'],
            18: ['accrate', 'x', 'z', 'mass'],
            19: ['accrate', 'x', 'z', 'mass'],
            20: ['accrate', 'x', 'z', 'mass'],
            21: ['accrate', 'x', 'z', 'qb', 'mass'],
            22: ['accrate', 'x', 'z', 'qb', 'mass'],
            23: ['accrate', 'x', 'z', 'mass'],
            24: ['accrate', 'x', 'z', 'mass'],
            25: ['accrate', 'x', 'z', 'mass'],
        },
    },

    'bprops': {
        'gs1826': {},
        'biggrid1': {},
        'biggrid2': {
            25: ('rate', 'u_rate', 'fluence', 'u_fluence', 'peak', 'u_peak'),
                    },
    },

    'batches_exclude': {
        'gs1826': {},
        'biggrid1': {
            1: {
                'batch': [255, 256, 257, 258, 259, 260, 471, 472, 473, 418, 419, 420]},
            },
        'biggrid2': {
            1: {},
        },
    },

    'params_exclude': {
        'gs1826': {
            1: {
                'qb': [0.5, 0.7, 0.9],
                'x': [0.6],
                'xi': [0.8, 0.9, 1.0, 1.1, 3.2],
                'z': [0.001, 0.003],
            },
        },
        'biggrid1': {},
        'biggrid2': {
            14: {
                'accrate': np.append(np.arange(5, 8) / 100, np.arange(9, 24, 2) / 100),
                'x': [0.5, 0.6, 0.8, 0.65, 0.77],
                'z': [0.001],
                'qb': [.075],
                'mass': [0.8, 3.2],
                },
            15: {
                'accrate': np.append(np.arange(5, 8) / 100, np.arange(9, 24, 2) / 100),
                'x': [0.5, 0.6, 0.8],
                'z': [0.001],
                'qb': [.075],
                'mass': [0.8, 3.2],
                },
            16: {
                'accrate': np.append(np.arange(5, 10) / 100, np.arange(11, 24, 2) / 100),
                'x': [0.5, 0.6, 0.8],
                'z': [0.001],
                'qb': [0.025, .075],
                'mass': [0.8, 3.2],
                },
            17: {
                'accrate': np.append(np.arange(5, 10) / 100, np.arange(11, 24, 2) / 100),
                'x': [0.5, 0.6, 0.8],
                'z': [0.001],
                'qb': [0.025],
                'mass': [0.8, 3.2],
                },
            18: {
                'accrate': np.append(np.arange(5, 10) / 100, np.arange(11, 24, 2) / 100),
                'x': [0.5, 0.6, 0.77, 0.8],
                'z': [0.001, 0.0175],
                'qb': [0.025, 0.075, 0.2],
                'mass': [0.8, 3.2],
                },
            19: {
                'accrate': np.append(np.arange(5, 10) / 100, np.arange(11, 24, 2) / 100),
                'x': [0.5, 0.6, 0.77, 0.8],
                'z': [0.001, 0.0175],
                'qb': [0.025, 0.075, 0.2],
                'mass': [0.8, 2.6, 3.2],
                },
            20: {
                'accrate': np.append(np.arange(5, 10) / 100, np.arange(11, 24, 2) / 100),
                'x': [0.5, 0.6, 0.77, 0.8],
                'z': [0.001, 0.0175],
                'qb': [0.025, 0.075, 0.2],
                'mass': [0.8, 2.6, 3.2],
                },
            21: {
                'accrate': np.arange(5, 8) / 100,
                'x': [0.5, 0.8],
                'z': [0.001],
                'mass': [0.8, 3.2],
                },
            22: {
                'accrate': np.arange(5, 8) / 100,
                'x': [0.5, 0.8],
                'z': [0.001],
                'mass': [0.8, 3.2],
                },
            23: {
                'accrate': np.append(np.arange(5, 10), np.arange(11, 24, 2))/100,
                'x': [0.5, 0.6, 0.75, 0.77, 0.8],
                'qb': [0.025, 0.075, 0.125, 0.2],
                'z': [0.001, 0.0015, 0.0125, 0.0175],
                'mass': [0.8, 1.4, 2.6, 3.2],
                },
            24: {
                'accrate': np.append(np.arange(5, 10), np.arange(11, 24, 2))/100,
                'x': [0.5, 0.6, 0.75, 0.77, 0.8],
                'qb': [0.025, 0.075, 0.125, 0.2],
                'z': [0.001, 0.0015, 0.0125, 0.0175],
                'mass': [0.8, 1.4, 2.6, 3.2],
                },
            25: {
                'accrate': np.append(np.arange(5, 10), np.arange(11, 24, 2)) / 100,
                'x': [0.5, 0.6, 0.75, 0.77, 0.8],
                'qb': [0.025, 0.075, 0.125, 0.2],
                'z': [0.001, 0.0015, 0.0125, 0.0175],
                'mass': [0.8, 1.4, 2.6, 3.2],
            },
        },
    },
}


class InterpVersion:
    """Class for defining different interpolator versions
    """
    def __init__(self, source, version):
        self.source = source
        self.version = version
        self.param_keys = get_param_keys(source, version)
        self.bprops = get_bprops(source, version)
        self.batches_exclude = get_batches_exclude(source, version)
        self.params_exclude = get_params_exclude(source, version)

    def __repr__(self):
        return (f'MCMC version definitions for {self.source} V{self.version}'
                + f'\nparam keys     : {self.param_keys}'
                + f'\nbprops         : {self.bprops}'
                + f'\nbatches exclude: {self.batches_exclude}'
                + f'\nparams_exclude : {self.params_exclude}'
                )


# ===== Convenience functions =====
def get_param_keys(source, version):
    default = version_defaults['param_keys'][source]
    return version_definitions['param_keys'][source].get(version, default)


def get_bprops(source, version):
    default = version_defaults['bprops'][source]
    return version_definitions['bprops'][source].get(version, default)


def get_batches_exclude(source, version):
    default = version_defaults['batches_exclude'][source]
    return version_definitions['batches_exclude'][source].get(version, default)


def get_params_exclude(source, version):
    default = version_defaults['params_exclude'][source]
    return version_definitions['params_exclude'][source].get(version, default)