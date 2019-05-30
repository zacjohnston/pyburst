import numpy as np

# Defines various versions/subsets of model grids

version_defaults = {
    'exclude_any': {
        'adelle': {},
        'biggrid2': {},
        'biggrid3': {},
        'grid4': {},
        'grid5': {
            'x': [0.74], 'qb': [0.05, 0.15], 'mass': [2.5],
        },
        'grid6': {},
        'synth5': {},
        'sample5': {},
        'sample2': {},
        'res1': {},
        'test1': {},
        'heat': {
            'batch': [1]
        },
        'triplets': {},
        'he1': {
            'qb': [0.0, 0.4],
        },
        'he2': {
            'qnuc': [3.0], 'qb': [0.5, 0.8], 'mass': [2.6], 'x': [0.001],
        },
        'alpha1': {},
        'alpha2': {},
        'mesa': {},
    },

    'exclude_all': {
        'adelle': [{}],
        'biggrid2': [{}],
        'biggrid3': [{}],
        'grid4': [{}],
        'grid5': [{}],
        'grid6': [{}],
        'synth5': [{}],
        'sample5': [{}],
        'sample2': [{}],
        'heat': [{}],
        'res1': [{}],
        'test1': [{}],
        'triplets': [{}],
        'he1': [{}],
        'he2': {},
        'alpha1': [{}],
        'alpha2': [{}],
        'mesa': {},
    },
}


version_definitions = {
    'exclude_any':
        {
            'adelle': {},
            'biggrid2': {},
            'biggrid3': {},
            'test1': {},
            'grid4': {},
            'grid5': {
                2: {  # sparse grid
                    'accrate': [0.2], 'z': [0.01], 'x': [0.74],
                    'qb': [0.05, 0.15, 0.3], 'mass': [1.4, 2.5],
                },
                3: {  # sparse grid
                    'accrate': [0.2], 'z': [0.01, 0.015], 'x': [0.74],
                    'qb': [0.05, 0.15, 0.3], 'mass': [1.4, 2.5],
                },
            },
            'grid6': {},
            'synth5': {},
            'sample5': {},
            'sample2': {},
            'heat': {
                1: {
                    'batch': [1, 10, 11, 12, 14, 15, 16, 17],  # gs1826 models
                },

                2: {
                    'batch': [1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 17],  # 4u1820 models
                    'accrate': 0.1,
                },

                3: {  # reduced gs1826
                    'batch': list(np.concatenate((np.arange(1, 4), np.arange(7, 18)))),
                    'accrate': [0.14]
                },
            },
            'res1': {
                1: {
                    'batch': [6, 7],  # gs1826 models
                    'accdepth': 1e21
                },
                2: {
                    'batch': [1, 2, 3, 4, 5]   # 4u1820 models
                },
            },
            'triplets': {},
            'he1': {},
            'he2': {
                1: {
                    'qnuc': [3.0], 'qb': [0.5, 0.8], 'mass': [2.6], 'x': [0.001],
                    'accrate': [0.4],
                },
                2: {
                    'qnuc': [3.0], 'qb': [0.5, 0.8], 'mass': [1.4, 2.6], 'x': [0.001],
                    'accrate': [0.4],
                },
            },
            'alpha1': {},
            'alpha2': {},
            'mesa': {},
        },
    'exclude_all':
        {
            'adelle': {},
            'biggrid2': {},
            'biggrid3': {},
            'grid4': {},
            'grid5': {},
            'grid6': {},
            'synth5': {},
            'sample5': {},
            'sample2': {},
            'heat': {},
            'res1': {},
            'test1': {},
            'triplets': {},
            'he1': {},
            'he2': {},
            'alpha1': {},
            'alpha2': {},
            'mesa': {},
        }
}


class GridVersion:
    """Class for defining different interpolator versions

    Conventions (NOT enforced)
    -----------
    version = -1: No models excluded from grid. Entire grid accessible.
    version = 0: Defaults excluded from grid
    version > 0: As defined in this file
    """
    def __init__(self, source, version):
        self.source = source
        self.version = version
        self.exclude_any = get_parameter(source, version, 'exclude_any')
        self.exclude_all = get_parameter(source, version, 'exclude_all')

    def __repr__(self):
        return (f'Grid version definitions for {self.source} V{self.version}'
                + f'\nexclude_any : {self.exclude_any}'
                + f'\nexclude_all : {self.exclude_all}'
                )


def get_parameter(source, version, parameter):
    if version == -1:
        return {'exclude_any': {}, 'exclude_all': [{}]}.get(parameter)

    default = version_defaults[parameter][source]
    out = version_definitions[parameter][source].get(version, default)

    if out == default:
        print(f'{parameter} not defined, using default')
    if type(out) is int:
        return version_definitions[parameter][source][out]
    else:
        return out
