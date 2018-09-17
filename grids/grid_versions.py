import numpy as np

# Defines various versions/subsets of model grids

version_defaults = {
    'exclude_any': {
        'adelle': {},

        'biggrid2': {
            'accrate': np.concatenate((np.arange(5, 10)/100, np.arange(11, 24, 2)/100)),
            'qb': [0.025, 0.075, 0.125, 0.2],
            'z': [0.001, 0.0175],
            'x': [0.5, 0.6, 0.75, 0.77, 0.8],
            'mass': [0.8, 1.4, 3.2, 2.6],
        },

        'biggrid3': {
            'batch': [1],
            'accrate': 0.1,
        },

        'grid4': {
            'accrate': [0.22],
            'mass': [2.0],
        },

        'res1': {},

        'heat': {'batch': [1]},
    },

    'exclude_all': {
        'adelle': [{}],
        'biggrid2': [{}],
        'biggrid3': [{}],
        'grid4': [
            {'x': 0.72, 'accdepth': 1e20},
            {'x': 0.73, 'accdepth': 1e20},
         ],

        'heat': [{}],
        'res1': [{}],
    },
}


version_definitions = {
    'exclude_any':
        {
            'adelle': {},
            'biggrid2': {},
            'biggrid3': {},
            'grid4': {},
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
                    'accdepth': 1e21
                },
                2: {},
            }
        },
    'exclude_all':
        {
            'adelle': {},
            'biggrid2': {},
            'biggrid3': {},
            'grid4': {},
            'heat': {},
            'res1': {},
        }
}


class GridVersion:
    """Class for defining different interpolator versions
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
    default = version_defaults[parameter][source]
    out = version_definitions[parameter][source].get(version, default)

    if out == default:
        print(f'{parameter} not defined, using default')
    if type(out) is int:
        return version_definitions[parameter][source][out]
    else:
        return out
