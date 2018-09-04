import numpy as np

# Defines various versions/subsets of model grids

version_defaults = {
    'params_exclude':
        {
            'biggrid2':
                {
                    'accrate': np.append(np.arange(5, 10) / 100, np.arange(11, 24, 2) / 100),
                    'x': [0.5, 0.6, 0.8],
                    'z': [0.001],
                    'qb': [.075],
                    'mass': [0.8, 3.2],
                },
            'grid4':
                {
                    'accrate': [0.22],
                    'mass': [2.0],
                },
        },

    'exclude_all':
        {
            'biggrid2': [{}],
            'grid4':
                [
                    {'x': 0.72, 'accdepth': 1e20},
                    {'x': 0.73, 'accdepth': 1e20},
                ],
        },
}


version_definitions = {
    'params_exclude':
        {
            'biggrid2': {},
            'grid4': {},
        },
    'exclude_all':
        {
            'biggrid2': {},
            'grid4': {},
        }
}


class GridVersion:
    """Class for defining different interpolator versions
    """
    def __init__(self, source, version):
        self.source = source
        self.version = version
        self.params_exclude = get_parameter(source, version, 'params_exclude')
        self.exclude_all = get_parameter(source, version, 'exclude_all')

    def __repr__(self):
        return (f'Grid version definitions for {self.source} V{self.version}'
                + f'\nparams_exclude : {self.params_exclude}'
                + f'\nexclude_all : {self.exclude_all}'
                )


def get_parameter(source, version, parameter):
    default = version_defaults[parameter][source]
    out = version_definitions[parameter][source].get(version, default)
    if type(out) is int:
        return version_definitions[parameter][source][out]
    else:
        return out