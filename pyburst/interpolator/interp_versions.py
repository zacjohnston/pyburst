import numpy as np

# TODO: outsource some of this to grid_versions?
version_defaults = {
    'param_keys': {
        'grid5': ['accrate', 'x', 'z', 'qb', 'mass'],
        'grid6': ['accrate', 'x', 'z', 'qb', 'mass'],
        'he1': ['accrate', 'x', 'z', 'qb', 'mass'],
    },
    'bprops': {
        'grid5': ('rate', 'u_rate', 'fluence', 'u_fluence', 'peak', 'u_peak'),
        'grid6': ('rate', 'u_rate', 'fluence', 'u_fluence', 'peak', 'u_peak'),
        'he1': ('rate', 'u_rate', 'fluence', 'u_fluence', 'peak', 'u_peak'),
    },
    'exclude_any': {
        'grid5': {},
        'grid6': {},
        'he1': {
            'qb': [0.0, 0.4],
        },

    },
    'exclude_all': {
        'grid5': [{}],
        'grid6': [{}],
        'he1': [{}],
    },
}


version_definitions = {
    'param_keys': {  # This will set the order of params when calling interpolator
        'grid5': {},
        'grid6': {},
        'he1': {},
    },

    'bprops': {
        'grid5': {},
        'grid6': {},
        'he1': {},
    },

    'exclude_any': {
        'grid5': {
            1: {
                'z': [0.01, 0.0125], 'x': [0.74], 'mass': [2.5], 'qb': [0.05, 0.15],
                # 'accrate': [0.1, 0.14],
            },
            2: {
                'z': [0.0025, 0.005], 'x': [0.74], 'qb': [0.05, 0.15], 'mass': [1.4, 2.5],
                'accrate': [0.08],
            },
            3: {
                'z': [0.0025], 'x': [0.73], 'qb': [0.05, 0.15], 'mass': [1.4],
                'accrate': [0.08],
            },
            4: {  # stripped-back grid
                'z': [0.0025, 0.005, 0.01], 'x': [0.67, 0.73, 0.76],
                'qb': [0.05, 0.1, 0.2], 'mass': [1.4, 2.3],
                'accrate': [0.08],
            },

        },
        'grid6': {},
        'he1': {},
    },

    'exclude_all': {
        'grid5': {},
        'grid6': {},
        'he1': {},
    },
}

class InterpVersion:
    """Class for defining different interpolator versions
    """
    def __init__(self, source, version):
        self.source = source
        self.version = version
        self.param_keys = get_parameter(source, version, 'param_keys')
        self.bprops = get_parameter(source, version, 'bprops')
        self.exclude_any = get_parameter(source, version, 'exclude_any')
        self.exclude_all = get_parameter(source, version, 'exclude_all')

    def __repr__(self):
        return (f'Interpolator version definitions for {self.source} V{self.version}'
                + f'\nparam keys     : {self.param_keys}'
                + f'\nbprops         : {self.bprops}'
                + f'\nexclude_any : {self.exclude_any}'
                )


def get_parameter(source, version, parameter):
    default = version_defaults[parameter][source]
    out = version_definitions[parameter][source].get(version, default)
    if type(out) is int:
        return version_definitions[parameter][source][out]
    else:
        return out

