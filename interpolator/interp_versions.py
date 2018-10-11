import numpy as np

# TODO: outsource some of this to grid_versions?
version_defaults = {
    'param_keys': {
        'grid5': ['accrate', 'x', 'z', 'qb', 'mass'],
        'heat': ['accrate', 'x', 'z', 'mass'],
    },
    'bprops': {
        'grid5': ('rate', 'u_rate', 'fluence', 'u_fluence', 'peak', 'u_peak'),
        'heat': ('rate', 'u_rate', 'fluence', 'u_fluence', 'peak', 'u_peak'),
    },
    'exclude_any': {
        'grid5': {},
        'heat': {},

    },
    'exclude_all': {
        'grid5': [{}],
        'heat': [{}],
    },
}


version_definitions = {
    'param_keys': {  # This will set the order of params when calling interpolator
        'grid5': {
            2: ['accrate', 'x', 'z', 'mass'],
        },
        'heat': {},
    },

    'bprops': {
        'grid5': {},
        'heat': {},
    },

    'exclude_any': {
        'grid5': {},

        'heat': {
            1: {  # gs1826 models
                'batch': np.concatenate((np.arange(1, 9), np.arange(10, 13),
                                        np.arange(14, 20))),
            },

            2: {  # 4u1820 models
                'batch': np.concatenate((np.arange(1, 10), [13])),
                'accrate': 0.1,
                },
            },
    },
    
    'exclude_all': {
        'grid5': {},
        'heat': {},
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

