

version_defaults = {
    'param_keys': {
        'grid5': ['accrate', 'x', 'z', 'qb', 'mass'],
        'grid6': ['accrate', 'x', 'z', 'qb', 'mass'],
        'he1': ['accrate', 'x', 'z', 'qb', 'mass'],
        'he2': ['accrate', 'qb', 'mass'],
    },
    'bprops': {
        'grid5': ('rate', 'u_rate', 'fluence', 'u_fluence', 'peak', 'u_peak'),
        'grid6': ('rate', 'u_rate', 'fluence', 'u_fluence', 'peak', 'u_peak'),
        'he1': ('rate', 'u_rate', 'fluence', 'u_fluence', 'peak', 'u_peak'),
        'he2': ('rate', 'u_rate', 'fluence', 'u_fluence'),
    },
}


version_definitions = {
    'param_keys': {  # The input params (and their order) when calling interpolator
        'grid5': {
            3: ['accrate', 'x', 'z', 'qb'],
        },
        'grid6': {},
        'he1': {},
        'he2': {},
    },

    # The burst properties being interpolated
    'bprops': {
        'grid5': {},
        'grid6': {},
        'he1': {},
        'he2': {
            1: ('rate', 'u_rate', 'fluence', 'u_fluence', 'tail_index', 'u_tail_index'),
            2: ('rate', 'u_rate'),
        },
    },

    # The base grid to interpolate over (see: grids/grid_versions.py)
    #       Note: if not defined, defaults to: grid_version = interp_version
    'grid_version': {
        'grid5': {},
        'grid6': {},
        'he1': {},
        'he2': {
            1: 0,
            2: 0,
        },
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
        self.grid_version = get_parameter(source, version, 'grid_version')

    def __repr__(self):
        return (f'Interpolator version definitions for {self.source} V{self.version}'
                + f'\nbase grid versions : {self.grid_version}'
                + f'\nparam keys         : {self.param_keys}'
                + f'\nbprops             : {self.bprops}'
                )


def get_parameter(source, version, parameter):
    if parameter == 'grid_version':
        default = version
    else:
        default = version_defaults[parameter][source]

    out = version_definitions[parameter][source].get(version, default)
    if type(out) is int and (parameter != 'grid_version'):
        return version_definitions[parameter][source][out]
    else:
        return out

