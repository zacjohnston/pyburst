import os

# ========================================================
# Miscellaneous printing functions
# ========================================================
GRIDS_PATH = os.environ['KEPLER_GRIDS']


class Debugger:
    def __init__(self, debug):
        self.debug = debug
        if self.debug:
            self.indent = Indenter()

    def start_function(self, name):
        if self.debug:
            self.indent.print_(f'>>>Function: {name}')
            self.indent.__enter__()

    def end_function(self):
        if self.debug:
            self.indent.__exit__()

    def variable(self, var_name, var_value, formatter='f'):
        if self.debug:
            string = f'{var_name} = {var_value:{formatter}}'
            self.indent.print_(string)

    def print_(self, string):
        if self.debug:
            self.indent.print_(string)


class Indenter:
    def __init__(self):
        self.level = 0

    def __enter__(self):
        self.level += 1
        return self

    def __exit__(self):
        self.level -= 1

    def print_(self, text):
        print('    ' * self.level + text)


def print_dashes(n=40):
    print('-' * n)


def print_squiggles(n=40):
    print('~' * n)


def print_title(string='', n=80):
    line = '=' * n
    print(line)
    if string != '':
        print(string)
        print(line)


def print_warning(n=40):
    print('X' * n)


def print_stars(n=40):
    print('*' * n)


def print_list(list_in, delim='  ', fmt='f', decimal=5):
    """Prints list/array on same line, with given delimiter and precision
    """
    for i in range(len(list_in)):
        print(f'{list_in[i]:.{decimal}{fmt}}', end=delim)
    print()


def print_labelled_list(var, title='str', decimal=5, fmt='f'):
    dashes()
    print(title)
    print_list(var, decimal=decimal, fmt=fmt)


def print_sci(val, decimal=3):
    """Print given value in scientific notation, to given decimal places
    """
    print('{:.{n}e}'.format(val, n=decimal))


def check_params_length(params, n=8):
    """Checks that n parameters have been provided
    """
    if len(params) != n:
        raise ValueError(f'"params" must be of length {n}')


def check_same_length(x1, x2, description=''):
    if len(x1) != len(x2):
        raise ValueError(f'length mismatch: {description}:'
                         + f'\n\tarray1: {x1}'
                         + f'\n\tarray2: {x2}')


def printv(string, verbose):
    if verbose:
        print(string)
