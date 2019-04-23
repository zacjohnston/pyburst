# standard
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import os
import time
import pickle

# kepler_grids
from pyburst.grids import grid_tools, grid_strings, grid_versions
from . import interp_versions

GRIDS_PATH = os.environ['KEPLER_GRIDS']
MODELS_PATH = os.environ['KEPLER_MODELS']

key_map = {'dt': 'tDel', 'u_dt': 'uTDel',
           'fluence': 'fluence', 'u_fluence': 'uFluence',
           'peak': 'peakLum', 'u_peak': 'uPeakLum'}

# TODO:
#   - function to re-generate interpolator files

class Kemulator:
    """Kepler emulator class. Creates a 'model' of kepler results

    parameters
    ----------
    source : str
        source object of grid to emaulate (e.g., gs1826)
    verbose : bool
        print diagnostics
    re_interp: bool
        setup interpolator
    """

    def __init__(self, source, version, verbose=True, re_interp=True,
                 lampe_analyser=False, check_complete=True):
        self.verbose = verbose
        source = grid_strings.source_shorthand(source)
        self.source = source
        self.version = version
        self.lampe_analyser = lampe_analyser
        self.interpolator = None

        summ = grid_tools.load_grid_table('summ', source=source, lampe_analyser=lampe_analyser)
        params = grid_tools.load_grid_table('params', source=source)

        self.version_def = interp_versions.InterpVersion(source=source, version=version)
        self.grid_def = grid_versions.GridVersion(source=source,
                                                  version=self.version_def.grid_version)
        self.bprops = self.version_def.bprops

        params = grid_tools.exclude_params(table=params,
                                           params=self.grid_def.exclude_any,
                                           logic='any')
        params = grid_tools.exclude_params(table=params,
                                           params=self.grid_def.exclude_all,
                                           logic='all')
        idxs_kept = params.index
        summ = summ.loc[idxs_kept]
        self.summ = summ
        self.params = params

        if check_complete:
            self.check_completeness()

        if re_interp:
            self.setup_interpolator(self.bprops)
        else:
            self.load_interpolator()

    def printv(self, string, **kwargs):
        """Prints string if self.verbose == True
        """
        if self.verbose:
            print(string, **kwargs)

    def reduce_summ(self, params):
        """Returns reduced summ table with specified parameters (e.g., mass, qb)
        """
        reduced_idxs = grid_tools.reduce_table_idx(self.params, params=params)
        return self.summ.iloc[reduced_idxs]

    def save_interpolator(self):
        """Saves (pickles) interpolator to file
        """
        self.printv(f'Saving interpolator')
        filename = f'interpolator_{self.source}_V{self.version}'
        filepath = os.path.join(GRIDS_PATH, 'sources', self.source,
                                'interpolator', filename)
        self.printv(f'Saving interpolator: {filepath}')
        pickle.dump(self.interpolator, open(filepath, 'wb'))

    def load_interpolator(self):
        """Loads previously-saved (pickled) interpolators from file
        """
        self.printv(f'Loading interpolator')
        filename = f'interpolator_{self.source}_V{self.version}'
        filepath = os.path.join(GRIDS_PATH, 'sources', self.source,
                                'interpolator', filename)
        self.interpolator = pickle.load(open(filepath, 'rb'))

    def setup_interpolator(self, bprops):
        """Creates interpolator object from kepler grid data

        bprops : [str]
            burst properties to interpolate (e.g., dt, fluence)
        """
        self.printv('Creating interpolator on grid: ')
        points = []

        for param in self.version_def.param_keys:
            param_points = np.array(self.params[param])
            points += [param_points]
            self.printv(f'{param}:  {np.unique(param_points)}')

        points = tuple(points)
        n_models = len(self.params)
        n_bprops = len(bprops)
        values = np.full((n_models, n_bprops), np.nan)
        self.printv(f'Number of models: {n_models}')

        t0 = time.time()
        self.printv(f'Creating interpolator:')

        for i, bp in enumerate(bprops):
            if self.lampe_analyser:
                key = key_map[bp]
            else:
                key = bp
            values[:, i] = np.array(self.summ[key])  # * 0.9
        self.interpolator = LinearNDInterpolator(points, values)
        t1 = time.time()
        self.printv(f'Setup time: {t1-t0:.1f} s')

    def emulate_burst(self, params):
        """Returns interpolated burst properties for given params

        params : [acc, x, z, qb, mass]
        """
        # check_params_length(params, length=len(self.version_def.param_keys))
        return self.interpolator(params)

    def check_completeness(self):
        """Checks for completeness of model grid, and raises an error if incomplete
        """
        self.printv('Checking model grid completeness')
        product = 1
        n_models = len(self.params)

        for param in self.version_def.param_keys:
            product *= len(np.unique(self.params[param]))

        if product != n_models:
            raise RuntimeError(f'Model grid is not complete! Expected {product} models, '
                               f'but only have {n_models}. '
                               'Some parameter combinations are missing. '
                               "Use arg check_complete=False to disable this check.")


def check_params_length(params, length=5):
    """Checks that five parameters have been provided
    """
    # TODO: generalise this for emulators with some parameters fixed
    def check(array):
        if len(array) != length:
            raise ValueError("'params' must specify each of (acc, x, z, qb, mass)")

    if len(params.shape) == 1:
        check(params)
    elif len(params.shape) == 2:
        check(params[0])
