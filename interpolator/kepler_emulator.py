# standard
import numpy as np
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
import os
import time
import pickle

# kepler_grids
from ..grids import grid_tools, grid_strings

GRIDS_PATH = os.environ['KEPLER_GRIDS']
MODELS_PATH = os.environ['KEPLER_MODELS']

batches_exclude = {'biggrid1': {'batch': [255, 256, 257, 258, 259, 260, 471,
                                          472, 473, 418, 419, 420]},
                   'biggrid2': {},
                   }

params_exclude = {'gs1826': {'qb': [0.5, 0.7, 0.9],
                             'x': [0.6],
                             'xi': [0.8, 0.9, 1.0, 1.1, 3.2],
                             'z': [0.001, 0.003],
                             },
                  'biggrid1': {},
                  'biggrid2': {'qb': [.075],
                               'x': [0.5, 0.6, 0.8],
                               # 'accrate': np.arange(5, 24, 2)/100,
                               'accrate': np.append(np.arange(5, 8)/100,
                                                    np.arange(9, 24, 2)/100),
                               'z': [0.001],
                               'z': [0.001, 0.0125, 0.0175],
                               # 'mass': [1.4, 2.6]
                               'mass': [0.8, 3.2]
                               },
                  }

key_map = {'dt': 'tDel', 'u_dt': 'uTDel',
           'fluence': 'fluence', 'u_fluence': 'uFluence',
           'peak': 'peakLum', 'u_peak': 'uPeakLum'}


class Kemulator:
    """========================================================
    Kepler emulator class. Creates a 'model' of kepler results
    ========================================================
    source              = str  : source object of grid to emaulate (e.g., gs1826)
    verbose             = bool : print diagnostics
    exclude_tests       = bool : exclude test batches from grid before interpolating
    create_interpolator = bool : setup interpolator (takes a while)
    ========================================================"""

    def __init__(self, source, version, verbose=True, exclude_tests=True,
                 recalculate_interpolators=False, burst_analyser=False,
                 bprops=('dt', 'u_dt', 'fluence', 'u_fluence', 'peak', 'u_peak')):
        self.verbose = verbose
        source = grid_strings.source_shorthand(source)
        self.source = source
        self.bprops = bprops
        self.version = version
        self.burst_analyser = burst_analyser
        self.interpolator = None

        summ = grid_tools.load_grid_table('summ', source=source, burst_analyser=burst_analyser)
        params = grid_tools.load_grid_table('params', source=source)

        if exclude_tests:
            exclude = {**batches_exclude[source], **params_exclude[source]}
            params = grid_tools.exclude_params(table=params, params=exclude)

            idxs_kept = params.index
            summ = summ.loc[idxs_kept]

        self.summ = summ
        self.params = params

        if recalculate_interpolators:
            self.setup_interpolator(bprops=bprops)
        else:
            self.load_interpolator()

    def printv(self, string, **kwargs):
        """=================================================
        Prints string if self.verbose == True
        ================================================="""
        if self.verbose:
            print(string, **kwargs)

    def reduce_summ(self, params):
        """========================================================
        Returns reduced summ table with specified parameters (e.g., mass, qb)
        ========================================================"""
        reduced_idxs = grid_tools.reduce_table_idx(self.params, params=params)
        return self.summ.iloc[reduced_idxs]

    def save_interpolator(self):
        """========================================================
        Saves (pickles) interpolator to file
        ========================================================"""
        self.printv(f'Saving interpolator')
        filename = f'interpolator_{self.source}_V{self.version}'
        filepath = os.path.join(GRIDS_PATH, 'sources', self.source,
                                'interpolator', filename)
        self.printv(f'Saving interpolator: {filepath}')
        pickle.dump(self.interpolator, open(filepath, 'wb'))

    def load_interpolator(self):
        """========================================================
        Loads previously-saved (pickled) interpolators from file
        ========================================================"""
        self.printv(f'Loading interpolator')
        filename = f'interpolator_{self.source}_V{self.version}'
        filepath = os.path.join(GRIDS_PATH, 'sources', self.source,
                                'interpolator', filename)
        self.interpolator = pickle.load(open(filepath, 'rb'))

    def setup_interpolator(self, bprops):
        """========================================================
        Creates interpolator object from kepler grid data
        ========================================================
        bprops = [str]  : burst properties to interpolate (e.g., dt, fluence)
        ========================================================"""
        acc = self.params['accrate']
        x = self.params['x']
        z = self.params['z']
        qb = self.params['qb']
        mass = self.params['mass']

        print('Creating interpolator on grid: ')
        print(f'x:    {np.unique(x)}')
        print(f'z:    {np.unique(z)}')
        print(f'qb:   {np.unique(qb)}')
        print(f'mass: {np.unique(mass)}')
        print(f'acc:  {np.unique(acc)}')

        # ==== ensure correct order of parameters ====
        points = (acc, x, z, qb, mass)
        # points = (acc, z, qb, mass)

        n_models = len(self.params)
        n_bprops = len(bprops)
        values = np.full((n_models, n_bprops), np.nan)

        t0 = time.time()
        self.printv(f'Creating interpolator:')
        for i, bp in enumerate(bprops):
            if not self.burst_analyser:
                key = key_map[bp]
            else:
                key = bp

            values[:, i] = self.summ[key]
        self.interpolator = LinearNDInterpolator(points, values)
        # self.interpolator = RegularGridInterpolator(points, values)
        t1 = time.time()
        self.printv(f'Setup time: {t1-t0:.1f} s')

    def emulate_burst(self, params):
        """========================================================
        Returns interpolated burst properties for given params
        ========================================================
        params: acc, x, z, qb, mass
        ========================================================"""
        check_params_length(params)
        # ==== ensure correct order of parameters ====
        if type(params) == dict:
            params = convert_params(params=params)

        return self.interpolator(params)


def convert_params(params):
    """========================================================
    Converts params from dict to list format, and vice versa (ensures order)
    ========================================================"""
    # check_params_length(params=params)
    ptype = type(params)
    pkeys = ['acc', 'x', 'z', 'qb', 'mass']
    # pkeys = ['acc', 'z', 'qb', 'mass']

    if ptype == dict:
        params_out = []
        for key in pkeys:
            params_out += [params[key]]

    elif (ptype == list) or (ptype == tuple) or (ptype == np.ndarray):
        params_out = {}
        for i, key in enumerate(pkeys):
            params_out[key] = params[i]

    return params_out


def check_params_length(params, length=5):
    """========================================================
    Checks that five parameters have been provided
    ========================================================"""

    def check(array):
        if len(array) != length:
            raise ValueError("'params' must specify each of (acc, x, z, qb, mass)")

    if len(params.shape) == 1:
        check(params)
    elif len(params.shape) == 2:
        check(params[0])
