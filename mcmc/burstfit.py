import numpy as np
import matplotlib.pyplot as plt
import os
import astropy.units as u
import astropy.constants as const
from scipy.stats import norm
import pickle
from matplotlib.ticker import NullFormatter

# kepler_grids
from ..interpolator import kepler_emulator
from .mcmc_versions import McmcVersion
from pygrids.misc import pyprint
from pygrids.mcmc.mcmc_tools import print_params

# concord
import ctools
import anisotropy

GRIDS_PATH = os.environ['KEPLER_GRIDS']
source_map = {
    'biggrid1': 'gs1826',  # alias for the source being modelled
    'biggrid2': 'gs1826',
    'sim_test': 'biggrid2',
}

c = const.c.to(u.cm / u.s)
msunyer_to_gramsec = (u.M_sun / u.year).to(u.g / u.s)
mdot_edd = 1.75e-8 * msunyer_to_gramsec


class BurstFit:
    """Class for comparing modelled bursts to observed bursts
    """

    def __init__(self, source, version,
                 bprops=('dt', 'u_dt', 'fluence', 'u_fluence', 'peak', 'u_peak'),
                 verbose=True, lhood_factor=1, debug=False, **kwargs):
        self.source = source
        self.version = version
        self.verbose = verbose
        self.debug = pyprint.Debugger(debug=debug)
        self.mcmc_version = McmcVersion(source=source, version=version)
        self.param_idxs = {}
        self.get_param_indexes()

        if self.source == 'sim_test':
            source = 'biggrid2'  # from here on effectively treat as biggrid2

        self.bprops = bprops
        self.n_bprops = len(bprops)
        self.lhood_factor = lhood_factor
        self.kemulator = kepler_emulator.Kemulator(source=source,
                                                   bprops=bprops,
                                                   version=self.mcmc_version.interpolator,
                                                   **kwargs)

        concord_source = source_map.get(source, source)
        self.obs = ctools.load_obs(concord_source)
        self.n_epochs = len(self.obs)
        self._mdots = None

        if self.source == 'sim_test':
            self.n_epochs = 1

        self.obs_data = None
        self.extract_obs_values()

        self.fper_ratios = None
        self.mdot_ratio_priors = None
        self.z_prior = None
        self.setup_priors()

    def printv(self, string, **kwargs):
        if self.verbose:
            print(string, **kwargs)

    def get_param_indexes(self):
        """Extracts indexes of parameters

        Expects params array to be in same order as param_keys
        """
        self.debug.start_function('get_param_indexes')
        pkeys = self.mcmc_version.param_keys

        for key in pkeys:
            self.param_idxs[key] = pkeys.index(key)
        self.debug.end_function()

    def setup_priors(self):
        self.debug.start_function('setup_priors')
        fper_ratios = self.obs_data['fper'] / self.obs_data['fper'][0]
        frac_uncert = self.obs_data['u_fper'] / self.obs_data['fper']

        ratio_priors = {}
        for i in range(self.n_epochs):
            u_ratio = fper_ratios[i] * np.sqrt(np.sum(frac_uncert[[0, i]] ** 2))
            ratio_priors[i] = norm(loc=fper_ratios[i], scale=u_ratio).pdf

        self.fper_ratios = fper_ratios
        self.mdot_ratio_priors = ratio_priors
        self.z_prior = norm(loc=-0.5, scale=0.25).pdf  # log-space [z/solar]
        self.debug.end_function()

    def extract_obs_values(self):
        """Unpacks observed burst properties (dt, fper, etc.) from data
        """
        self.debug.start_function('extract_obs_values')
        key_map = {'dt': 'tdel', 'u_dt': 'tdel_err',
                   'fper': 'fper', 'u_fper': 'fper_err',
                   'fluence': 'fluen', 'u_fluence': 'fluen_err',
                   'peak': 'F_pk', 'u_peak': 'F_pk_err'}

        if self.source == 'sim_test':
            filepath = os.path.join(GRIDS_PATH, 'obs_data', 'sim_test_summary.p')
            self.obs_data = pickle.load(open(filepath, 'rb'))
            self.debug.end_function()
            return

        self.obs_data = dict.fromkeys(key_map)
        for key in self.obs_data:
            self.obs_data[key] = np.zeros(self.n_epochs)
            key_old = key_map[key]

            for i in range(self.n_epochs):
                self.obs_data[key][i] = self.obs[i].__dict__[key_old].value

        self.debug.end_function()

    def lhood(self, params, plot=False):
        """Return lhood for given params

        Parameters
        ----------
        params : ndarray
            set of parameters to try (see "param_keys" for labels)
        plot : bool
            whether to plot the comparison
        """
        self.debug.start_function('lhood')
        if self.debug.debug:
            print_params(params, source=self.source, version=self.version)

        self._mdots = params[self.param_idxs['mdot1']: self.n_epochs]
        self.debug.variable('mdots', self._mdots, '')

        # ===== check priors =====
        lp = self.lnprior(params=params)
        if np.isinf(lp):
            self.debug.end_function()
            return -np.inf

        # ===== strip non-model params for interpolator =====
        # note: interpolate_epochs() will overwrite the mdot parameter
        reference_mass = 1.4  # solmass
        interp_params = np.array(params[self.n_epochs - 1: self.param_idxs['g'] + 1])
        interp_params[-1] *= reference_mass

        # ===== compare model burst properties against observed =====
        lh = 0.0
        interp = self.interpolate_epochs(interp_params=interp_params,
                                         mdots=self._mdots)

        # Check if outside of interpolator domain
        if True in np.isnan(interp):
            self.debug.print_('Outside interpolator bounds')
            self.debug.end_function()
            return -np.inf

        for i, bprop in enumerate(['dt', 'fluence', 'peak']):
            u_bprop = f'u_{bprop}'
            bprop_col = 2 * i
            u_bprop_col = bprop_col + 1

            # ===== shift values to observer frame and units =====
            for j, key in enumerate([bprop, u_bprop]):
                col = bprop_col + j
                interp[:, col] = self.shift_to_observer(values=interp[:, col],
                                                        bprop=key, params=params)
            lh += self.compare(model=interp[:, bprop_col],
                               u_model=interp[:, u_bprop_col], obs=self.obs_data[bprop],
                               u_obs=self.obs_data[u_bprop], label=bprop, plot=plot)

        # ===== compare predicted persistent flux with observed =====
        fper = self.shift_to_observer(values=self._mdots, bprop='fper', params=params)
        u_fper = fper * 1e-2  # Give 1% uncertainty to model persistent flux

        lh += self.compare(model=fper, u_model=u_fper, label='fper',
                           obs=self.obs_data['fper'], u_obs=self.obs_data['u_fper'],
                           plot=plot)

        self.debug.end_function()
        return (lp + lh) * self.lhood_factor

    def shift_to_observer(self, values, bprop, params):
        """Returns burst property shifted to observer frame/units

        Parameters
        ----------
        values : ndarray|flt
            model frame value(s)
        bprop : str
            name of burst property being converted/calculated
        params : 1darray
            parameters (see param_keys)

        Note: in special case bprop='fper', 'values' must be local accrate
                as fraction of Eddington rate.
        """
        self.debug.start_function('shift_to_observer')

        redshift = params[self.param_idxs['redshift']]
        d = params[self.param_idxs['d']]
        inc = params[self.param_idxs['inc']]

        if bprop in ('dt', 'u_dt'):
            shifted = values * redshift / 3600
        else:
            d *= u.kpc.to(u.cm)
            area = (4 * np.pi * d ** 2)
            xi_b, xi_p = anisotropy.anisotropy(inclination=inc * u.deg,
                                               model=self.mcmc_version.disc_model)

            if bprop in ('fluence', 'u_fluence'):  # (erg) --> (erg / cm^2)
                shifted = values / (xi_b * area)

            elif bprop in ('peak', 'u_peak'):  # (erg/s) --> (erg / cm^2 / s)
                shifted = values / (redshift * xi_b * area)

            elif bprop in 'fper':  # mdot --> (erg / cm^2 / s)
                phi = (redshift - 1) * c.value ** 2 / redshift  # gravitational potential
                shifted = (values * mdot_edd * phi) / (redshift * xi_p * area)
            else:
                raise ValueError('bprop must be one of (dt, u_dt, fluence, u_fluence, '
                                 + 'peak, u_peak, fper)')
        self.debug.end_function()
        return shifted

    def interpolate(self, interp_params):
        """Returns interpolated burst properties for given parameters

        Parameters
        ----------
        interp_params : 1darray
            parameters specific to the model (mdot1, x, z, qb, mass)
        """
        self.debug.start_function('interpolate')
        interpolated = self.kemulator.emulate_burst(params=interp_params)[0]
        self.debug.end_function()
        return interpolated

    def interpolate_epochs(self, interp_params, mdots):
        """Iterates over interpolate for multiple accretion epochs

        Parameters
        ----------
        interp_params : 1darray
            parameters specific to the model (mdot1, x, z, qb, mass)
        mdots : 1darray
            accretion rates for each epoch (as fraction of Eddington rate)
        """
        self.debug.start_function('interpolate_epochs')

        if self.n_epochs == 3:
            interp_params = np.array((interp_params, interp_params, interp_params))
            interp_params[:, 0] = mdots
        else:
            interp_params = np.array(interp_params)
            interp_params[0] = mdots

        output = self.kemulator.emulate_burst(params=interp_params)
        self.debug.end_function()
        return output

    def lnprior(self, params):
        """Return logarithm prior lhood of params
        """
        self.debug.start_function('lnprior')
        lower_bounds = self.mcmc_version.prior_bounds[:, 0]
        upper_bounds = self.mcmc_version.prior_bounds[:, 1]
        inside_bounds = np.logical_and(params > lower_bounds,
                                       params < upper_bounds)

        if False in inside_bounds:
            return -np.inf

        mdot_ratios = self._mdots / self._mdots[0]
        z = params[self.param_idxs['z']]
        inc = params[self.param_idxs['inc']]

        mdot_prior = 0
        for i in range(1, self.n_epochs):
            mdot_prior += np.log(self.mdot_ratio_priors[i](mdot_ratios[i]))

        z_sun = 0.015
        prior_lhood = (np.log(np.sin(inc * u.deg)).value
                       + np.log(self.z_prior(np.log10(z / z_sun)))
                       + mdot_prior
                       )
        self.debug.variable('prior_lhood', prior_lhood, 'f')
        self.debug.end_function()
        return prior_lhood

    def compare(self, model, u_model, obs, u_obs, label='', plot=False):
        """Returns logarithmic likelihood of given model values

        Calculates difference between modelled and observed values.
        All provided arrays must be the same length

        Parameters
        ----------
        model : 1darray
            Model values for particular property
        obs : 1darray
            Observed values for particular property.
        u_model : 1darray
            Corresponding model uncertainties
        u_obs : 1darray
            corresponding observed uncertainties
        label : str
            label of parameter to print
        plot : bool
            whether to plot the comparison
        """
        self.debug.start_function('compare')
        pyprint.check_same_length(model, obs, 'model and obs arrays')
        pyprint.check_same_length(u_model, u_obs, 'u_model and u_obs arrays')

        inv_sigma2 = 1 / (u_model ** 2 + u_obs ** 2)
        lh = -0.5 * ((model - obs) ** 2 * inv_sigma2
                     + np.log(2 * np.pi / inv_sigma2))
        self.debug.print_(f'lhood breakdown: {label} {lh}')

        if plot:
            self.plot_compare(model=model, u_model=u_model, obs=obs,
                              u_obs=u_obs, bprop=label)
        self.debug.end_function()
        return lh.sum()

    def plot_compare(self, model, u_model, obs, u_obs, bprop):
        """Plots comparison of modelled and observed burst property

        Parameters
        ----------
        (others same as compare)
        bprop : str
            burst property being compared
        """
        dx = 0.04  # horizontal offset of plot points
        fig, ax = plt.subplots()
        x = np.arange(1, self.n_epochs + 1)

        ax.errorbar(x=x - dx, y=model, yerr=u_model, ls='none', marker='o',
                    capsize=3, color='r', label='Model')
        ax.errorbar(x=x + dx, y=obs, yerr=u_obs, ls='none',
                    marker='o', capsize=3, color='b', label='Observed')

        ax.set_ylabel(f'{bprop}')
        ax.set_xlabel('epoch number')
        ax.set_title(bprop)
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_mdot_priors(self):
        """Plots priors on mdot ratio
        """
        x = np.linspace(0., 1.5, 1000)
        fig, ax = plt.subplots()

        for i, mdot in enumerate(['mdot1', 'mdot2', 'mdot3']):
            y = self.mdot_ratio_priors[i](x)
            # peak = np.max(y)
            # y /= peak
            ax.plot(x, y, label=mdot)

        ax.set_xlabel('mdot / mdot1')
        ax.set_xlim([0.5, 1.1])
        ax.legend()
        plt.tight_layout()
        plt.show(block=False)

    def plot_z_prior(self):
        z_sun = 0.015
        x = np.linspace(0, 0.02, 1000)
        fig, ax = plt.subplots()
        y = self.z_prior(np.log10(x / z_sun))

        ax.set_xlabel('z (mass fraction)')
        ax.plot(x, y, label='z')
        ax.legend()
        plt.tight_layout()
        plt.show(block=False)


def plot_obs(bfit, obs=0):
    fig, ax = plt.subplots(1, 3)
    nullfmt = NullFormatter()
    for i, bprop in enumerate(('dt', 'fluence', 'peak')):
        y_scale = {'dt': 1.0,
                   'fluence': 1e-6,
                   'peak': 1e-8,
                   'fper': 1e-9,
                   }[bprop]
        u_bprop = f'u_{bprop}'
        ax[i].errorbar([0], bfit.obs_data[bprop][obs] / y_scale,
                       yerr=bfit.obs_data[u_bprop][obs] / y_scale,
                       marker='o', capsize=3, color='C0')
        ax[i].xaxis.set_major_formatter(nullfmt)
    plt.show(block=False)