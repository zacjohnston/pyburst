import numpy as np
import matplotlib.pyplot as plt
import os
import astropy.units as u
import astropy.constants as const
import pickle
from matplotlib.ticker import NullFormatter

# kepler_grids
from pygrids.interpolator import interpolator
from .mcmc_versions import McmcVersion
from pygrids.mcmc.mcmc_tools import print_params
from pygrids.misc import pyprint
from pygrids.synth import synth_data

# concord
import ctools
import anisotropy

GRIDS_PATH = os.environ['KEPLER_GRIDS']
source_map = {
    'biggrid1': 'gs1826',  # alias for the source being modelled
    'biggrid2': 'gs1826',
    'grid4': 'gs1826',
    'grid5': 'gs1826',
    'sim_test': 'biggrid2',
    'sim10': 'biggrid2',
    'heat': 'gs1826',
}

c = const.c.to(u.cm / u.s)
msunyer_to_gramsec = (u.M_sun / u.year).to(u.g / u.s)
mdot_edd = 1.75e-8 * msunyer_to_gramsec
z_sun = 0.01


def default_plt_options():
    """Initialise default plot parameters"""
    params = {'mathtext.default': 'regular',
              'font.family': 'serif', 'text.usetex': False}
    plt.rcParams.update(params)


default_plt_options()


class BurstFit:
    """Class for comparing modelled bursts to observed bursts
    """

    def __init__(self, source, version, verbose=True,
                 lhood_factor=1, debug=False, priors_only=False,
                 re_interp=False, u_fper_frac=0.0, zero_lhood=-np.inf, **kwargs):

        self.source = source
        self.version = version
        self.verbose = verbose
        self.debug = pyprint.Debugger(debug=debug)
        self.mcmc_version = McmcVersion(source=source, version=version)
        self.param_idxs = {}
        self.interp_idxs = {}
        self.get_param_indexes()
        self.has_g = 'g' in self.mcmc_version.param_keys
        self.has_logz = 'logz' in self.mcmc_version.param_keys
        self.has_inc = 'inc' in self.mcmc_version.param_keys
        self.has_ratio = 'f_ratio' in self.mcmc_version.param_keys
        self.has_one_f = 'f' in self.mcmc_version.param_keys
        self.has_two_f = ('f_b' in self.mcmc_version.param_keys
                          and 'f_p' in self.mcmc_version.param_keys)
        self.has_d = ('d_b' in self.mcmc_version.param_keys
                      and 'xi_ratio' in self.mcmc_version.param_keys)

        self.kpc_to_cm = u.kpc.to(u.cm)
        self.zero_lhood = zero_lhood
        self.u_fper_frac = u_fper_frac
        self.lhood_factor = lhood_factor
        self.priors_only = priors_only
        self.kemulator = interpolator.Kemulator(source=source,
                                                version=self.mcmc_version.interpolator,
                                                re_interp=re_interp,
                                                **kwargs)
        concord_source = source_map.get(source, source)
        self.obs = ctools.load_obs(concord_source)
        self.n_epochs = len(self.obs)

        if self.source == 'sim_test':
            self.n_epochs = 1

        self.obs_data = None
        self.extract_obs_values()

        self.z_prior = None
        self.f_ratio_prior = None
        self.xi_ratio_prior = None
        self.inc_prior = None
        self.setup_priors()

    def printv(self, string, **kwargs):
        if self.verbose:
            print(string, **kwargs)

    def get_param_indexes(self):
        """Extracts indexes of parameters

        Expects params array to be in same order as param_keys
        """
        self.debug.start_function('get_param_indexes')

        for i, key in enumerate(self.mcmc_version.param_keys):
            self.param_idxs[key] = i
        for i, key in enumerate(self.mcmc_version.interp_keys):
            self.interp_idxs[key] = i

        self.debug.end_function()

    def setup_priors(self):
        self.debug.start_function('setup_priors')
        self.z_prior = self.mcmc_version.prior_pdfs['z']
        self.f_ratio_prior = self.mcmc_version.prior_pdfs['f_ratio']
        self.xi_ratio_prior = self.mcmc_version.prior_pdfs['xi_ratio']
        self.inc_prior = self.mcmc_version.prior_pdfs['inc']
        self.debug.end_function()

    def extract_obs_values(self):
        """Unpacks observed burst properties (dt, fper, etc.) from data
        """
        def strip_units(x):
            if type(x) is u.quantity.Quantity:
                x = x.value
            return x

        self.debug.start_function('extract_obs_values')
        hr_day = 24
        key_map = {'dt': 'tdel', 'u_dt': 'tdel_err',
                   'fper': 'fper', 'u_fper': 'fper_err',
                   'fluence': 'fluen', 'u_fluence': 'fluen_err',
                   'peak': 'F_pk', 'u_peak': 'F_pk_err',
                   'cbol': 'cbol', 'u_cbol': 'cbol_err'}

        if self.source == 'sim_test':
            filepath = os.path.join(GRIDS_PATH, 'obs_data', 'sim1', 'sim_test_summary.p')
            self.obs_data = pickle.load(open(filepath, 'rb'))
            self.debug.end_function()
            return
        elif 'sim' in self.source:
            self.obs_data = synth_data.extract_obs_data(self.source)
            return
        else:
            self.obs_data = dict.fromkeys(key_map)
            for key in self.obs_data:
                self.obs_data[key] = np.zeros(self.n_epochs)
                key_old = key_map[key]

                for i in range(self.n_epochs):
                    self.obs_data[key][i] = strip_units(self.obs[i].__dict__[key_old])

            # ====== extract burst rates ======
            self.obs_data['rate'] = hr_day / self.obs_data['dt']
            self.obs_data['u_rate'] = hr_day * self.obs_data['u_dt'] / self.obs_data['dt']**2

            for var in ('fper', 'u_fper'):
                self.obs_data[var] *= self.obs_data['cbol']

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

        # ===== check priors =====
        lp = self.lnprior(params=params)
        if self.priors_only:
            self.debug.end_function()
            return lp * self.lhood_factor

        if lp == self.zero_lhood:
            self.debug.end_function()
            return self.zero_lhood * self.lhood_factor

        # ===== interpolate bursts from model params =====
        epoch_params = self.get_epoch_params(params)
        interp = self.interpolate(interp_params=epoch_params)

        if True in np.isnan(interp):
            self.debug.print_('Outside interpolator bounds')
            self.debug.end_function()
            return self.zero_lhood * self.lhood_factor

        plot_map = np.arange(4)
        if plot:
            fig, ax = plt.subplots(4, 1, figsize=(6, 9))
        else:
            fig = ax = None

        # ===== compare model burst properties against observed =====
        lh = 0.0
        for i, bprop in enumerate(self.mcmc_version.bprops):
            u_bprop = f'u_{bprop}'
            bprop_col = 2*i
            u_bprop_col = bprop_col + 1

            # ===== shift values to observer frame and units =====
            for j, key in enumerate([bprop, u_bprop]):
                col = bprop_col + j
                interp[:, col] = self.shift_to_observer(values=interp[:, col],
                                                        bprop=key, params=params)
            model = interp[:, bprop_col]
            u_model = interp[:, u_bprop_col]

            lh += self.compare(model=model, u_model=u_model,
                               obs=self.obs_data[bprop], bprop=bprop,
                               u_obs=self.obs_data[u_bprop], label=bprop)
            if plot:
                self.plot_compare(model=model, u_model=u_model, obs=self.obs_data[bprop],
                                  u_obs=self.obs_data[u_bprop], bprop=bprop,
                                  ax=ax[np.where(plot_map == i)][0], display=False,
                                  legend=True if i==0 else False)

        # ===== compare predicted persistent flux with observed =====
        fper = self.shift_to_observer(values=epoch_params[:, self.interp_idxs['mdot']],
                                      bprop='fper', params=params)
        u_fper = fper * self.u_fper_frac  # Assign uncertainty to model persistent flux

        lh += self.compare(model=fper, u_model=u_fper, label='fper',
                           obs=self.obs_data['fper'], bprop='fper',
                           u_obs=self.obs_data['u_fper'])

        lhood = (lp + lh) * self.lhood_factor

        if plot:
            self.plot_compare(model=fper, u_model=u_fper, bprop='fper',
                              obs=self.obs_data['fper'], u_obs=self.obs_data['u_fper'],
                              ax=ax[np.where(plot_map == 3)][0], display=False,
                              xlabel=True)
            plt.show(block=False)
            self.debug.end_function()
            return lhood, fig
        else:
            self.debug.end_function()
            return lhood

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

        if bprop in ('dt', 'u_dt'):
            shifted = values * redshift / 3600
        elif bprop in ('rate', 'u_rate'):
            shifted = values / redshift
        else:
            if self.has_two_f:  # model uses generalised flux_factors xi*d^2 (x10^45)
                flux_factor_b = 1e45 * params[self.param_idxs['f_b']]
                flux_factor_p = 1e45 * params[self.param_idxs['f_p']]
            elif self.has_one_f:
                flux_factor_b = 1e45 * params[self.param_idxs['f']]
                flux_factor_p = flux_factor_b
            elif self.has_ratio:
                flux_factor_b = 1e45 * params[self.param_idxs['f_b']]
                flux_factor_p = flux_factor_b * params[self.param_idxs['f_ratio']]
            elif self.has_d:
                flux_factor_b = (self.kpc_to_cm * params[self.param_idxs['d_b']])**2
                flux_factor_p = flux_factor_b * params[self.param_idxs['xi_ratio']]
            else:
                if self.has_inc:  # model explicitly uses inclination
                    inc = params[self.param_idxs['inc']]
                    xi_b, xi_p = anisotropy.anisotropy(inclination=inc*u.deg,
                                                       model=self.mcmc_version.disc_model)
                else:   # model explicitly uses xi
                    xi_b = params[self.param_idxs['xi_b']]
                    xi_p = params[self.param_idxs['xi_p']]

                d = params[self.param_idxs['d']]
                d *= u.kpc.to(u.cm)
                flux_factor_p = xi_p * d**2
                flux_factor_b = xi_b * d**2

            if bprop in ('fluence', 'u_fluence'):  # (erg) --> (erg / cm^2)
                shifted = values / (4*np.pi * flux_factor_b)

            elif bprop in ('peak', 'u_peak'):  # (erg/s) --> (erg / cm^2 / s)
                shifted = values / (redshift * 4*np.pi * flux_factor_b)

            elif bprop in 'fper':  # mdot --> (erg / cm^2 / s)
                phi = (redshift - 1) * c.value ** 2 / redshift  # gravitational potential
                shifted = (values * mdot_edd * phi) / (redshift * 4*np.pi * flux_factor_p)
            else:
                raise ValueError('bprop must be one of (dt, u_dt, rate, u_rate, '
                                 + 'fluence, u_fluence, '
                                 + 'peak, u_peak, fper)')
        self.debug.end_function()
        return shifted

    def interpolate(self, interp_params):
        """Interpolates burst properties for N epochs

        Parameters
        ----------
        interp_params : 1darray
            parameters specific to the model (e.g. mdot1, x, z, qb, mass)
        """
        self.debug.start_function('interpolate')
        # TODO: generalise to N-epochs
        self.debug.variable('interp_params', interp_params, formatter='')
        output = self.kemulator.emulate_burst(params=interp_params)
        self.debug.end_function()
        return output

    def get_epoch_params(self, params):
        """Extracts array of model parameters for each epoch
        """
        self.debug.start_function('extract_epoch_params')
        # TODO: use base set of interp params (without epoch duplicates)
        n_interp = len(self.mcmc_version.interp_keys)
        epoch_params = np.full((self.n_epochs, n_interp), np.nan, dtype=float)

        for i in range(self.n_epochs):
            for j in range(n_interp):
                key = self.mcmc_version.interp_keys[j]
                epoch_params[i, j] = self.get_interp_param(key, params, epoch_idx=i)

        self.transform_aliases(epoch_params)
        self.debug.variable('epoch_params', epoch_params, formatter='')
        self.debug.end_function()
        return epoch_params

    def get_interp_param(self, key, params, epoch_idx):
        """Extracts interp param value from full params
        """
        self.debug.start_function('get_interp_param')
        self.debug.variable('interp key', key, formatter='')
        key = self.mcmc_version.param_aliases.get(key, key)

        if key in self.mcmc_version.epoch_unique:
            key = f'{key}{epoch_idx + 1}'

        self.debug.variable('param key', key, formatter='')
        self.debug.end_function()
        return params[self.param_idxs[key]]

    def transform_aliases(self, epoch_params, reference_mass=1.4):
        """Transforms any alias params into the correct model form

        parameters
        ----------
        epoch_params : nparray
            set of parameters to be parsed to interpolator
        reference_mass : float
            mass (Msun) that 'g' factor is relative to
        """
        self.debug.start_function('transform_aliases')
        self.debug.variable('epoch params in', epoch_params, formatter='')

        if self.has_g:
            epoch_params[:, self.interp_idxs['mass']] *= reference_mass
        if self.has_logz:
            idx = self.interp_idxs['z']
            epoch_params[:, idx] = z_sun * 10**epoch_params[:, idx]

        self.debug.variable('epoch params out', epoch_params, formatter='')
        self.debug.end_function()

    def lnprior(self, params):
        """Return logarithm prior lhood of params
        """
        self.debug.start_function('lnprior')
        lower_bounds = self.mcmc_version.prior_bounds[:, 0]
        upper_bounds = self.mcmc_version.prior_bounds[:, 1]
        inside_bounds = np.logical_and(params > lower_bounds,
                                       params < upper_bounds)

        if False in inside_bounds:
            self.debug.end_function()
            return self.zero_lhood

        if self.has_logz:
            z_input = params[self.param_idxs['logz']]
        else:
            z = params[self.param_idxs['z']]
            z_input = np.log10(z / z_sun)

        prior_lhood = np.log(self.z_prior(z_input))

        if self.has_two_f:
            f_ratio = params[self.param_idxs['f_p']] / params[self.param_idxs['f_b']]
            prior_lhood += np.log(self.f_ratio_prior(f_ratio))
        elif self.has_ratio:
            f_ratio = params[self.param_idxs['f_ratio']]
            prior_lhood += np.log(self.f_ratio_prior(f_ratio))
        elif self.has_d:
            xi_ratio = params[self.param_idxs['xi_ratio']]
            prior_lhood += np.log(self.xi_ratio_prior(xi_ratio))
        elif self.has_inc:
            inc = params[self.param_idxs['inc']]
            prior_lhood += np.log(self.inc_prior(inc * u.deg)).value

        self.debug.variable('prior_lhood', prior_lhood, formatter='f')
        self.debug.end_function()
        return prior_lhood

    def compare(self, model, u_model, obs, u_obs, bprop, label='', plot=False):
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
        bprop : str
            burst property being compared
        label : str
            label of parameter to print
        plot : bool
            whether to plot the comparison
        """
        self.debug.start_function('compare')
        pyprint.check_same_length(model, obs, 'model and obs arrays')
        pyprint.check_same_length(u_model, u_obs, 'u_model and u_obs arrays')

        weight = self.mcmc_version.weights[bprop]
        inv_sigma2 = 1 / (u_model ** 2 + u_obs ** 2)
        lh = -0.5 * weight * ((model - obs) ** 2 * inv_sigma2
                     + np.log(2 * np.pi / inv_sigma2))
        self.debug.print_(f'lhood breakdown: {label} {lh}')

        if plot:
            self.plot_compare(model=model, u_model=u_model, obs=obs,
                              u_obs=u_obs, bprop=label)
        self.debug.end_function()
        return lh.sum()

    def plot_compare(self, model, u_model, obs, u_obs, bprop, ax=None, title=False,
                     display=True, xlabel=False, legend=False):
        """Plots comparison of modelled and observed burst property

        Parameters
        ----------
        (others same as compare)
        bprop : str
            burst property being compared
        """
        # TODO: move to mcmc_plot?
        fontsize = 12
        markersize = 6
        capsize = 3
        n_sigma = 3
        dx = 0.13  # horizontal offset of plot points
        yscale = {'dt': 1.0, 'rate': 1.0,
                  'fluence': 1e-6, 'peak': 1e-8, 'fper': 1e-9}.get(bprop)
        ylabel = {'dt': r'$\Delta t$',
                  'rate': 'Burst rate',
                  'fluence': r'$E_b$',
                  'peak': r'$F_{peak}$',
                  'fper': r'$F_p$'}.get(bprop, bprop)
        y_units = {'dt': 'hr',
                   'rate': 'day$^{-1}$',
                   'fluence': r'$10^{-6}$ erg cm$^{-2}$',
                   'peak': r'$10^{-8}$ erg cm$^{-2}$ s$^{-1}$',
                   'fper': r'$10^{-9}$ erg cm$^{-2}$ s$^{-1}$'}.get(bprop)

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))

        epochs = np.array([1998, 2000, 2007])
        x = epochs

        ax.errorbar(x=x - dx, y=model/yscale, yerr=n_sigma*u_model/yscale, ls='none', marker='o',
                    capsize=capsize, color='C3', label='Model', markersize=markersize)
        ax.errorbar(x=x + dx, y=obs/yscale, yerr=n_sigma*u_obs/yscale, ls='none',
                    marker='o', capsize=capsize, color='C0', label='Observed',
                    markersize=markersize)

        ax.set_xlim([2008, 1997])
        ax.set_ylabel(f'{ylabel} ({y_units})', fontsize=fontsize)

        ax.set_xticks(epochs)
        if xlabel:
            ax.set_xticklabels(['2007', '2000', '1998'])
            ax.set_xlabel('Epoch year', fontsize=fontsize)
        else:
            ax.set_xticklabels([])

        if title:
            ax.set_title(ylabel, fontsize=fontsize)
        if legend:
            ax.legend()
        plt.tight_layout()
        if display:
            plt.show(block=False)

    def plot_z_prior(self):
        z_sun = 0.01
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
    plt.tight_layout()
    plt.show(block=False)