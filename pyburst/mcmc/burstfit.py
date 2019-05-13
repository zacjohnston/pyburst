import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const

# pyburst
from pyburst.interpolator import interpolator
from .mcmc_versions import McmcVersion
from pyburst.mcmc.mcmc_tools import print_params
from pyburst.misc import pyprint
from pyburst.synth import synth
from pyburst.physics import gravity

GRIDS_PATH = os.environ['KEPLER_GRIDS']
PYBURST_PATH = os.environ['PYBURST']

obs_source_map = {
    'biggrid1': 'gs1826',  # alias for the source being modelled
    'biggrid2': 'gs1826',
    'grid4': 'gs1826',
    'grid5': 'gs1826',
    'grid6': 'gs1826',
    'heat': 'gs1826',
    'he1': '4u1820',
    'he2': '4u1820',
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


# TODO: Docstrings

class BurstFit:
    """Class for comparing modelled bursts to observed bursts
    """

    def __init__(self, source, version, verbose=True,
                 lhood_factor=1, debug=False, priors_only=False,
                 re_interp=False, u_fper_frac=0.0, zero_lhood=-np.inf,
                 reference_radius=10, **kwargs):
        """
        reference_radius : float
            Newtonian radius (km) used in Kepler
        """
        self.source = source
        self.source_obs = obs_source_map.get(self.source, self.source)
        self.version = version
        self.verbose = verbose
        self.debug = pyprint.Debugger(debug=debug)
        self.mcmc_version = McmcVersion(source=source, version=version)
        self.param_idxs = {}
        self.interp_idxs = {}
        self.get_param_indexes()
        self.reference_radius = reference_radius

        self.kpc_to_cm = u.kpc.to(u.cm)
        self.zero_lhood = zero_lhood
        self.u_fper_frac = u_fper_frac
        self.lhood_factor = lhood_factor
        self.priors_only = priors_only

        if self.mcmc_version.synthetic:
            interp_source = self.mcmc_version.interp_source
        else:
            interp_source = self.source

        self.kemulator = interpolator.Kemulator(source=interp_source,
                                                version=self.mcmc_version.interpolator,
                                                re_interp=re_interp,
                                                **kwargs)
        self.obs = None
        self.n_epochs = None
        self.obs_data = None
        self.extract_obs_values()

        self.priors = self.mcmc_version.priors

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

    def extract_obs_values(self):
        """Unpacks observed burst properties (dt, fper, etc.) from data
        """
        self.debug.start_function('extract_obs_values')

        if self.mcmc_version.synthetic:
            self.obs_data = synth.extract_obs_data(self.source,
                                                   self.mcmc_version.synth_version,
                                                   group=self.mcmc_version.synth_group)
            self.n_epochs = len(self.obs_data['fluence'])
        else:
            filename = f'{self.source_obs}.dat'
            filepath = os.path.join(PYBURST_PATH, 'files', 'obs_data',
                                    self.source_obs, filename)

            self.obs = pd.read_csv(filepath, delim_whitespace=True)
            self.n_epochs = len(self.obs)
            self.obs_data = self.obs.to_dict(orient='list')

            for key, item in self.obs_data.items():
                self.obs_data[key] = np.array(item)

            # ===== Apply bolometric corrections (cbol) to fper ======
            u_fper_frac = np.sqrt((self.obs_data['u_cbol'] / self.obs_data['cbol']) ** 2
                                  + (self.obs_data['u_fper'] / self.obs_data['fper']) ** 2)

            self.obs_data['fper'] *= self.obs_data['cbol']
            self.obs_data['u_fper'] = self.obs_data['fper'] * u_fper_frac

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
        bprop_values = np.zeros((len(self.mcmc_version.bprops) + 1, self.n_epochs))
        epoch_params = self.get_epoch_params(params)
        interp = self.interpolate(interp_params=epoch_params)

        if True in np.isnan(interp):
            self.debug.print_('Outside interpolator bounds')
            self.debug.end_function()
            return self.zero_lhood * self.lhood_factor

        n_bprops = len(self.mcmc_version.bprops) + 1
        if plot:
            plot_width = 6
            plot_height = 2.25
            fig, ax = plt.subplots(n_bprops, 1, sharex=True,
                                   figsize=(plot_width, plot_height * n_bprops))
        else:
            fig = ax = None

        # ===== compare model burst properties against observed =====
        lh = 0.0
        for i, bprop in enumerate(self.mcmc_version.bprops):
            u_bprop = f'u_{bprop}'
            bprop_col = 2 * i
            u_bprop_col = bprop_col + 1

            # ===== shift values to observer frame and units =====
            for j, key in enumerate([bprop, u_bprop]):
                col = bprop_col + j
                interp[:, col] = self.shift_to_observer(values=interp[:, col],
                                                        bprop=key, params=params)
            model = interp[:, bprop_col]
            u_model = interp[:, u_bprop_col]

            bprop_values[i, :] = model
            lh += self.compare(model=model, u_model=u_model,
                               obs=self.obs_data[bprop], bprop=bprop,
                               u_obs=self.obs_data[u_bprop], label=bprop)
            if plot:
                self.plot_compare(model=model, u_model=u_model, obs=self.obs_data[bprop],
                                  u_obs=self.obs_data[u_bprop], bprop=bprop,
                                  ax=ax[i], display=False,
                                  legend=True if i == 0 else False)

        # ===== compare predicted persistent flux with observed =====
        # TODO: fold this into above loop?
        fper = self.shift_to_observer(values=epoch_params[:, self.interp_idxs['mdot']],
                                      bprop='fper', params=params)
        u_fper = fper * self.u_fper_frac  # Assign uncertainty to model persistent flux

        bprop_values[-1, :] = fper
        lh += self.compare(model=fper, u_model=u_fper, label='fper',
                           obs=self.obs_data['fper'], bprop='fper',
                           u_obs=self.obs_data['u_fper'])

        lhood = (lp + lh) * self.lhood_factor

        if plot:
            self.plot_compare(model=fper, u_model=u_fper, bprop='fper',
                              obs=self.obs_data['fper'], u_obs=self.obs_data['u_fper'],
                              ax=ax[n_bprops - 1], display=False,
                              xlabel=True)
            plt.show(block=False)
            self.debug.end_function()
            return lhood, fig
        else:
            self.debug.end_function()
            return lhood#, bprop_values

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


        Notes
        ------
        In special case bprop='fper', 'values' must be local accrate
                as fraction of Eddington rate.
        """

        def gr_factors():
            mass_nw = params[self.param_idxs['m_nw']]
            mass_gr = params[self.param_idxs['m_gr']]
            m_ratio = mass_gr / mass_nw
            red = gravity.gr_corrections(r=self.reference_radius, m=mass_nw,
                                         phi=m_ratio)[1]
            return m_ratio, red

        # TODO: cache other reused values
        self.debug.start_function('shift_to_observer')
        mass_ratio, redshift = gr_factors()

        if bprop in ('dt', 'u_dt'):
            shifted = values * redshift / 3600
        elif bprop in ('rate', 'u_rate'):
            shifted = values / redshift
        elif bprop in ('tail_50', 'u_tail_50'):
            shifted = values * redshift
        else:
            flux_factor_b = (self.kpc_to_cm * params[self.param_idxs['d_b']]) ** 2
            flux_factor_p = flux_factor_b * params[self.param_idxs['xi_ratio']]

            if bprop in ('fluence', 'u_fluence'):  # (erg) --> (erg / cm^2)
                shifted = (values * mass_ratio) / (4 * np.pi * flux_factor_b)

            elif bprop in ('peak', 'u_peak'):  # (erg/s) --> (erg / cm^2 / s)
                shifted = (values * mass_ratio) / (redshift * 4 * np.pi * flux_factor_b)

            elif bprop in 'fper':  # mdot --> (erg / cm^2 / s)
                phi = (redshift - 1) * c.value ** 2 / redshift  # gravitational potential
                lum_acc = values * mdot_edd * phi
                shifted = (lum_acc * mass_ratio) / (redshift * 4 * np.pi * flux_factor_p)
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

    def lnprior(self, params):
        """Return logarithm prior lhood of params
        """
        self.debug.start_function('lnprior')
        lower_bounds = self.mcmc_version.grid_bounds[:, 0]
        upper_bounds = self.mcmc_version.grid_bounds[:, 1]
        inside_bounds = np.logical_and(params > lower_bounds,
                                       params < upper_bounds)

        if False in inside_bounds:
            self.debug.end_function()
            return self.zero_lhood

        prior_lhood = 0.0

        for param, prior_pdf in self.priors.items():
            idx = self.param_idxs[param]
            prior_lhood += np.log(prior_pdf(params[idx]))

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
        yscale = {'dt': 1.0, 'rate': 1.0, 'tail_50': 1.0,
                  'fluence': 1e-6, 'peak': 1e-8, 'fper': 1e-9}.get(bprop)
        ylabel = {'dt': r'$\Delta t$',
                  'rate': 'Burst rate',
                  'fluence': r'$E_b$',
                  'peak': r'$F_{peak}$',
                  'fper': r'$F_p$',
                  'tail_50': r'$t_{50}$'}.get(bprop, bprop)
        y_units = {'dt': 'hr',
                   'rate': 'day$^{-1}$',
                   'fluence': r'$10^{-6}$ erg cm$^{-2}$',
                   'peak': r'$10^{-8}$ erg cm$^{-2}$ s$^{-1}$',
                   'fper': r'$10^{-9}$ erg cm$^{-2}$ s$^{-1}$',
                   'tail_50': 's'}.get(bprop)
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))

        epochs = np.array(self.obs.epoch)
        x = epochs

        ax.errorbar(x=x - dx, y=model / yscale, yerr=n_sigma * u_model / yscale, ls='none', marker='o',
                    capsize=capsize, color='C3', label='Model', markersize=markersize)
        ax.errorbar(x=x + dx, y=obs / yscale, yerr=n_sigma * u_obs / yscale, ls='none',
                    marker='o', capsize=capsize, color='C0', label='Observed',
                    markersize=markersize)

        ax.set_ylabel(f'{ylabel} ({y_units})', fontsize=fontsize)
        ax.set_xticks(epochs)

        if xlabel:
            ax.set_xticklabels([f'{year}' for year in epochs])
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
        x = np.linspace(0, 0.02, 1000)
        fig, ax = plt.subplots()
        y = self.priors['z'](np.log10(x / z_sun))

        ax.set_xlabel('z (mass fraction)')
        ax.plot(x, y, label='z')
        ax.legend()
        plt.tight_layout()
        plt.show(block=False)
