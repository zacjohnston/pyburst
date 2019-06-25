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
from pyburst.physics import gravity, accretion

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


# TODO:
#       - Docstrings
#       - replace "debug" with "verbose"


class ZeroLhood(Exception):
    pass


class BurstFit:
    """Class for comparing modelled bursts to observed bursts
    """

    def __init__(self, source, version, verbose=True,
                 lhood_factor=1, debug=False, priors_only=False,
                 re_interp=False, u_fper_frac=0.0, u_fedd_frac=0.0, zero_lhood=-np.inf,
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
        self.get_indexes()

        self.reference_radius = reference_radius

        self.n_bprops = len(self.mcmc_version.bprops)
        self.n_analytic_bprops = len(self.mcmc_version.analytic_bprops)
        self.n_interp_params = len(self.mcmc_version.interp_keys)
        self.has_xedd_ratio = ('xedd_ratio' in self.mcmc_version.param_keys)

        self.kpc_to_cm = u.kpc.to(u.cm)
        self.zero_lhood = zero_lhood
        self.u_fper_frac = u_fper_frac
        self.u_fedd_frac = u_fedd_frac
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

    def get_indexes(self):
        """Extracts indexes of parameters and burst properties

        Expects params array to be in same order as param_keys
        """
        def idx_dict(dict_in):
            dict_out = {}
            for i, key in enumerate(dict_in):
                dict_out[key] = i
            return dict_out

        self.debug.start_function('get_param_indexes')
        self.param_idxs = idx_dict(self.mcmc_version.param_keys)
        self.interp_idxs = idx_dict(self.mcmc_version.interp_keys)
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
            self.obs.set_index('epoch', inplace=True, verify_integrity=True)

            # Select single epoch (if applicable)
            if self.mcmc_version.epoch is not None:
                # TODO: define/specify epochs for all mcmc versions?
                try:
                    self.obs = self.obs.loc[[self.mcmc_version.epoch]]
                except KeyError:
                    raise KeyError(f'epoch [{self.mcmc_version.epoch}] '
                                   f'not in obs_data table')

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

    def lhood(self, x, plot=False):
        """Return lhood for given params

        Parameters
        ----------
        x : 1D array
            set of parameter values to try (must match order of mcmc_version.param_keys)
        plot : bool
            whether to plot the comparison
        """
        self.debug.start_function('lhood')

        params = self.get_params_dict(x=x)
        zero_lhood = self.zero_lhood * self.lhood_factor

        if self.debug.debug:
            for key, val in params.items():
                print(f'{key:10}  {val:.4f}')

        # ===== check priors =====
        try:
            lp = self.lnprior(x=x, params=params)
        except ZeroLhood:
            return zero_lhood

        if self.priors_only:
            self.debug.end_function()
            return lp * self.lhood_factor

        # ===== Interpolate and calculate local model burst properties =====
        try:
            interp_local, analytic_local = self.get_model_local(params=params)
        except ZeroLhood:
            self.debug.end_function()
            return zero_lhood

        # ===== Shift all burst properties to observable quantities =====
        interp_shifted, analytic_shifted = self.get_model_shifted(
                                                    interp_local=interp_local,
                                                    analytic_local=analytic_local,
                                                    params=params)

        # ===== Setup plotting =====
        n_bprops = len(self.mcmc_version.bprops)
        if plot:
            plot_width = 6
            plot_height = 2.25
            fig, ax = plt.subplots(n_bprops, 1, sharex=True,
                                   figsize=(plot_width, plot_height * n_bprops))
        else:
            fig = ax = None

        # ===== Evaluate likelihoods against observed data =====
        lh = self.compare_all(interp_shifted, analytic_shifted, ax=ax, plot=plot)
        lhood = (lp + lh) * self.lhood_factor

        # ===== Finalise plotting =====
        if plot:
            plt.show(block=False)
            self.debug.end_function()
            return lhood, fig
        else:
            self.debug.end_function()
            return lhood

    def get_params_dict(self, x):
        """Returns params in form of dict
        """
        keys = self.mcmc_version.param_keys
        params_dict = dict.fromkeys(keys)

        for i, key in enumerate(keys):
            params_dict[key] = x[i]

        return params_dict

    def get_model_local(self, params):
        """Calculates predicted model values (bprops) for given params
            Returns: interp_local, analytic_local
        """
        self.debug.start_function('predict_model_values')

        epoch_params = self.get_epoch_params(params=params)
        interp_local = self.get_interp_bprops(interp_params=epoch_params)
        analytic_local = self.get_analytic_bprops(params=params, epoch_params=epoch_params)

        return interp_local, analytic_local

    def get_analytic_bprops(self, params, epoch_params):
        """Returns calculated analytic burst properties for given params
        """
        def get_fedd():
            """Returns Eddington flux array (n_epochs, 2)
                Note: Actually the luminosity at this stage, as this is the local value
            """
            out = np.full([self.n_epochs, 2], np.nan, dtype=float)

            if self.has_xedd_ratio:
                x_edd = params['x'] * params['xedd_ratio']
            elif self.mcmc_version.x_edd_option == 'x_0':
                x_edd = params['x']
            else:
                x_edd = self.mcmc_version.x_edd_option

            l_edd = accretion.eddington_lum(mass=params['m_nw'], x=x_edd)
            out[:, 0] = l_edd
            out[:, 1] = l_edd * self.u_fedd_frac
            return out

        def get_fper():
            """Returns persistent accretion flux array (n_epochs, 2)
                Note: Actually the luminosity, because this is the local value
            """
            out = np.full([self.n_epochs, 2], np.nan, dtype=float)
            mass_ratio, redshift = self.get_gr_factors(params=params)

            phi = (redshift - 1) * c.value ** 2 / redshift  # gravitational potential
            mdot = epoch_params[:, self.interp_idxs['mdot']]
            l_per = mdot * mdot_edd * phi

            out[:, 0] = l_per
            out[:, 1] = out[:, 0] * self.u_fper_frac
            return out

        function_map = {'fper': get_fper, 'fedd': get_fedd}
        analytic = np.full([self.n_epochs, 2*self.n_analytic_bprops], np.nan, dtype=float)

        for i, bprop in enumerate(self.mcmc_version.analytic_bprops):
            analytic[:, 2*i: 2*(i+1)] = function_map[bprop]()

        return analytic

    def get_model_shifted(self, interp_local, analytic_local, params):
        """Returns predicted model values (+ uncertainties) shifted to an observer frame
        """
        interp_shifted = np.full_like(interp_local, np.nan, dtype=float)
        analytic_shifted = np.full_like(analytic_local, np.nan, dtype=float)

        # ==== shift interpolated bprops ====
        # TODO: concatenate bprop arrays and handle together
        for i, bprop in enumerate(self.mcmc_version.interp_bprops):
            i0 = 2 * i
            i1 = 2 * (i + 1)
            interp_shifted[:, i0:i1] = self.shift_to_observer(
                                                    values=interp_local[:, i0:i1],
                                                    bprop=bprop, params=params)

        # ==== shift analytic bprops ====
        for i, bprop in enumerate(self.mcmc_version.analytic_bprops):
            i0 = 2 * i
            i1 = 2 * (i + 1)
            analytic_shifted[:, i0:i1] = self.shift_to_observer(
                                                    values=analytic_local[:, i0:i1],
                                                    bprop=bprop, params=params)
        return interp_shifted, analytic_shifted

    def compare_all(self, interp_shifted, analytic_shifted, ax, plot=False):
        """Compares all bprops against observations and returns total likelihood
        """
        lh = 0.0
        all_shifted = np.concatenate([interp_shifted, analytic_shifted], axis=1)

        for i, bprop in enumerate(self.mcmc_version.bprops):
            u_bprop = f'u_{bprop}'
            bprop_idx = 2 * i
            u_bprop_idx = bprop_idx + 1

            model = all_shifted[:, bprop_idx]
            u_model = all_shifted[:, u_bprop_idx]

            lh += self.compare(model=model, u_model=u_model,
                               obs=self.obs_data[bprop], bprop=bprop,
                               u_obs=self.obs_data[u_bprop], label=bprop)
            if plot:
                self.plot_compare(model=model, u_model=u_model, obs=self.obs_data[bprop],
                                  u_obs=self.obs_data[u_bprop], bprop=bprop,
                                  ax=ax[i], display=False,
                                  legend=True if i == 0 else False,
                                  xlabel=True if i == self.n_bprops-1 else False)

        return lh

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
        # TODO:
        #       - cache other reused values
        #       - generalise to type of units (eg: lum --> flux)
        #       - add "GR factor" along with flux_factor"
        self.debug.start_function('shift_to_observer')
        mass_ratio, redshift = self.get_gr_factors(params=params)

        if bprop == 'dt':
            shifted = values * redshift / 3600
        elif bprop == 'rate':
            shifted = values / redshift
        else:
            #  ---- convert (erg) ==> (erg / cm ^ 2) ----
            flux_factor_b = 4 * np.pi * (self.kpc_to_cm * params['d_b']) ** 2
            flux_factor_p = flux_factor_b * params['xi_ratio']

            flux_factor = {'fluence': flux_factor_b,
                           'peak': flux_factor_b,
                           'fedd': flux_factor_b,
                           'fper': flux_factor_p,
                           }.get(bprop)

            gr_correction = {'fluence': mass_ratio,
                             'peak': mass_ratio / redshift,
                             'fedd': 1,
                             'fper': mass_ratio / redshift,
                             }.get(bprop)

            if flux_factor is None:
                raise ValueError('bprop must be one of (dt, rate, fluence, peak, '
                                 'fper, f_edd)')

            shifted = (values * gr_correction) / flux_factor

        self.debug.end_function()
        return shifted

    def get_interp_bprops(self, interp_params):
        """Interpolates burst properties for N epochs

        Parameters
        ----------
        interp_params : 1darray
            parameters specific to the model (e.g. mdot1, x, z, qb, mass)
        """
        self.debug.start_function('interpolate')
        self.debug.variable('interp_params', interp_params, formatter='')

        output = self.kemulator.emulate_burst(params=interp_params)

        if True in np.isnan(output):
            self.debug.print_('Outside interpolator bounds')
            self.debug.end_function()
            raise ZeroLhood

        self.debug.end_function()
        return output

    def get_epoch_params(self, params):
        """Extracts array of model parameters for each epoch
        """
        self.debug.start_function('get_epoch_params')
        epoch_params = np.full((self.n_epochs, self.n_interp_params), np.nan, dtype=float)

        for i in range(self.n_epochs):
            for j, key in enumerate(self.mcmc_version.interp_keys):
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
        return params[key]

    def get_gr_factors(self, params):
        """Returns GR factors (m_ratio, redshift) given (m_nw, m_gr)"""
        mass_nw = params['m_nw']
        mass_gr = params['m_gr']
        m_ratio = mass_gr / mass_nw
        redshift = gravity.gr_corrections(r=self.reference_radius, m=mass_nw, phi=m_ratio)[1]
        return m_ratio, redshift

    def lnprior(self, x, params):
        """Return logarithm prior lhood of params
        """
        self.debug.start_function('lnprior')
        lower_bounds = self.mcmc_version.grid_bounds[:, 0]
        upper_bounds = self.mcmc_version.grid_bounds[:, 1]
        inside_bounds = np.logical_and(x > lower_bounds,
                                       x < upper_bounds)

        if False in inside_bounds:
            self.debug.end_function()
            raise ZeroLhood

        prior_lhood = 0.0
        for key, val in params.items():
            prior_lhood += np.log(self.priors[key](val))

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
        yscale = {'dt': 1.0, 'rate': 1.0, 'tail_50': 1.0, 'fedd': 1e-8,
                  'fluence': 1e-6, 'peak': 1e-8, 'fper': 1e-9}.get(bprop)
        ylabel = {'dt': r'$\Delta t$',
                  'rate': 'Burst rate',
                  'fluence': r'$E_b$',
                  'peak': r'$F_{peak}$',
                  'fper': r'$F_p$',
                  'tail_50': r'$t_{50}$',
                  'fedd': r'$F_\mathrm{Edd}$',
                  }.get(bprop, bprop)

        y_units = {'dt': 'hr',
                   'rate': 'day$^{-1}$',
                   'fluence': r'$10^{-6}$ erg cm$^{-2}$',
                   'peak': r'$10^{-8}$ erg cm$^{-2}$ s$^{-1}$',
                   'fper': r'$10^{-9}$ erg cm$^{-2}$ s$^{-1}$',
                   'tail_50': 's',
                   'fedd': r'$10^{-8}$ erg cm$^{-2}$ s$^{-1}$',
                   }.get(bprop)
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))

        epochs = np.array(self.obs.index)
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
