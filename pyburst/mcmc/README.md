# mcmc

Modules for performing Markov chain Monte Carlo (MCMC) analysis on burst models.

Uses the `emcee` toolkit (https://github.com/dfm/emcee) for the actual sampling.

The `BurstFit` class in `burstfit.py` implements the likelihood function and the primary methods for comparing burst models to observations. It uses an "interpolator" object from `interpolator` to sample points in parameter space.

