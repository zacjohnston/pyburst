Notes/descriptions of different MCMC versions (defined in mcmc_versions.py)

Source: bigrid2
---------------
Version     Note
----------------
56          extended to z=0.0015
57          mass bounds 1.4 - 2.3
58          same as 57, with forced 2% bprop uncertainties
59          use almost entire grid, with forced 2% bprop uncertainties
60          same as 59, without forced uncertainty
61          new subgrid of models (qb=0.05)
62          same as 61 but using interpolator 24 (forced 2% bprop uncertainties)
64          using only one f-factor (i.e. f_p=f_b)
65          same as 63 but using f-factor ratio priors
66          increasing redshift bound to 1.6

Source: grid4
---------------
Version     Note
----------------
1           initial test of new grid with qnuc burn-in heating
2           same as 2, but without f_ratio prior