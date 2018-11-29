# qnuc

Modules for implementing nuclear heating (Q_nuc, MeV/nucleon) in the initial setup-phase of `Kepler` simulations.

When a `Kepler` model begins, it initialises the thermal profile of the envelope, mostly by letting it "relax" with a given base heating (Q_b). It was assumed that additional heating from nuclear burning (Q_nuc) would stabilise after the first few bursts, but this was discovered to not be true.

A solution was to add an additional heating term in the envelope while it relaxes, so that when the full simulation calculations begin, the envelope is closer to its proper thermal equilibrium.
