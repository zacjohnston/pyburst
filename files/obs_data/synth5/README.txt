Synthetic data tables, using kepler_grids source "synth5"

synth5.dat was generated with:

    pyburst.synth.synth_new.generate_synth_data(source='synth5',
    batches=[7,8,9], run=2, mc_source='grid5', mc_version=6,
    reproduce=True)

Which should have used the params saved as param_used in synth_new.py

The free parameters were randomly chosen within the grid5_v6 grid bounds,
(saved in param_used as m_gr, d_b, and xi_ratio)

observables were predicted using pyburst.mcmc.burstfit.bprop_sample(),
and then gaussian noise with sigma of 1% added to all values.