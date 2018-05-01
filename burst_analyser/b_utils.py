#===============================================================
# Various utilities for burstfit_1808.py
# Author: Zac Johnston 2016
# zac.johnston@monash.edu
#===============================================================
import numpy as np
import subprocess
import os, sys
import matplotlib.pyplot as plt

from printing import *
import burstfit_1808
import lcdata

KEPLER_MODELS = os.environ['KEPLER_MODELS']

def load(run, basename='run', path='/home/zac/kepler/runs/mdot/', re_load=False, save=True):
    """Attempts to load pre-saved lumfile, or load binary. Returns luminosity [t, lum]"""
    # run   = int  : Run number of kepler simulation
    # basename = str : base filename of runs, eg. 'run' or 'xrb'
    # path  = str  : path to kepler simulation directory (include trailing slash)
    # save  = bool : Save txt version for faster loading next time? Set False for still-running models
    #-------------------------------------------------
    preload_file = 'preload{run}.txt'.format(run=run)
    run_name = '{base}{run}'.format(base=basename, run=run)
    preload_filepath = os.path.join(path, run_name, preload_file)

    # ===== Force reload =====
    if re_load:
        print('Force-reloading binary file: ')
        try:
            print('Deleting old preload')
            subprocess.run(['rm', preload_filepath], check=True)
        except:
            pass

    # ===== Try loading pre-saved data =====
    try:
        print('Looking for pre-loaded luminosity file: {path}'.format(path=preload_filepath))
        lum = np.loadtxt(preload_filepath, skiprows=1)
        print('Pre-loaded data found, loaded.')

    except FileNotFoundError:
        print('No preload file found. Reloading binary')
        dashes()
        lumpath = os.path.join(path, run_name, run_name+'.lc')

        if os.path.exists(lumpath):
            lum_temp = lcdata.load(lumpath)
            model_exists = True
            n = len(lum_temp.time)
            lum = np.ndarray((n,2))
            lum[:,0] = lum_temp.time
            lum[:,1] = lum_temp.xlum

            # ===== Save for faster loading next time (NOTE: still in Kepler reference frame) =====
            dashes()
            if save:
                save_file = 'preload{run}.txt'.format(run=run)
                save_filepath = os.path.join(path, run_name, save_file)

                print('Saving data for faster loading in: {fpath}'.format(fpath=save_filepath))
                if lum[-1,0] < 3.e5:
                    print('WARNING! Did you mean to save? Run may not be finished')
                head = '[time (s), luminosity (erg/s)]'

                np.savetxt(save_filepath, lum, header=head)
            else:
                print('Saving disabled')

        else:
            print('File not found: {file}'.format(file=lumpath))
            model_exists = False
            lum = np.array([np.nan])


    dashes()

    return lum



def plot_std(runs,
             batches,
             basename='xrb',
             source='gs1826',
             var='dt',
             **kwargs):
    """
    Plots how standard deviation (STD) of recurrence times (DT) evolves
    over a train of bursts
    """
    path = kwargs.get('path', KEPLER_MODELS)
    runs = expand_runs(runs)
    fig, ax = plt.subplots()

    for batch in batches:
        batch_str = '{source}_{batch}'.format(source=source, batch=batch)
        batch_path = os.path.join(path, batch_str)

        for run in runs:
            model = burstfit_1808.BurstRun(run, flat_run=True, runs_home=batch_path,
                                        verbose=False, **kwargs)
            model.analyse_all()
            N = len(model.bursts[var])
            std_frac = np.zeros(N-1)    # skip first burst (zero std)
            x = np.arange(2, N+1)

            # ==== iterate along burst train ====
            for b in x:
                std = np.std(model.bursts['dt'][:b])
                mean = np.mean(model.bursts['dt'][:b])
                std_frac[b-2] = std/mean

            label = 'B{batch}_{run}'.format(batch=batch, run=run)
            ax.plot(x,std_frac, label=label, ls='', marker='o')

    ax.legend()
    plt.show(block=False)



def multi_save(runs,
                basename='xrb',
                re_load=True,
                **kwargs):
    """Loads a collection of models and saves their lightcurves"""
    # ==========================================
    # runs     = []   : list of model IDs to load/save
    # basename = str  : base filename of models, e.g. 'xrb' for xrb3
    # re_load  = bool : whether to force-reload binaries
    # path     = str  : path to location of model folders
    # ==========================================
    path = kwargs.get('path', KEPLER_MODELS)
    runs=expand_runs(runs)

    for r in runs:
        load(run=r, basename=basename, re_load=re_load, path=path)



def multi_batch_save(runs,
                        batches,
                        source='gs1826',
                        **kwargs):
    """
    Loads multiple batches of models and saves lightcurves
    """
    # ==========================================
    # batches = []  : batches to iterate over
    #   - see multi_save()
    # ==========================================
    path = kwargs.get('path', KEPLER_MODELS)
    runs = expand_runs(runs)
    batches = expand_batches(batches, source)
    for batch in batches:
        bname = '{source}_{batch}'.format(source=source, batch=batch)
        bpath = os.path.join(path, bname)

        multi_save(runs=runs, path=bpath, **kwargs)
