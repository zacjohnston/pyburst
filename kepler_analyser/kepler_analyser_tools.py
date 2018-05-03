import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import subprocess
import multiprocessing as mp

# kepler
import lcdata

# kepler_grids
from ..grids import grid_tools
from pygrids.misc.pyprint import print_title, print_dashes

# ============================================
# Author: Zac Johnston (2017)
# zac.johnston@monash.edu
#
# Tools for using Nathanael Lampe's kepler_analyser.py
# https://github.com/natl/kepler-analyser
# ============================================

# === GLOBAL SETTINGS ===
GRIDS_PATH = os.environ['KEPLER_GRIDS']
MODELS_PATH = os.environ['KEPLER_MODELS']
OUTPUT_SUFFIX = '_output'
INPUT_SUFFIX = '_input'


def multi_setup_analyser(batches, source, multithread=True, **kwargs):
    batches = grid_tools.expand_batches(batches, source)

    if multithread:
        args = []
        for batch in batches:
            args.append((batch, source))
        with mp.Pool(processes=8) as pool:
            pool.starmap(setup_analyser, args)

    else:
        for batch in batches:
            runs = grid_tools.get_nruns(batch, source)
            setup_analyser(runs=runs, batch=batch, source=source, **kwargs)


def setup_analyser(batch, source, runs=None, basename='xrb', **kwargs):
    """
    Sets up directories and files for kepler_analyser
    --------------------------------------------------------------------------
     runs          = [int]  : list of model IDs
     batch         = int    : batch ID

     NOTE: If any of the above lists contain only a single entry,
           it will be assumed that all runs have the same value.
    """
    source = grid_tools.source_shorthand(source=source)
    path = kwargs.get('path', GRIDS_PATH)
    models_path = kwargs.get('models_path', MODELS_PATH)
    analyser_path = os.path.join(path, 'analyser', source)

    batch_name = f'{source}_{batch}'
    runs_path = os.path.join(models_path, batch_name)

    if runs is None:
        runs = grid_tools.get_nruns(batch, source)
    runs = grid_tools.expand_runs(runs)

    # ===== load input params from MODEL file =====
    paramfile = os.path.join(runs_path, 'MODELS.txt')
    params = np.loadtxt(paramfile, usecols=[1, 3, 5, 7], skiprows=1)

    # kepler_analyser cyrrently doesn't work with only one model.
    # Hack fix is just double-up a single model
    N = len(runs)
    if N == 1:
        N = 2
        runs = np.full(2, runs[0], dtype='int')
        params = np.stack([params, params])

    z = params[:, 0]
    x = params[:, 1]
    accrate = params[:, 2] * params[:, 3]  # multiply accrate by xi factor

    # ====== Check for single-length arrays (see Note for Input Parameters) ======
    arrays = {'x': x, 'z': z, 'accrate': accrate}
    for var in arrays:
        if len(arrays[var]) == 1:
            print(f'Taking {var} to be constant')
            arrays[var] = np.full(N, arrays[var][0])  # expand array to appropriate length

    x = arrays['x']
    z = arrays['z']
    accrate = arrays['accrate']

    # ====== Create input/output directories ======
    indir = f'{batch_name}{INPUT_SUFFIX}'
    outdir = f'{batch_name}{OUTPUT_SUFFIX}'
    inpath = os.path.join(analyser_path, indir)
    outpath = os.path.join(analyser_path, outdir)

    for folder in [inpath, outpath]:
        grid_tools.try_mkdir(folder, skip=True)

    # ====== Extract and write data files (and get no. of cycles) ======
    cycles = extract_lightcurves(runs,
                                 basename=basename,
                                 path_data=runs_path,
                                 path_target=inpath,
                                 **kwargs)

    write_modelfile(runs=runs,
                    x=x,
                    z=z,
                    accrate=accrate,
                    cycles=cycles,
                    basename=basename,
                    batch_name=batch_name,
                    path=inpath)


def write_modelfile(runs, x, z, accrate, cycles, basename, batch_name, path):
    """
    Writes MODELS.txt file for kepler_analyser input
    """
    # --------------------------------------------------------------------------
    # Input Parameters
    # --------------------------------------------------------------------------
    # runs         : as above (setup_analyser)
    # x, z         : as above
    # accrate      : as above
    # basename     : as above
    # batch        : as above
    # cycles = []  : list of no. of cycles in each model
    # path   = str : path to input directory
    # -------------------------------------------------------------------------  -
    filepath = os.path.join(path, 'MODELS.txt')
    N = len(runs)
    M_edd = 1.75e-8  # * 1.7/(x+1)
    acc_sol = np.array(accrate) * M_edd  # accretion rate in solar masses per year

    print('Writing MODELS.txt file')
    print_dashes()

    with open(filepath, 'w') as f:
        f.write('#mod     acc rate        Z         H       Lacc/Ledd   #pul   #cycle   Comment\n')
        for i in range(N):
            f.write('{base}{run}   {acc:.6e}    {z:.4f}    {x:.4f}  {acc_edd:.6f}      {run}    {cycle}    {batch}\n' \
                    .format(base=basename, run=runs[i], acc=acc_sol[i], z=z[i], x=x[i],
                            acc_edd=accrate[i], cycle=cycles[i], batch=batch_name))


def extract_lightcurves(runs, path_data, path_target, basename='xrb'):
    """========================================================
    Loads Kepler .lum binaries and saves txt file of [time, luminosity, radius] for input to kepler_analyser
    Returns No. of cycles for each run
    ========================================================
    runs = []         : list of run numbers, eg. [324,325,340]
    path_data = str   : path to directory where kepler output folders are located (include trailing slash)
    path_target = str : path to directory where output .txt files will be written (include trailing slash)
    in_name = str     : base filename of input, eg. 'run' for run324
    out_name = str    : base filename of output, eg. 'xrb' for xrb324

    NOTE: kepler_analyser overwrites summ.csv and db.csv, should put path_target as new directory
    ========================================================"""

    print_title()
    print('Loading binaries from {}'.format(path_data))
    print('Writing txt files to {}'.format(path_target))
    print_title()

    cycles = np.zeros(len(runs), dtype=int)

    for i, run in enumerate(runs):
        rname = '{base}{r}'.format(base=basename, r=run)
        lcfile = '{}.lc'.format(rname)
        savefile = '{}.data'.format(rname)

        print('Loading kepler binary for {rname}'.format(rname=rname))
        lcpath = os.path.join(path_data, rname, lcfile)
        data = lcdata.load(lcpath)

        print('Writing txt file')
        savepath = os.path.join(path_target, savefile)
        data.write_lc_txt(savepath)

        # --- Save number of cycles in model ---
        cycles[i] = len(data.time)
        print_dashes()

    return cycles


def collect_output(runs, batches, source, basename='xrb',
                   mean_name='mean.data', **kwargs):
    """=======================================================================
    Collects output files from kepler-analyser output and organises them into batches
    =======================================================================
    runs        = int,[int]  : list of model IDs ()
    batches     = [int]      : list of batch IDs/numbers (assumes same run-IDs for eachs)
    basename    = str        : basename of kepler models
    path        = str        : path to location of all collected batches
    ======================================================================="""
    source = grid_tools.source_shorthand(source=source)
    print_title('Collecting mean lightcurve and summ.csv files')

    batches = grid_tools.expand_batches(batches, source)
    runs = grid_tools.expand_runs(runs)
    path = kwargs.get('path', GRIDS_PATH)
    analyser_path = os.path.join(path, 'analyser', source)

    for batch in batches:
        batch_str = '{source}_{batch}'.format(source=source, batch=batch)
        analyser_output_path = os.path.join(analyser_path, batch_str + OUTPUT_SUFFIX)
        source_path = os.path.join(path, 'sources', source)
        save_path = os.path.join(source_path, 'mean_lightcurves', batch_str)
        grid_tools.try_mkdir(save_path, skip=True)

        print('Copying from: ', analyser_output_path)
        print('          to: ', source_path)

        print('Copying/reformatting summ files')
        reformat_summ(batch=batch, source=source)

        print('Copying mean lightcurves')
        final_run_str = f'{basename}{runs[-1]}'
        for run in runs:
            run_str = '{base}{run}'.format(base=basename, run=run)
            sys.stdout.write(f'\r{run_str}/{final_run_str}')

            mean_filepath = os.path.join(analyser_output_path, run_str, mean_name)
            save_file = '{batch}_{run}_{mean}'.format(batch=batch_str, run=run_str, mean=mean_name)
            save_filepath = os.path.join(save_path, save_file)

            subprocess.run(['cp', mean_filepath, save_filepath])
        sys.stdout.write('\n')


def print_summary(runs, batches, source, basename='xrb', skip=1, redshift=1.259,
                  **kwargs):
    """
    prints summary analyser output of model
    """
    source = grid_tools.source_shorthand(source=source)
    batches = grid_tools.expand_batches(batches, source)
    path = kwargs.get('path', GRIDS_PATH)
    analyser_path = os.path.join(path, 'analyser', source)
    runs = grid_tools.expand_runs(runs)

    for batch in batches:
        batch_name = '{source}_{batch}'.format(source=source, batch=batch)
        output_str = '{batch}{output}'.format(batch=batch_name, output=OUTPUT_SUFFIX)

        print_title(f'Batch {batch}')

        summ_filepath = os.path.join(analyser_path, output_str, 'summ.csv')
        summ = pd.read_csv(summ_filepath)
        summ_names = np.genfromtxt(summ_filepath, delimiter="'", usecols=[1], dtype='str')

        for run in runs:
            idx = np.where(summ_names == '{base}{run}'.format(base=basename, run=run))[0][
                0]  # index of row for this run
            N = int(summ['num'][idx])  # Number of bursts

            dt = summ['tDel'][idx] * redshift / 3600
            u_dt = summ['uTDel'][idx] * redshift / 3600

            # ===== Print info =====
            print(basename + str(run))
            print('Total bursts = {n}'.format(n=N))
            print('Exluding first {skip} bursts'.format(skip=skip))
            print('Using redshift (1+z)={:.3f}'.format(redshift))
            print('Delta_t = {:.2f} (hr)'.format(dt))
            print('u(Delta_t) = {:.2f} (hr)'.format(u_dt))
            print_dashes()

    return {'N': N, 'dt': dt, 'u_dt': u_dt}


def reformat_summ(batch, source, basename='xrb', **kwargs):
    """
    Saves a summ.csv file that is human-readable
    """
    source = grid_tools.source_shorthand(source=source)
    path = kwargs.get('path', GRIDS_PATH)

    batch_str = '{source}_{batch}'.format(source=source, batch=batch)
    output_dir = '{batch_str}_output'.format(batch_str=batch_str)

    output_path = os.path.join(path, 'analyser', source, output_dir)
    summ_filepath = os.path.join(output_path, 'summ.csv')

    summ = pd.read_table(summ_filepath, delimiter=',')

    # ==== fix first column name ====
    new_columns = summ.columns.values
    new_columns[0] = 'run'
    summ.columns = new_columns

    # ===== strip model ids to integers =====
    for i, model in enumerate(summ['run']):
        string = model.strip("'")
        string = string.strip(basename)
        summ.iloc[i, 0] = int(string)  # e.g. 'xrb12' --> 12

    summ_str = summ.to_string(index=False, justify='left')
    save_file = 'summ_{batch_str}.txt'.format(batch_str=batch_str)
    save_filepath = os.path.join(path, 'sources', source, 'summ', save_file)

    with open(save_filepath, 'w') as f:
        f.write(summ_str)


def plot_analyser(runs,
                  batch,
                  source,
                  basename='xrb',
                  skip=2,
                  skip_all=False,
                  mean=True,
                  redshift=1.259,
                  gr=True,
                  labels=True,
                  legend=True,
                  **kwargs):
    """Plots all bursts after first 4 on a single axis, from kepler_analyser output"""
    # runs    = [int] : which run to plot
    # name    = str   : base filename of runs, eg. 'xrb'
    # skip    = int   : exclude this many bursts from start
    # skip_all = bool : Only plot mean curves (overrides 'skip')
    # labels  = bool  : place labels at peaks of bursts
    # mean    = bool  : Plot mean lightcurve too
    # gr      = bool  : use GR corrections
    # path    = str   : path to working directory
    # output  = str   : name of folder containing kepler_analyser output
    source = grid_tools.source_shorthand(source=source)
    path = kwargs.get('path', GRIDS_PATH)
    analyser_path = os.path.join(path, 'analyser', source)

    if gr:
        red = redshift
    else:
        red = 1.0

    batch_name = '{source}_{batch}'.format(source=source, batch=batch)
    fig, ax = plt.subplots()

    for run in runs:
        run_str = '{base}{run}'.format(base=basename, run=run)
        # ===== Auto-find the number of bursts in this run (and print recurrence time) =====
        output_str = '{batch}{output}'.format(batch=batch_name, output=OUTPUT_SUFFIX)

        N = print_summary(runs=[run], batches=[batch], source=source,
                          **kwargs)['N']

        if skip_all:
            skip = N

        # ===== Load each burst lightcurve for plotting =====
        for i in range(skip, N):
            burst_file = '{i}.data'.format(i=i)
            burst_filepath = os.path.join(analyser_path, output_str, run_str, burst_file)
            data = np.loadtxt(burst_filepath)

            ax.plot(data[:, 0] * red, data[:, 1] / red)

            if labels:
                label = 'b{}'.format(i)

                peak_idx = np.argmax(data[:, 1])
                peak_time = data[peak_idx, 0] * red
                peak_lum = data[peak_idx, 1] / red

                ax.text(peak_time, peak_lum, label, horizontalalignment='center', verticalalignment='bottom')

        # ===== Add averaged lightcurve to plot =====
        if mean:
            mean_filepath = os.path.join(analyser_path, output_str, run_str, 'mean.data')

            mean_data = np.loadtxt(mean_filepath, usecols=[0, 1], skiprows=1)
            ax.plot(mean_data[:, 0] * red, mean_data[:, 1] / red, c='black', ls='--', linewidth=1, label=run_str)

    if legend:
        ax.legend()

    txtsize = 22
    ax.set_xlabel('(sec)', fontsize=txtsize)
    ax.set_ylabel('L (erg)', fontsize=txtsize)
    ax.set_xlim([-10, 60])
    plt.show(block=False)
