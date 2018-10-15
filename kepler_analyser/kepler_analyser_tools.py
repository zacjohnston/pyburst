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
from ..grids import grid_tools, grid_strings
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

# xxxxxxxxxxxxxxxx !!!CAUTION!!! xxxxxxxxxxxxxxxxxxxxxxx
# this module is very outdated and very un-maintained
# nothing is gaurunteed to work anymore
# proceed with caution
# xxxxxxxxxxxxxxxx !!!CAUTION!!! xxxxxxxxxxxxxxxxxxxxxxx

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
    source = grid_strings.source_shorthand(source=source)
    path = kwargs.get('path', GRIDS_PATH)
    analyser_path = os.path.join(path, 'analyser', source)

    batch_name = grid_strings.get_batch_string(batch, source)
    runs_path = grid_strings.get_batch_models_path(batch, source)

    if runs is None:
        runs = grid_tools.get_nruns(batch, source)
    runs = grid_tools.expand_runs(runs)

    # ===== load input params from MODEL file =====
    paramfile = os.path.join(runs_path, 'MODELS.txt')
    params = pd.read_csv(paramfile, delim_whitespace=True)

    # ====== Create input/output directories ======
    indir = f'{batch_name}{INPUT_SUFFIX}'
    outdir = f'{batch_name}{OUTPUT_SUFFIX}'
    inpath = os.path.join(analyser_path, indir)
    outpath = os.path.join(analyser_path, outdir)

    for folder in [inpath, outpath]:
        grid_tools.try_mkdir(folder, skip=True)

    # ====== Extract and write data files (and get no. of cycles) ======
    extract_lightcurves(runs,
                        basename=basename,
                        path_data=runs_path,
                        path_target=inpath)

    write_model_table(table=params,
                      basename=basename,
                      batch_name=batch_name,
                      path=inpath)


def write_model_table(table, basename, batch_name, path):
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
    m_edd = 1.75e-8  # * 1.7/(x+1)

    print('Writing MODELS.txt file')
    with open(filepath, 'w') as f:
        f.write('#mod     acc rate        Z         H       Lacc/Ledd   #pul   #cycle   Comment\n')
        for row in table.itertuples():
            f.write(f'{basename}{row.run}   {row.accrate*m_edd:.6e}    {row.z:.4f}    {row.x:.4f}'
                    + f'  {row.accrate:.6f}      {row.run}    0    {batch_name}\n')


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
    print(f'Loading binaries from {path_data}')
    print(f'Writing txt files to {path_target}')
    print_title()

    cycles = np.zeros(len(runs), dtype=int)

    for i, run in enumerate(runs):
        rname = grid_strings.get_run_string(run, basename)
        lcfile = f'{rname}.lc'
        savefile = f'{rname}.data'

        print(f'Loading kepler binary for {rname}')
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
    print_title('Collecting mean lightcurve and summ.csv files')
    source = grid_strings.source_shorthand(source=source)
    batches = grid_tools.expand_batches(batches, source)
    runs = grid_tools.expand_runs(runs)

    path = kwargs.get('path', GRIDS_PATH)
    analyser_path = os.path.join(path, 'analyser', source)

    for batch in batches:
        batch_str = grid_strings.get_batch_string(batch, source)
        analyser_output_path = os.path.join(analyser_path, batch_str + OUTPUT_SUFFIX)
        source_path = grid_strings.get_source_path(source)
        save_path = os.path.join(source_path, 'mean_lightcurves', batch_str)
        grid_tools.try_mkdir(save_path, skip=True)

        print('Copying from: ', analyser_output_path)
        print('          to: ', source_path)

        print('Copying/reformatting summ files')
        reformat_summ(batch=batch, source=source, basename=basename)

        print('Copying mean lightcurves')
        final_run_str = grid_strings.get_run_string(runs[-1], basename)
        for run in runs:
            run_str = grid_strings.get_run_string(run, basename)
            sys.stdout.write(f'\r{run_str}/{final_run_str}')

            mean_filepath = os.path.join(analyser_output_path, run_str, mean_name)
            save_file = f'{batch_str}_{run_str}_{mean_name}'
            save_filepath = os.path.join(save_path, save_file)

            subprocess.run(['cp', mean_filepath, save_filepath])
        sys.stdout.write('\n')


def print_summary(runs, batches, source, basename='xrb', skip=1, redshift=1.259,
                  **kwargs):
    """
    prints summary analyser output of model
    """
    source = grid_strings.source_shorthand(source=source)
    batches = grid_tools.expand_batches(batches, source)
    path = kwargs.get('path', GRIDS_PATH)
    analyser_path = os.path.join(path, 'analyser', source)
    runs = grid_tools.expand_runs(runs)

    for batch in batches:
        batch_name = grid_strings.get_batch_string(batch, source)
        output_str = f'{batch_name}{OUTPUT_SUFFIX}'

        print_title(f'Batch {batch}')

        summ_filepath = os.path.join(analyser_path, output_str, 'summ.csv')
        summ = pd.read_csv(summ_filepath)
        summ_names = np.genfromtxt(summ_filepath, delimiter="'", usecols=[1], dtype='str')

        for run in runs:
            run_str = grid_strings.get_run_string(run, basename)
            idx = np.where(summ_names == run_str)[0][0]  # index of row for this run
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
    source = grid_strings.source_shorthand(source=source)
    path = kwargs.get('path', GRIDS_PATH)

    batch_str = '{source}_{batch}'.format(source=source, batch=batch)
    output_dir = '{batch_str}_output'.format(batch_str=batch_str)

    output_path = os.path.join(path, 'analyser', source, output_dir)
    summ_filepath = os.path.join(output_path, 'summ.csv')

    summ = pd.read_table(summ_filepath, delimiter=',')
    summ = summ.rename(index=str, columns={'# model': 'run'})

    # ===== convert model ids to integers =====
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
    source = grid_strings.source_shorthand(source=source)
    path = kwargs.get('path', GRIDS_PATH)
    analyser_path = os.path.join(path, 'analyser', source)

    if gr:
        red = redshift
    else:
        red = 1.0

    batch_name = grid_strings.get_batch_string(batch, source)
    fig, ax = plt.subplots()

    for run in runs:
        run_str = grid_strings.get_run_string(run, basename)
        # ===== Auto-find the number of bursts in this run (and print recurrence time) =====
        output_str = f'{batch_name}{OUTPUT_SUFFIX}'

        N = print_summary(runs=[run], batches=[batch], source=source,
                          **kwargs)['N']

        if skip_all:
            skip = N

        # ===== Load each burst lightcurve for plotting =====
        for i in range(skip, N):
            burst_file = f'{i}.data'
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
