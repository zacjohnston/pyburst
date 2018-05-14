"""
Standardised strings/labels/paths for grid models/batches/files
"""

import os

MODELS_PATH = os.environ['KEPLER_MODELS']
GRIDS_PATH = os.environ['KEPLER_GRIDS']


def source_shorthand(source):
    """Expands source aliases (e.g. 4u ==> 4u1820)
    """
    if source == '4u':
        return '4u1820'
    elif source == 'gs':
        return 'gs1826'
    else:
        return source


# ======================================================
# Basic strings
# ======================================================
def get_batch_string(batch, source, b_tag=''):
    return f'{source}_{b_tag}{batch}'


def get_run_string(run, basename='xrb'):
    return f'{basename}{run}'


def get_model_string(run, batch, source, b_tag='', r_tag=''):
    return f'{source}_{b_tag}{batch}_{r_tag}{run}'


# ======================================================
# Top level paths
# ======================================================
def get_source_path(source):
    return os.path.join(GRIDS_PATH, 'sources', source)


def get_analyser_path(source):
    return os.path.join(GRIDS_PATH, 'analyser', source)


def get_obs_data_path(source):
    return os.path.join(GRIDS_PATH, 'obs_data', source)


# ======================================================
# Misc. paths
# ======================================================
def get_batch_models_path(batch, source):
    batch_str = get_batch_string(batch, source)
    return os.path.join(MODELS_PATH, batch_str)


def get_model_path(run, batch, source, basename='xrb'):
    batch_path = get_batch_models_path(batch, source)
    run_str = get_run_string(run, basename=basename)
    return os.path.join(batch_path, run_str)


def get_source_subdir(source, dir_):
    source_path = get_source_path(source)
    return os.path.join(source_path, dir_)


# ======================================================
# Misc. files
# ======================================================
def get_batch_filename(batch, source, prefix, extension=''):
    batch_str = get_batch_string(batch, source)
    return f'{prefix}_{batch_str}{extension}'


def get_source_filename(source, prefix, extension=''):
    return f'{prefix}_{source}{extension}'


def get_table_filepath(source, table, batch=None):
    """Return filepath of source (and/or batch) table (summ, params)

    Optional: provide batch, for batch-specific table
    """
    if batch is None:
        extension = '.txt'
    else:
        extension = f'_{batch}.txt'

    table_path = get_source_subdir(source, table)
    table_filename = get_source_filename(source, prefix=table, extension=extension)
    return os.path.join(table_path, table_filename)


def get_model_table_filepath(batch, source, filename='MODELS.txt'):
    path = get_batch_models_path(batch, source)
    return os.path.join(path, filename)
