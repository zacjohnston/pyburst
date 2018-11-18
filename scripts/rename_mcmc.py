import os, sys

# =================================================================
# Script to rename old mcmc files (from concord)
# =================================================================
GRIDS_PATH = os.environ['KEPLER_GRIDS']
n_arg = len(sys.argv)

if n_arg != 2:
    raise ValueError('Must provide 1 parameters: \n\t1. [source]')

source = sys.argv[1]
source_path = os.path.join(GRIDS_PATH, 'sources', source, 'concord_summ')
file_list = os.listdir(source_path)

print(f'Renaming from: {source_path}')
for file_ in file_list:
    if 'mcmc' in file_:
        new_file = 'concord_summ' + file_.strip('mcmc')
        print(file_, ' -----> ', new_file)

        old_filepath = os.path.join(source_path, file_)
        new_filepath = os.path.join(source_path, new_file)
        os.rename(old_filepath, new_filepath)
