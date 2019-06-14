import sys
import os

from pyburst.grids import grid_strings
# =================================================================
# Script callable from terminal to clean up chain/sampler files
# =================================================================
GRIDS_PATH = os.environ['KEPLER_GRIDS']
n_arg = len(sys.argv)

if n_arg != 4:
    print('Must provide 3 parameters: \n\t1. [source]'
          + '\n\t2. [version]\n\t3. [keep_step]')
    sys.exit()

source = sys.argv[1]
version = sys.argv[2]
keep_step = sys.argv[3]

source_path = grid_strings.get_source_subdir(source, 'mcmc')
file_list = os.listdir(source_path)
discard = []
keep = []

print(source)
print(source_path)

for file_ in file_list:
    if (source in file_
            and file_ not in ['.keep', '.gitignore']
            and f'_V{version}_' in file_
            and f'_S{keep_step}.' not in file_):
        discard += [file_]

    elif (source in file_
            and keep_step in file_
            and f'_V{version}_' in file_):
        keep += [file_]

if len(keep) == 0:
    print(f'ABORTING! This would delete every file of V{version}!'
          + '\n\tcheck that keep_step is an existing step file')
    sys.exit()

if len(discard) == 0:
    print('No files to delete')
    print('Nothing to do')
    sys.exit()

print(f'Files in: {source_path}')
print('These will be kept:')
for k in keep:
    print(f'\t{k}')

print(f'These will be deleted:')
for d in discard:
    print(f'\t{d}')

cont = input('Confirm deletion (y/[n]): ')
if cont not in ('y', 'Y', 'yes', 'Yes'):
    print('Aborting')
    sys.exit()

print('Deleting files...')
for d in discard:
    filepath = os.path.join(source_path, d)
    os.remove(filepath)
