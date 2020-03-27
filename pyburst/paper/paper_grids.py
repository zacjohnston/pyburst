from pyburst.grids import grid_analyser

source = 'grid5'


def load_grid():
    return grid_analyser.Kgrid(source=source)