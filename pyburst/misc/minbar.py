import numpy as np
import pandas as pd
import os

"""
Uses minbar table from:
https://burst.sci.monash.edu/wiki/uploads/MINBAR/minbar.txt 
"""

pyburst_path = os.environ['PYBURST']
filename = 'minbar.txt'
filepath = os.path.join(pyburst_path, 'files', filename)

def load_table():
    """Loads minbar table from URL
    """
    table = pd.read_table(filepath, delim_whitespace=True, header=9)
    return table
