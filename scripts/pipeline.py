import numpy as np
import sys

from pygrids.burst_analyser import burst_pipeline

# =================================================================
# Script callable from terminal to run burst analysis
# Usage:
#   python pipeline [source] [batch0] [batch1]
# =================================================================


def main(source, batch_first, batch_last, **kwargs):
    bool_map = {'True': True, 'False': False}
    for k in kwargs:
        kwargs[k] = bool_map[kwargs[k]]
    batches = np.arange(batch_first, batch_last + 1)
    burst_pipeline.run_analysis(batches, source, **kwargs)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Parameters:'
              + '\n1. source'
              + '\n2. first batch'
              + '\n3. last batch'
              + '\n4. reload (optional)'
              )
        sys.exit(0)

    if len(sys.argv) == 3:
        main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    else:
        main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
             **dict(arg.split('=') for arg in sys.argv[4:]))
