import numpy as np
import matplotlib.pyplot as plt
import os

# kepler
# import kepdump
import lcdata


def substrate_lc():
    path = '/home/zacpetej/archive/kepler/biggrid3_1/xrb11'
    fp0 = os.path.join(path, 'xrb11_lc.txt')
    fp1 = os.path.join(path, 'test11', 'test11.lc')
    sday = 8.64e4

    l0 = np.loadtxt(fp0)
    l1 = lcdata.load(fp1)

    fig, ax = plt.subplots()
    ax.plot(l0[:, 0]/sday, l0[:, 1])
    ax.plot(l1.time/sday, l1.xlum)
    ax.set_yscale('log')

    plt.show(block=False)


def substrate_temp(cycle0, cycle1):
    pass
