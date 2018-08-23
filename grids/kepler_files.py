import numpy as np
import os


# ========================================================
# Functions for writing input files for kepler models
# ========================================================


def write_genfile(h1, he4, n14, qb, xi, lburn,
                  geemult, path, lumdata, header,
                  t_end=1.3e5, accdepth=1.0e19, accrate0=5.7E-04,
                  accmass=1.0e18, zonermax=10, zonermin=-1,
                  accrate1_str='', nsdump=500, nuc_heat=False, cnv=0):
    """========================================================
    Creates a model generator file with the given params inserted
    ========================================================
    h1       = flt   : hydrogen mass fraction
    he4      = flt   : helium    "    "
    z        = flt   : metals    "    "
    qb       = flt   : base heating (MeV/nucleon)
    lburn    = int   : switch for full network energy generation (0/1  = off/on)
    lumdata  = int   : switch for time-dependent base-flux (0/1 = off/on)
    accrate0 = flt   : accretion rate at model start (as fraction of Eddington),
                          for getting profile into equilibrium (essentially setting base flux)
    accrate1_str = str   : optional string to redefine accrate  (-1 = time-dependent)
    nsdump   = int   : keep every 'nsdump' savefiles
    path     = str   : target directory for generator file
    ========================================================"""
    genpath = os.path.join(path, 'xrb_g')
    if nuc_heat:
        qnuc1 = """
o qnuc {1.31 + 6.95 * x + 1.92 * x ** 2} def
o qnuc {1.6e-6} *
o qnuc {accrate} *
o qnuc {2.e33 * 6.7e23 / 3.15e7} *
p xheatl {qnuc}
p xheatym 1.e21
p xheatdm 2.e20"""
        qnuc2 = "p xheatl 0."
    else:
        qnuc1 = ''
        qnuc2 = ''

    with open(genpath, 'w') as f:
        f.write(f"""c ==============================================
c {header}
c ==============================================
net 1 h1 he3 he4 n14 c12 o16 ne20 mg24
net 1 si28 s32 ar36 ca40 ti44 cr48 fe52
net 1 ni56 fe54 pn1 nt1
m nstar 1.00 fe54
c He star abundances
m acret {h1:.4f} h1 {he4:.4f} he4 0.0 c12 {n14:.4f} n14 0. o16
c THIS GRID FOR He ACCRETION
g 0   2.0000e25  1 nstar  4.0e+8  1.0e+9
g 1   1.9000e25  1 nstar  4.0e+8  1.0e+9
g 40  1.0000e22  1 nstar  4.0e+8  1.0e+8
g 50  1.0000e21  1 nstar  4.0e+8  1.0e+8
g 51  8.0000e20  1 acret  2.0e+8  1.0e+8
g 54  2.0000e20  1 acret  1.0e+8  1.0e+6
g 55  0.         1 acret  5.0e+7  1.0e+4
dstat
genburn  rpabg
mapburn
p geemult {geemult:.5f}
p 1 1.e-4
p 5 40
p 6 .05
p 7 .05
p 8 .10
p 9 .1
p 10 .99
p 14 1000000
p 16 100000
p 18 10
p 28 2
p 39 50.
p 40 2.
p 46 .15
p 47 3.e-3
p 48 1.
p 49 1.e+50
p 53 .1
p 54  2.
p 55  10.
p 59 .05
p 60 1.0e+06
p 61 2.8e+33
p 62 1.6e+34
p 65 1.0e+99
p 70 1.e+99
p 73 1.e+99
p 75 1.e+99
p 80 .25
p 82 1.e+6
p 83 1.e+4
p 84 2.e-5
p 86 0
p 87 0
p 93 51
p 88 1.e+14
p 105 3.e+9
p 132 6
p 138 .33
p 139 .5
p 144 1.3
p 146 .0
p 147 .0
p 148 0.
p 150 .01
p 151 .015
p 152 .03
p 156 100
p 159 5
p 160 0
p 189 .02
p 206 .003
p 42 14001000
p 199 -1.
p 388 1
p 377 0
p 233 1.1e8
p 299 100000
p 265 -1
c p 425 0.
p 64 1

p 405 -1.d0
p 406 -1.d0
p 420 -1.d0
p 64 1
p 434 1
p 443 2
c p 419 2.80000019999895D33
p bmasslow 2.8000000199990d33
p 147 1.
p 146 1.
p 233 1.d7
p 65 1.d7

p 211 1.75d-9
#
p 444 51
p 119 40
p 132 4
p 336 1.5d19
p 445 1.d20
p 437 10
p 376 1
p 11 1.d-8
p 12 1.d-8
p 128 1.d-4

p 137 1

c no h/he burn dumps
p 454 -1.
p 456 -1.

c=======================================================================
c Now follows the command file
c=======================================================================
//*
c .... accretion rate 1.75D-8 (L/Ledd) * 1.7/(X + 1)
p accrate 1.75D-8
c .... substrate luminosity - accrate * 6.0816737e+43 * (Q/MeV)
c .... 1.0642929e+36 (L/Ledd) * (Q/MeV)
p xlum0   1.0642929e+36
c -------------------------
c substrate L, Q/MeV
p xlum0 {qb:.4f} *
c .......
c ..... SCALE to He/C/O L_Edd accretion: factor 1.7 / (X + 1)
c --- h mass fraction ---
o x {h1:.4f} def
# o xeddf {{1.7 / (1. + x)}} def
# p accrate {{xeddf}} *
# p xlum0 {{xeddf}} *

c set fraction of Eddington accretion rate
o xledd {accrate0:.4f} def
p accrate {{xledd}} *
p xlum0 {{xledd}} *

c apply anisotropy multiplier/factor
o xip {xi:.4f} def
p accrate {{xip}} *
p xlum0 {{xip}} *

c -------------------------
c get model in equilibrium
p ncnvout 0
p nstop 1000000000
p tnucmin 1.d10
p tnumin 1.d7
p accmass 1.d13
p optconv 0.67
p 521 0
p 520 1
c for APPROX ONLY
p jp0 0
p 132 4

c plot refresh time (s)
p ipdtmin 0
c plot

c ================================
c MATCH TO accmass
@xm(jm)<{accmass:.2e}
p 52 20
p accdepth 1.d99
p iterbarm 999999
c .........................
{qnuc1}

c =================================
@time>1.d17
{qnuc2}
p ncnvout {cnv}

c overwrites accreted composition (if need to change)
compsurb {n14:.4f} n14 {he4:.4f} he4 {h1:.4f} h1

p xlum0 1. *
p lumdata {lumdata}

{accrate1_str}

c multiplier (only on time-dependent files!)
p accratef {xi:.4f}
p xl0ratef {xi:.4f}

c use accdepth 5.d20 for He
c use accdepth 1.d20 for H
p accdepth {accdepth:.2e}

mapsurfb
p ibwarn 0

zerotime
p toffset 0.
setcycle 0
cutbin
resetacc

p lburn {lburn}
p 1 1.
p 521 1
p tnucmin 1.d7

p 86 1
p 87 1
p 452 0
p zonermin {zonermin:.2f}
p zonermax {zonermax:.2f}
p zonemmax 1.d99
p ddmin 1.d4
c decretion
p decrate -1.D0
p idecmode 1
p jshell0 0
p ipup 5

c some other stuff
c p 69 5.d18
p pbound {{6.67259e-8 * zm(0) * xm(0) / (4. * 3.14159 * rn(0) ^ 4 ) * 0.5}}

p 132 11
p nsdump {nsdump:.0f}

p abunlim 0.01

@time>{t_end:.3e}
end""")


def write_rpabg(x, z, path):
    """=================================================
    Writes burn generator file, rpabg. Sets initial grid structure
    =================================================
    x    = flt  : hydrogen mass fraction
    z    = flt  : metal mass fraction (here N14)
    path = str  : path to write file to
    ================================================="""
    print('Writing rpabg file')
    filepath = os.path.join(path, 'rpabg')
    y = 1 - x - z

    with open(filepath, 'w') as f:
        f.write(f"""c rpa3bg -- burn generator deck for x-ray burst calculations
c
net 1    nt1    h1    h2    h3   he3   he4   li6   li7   be7
net 1    be9    b8   b10   b11   c11   c12   c13   c14   n13
net 1    n14   n15   o14   o15   o16   o17   o18
net 1    f16   f17   f18   f19
net 1   ne19  ne20
net 1   ne21  ne22
net 1   na21  na22  na23  mg23  mg24  mg25  mg26
net 1   al25  al26  al27  si27  si28  si29  si30
net 1   p30   p31   s31   s32   s33   s34   s35   s36
net 1   fe56
c define composition
m fecomp    1.0e+00   fe56
m hcomp     {x:.4f} h1  {y:.4f} he4 {z:.4f} n14
c
c specify grid composition
g    0  1  fecomp
g   50  1  fecomp
g   51  1  hcomp
g   60  1  hcomp""")


def base(qb=0.3, acc_file='outburst.acc', qb_delay=0,
         save_file='outburst.lum'):
    """=================================================
    Creates a base luminosity file from an accretion rate file
    =================================================
    acc_file : output from accrise()
    qb_delay: time delay added between accretion and baselum curve (hrs, observer frame)
    ================================================="""
    # TODO: NOTE: this has not been touched/used in a long time:
    red = 1.259
    path = '/home/zacpetej/projects/codes/mdot'
    pull_path = os.path.join(path, 'tmp', acc_file)
    target = os.path.join(path, 'tmp', save_file)

    xlum0 = 1.0642929e+36  # Base luminosity (erg/s) for 1MeV/nuc at Eddington
    xlum0 *= qb  # MeV/nuc

    mdot_edd = 1.75e-8  # Eddington rate Msun/yr
    mdot_edd *= Msun / (8.64e4 * 365.25)  # g/s

    Qb = np.loadtxt(pull_path, skiprows=2)
    Qb[:, 0] += qb_delay * 3600 / red  # Add time delay to base luminosity
    Qb[:, 1] *= 1 / mdot_edd  # Fraction of eddington accretion
    Qb[:, 1] *= xlum0  # Base erg/s

    n = len(Qb[:, 0])
    head = f'# Qb (erg/s) from {qb:.3f} MeV/nuc, +{qb_delay} hr delay from accretion \n{n}'

    print(f'Saving ({qb} MeV):  {target}')
    np.savetxt(target, Qb, fmt='%25.17E%25.17E', header=head, comments='')


def accrise(tshift, plot=False, save_file='outburst.acc', rise_file='rise_spline2.txt',
            pca_file='pca_GR.acc'):
    """=================================================
    Appends accretion rise curve/spline to PCA data (already redshifted)
    ================================================="""
    # TODO: NOTE: this has not been touched/used in a long time:
    red = 1.259
    path = '/home/zacpetej/projects/codes/mdot'
    pull_path = os.path.join(path, 'files/', )
    pull_rise = os.path.join(pull_path, rise_file)
    pull_pca = os.path.join(pull_path, pca_file)
    target = os.path.join(path, 'tmp/', save_file)
    print(f'Loading {rise_file} and {pca_file} from:  {pull_path}')

    # ARRAYS: [time (s), mdot (g/s)]
    # Time zeroed to 10 days (observer frame) prior to PCA data
    start = np.array([0, 8.86e11], ndmin=2)  # First point
    spline = np.loadtxt(pull_rise, skiprows=1)  # Accretion onset/rise
    pca = np.loadtxt(pull_pca, skiprows=4)  # PCA accretion

    dt = tshift / red  # Amount to shift onset forward (hr)
    spline_old = np.array(spline)

    print(f'Shifting spline later by {tshift:.2f} hrs (observer frame)')

    spline[:, 0] += dt * 3600  # apply time shift
    keep = spline[:, 0] < pca[0, 0]  # Only keep points still prior to PCA data
    spline = spline[keep]
    throwaway = np.sum(np.invert(keep))  # No. of points discarded

    print('Discarding {throwaway} overlapping points from spline')
    acc = np.concatenate((start, spline, pca))

    tf = 8.64e4

    if plot:
        plt.figure()
        plt.plot(acc[:, 0] * red / tf, acc[:, 1])
        plt.plot(spline_old[:, 0] * red / tf, spline_old[:, 1])
        plt.plot(spline_old[[0, -1], 0] * red / tf, spline_old[[0, -1], 1])
        plt.show(block=False)

    n = len(acc[:, 0])
    head = f'# Accretion rate (g/s) with spline onset, shifted by {tshift:.2f} hrs (observer frame)\n{n}'

    print(f'Saving:   {target}')
    np.savetxt(target, acc, fmt='%25.17E%25.17E', header=head, comments='')
