import numpy as np
import os


# ========================================================
# Functions for writing input files for kepler models
# ========================================================
# TODO: rename this kepler_setup.py

def write_genfile(h1, he4, n14, qb, acc_mult, lburn,
                  geemult, path, header, lumdata=0, qnuc=5.,
                  t_end=1.3e5, accdepth=1.0e19, accrate0=5.7E-04,
                  accmass=1.0e18, zonermax=10, zonermin=-1, nstop=10000000,
                  accrate1_str='', nsdump=500, nuc_heat=False, cnv=0,
                  minzone=51, thickfac=0.001, setup_test=False, substrate_off=True,
                  ibdatov=0):
    """========================================================
    Creates a model generator file with the given params inserted
    ========================================================
    h1       = flt   : hydrogen mass fraction
    he4      = flt   : helium    "    "
    z        = flt   : metals    "    "
    qb       = flt   : base heating (MeV/nucleon)
    nstop    = int   : max number of timesteps (cycles)
    qnuc     = flt   : nuclear heating (MeV/nucleon, for thermal setup)
    lburn    = int   : switch for full network energy generation (0/1  = off/on)
    lumdata  = int   : switch for time-dependent base-flux (0/1 = off/on)
    accrate0 = flt   : accretion rate at model start (as fraction of Eddington),
                          for getting profile into equilibrium (essentially setting base flux)
    accrate1_str = str   : optional string to redefine accrate  (-1 = time-dependent)
    nsdump   = int   : keep every 'nsdump' savefiles
    path     = str   : target directory for generator file
    ========================================================"""
    genpath = os.path.join(path, 'xrb_g')

    qnuc_str1 = ''
    qnuc_str2 = ''
    kill_setup = ''
    bmasslow = 'p bmasslow 2.8000000199990d33'

    if nuc_heat:
        qnuc_str1 = f"""
c Convert qnuc from MeV/nucleon to erg/g, then to erg/s (with accrate)
c (Note accrate is in Msun/yr)
o qnuc {qnuc:.2f} def
o qnuc {{1.602e-6}} *
o qnuc {{accrate}} *
o qnuc {{1.99e33 * 5.979e23 / 3.156e7}} *
p xheatl {{qnuc}}
p xheatym 1.e21
p xheatdm 2.e20"""
        qnuc_str2 = "p xheatl 0."

    if setup_test:
        kill_setup = "end\n"
    if not substrate_off:
        bmasslow = ''

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
p thickfac {thickfac:.2f}
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
{bmasslow}
p 147 1.
p 146 1.
p 233 1.d7
p 65 1.d7

p 211 1.75d-9

p minzone {minzone:.0f}
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

# Overwrite hard-coded rates with bdat
p ibdatov {ibdatov:.0f}

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
p xlum0 {qb:.6f} *
c .......
c ..... SCALE to He/C/O L_Edd accretion: factor 1.7 / (X + 1)
c --- h mass fraction ---
o x {h1:.6f} def
# o xeddf {{1.7 / (1. + x)}} def
# p accrate {{xeddf}} *
# p xlum0 {{xeddf}} *

c set fraction of Eddington accretion rate
o xledd {accrate0:.6f} def
p accrate {{xledd}} *
p xlum0 {{xledd}} *

c apply anisotropy multiplier/factor
o accmult {acc_mult:.6f} def
p accrate {{accmult}} *
p xlum0 {{accmult}} *

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
@xm(jm)<{accmass:.4e}
p 52 20
p accdepth 1.d99
p iterbarm 999999
c .........................
{qnuc_str1}

c =================================
@time>1.d17
{qnuc_str2}
p ncnvout {cnv}

c overwrites accreted composition (if need to change)
compsurb {n14:.6f} n14 {he4:.6f} he4 {h1:.6f} h1

p xlum0 1. *
p lumdata {lumdata}

{accrate1_str}

c multiplier (only on time-dependent files!)
p accratef {acc_mult:.6f}
p xl0ratef {acc_mult:.6f}

c use accdepth 5.d20 for He
c use accdepth 1.d20 for H
p accdepth {accdepth:.4e}

mapsurfb
p ibwarn 0

zerotime
p toffset 0.
setcycle 0
cutbin
resetacc
d #

p lburn {lburn}
p 1 1.
p 521 1
p tnucmin 1.d7

p 86 1
p 87 1
p 452 0
p zonermin {zonermin:.4f}
p zonermax {zonermax:.4f}
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

p nstop {nstop:.0f}
p abunlim 0.01
{kill_setup}

@time>{t_end:.4e}
d #
end""")


def write_rpabg(x, z, path, substrate='fe54'):
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
net 1   {substrate}
c define composition
m fecomp    1.0e+00   {substrate}
m hcomp     {x:.4f} h1  {y:.4f} he4 {z:.4f} n14
c
c specify grid composition
g    0  1  fecomp
g   50  1  fecomp
g   51  1  hcomp
g   60  1  hcomp""")
