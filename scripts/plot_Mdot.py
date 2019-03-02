from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rcParams
rcParams['image.cmap'] = 'plasma'
from snap_reader import DiscReader


class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)

if __name__ ==  "__main__":
    from DiscEvolution.constants import *
    import sys
    #DIR = os.path.join('..', 'planets', 'TimeDep', 'irradiated', 'Model2')
    #DIR = os.path.join('..', 'output', 'TimeDep', 'irradiated', 'Model1')
    #DIR = os.path.join('..', 'output', 'TimeDep', 'isothermal', 'Model1')
    DIR = os.path.join('.', 'dusty_disc_0') ## CHANGED to match my file structure - ADS
       
    try:
        DIR = sys.argv[1]
    except IndexError:
        pass
    
    """params = {}
    print('Model params:')
    for line in open(os.path.join(DIR, 'model.dat')):
        print('\t', line.strip())
        k, val = line.strip().split()
        params[k] = val
    print()

    alpha = float(params['alpha'])""" ## Not compatible with .json inputs - ADS
    alpha = 1e-3 ## CHANGED to be hard coded for now - ADS

    chemistry_on = False ## Added chemistry switch - ADS

    reader = DiscReader(DIR, 'disc', chem_on=chemistry_on)
    time = []
    Mdot = []
    M    = []
    if chemistry_on:
        solar = reader[0].chem.gas.atomic_abundance()
        solar.set_solar_abundances()

    #snaps = [0, 10, 100, 300]
    
    for n in range(0, reader.Num_Snaps+1):
        disc = reader[n]

        cs =  (GasConst * disc.T / 2.4)**0.5
        H = ((cs  / (AU * Omega0)) * disc.R**1.5)
        h = H / disc.R
        nu = alpha * cs * H * AU
        mdot = (3*np.pi * nu * disc.Sigma).mean()
        mdot /= Msun / (365.25 * 3600*24)
        time.append(disc.time)
        Mdot.append(mdot)
        M.append(np.trapz(disc.Sigma*disc.R, disc.R)*2*np.pi*AU**2/Msun)

    print(Mdot[0], M[0])
    plt.subplot(211)
    plt.loglog(time, Mdot)
    plt.xlabel(r'$t\,[\mathrm{yr}]$')
    plt.ylabel(r'$\dot{M}\,[M_\odot\,\mathrm{yr}^{-1}]$')

    plt.subplot(212)
    plt.loglog(time, M)
    plt.xlabel(r'$t\,[\mathrm{yr}]$')
    plt.ylabel(r'${M}\,[M_\odot]$')
    plt.show()
