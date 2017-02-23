import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rcParams
rcParams['image.cmap'] = 'plasma'
from snap_reader import DiscReader


class Formatter(object):
    def __init__(self, x,y,z):
        self._x, self._y, self._z = x,y,z
    def __call__(self, x, y):
        try:
            i, j = np.searchsorted(self._x, x), np.searchsorted(self._y, y)
            z = self._z[j,i]
            return 'x={:5g}, y={:5g}, z={:5g}'.format(x, y, z)
        except IndexError:
            return 'x={:5g}, y={:5g}'.format(x, y)

def pcolor_plot(x,y,z, **kwargs):
    ax = plt.gca()
    plt.pcolormesh(x,y,z, **kwargs)
    ax.format_coord = Formatter(x,y,z)
    
if __name__ ==  "__main__":
    import sys
    DIR = os.path.join('..', 'output', 'TimeDep', 'irradiated', 'Model1')
    #DIR = os.path.join('..', 'output', 'Madhu', 'irradiated', 'Model2')
    #DIR = os.path.join('..', 'output', 'Oberg', 'irradiated', 'Model2')
    #DIR = os.path.join('..', 'output', 'isothermal', 'Model2')
    DIR = os.path.join('..', 'planets', 'TimeDep', 'irradiated', 'Model2')

    try:
        DIR = sys.argv[1]
    except IndexError:
        pass

    
    reader = DiscReader(DIR, 'disc')
    time = []
    T = []
    S = []
    eps = []
    size = []
    C = [], []
    O = [], []

    solar = reader[0].chem.gas.atomic_abundance()
    solar.set_solar_abundances()
    
    for n in range(0, reader.Num_Snaps+1):
        disc  = reader[n]
        time.append(disc.time)
        T.append(disc.T)

        S.append(disc.Sigma)
        eps.append(disc.dust_frac.sum(0))
        size.append(disc.grain_size[1])

        gas,ice = [X.atomic_abundance() for X in (disc.chem.gas,disc.chem.ice)]

        C[0].append(gas['C'] / solar['C'])
        C[1].append(ice['C'] / solar['C'])
        
        O[0].append(gas['O'] / solar['O'])
        O[1].append(ice['O'] / solar['O'])
        
        

    time = np.array(time) / 1e6
    T = np.array(T)
    S = np.array(S)
    eps = np.array(eps)
    size = np.array(size)
    R = disc.R

    St = (size / S) * np.pi / 2
    
    C = [np.array(X) for X in C]
    O = [np.array(X) for X in O]

    # Temperature
    plt.subplot(421)
    pcolor_plot(R, time, T, norm=LogNorm(), vmin=10, vmax=1500)
    cbar = plt.colorbar()
    cbar.set_label('$T\,[K]$')
    #plt.xlabel('$R\,[\mathrm{au}]$')
    plt.ylabel('$t\,[10^6\mathrm{yr}]$')
    plt.xscale('log')
    plt.xlim(R.min(), R.max())


    # Dust-to-gas ratio
    plt.subplot(422)
    pcolor_plot(R, time, eps/(1-eps), norm=LogNorm())
    cbar = plt.colorbar()
    cbar.set_label(r'Dust-to-gas ratio')
    #plt.xlabel(r'$R\,[\mathrm{au}]$')
    plt.ylabel('$t\,[10^6\mathrm{yr}]$')
    plt.xscale('log')
    plt.xlim(R.min(), R.max())

    # Stokes number
    plt.subplot(423)
    pcolor_plot(R, time, St, norm=LogNorm(), vmin=1e-6)
    cbar = plt.colorbar()
    cbar.set_label(r'$St$')
    #plt.xlabel(r'$R\,[\mathrm{au}]$')
    plt.ylabel('$t\,[10^6\mathrm{yr}]$')
    plt.xscale('log')
    plt.xlim(R.min(), R.max())

    
    # Grain Size
    plt.subplot(424)
    pcolor_plot(R, time, size, norm=LogNorm())
    cbar = plt.colorbar()
    cbar.set_label(r'$a\,[\mathrm{cm}]$')
    #plt.xlabel(r'$R\,[\mathrm{au}]$')
    plt.ylabel('$t\,[10^6\mathrm{yr}]$')
    plt.xscale('log')
    plt.xlim(R.min(), R.max())


    # Chemistry
    plt.subplot(425)
    pcolor_plot(R, time, C[0])
    cbar = plt.colorbar()
    cbar.set_label(r'$[C]_\mathrm{solar}$')
    #plt.xlabel(r'$R\,[\mathrm{au}]$')
    plt.ylabel('$t\,[10^6\mathrm{yr}]$')
    plt.xscale('log')
    plt.xlim(R.min(), R.max())


    plt.subplot(426)
    pcolor_plot(R, time, C[0]/O[0])
    cbar = plt.colorbar()
    cbar.set_label(r'$[C/O]_\mathrm{solar}$')
    #plt.xlabel(r'$R\,[\mathrm{au}]$')
    plt.ylabel('$t\,[10^6\mathrm{yr}]$')
    plt.xscale('log')
    plt.xlim(R.min(), R.max())



    plt.subplot(427)
    pcolor_plot(R, time, C[1])
    cbar = plt.colorbar()
    cbar.set_label(r'$[C]_\mathrm{solar}$')
    plt.xlabel(r'$R\,[\mathrm{au}]$')
    plt.ylabel('$t\,[10^6\mathrm{yr}]$')
    plt.xscale('log')
    plt.xlim(R.min(), R.max())


    plt.subplot(428)
    pcolor_plot(R, time, C[1]/O[1])
    cbar = plt.colorbar()
    cbar.set_label(r'$[C/O]_\mathrm{solar}$')
    plt.xlabel(r'$R\,[\mathrm{au}]$')
    plt.ylabel('$t\,[10^6\mathrm{yr}]$')
    plt.xscale('log')
    plt.xlim(R.min(), R.max())


    
    plt.show()
