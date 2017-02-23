import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rcParams
rcParams['image.cmap'] = 'plasma'
from snap_reader import DiscReader, PlanetReader
from chemistry import SimpleCOMolAbund, SimpleCOAtomAbund
from planet_formation import Bitsch2015Model
import star, grid, eos
from dust import FixedSizeDust
from scipy.interpolate import InterpolatedUnivariateSpline as spline

class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)

if __name__ ==  "__main__":
    import sys
    DIR = os.path.join('..', 'planets', 'TimeDep', 'irradiated', 'Model2')
    try:
        DIR = sys.argv[1]
    except IndexError:
        pass
    n = 0
    try:
        n = int(sys.argv[2])
    except IndexError:
        pass
    
    reader = DiscReader(DIR)
    
    # Setup a new disc model from the file
    disc = reader[n]

    fname = reader.filename(0)
    #grid = grid.from_file(fname)
    grid = grid.Grid(0.1, 500, 1000, spacing='natural')
    eos = eos.from_file(fname)
    #eos._accrete = False
    star = eos.star

    eos.set_grid(grid)
    eos.update(0,disc.Sigma)
    
    disc = FixedSizeDust(grid, star, eos, 
                         disc.dust_frac.sum(0), disc.grain_size[1],
                         Sigma=disc.Sigma)

    planet_model = Bitsch2015Model(disc)

    Rs = np.logspace(-1, 2, 1000)
    Ms = np.logspace(-2, 3, 1000)

    R, M = np.meshgrid(Rs, Ms)

    mig = planet_model._migrate
    peb = planet_model._peb_acc
    
    mig_dir = mig.migration_rate(R, M) >  0

    plt.pcolormesh(Rs, Ms, mig_dir)
    plt.xscale('log')
    plt.yscale('log')
    plt.title(r'$t={}\,\mathrm{{Myr}}$'.format(reader[n].time/1e6))
    plt.xlabel('$R\,[\mathrm{au}]$')
    plt.ylabel('$M\,[M_\oplus]$')

    _lgSig = spline(np.log(disc.R), np.log(disc.Sigma)).derivative(1)
    _lgT   = spline(np.log(disc.R), np.log(disc.T)).derivative(1)

    plt.figure()
    ax = plt.subplot(311)
    #plt.loglog(disc.R, disc.T)
    plt.semilogx(disc.R, _lgSig(np.log(disc.R)))
    plt.semilogx(disc.R, _lgT(np.log(disc.R)))
    plt.xlabel('$R\,[\mathrm{au}]$')
    plt.ylabel('$T\,[\mathrm{K}]$')
    plt.xlim(0.1, 1e2)

    plt.subplot(312, sharex=ax)
    plt.loglog(disc.R, disc.Sigma)
    plt.loglog(disc.R, 1000*disc.R**-0.7)
    plt.xlabel('$R\,[\mathrm{au}]$')
    plt.ylabel('$\Sigma\,[\mathrm{g\,cm}^{-2}]$')

    plt.subplot(313, sharex=ax)
    plt.loglog(disc.R, peb.M_iso(disc.R))
    plt.xlabel('$R\,[\mathrm{au}]$')
    plt.ylabel('$M_\mathrm{iso}\,[M_\oplus]$')

    plt.show()
