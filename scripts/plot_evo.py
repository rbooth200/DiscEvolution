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
    import sys
    DIR = os.path.join('../planets/pb_gas_acc_f_0.0/TimeDep/'
                       'irradiated/Rc_100/Mdot_1e-08/') 
    
    try:
        DIR = sys.argv[1]
    except IndexError:
        pass

    print('Model params:')
    for l in open(os.path.join(DIR, 'model.dat')):
        print('\t', l.strip())
    print()
 
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

    snaps = [0, 10, 50, 100, 300]
    
    for n in snaps:
        try:
            disc = reader[n]
        except KeyError:
            continue

        Sigma_G = disc.Sigma*(1 - disc.dust_frac.sum(0))
        Sigma_D = disc.Sigma*(    disc.dust_frac       )
        Stokes = (disc.grain_size / disc.Sigma) * np.pi / 2
        
        plt.subplot(321)
        l, = plt.loglog(disc.R, Sigma_G)
        plt.loglog(disc.R, Sigma_D.sum(0), l.get_color() + '--')
        plt.ylabel('$\Sigma_\mathrm{G, D}$')
        
        plt.subplot(322)
        l, = plt.loglog(disc.R, disc.dust_frac.sum(0))
        plt.ylabel('$\epsilon$')
        plt.subplot(323)
        l, = plt.loglog(disc.R, Stokes[1])
        plt.ylabel('$St$')
        plt.subplot(324)
        l, = plt.loglog(disc.R, disc.grain_size[1])
        plt.ylabel('$a\,[\mathrm{cm}]$')


        plt.subplot(325)
        gCO = disc.chem.gas.atomic_abundance()
        sCO = disc.chem.ice.atomic_abundance()
        gCO.data[:] /= solar.data
        sCO.data[:] /= solar.data
        c = l.get_color()
        plt.semilogx(disc.R, gCO['N'] , c+ '-', linewidth=1)
        plt.semilogx(disc.R, sCO['N'] , c+ ':', linewidth=1)
        plt.xlabel('$R\,[\mathrm{au}]}$')
        plt.ylabel('$[C]_\mathrm{solar}$')
        
        plt.subplot(326)
        plt.semilogx(disc.R, gCO['N'] / gCO['O'] , c+ '-')
        plt.semilogx(disc.R, sCO['N'] / sCO['O'] , c+ ':')
        plt.xlabel('$R\,[\mathrm{au}]}$')
        plt.ylabel('$[C/O]_\mathrm{solar}$')


    plt.show()
