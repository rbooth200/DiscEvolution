from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
from matplotlib import rcParams
rcParams['image.cmap'] = 'plasma'
rcParams['font.family'] = 'serif'
rcParams['legend.fontsize'] = 10
rcParams['legend.frameon'] = 0
from snap_reader import DiscReader

def update_plots(num, x, y_data, line):
    line.set_data(x, y_data[num])
    return line,
        
class ChemistryEvolutionAnimation(object):
    def __init__(self, r, t, epsilon, C, O):
        self.r = r
        self.t = t
        self.eps = epsilon
        self.C = C
        self.O = O

        # Set up the initial figures
        self._f, self._subs = plt.subplots(3,1,sharex=True)
        
        rmin = 0.3
        rmax = r.max()

        # Dust to gas ratio:
        s = self._subs[0]
        self._leps, = s.loglog([],[], 'k')
        s.set_xlim(rmin, rmax)
        s.set_ylim(1e-4, 1e-1)
        s.set_ylabel('$\epsilon$')
                   
        # Carbon / Oxygen abundance
        s = self._subs[1]
        self._lC,    = s.loglog([],[], 'k',  label='C')
        self._lO,    = s.loglog([],[], 'c',  label='O')
        self._lCg,   = s.loglog([],[], 'k:', label='C (grain)')
        self._lOg,   = s.loglog([],[], 'c:', label='O (grain)')
        
        s.legend(ncol=2)

        s.set_xlim(rmin, rmax)
        s.set_ylim(0.1, 50)
        s.set_ylabel('$[X]_\mathrm{solar}$')

        # C/O ratio
        s = self._subs[2]
        self._lCO,  = s.semilogx([],[], 'k')
        self._lCOg, = s.semilogx([],[], 'k:')
        s.set_xlim(rmin, rmax)
        s.set_ylim(0, 3)
        s.set_ylabel('$[C/O]_\mathrm{solar}$')
        s.set_xlabel('$R\,[\mathrm{au}]$')
        
    def set_title(self, num):
        self._subs[0].set_title(r'$t = {:2g}\,\mathrm{{Myr}}$'
                                ''.format(self.t[num]))


    def __call__(self, num):
        self.set_title(num)
        
        self._leps.set_data(self.r, self.eps[num])

        self._lC.set_data (self.r, self.C[0][num])
        self._lO.set_data (self.r, self.O[0][num])
        self._lCg.set_data(self.r, self.C[1][num])
        self._lOg.set_data(self.r, self.O[1][num])

        self._lCO.set_data (self.r, self.C[0][num]/self.O[0][num])
        self._lCOg.set_data(self.r, self.C[1][num]/self.O[1][num])

        return [self._leps, 
                self._lC, self._lO, self._lCg, self._lOg, 
                self._lCO, self._lCOg]

    @property
    def N(self):
        return len(self.t)
    @property
    def figure(self):
        return self._f


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

    params = {}
    print('Model params:')
    for line in open(os.path.join(DIR, 'model.dat')):
        print('\t', line.strip())
        k, val = line.strip().split()
        params[k] = val
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

    chem = ChemistryEvolutionAnimation(R, time, eps, C, O)


    chem(int(sys.argv[2]))
    plt.show()
