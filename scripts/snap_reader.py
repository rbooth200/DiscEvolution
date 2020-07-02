import sys, os
import re
import numpy as np

sys.path.append('/data/rab200/ChemoDrift/new/')
import DiscEvolution.chemistry as chem
from DiscEvolution.planet_formation import Planets


class DiscSnap(object):

    def __init__(self, filename, chem_on=False):
        self.read(filename, chem_on)

    def read(self, filename, chem_on=False):
        """Read disc data from file"""
        # read the header
        head = ''
        vars  = False
        count = 0
        with open(filename) as f:
            for line in f:
                if not vars:
                    if not (line.startswith('# time') or line.startswith('# InternalEvaporation')):
                        head += line
                    elif line.startswith('# InternalEvaporation'):
                        # Get internal photoevaporation type
                        self._IPE = line.strip().split(',')[1].split(':')[-1]
                        print(self._IPE)
                    elif line.startswith('# time'):
                        vars = True
                        # Get the time
                        self._t = float(line.strip().split(':')[1][:-2])                    

                    count += 1
                    continue
                # Get data variables stored
                vars = line[2:].split(' ')
                assert(len(vars) % 2 == 1)

                # Get the number of dust species
                Ndust = len([x for x in vars  if x.startswith('epsilon')])

                # If chemistry was used, get the number of chemical species
                if chem_on:
                    Nchem = (len(vars) - 3 - 2*Ndust) / 2
                    
                    iChem = 2*Ndust + 3
                    chem_spec = vars[iChem:iChem + Nchem]
                break
            
        # Parse the actual data:
        data = np.genfromtxt(filename, skip_header=count, names=True)
        Ndata = data.shape[0]
        names = data.dtype.names
        self._R     = data['R']
        self._Sigma = data['Sigma']
        self._T     = data['T']

        self._eps = np.empty([Ndust, Ndata], dtype='f8')
        self._a   = np.empty([Ndust, Ndata], dtype='f8')
        for i in range(Ndust):
            self._eps[i] = data['epsilon{}'.format(i)]
            self._a[i]   = data['a{}'.format(i)]

        # Only if chemistry used
        if chem_on:
            if Nchem == 6:
                self._chem = chem.MolecularIceAbund(chem.SimpleCOMolAbund(Ndata),
                                                    chem.SimpleCOMolAbund(Ndata))
            else:
                raise AttributeError('Nchem = {}'.format(Nchem) + 
                                     '. Chemistry not recognized')

            for i in range(Nchem):
                self._chem.gas.data[i] = data[names[iChem+i]]
                self._chem.ice.data[i] = data[names[iChem+Nchem+i]]
                                                        
    @property
    def photo_type(self):
        if hasattr(self,"_IPE"):
            return self._IPE        
        else:
            return None
    @property
    def time(self):
        return self._t
    @property
    def R(self):
        return self._R
    @property
    def Sigma(self):
        return self._Sigma
    @property
    def T(self):
        return self._T
    @property
    def dust_frac(self):
        return self._eps
    @property
    def grain_size(self):
        return self._a
    @property
    def chem(self):
        return self._chem

class PlanetSnap(object):

    def __init__(self, filename):
        self.read(filename)


    def read(self, filename):
        """Read disc data from file"""
        # read the header
        head = ''
        vars  = False
        count = 0
        with open(filename) as f:
            for line in f:
                if not vars:
                    if not line.startswith('# time'):
                        head += line
                    else:
                        vars = True
                        # Get the time
                        self._t = float(line.strip().split(':')[1][:-2])                    
                    count += 1
                    continue
                # Get data variables stored
                vars = line[2:].split(' ')
                assert(len(vars) % 2 == 0)

                # Get the number of dust species
                Nchem = (len(vars) - 4) / 2
                iChem = 4
                chem_spec = vars[iChem:iChem + Nchem]
                break
            
        # Parse the actual data:
        data = np.genfromtxt(filename, skip_header=count, names=True)
        planets = Planets(Nchem)
        planets.R      = data['R']
        planets.M_core = data['M_core']
        planets.M_env  = data['M_env']
        planets.t_form = data['t_form']
        planets._N = data.shape[0]


        if Nchem:
            assert(Nchem == 6 or Nchem == 8)
            Ndata = data.shape[0]
            planets.X_core = np.empty([Nchem, Ndata], dtype='f8')
            planets.X_env  = np.empty([Nchem, Ndata], dtype='f8')

            names = data.dtype.names
            for i in range(Nchem):
                planets.X_core[i] = data[names[iChem+i]]
                planets.X_env[i]  = data[names[iChem+Nchem+i]]
                                                        
        self._planets = planets

    @property
    def time(self):
        return self._t
    @property
    def planets(self):
        return self._planets


class Reader(object):

    def __init__(self, SnapType, DIR, base='*', chem_on=False):
        self._SnapType = SnapType
        self._DIR = DIR
        self._chem_on = chem_on

        m = re.compile(r'^'+base+r'_\d\d\d\d.dat$')
        self._files = [ f for f in os.listdir(DIR) if m.findall(f)]
        
        snaps = {}
        Nmax = 0
        for f in self._files:
            n = int(f[-8:-4])
            snaps[n] = os.path.join(self._DIR, f)
            Nmax = max(n, Nmax)
        self._snaps = snaps
        self._Nmax = Nmax
        
            
    def __getitem__(self, n):
        return self._SnapType(self._snaps[n], self._chem_on)

    def filename(self, n):
        return self._snaps[n]

    @property
    def Num_Snaps(self):
        return self._Nmax

                    
class DiscReader(Reader):
    """Read disc snaphshots from file"""
    # By default assume no chemistry
    def __init__(self, DIR, base='disc', chem_on=False):
        super(DiscReader, self).__init__(DiscSnap, DIR, base, chem_on)

class PlanetReader(Reader):
    """Read disc snaphshots from file"""
    def __init__(self, DIR, base='planets'):
        super(PlanetReader, self).__init__(PlanetSnap, DIR, base)

    def compute_planet_evo(self):
        """Compute the time evolution of each planet"""
        planets = []
        times = []
        Np = 0
        for n in range(self.Num_Snaps):
            try:
                pf = self[n]
                for i in range(pf.planets.N):
                    if i == Np:
                        try:
                            p = Planets(pf.planets[0].X_core.shape[0])
                        except AttributeError:
                            p = Planets()
                        p.time = []
                        planets.append(p)
                        Np += 1
                    planets[i].append(pf.planets[i])
                    planets[i].time = np.append(planets[i].time, pf.time)
            except KeyError:
                pass
        return planets
