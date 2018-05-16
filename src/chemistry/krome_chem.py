import sys 
sys.path.append('../')
KROME_PATH = '/home/rich/Working_Copies/krome_ilee/build'
sys.path.append(KROME_PATH)
import numpy as np
import ctypes

from base_chem import ChemicalAbund
from pykrome import PyKROME

# Alias for ctypes by reference
byref = lambda x: ctypes.byref(ctypes.c_double(x))

_krome = PyKROME(path_to_lib=KROME_PATH)
_krome.lib.krome_init()
_nmols = _krome.krome_nmols
_krome_names = np.array(_krome.krome_names[:_nmols])
_krome_masses = np.empty(_nmols, dtype='f8')
_krome.lib.krome_get_mass(_krome_masses)

# Gas / Ice species
_krome_ices = np.array([n.endswith('_DUST') for n in _krome_names])
_krome_ice_names = _krome_names.copy()
_krome_ice_names[_krome_ices] = map(lambda x:x[:-5], _krome_names[_krome_ices])
_krome_ice_indexes = \
    dict([(_krome_ice_names[i],i) for i in range(_nmols) if _krome_ices[i]])

_krome_gas = ~_krome_ices
_krome_gas_indexes = \
    dict([(_krome_names[i],i) for i in range(_nmols) if _krome_gas[i]])

class KromeAbund(ChemicalAbund):
    """Wrapper for chemical species used by the KROME package.
 
    args:
        size : Number of data points to hold
    """
    def __init__(self, size=0):
        super(KromeAbund, self).__init__(_krome_names, _krome_masses, size)


class KromeAbundProxy(ChemicalAbund):
    """Proxy reference for a sub-set of species in a full set of abundances

    args:
        parent  : parent object that we will reference the data of.
        species : indices of the species to reference.
        names   : names to use internally. If None, we default names.
    """
    def __init__(self, parent, species, names, indexes=None):
        masses = _krome_masses
        super(KromeAbundProxy, self).__init__(names, masses,indexes=indexes)

        self._data = parent.to_array()

    # Delete modifiers
    def from_array(self, _):
        raise AttributeError("from_array is deleted in KromeAbundProxy")
    def resize(self, _):
        raise AttributeError("resize is deleted in KromeAbundProxy")
    def append(self, _):
        raise AttributeError("append is deleted in KromeAbundProxy")

class KromeIceAbundProxy(KromeAbundProxy):
    """Proxy class which gives reference to the ice species
    
    args:
        parent  : parent object that we will reference the data of.
    """
    def __init__(self, parent):
        super(KromeIceAbundProxy, self).__init__(parent,
                                                 _krome_ices, _krome_ice_names,
                                                 indexes=_krome_ice_indexes)
        

class KromeGasAbundProxy(KromeAbundProxy):
    """Proxy class which gives reference to the gas species
    
    args:
        parent  : parent object that we will reference the data of.
    """
    def __init__(self, parent):
        super(KromeGasAbundProxy, self).__init__(parent,
                                                 _krome_gas, _krome_names,
                                                 indexes=_krome_gas_indexes)
        

################################################################################
# Wrapper for combined gas/ice phase data
################################################################################
class KromeMolecularIceAbund(KromeAbund):
    """Wrapper for seperating gas/ice species"""
    def __init__(self, size=0):
        super(KromeMolecularIceAbund, self).__init__(size)

        self.gas = KromeGasAbundProxy(self)
        self.ice = KromeIceAbundProxy(self)



################################################################################
# Wrapper for KROME Chemistry solver
################################################################################
class KromeChem(object):
    """Time-dependent chemistry integrated with the KROME pacakage

    args:
        amin : minimum dust grain size, cm. Default=5e-7
        amax : maximum dust grain size, cm. Default=2e-5
        phi  : slope of size distribution. Default=-3.5 (MRN)
    """
    def __init__(self, amin=5e-7, amax=2e-5, phi=-3.5):
        self._amin = byref(amin)
        self._amax = byref(amax)
        self._phi  = byref(phi)
        
    def update(self, dt, T, rho, dust_frac, chem):
        """Integrate the chemistry for time dt"""


        # Convert dt to seconds:
        dt *= _krome.krome_seconds_per_year/(2*np.pi)

        m = chem.masses
        
        chem_data = chem.to_array().T
        for i in range(len(T)):
            Ti, rho_i, eps_i = T[i], rho[i], dust_frac[i]

            n = (chem_data[i] / m) * (rho_i / _krome.krome_p_mass)
            
            _krome.lib.krome_init_dust_distribution(n, eps_i,
                                                    self._amin, self._amax,
                                                    self._phi)
            _krome.lib.krome_set_tdust(Ti)
            _krome.lib.krome(n, byref(Ti), byref(dt))

            chem_data[i] = n * m / (rho_i / _krome.krome_p_mass)

            

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from eos import LocallyIsothermalEOS
    from star import SimpleStar
    from grid import Grid
    from constants import Msun, AU

    Ncell = 1000

    Rmin  = 0.01
    Rmax  = 1000.

    cs0   = 1/30.
    q     = -0.25
    alpha = 1e-3

    Mdot = 1e-8 * (Msun/(2*np.pi)) / AU**2
    Rd = 100.
    
    d2g = 0.01
    

    CO_frac = 1e-4

    grid = Grid(Rmin, Rmax, Ncell, spacing='log')
    R = grid.Rc
    
    eos = LocallyIsothermalEOS(SimpleStar(), cs0, q, alpha)
    eos.set_grid(grid)
    
    Sigma = (Mdot / (3 * np.pi * eos.nu)) * np.exp(-R/Rd)
    rho = Sigma / (np.sqrt(2 * np.pi) * eos.H * AU)
    
    T = eos.T
    dust_frac = np.ones_like(Sigma) * d2g

    abund = KromeMolecularIceAbund(Ncell)

    abund.gas['CO'] = CO_frac*abund.mass('CO')
    abund.ice['CO'] = 1e-20
    
    times = np.array([1e2, 1e3, 1e4, 1e5, 1e6])*2*np.pi

    KC = KromeChem()
    
    t = 0.
    for ti in times:
        dt = ti - t
        KC.update(dt, T, rho, dust_frac, abund)
        t = ti
        
        l, = plt.loglog(R, abund.gas.number_abund('CO'), ls='-',
                        label=str(round(t/(2*np.pi),2)) + 'yr')
        plt.loglog(R, abund.ice.number_abund('CO'), ls='--', c=l.get_color())

    plt.legend()
    plt.show()
