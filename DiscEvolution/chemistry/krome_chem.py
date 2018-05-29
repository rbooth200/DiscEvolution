from __future__ import print_function
from ..constants import m_H

__all__ = [ "KromeAbund", "KromeIceAbund", "KromeGasAbund",
            "KromeMolecularIceAbund", "KromeChem" ]

# Locate the KROME library code
import sys, os
KROME_PATH = os.environ["KROME_PATH"]
sys.path.append(KROME_PATH)

import numpy as np
import ctypes

from .base_chem import ChemicalAbund
from pykrome import PyKROME

# Alias for ctypes by reference
def byref(x): return ctypes.byref(ctypes.c_double(x))

_krome = PyKROME(path_to_lib=KROME_PATH)
_krome.lib.krome_init()
_nmols = _krome.krome_nmols
_krome_names = np.array(_krome.krome_names[:_nmols])
_krome_masses = np.empty(_nmols, dtype='f8')
_krome.lib.krome_get_mass(_krome_masses)
_krome_masses /= m_H

# Gas / Ice species
_krome_ice = np.array([n.endswith('_DUST') for n in _krome_names])
_krome_gas = ~_krome_ice
_krome_ice_names = np.array(map(lambda x:x[:-5], _krome_names[_krome_ice]))


class KromeAbund(ChemicalAbund):
    """Wrapper for chemical species used by the KROME package.
 
    args:
        size : Number of data points to hold
    """
    def __init__(self, size=0):
        super(KromeAbund, self).__init__(_krome_names, _krome_masses, size)


class KromeIceAbund(ChemicalAbund):
    """Wrapper for ice phase chemical species used by the KROME package.
 
    args:
        size : Number of data points to hold
    """
    def __init__(self, size=0):
        super(KromeIceAbund, self).__init__(_krome_ice_names,
                                            _krome_masses[_krome_ice],
                                            size)

class KromeGasAbund(ChemicalAbund):
    """Wrapper for gas phase chemical species used by the KROME package.
 
    args:
        size : Number of data points to hold
    """
    def __init__(self, size=0):
        super(KromeGasAbund, self).__init__(_krome_names[_krome_gas],
                                            _krome_masses[_krome_gas],
                                            size)
################################################################################
# Wrapper for combined gas/ice phase data
################################################################################
class KromeMolecularIceAbund(object):
    """Wrapper for holding the fraction of species on/off the grains"""
    def __init__(self, gas=None, ice=None):
        self.gas = gas
        self.ice = ice

    def mu(self):
        """Mean molecular weight of the data"""
        n_g = (self.gas.data.T / self.gas.masses).sum(1)
        n_i = (self.ice.data.T / self.ice.masses).sum(1)
        
        return (self.gas.data.sum(0) + self.ice.data.sum(0)) / (n_g + n_i)

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

        m_gas = chem.gas.masses
        m_ice = chem.ice.masses

        n = np.empty(len(m_gas) + len(m_ice), dtype='f8')

        mu = chem.mu() * m_H

        gas_data = chem.gas.data.T
        ice_data = chem.ice.data.T
        for i in range(len(T)):
            Ti, rho_i, eps_i, mu_i = T[i], rho[i], dust_frac[i], mu[i]
            
            nH = rho_i / mu_i

            # Compute the number density
            n[_krome_gas] = (gas_data[i] / m_gas) * nH
            n[_krome_ice] = (ice_data[i] / m_ice) * nH

            _krome.lib.krome_set_dust_to_gas(eps_i)

            _krome.lib.krome(n, byref(Ti), byref(dt))

            gas_data[i] = n[_krome_gas] * m_gas / nH
            ice_data[i] = n[_krome_ice] * m_ice / nH



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ..eos import LocallyIsothermalEOS
    from ..star import SimpleStar
    from ..grid import Grid
    from ..constants import Msun, AU, m_H
    import time

    Ncell = 100

    Rmin  = 0.1
    Rmax  = 1000.

    cs0   = 1/30.
    q     = -0.25
    alpha = 1e-3

    Mdot = 1e-8 * (Msun/(2*np.pi)) / AU**2
    Rd = 100.

    d2g = 0.01
    mu = 1.28

    grid = Grid(Rmin, Rmax, Ncell, spacing='log')
    R = grid.Rc

    eos = LocallyIsothermalEOS(SimpleStar(), cs0, q, alpha)
    eos.set_grid(grid)

    Sigma = (Mdot / (3 * np.pi * eos.nu)) * np.exp(-R/Rd)
    rho = Sigma / (np.sqrt(2 * np.pi) * eos.H * AU)

    T = np.minimum(eos.T, 1500.)
    dust_frac = np.ones_like(Sigma) * d2g

    gas = KromeGasAbund(Ncell)
    ice = KromeIceAbund(Ncell)

    abund = KromeMolecularIceAbund(gas,ice)

    abund.gas.data[:] = 0
    abund.ice.data[:] = 0

    abund.gas.set_number_abund('H2',  0.5)
    abund.gas.set_number_abund('HE',  1.00e-1)
    abund.gas.set_number_abund('C',   3.75e-4) 
    abund.gas.set_number_abund('CO',  3.66e-5)
    abund.gas.set_number_abund('CH4', 1.10e-6)
    abund.gas.set_number_abund('N',   1.15e-4)
    abund.gas.set_number_abund('NH3', 3.30e-6)
    abund.gas.set_number_abund('O',   6.74e-4)
    abund.gas.set_number_abund('H2O', 1.83e-4)
    abund.gas.set_number_abund('NA',  3.50e-5)
    abund.gas.set_number_abund('H2CO',1.83e-6)
    abund.gas.set_number_abund('CO2', 3.67e-5)
    abund.gas.set_number_abund('HCN', 4.59e-7)
    abund.gas.set_number_abund('HNC', 7.34e-8)
    abund.gas.set_number_abund('S',   1.62e-5)
    abund.gas.set_number_abund('H2S', 2.75e-6)
    abund.gas.set_number_abund('SO',  1.47e-6)
    abund.gas.set_number_abund('SO2', 1.84e-7)
    abund.gas.set_number_abund('OCS', 3.30e-6)

    abund.gas.data[:] /= abund.mu() 

    times = np.array([1e0, 1e2, 1e4, 1e6])*2*np.pi

    KC = KromeChem()

    plt.subplot(311)
    plt.loglog(R, Sigma)
    plt.xlabel('R [au]')
    plt.ylabel('Sigma [g cm^-2]')

    plt.subplot(312)
    plt.loglog(R, T)
    plt.xlabel('R [au]')
    plt.ylabel('T [K]')

    plt.subplot(313)
    plt.xlabel('R [au]')
    plt.ylabel('X_i')

    l, = plt.loglog(R, abund.gas.number_abund('CO'), ls='-',
                    label=str(0.) + 'yr')
    plt.loglog(R, abund.ice.number_abund('CO'), ls='--', c=l.get_color())

    t = 0.
    tStart = time.time()
    for ti in times:        
        dt = ti - t
        KC.update(dt, T, rho, dust_frac, abund)
        t = ti
        print ('Time {} ({})'.format(t,(time.time()-tStart)/60.))
        
        l, = plt.loglog(R, abund.gas.number_abund('CO'), ls='-',
                        label=str(round(t/(2*np.pi),2)) + 'yr')
        plt.loglog(R, abund.ice.number_abund('CO'), ls='--',
                   c=l.get_color())


    plt.legend()
    plt.show()