from __future__ import print_function

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
_krome_masses /= _krome.krome_p_mass

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

        gas_data = chem.gas.data.T
        ice_data = chem.ice.data.T
        for i in range(len(T)):
            Ti, rho_i, eps_i = T[i], rho[i], dust_frac[i]

            nH = rho_i / _krome.krome_p_mass

            n_gas = (gas_data[i] / m_gas) * nH
            n_ice = (ice_data[i] / m_ice) * nH

            n[_krome_gas] = n_gas
            n[_krome_ice] = n_ice

            _krome.lib.krome_init_dust_distribution(n, eps_i,
                                                    self._amin, self._amax,
                                                    self._phi)
            _krome.lib.krome_set_tdust(Ti)
            _krome.lib.krome(n, byref(Ti), byref(dt))

            gas_data[i] = n[_krome_gas] * m_gas / nH
            ice_data[i] = n[_krome_ice] * m_ice / nH



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ..eos import LocallyIsothermalEOS
    from ..star import SimpleStar
    from ..grid import Grid
    from ..constants import Msun, AU

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

    gas = KromeGasAbund(Ncell)
    ice = KromeIceAbund(Ncell)

    abund = KromeMolecularIceAbund(gas,ice)

    abund.gas['CO'] = CO_frac*gas.mass('CO')
    abund.ice['CO'] = 0.

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
    for ti in times:
        dt = ti - t
        KC.update(dt, T, rho, dust_frac, abund)
        t = ti

        l, = plt.loglog(R, abund.gas.number_abund('CO'), ls='-',
                        label=str(round(t/(2*np.pi),2)) + 'yr')
        plt.loglog(R, abund.ice.number_abund('CO'), ls='--',
                   c=l.get_color())

    plt.legend()
    plt.show()
