from __future__ import print_function
from ..constants import m_H

from multiprocessing import Pool

__all__ = [ "KromeAbund", "KromeIceAbund", "KromeGasAbund",
            "KromeMolecularIceAbund", "KromeChem", 
            "KromeCallBack", "UserDust2GasCallBack",
            "main" ]

# Locate the KROME library code
import sys, os
try:
    KROME_PATH = os.environ["KROME_PATH"]
    sys.path.append(KROME_PATH)
except KeyError:
    raise ImportError("krome_chem module requires KROME_PATH environment "
                      "variable to be set")

import numpy as np
import ctypes

from .base_chem import ChemicalAbund
from pykrome import PyKROME

# Alias for ctypes by reference
def byref(x): return ctypes.byref(ctypes.c_double(x))

# Setup the KROME Library
_krome = PyKROME(path_to_lib=KROME_PATH)
_krome.lib.krome_init()

# Here we setup the species loaded from KROME.
# Note:
#   - KROME distinguishes between gas and ice phase species through the "_DUST"
#     tail on the names, so we use that to seperate the phases.
#   - KROME does not explicitly include refractory dust, so we add that those
#     to the end of the array. For now we hard code a single dust species.

# Number of species
_nmols = _krome.krome_nmols
_ngrain = 1

# Load the names / masses from the KROME library and convert mass to internal
# units (Hydrogen masses)
_krome_names = np.array(_krome.krome_names[:_nmols])
_krome_masses = np.empty(_nmols, dtype='f8')
_krome.lib.krome_get_mass(_krome_masses)
_krome_masses /= m_H

#if "M" in _krome_names:
#    _krome_masses[_krome_names == "M"] = 1.0

# Add grains to the end of the names, with an arbitrary mass
_krome_names  = np.append(_krome_names, "grain")
_krome_masses = np.append(_krome_masses, 100.)

# Seperate solid / gas species
def is_solid(species):
    return species.endswith("_DUST") or species.endswith("grain")

def to_ice_name(species):
    if species.endswith("_DUST"):
        return species[:-5]
    elif species.endswith("grain"):
        return species
    else:
        raise ValueError("Expect name ending in '_DUST' or 'grain'")

_krome_ice = np.array([is_solid(n) for n in _krome_names])
_krome_gas = ~_krome_ice
_krome_ice_names = np.array([to_ice_name(n) for n in _krome_names[_krome_ice]])

class KromeAbund(ChemicalAbund):
    """Wrapper for chemical species used by the KROME package.
 
    args:
        size : Number of data points to hold
    """
    def __init__(self, size=0):
        super(KromeAbund, self).__init__(_krome_names,
                                         _krome_masses, size)


class KromeIceAbund(ChemicalAbund):
    """Wrapper for ice phase chemical species used by the KROME package.
 
    args:
        sizes : Number of data points to hold
    """
    def __init__(self, *sizes):
        super(KromeIceAbund, self).__init__(_krome_ice_names,
                                            _krome_masses[_krome_ice], *sizes)

class KromeGasAbund(ChemicalAbund):
    """Wrapper for gas phase chemical species used by the KROME package.
 
    args:
        sizes : Number of data points to hold
    """
    def __init__(self, *sizes):
        super(KromeGasAbund, self).__init__(_krome_names[_krome_gas],
                                            _krome_masses[_krome_gas],
                                            *sizes)
################################################################################
# Wrapper for combined gas/ice phase data
################################################################################
class KromeMolecularIceAbund(object):
    """Wrapper for holding the fraction of species on/off the grains"""
    def __init__(self, gas=None, ice=None):
        self.gas = gas
        self.ice = ice

    def mu(self):
        """Total mean molecular weight of the data"""
        n_g = (self.gas.data.T / self.gas.masses).sum(1)
        n_i = (self.ice.data.T / self.ice.masses).sum(1)
        
        return (self.gas.data.sum(0) + self.ice.data.sum(0)) / (n_g + n_i)

################################################################################
# Wrapper for KROME Chemistry solver
################################################################################
class KromeCallBack(object):
    """Interface class for user call-backs to KROME"""
    def init_krome(self, krome):
        pass
    def __call__(self,  krome, T, rho, dust_frac, **kwargs):
        pass

class UserDust2GasCallBack(KromeCallBack):
    """Sets the user_dust2gas ratio argument"""
    def __init__(self, grain_size = 1e-5):
        self._asize = grain_size
    def init_krome(self, krome, grain_size=1e-5):
        try:
            krome.lib.krome_set_user_asize(grain_size)
        except AttributeError:
            pass
    def __call__(self, krome, T, rho, dust_frac, **kwargs):
        krome.lib.krome_set_user_dust2gas(dust_frac/(1-dust_frac))

def _krome_update(params):
    [dt, T_i, eps_i, n, call_back, opt] = params

    if call_back is not None:
        call_back(_krome, T_i, n, eps_i, **opt)

    # Do not send dummy grain species.
    _krome.lib.krome(n[:-_ngrain], byref(T_i), byref(dt))

    return n


class KromeChem(object):
    """Time-dependent chemistry integrated with the KROME pacakage

    args:
        renormalize : boolean, default = True
            If true, the total abdunances will be renormalized to 1 after the
            update.
        fixed_mu : float, default 0
            The mean molecular weight (in hydrogen masses). If not specified,
            this will be computed on the fly; however, if fixed_mu > 0, then 
            this value will be used instead.
        call_back: KromeCallBack object:
            Provides an interface to allow the user to call krome_user_*
            routines. The call_back object should provide two methods:
                init_krome(self, krome) : 
                    which is called once during the __init__ routine
                __call__(self, krome, T, n, dust_frac, **kwargs),
                    which is called before each call to krome.
                In both functions krome is the krome python library object
                created by the call to pyKROME
    """
    def __init__(self, renormalize=True, fixed_mu=0., call_back=None):
        self._renormalize = renormalize
        self._mu = fixed_mu
        self._call_back = call_back

        if self._call_back is not None:
            self._call_back.init_krome(_krome)
            
        try:
            N = int(os.environ['OMP_NUM_THREADS'])
        except KeyError:
            N = None
        self._pool = Pool(N)

    def ASCII_header(self):
        """Header for ASCII dump file"""
        return ('# {} '.format(self.__class__.__name__) +
                'renormalize: {}, fixed_mu: {}, '.format(self._renormalize,
                                                         self._mu) + 
                'KROME_PATH {}'.format(KROME_PATH))

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        header = { 
            'renormalize' : "{}".format(self._renormalize),
            'fixed_mu'    : "{}".format(self._mu),
            'KROME_PATH'  : "{}".format(KROME_PATH),
            }

        return self.__class__.__name__, header

    def update(self, dt, T, rho, dust_frac, chem, **kwargs):
        """Integrate the chemistry for time dt"""


        # Convert dt to seconds:
        dt *= _krome.krome_seconds_per_year/(2*np.pi)

        m_gas = chem.gas.masses * m_H
        m_ice = chem.ice.masses * m_H

        gas_data = chem.gas.data.T
        ice_data = chem.ice.data.T

        params = []
        for i in range(len(T)):
            T_i, rho_i, eps_i = T[i], rho[i], dust_frac[i]
            
            rho_i /= 1 - eps_i

            # Compute the number density
            n = np.empty(_nmols + _ngrain, dtype='f8')
            n[_krome_gas] = (gas_data[i] / m_gas) * rho_i
            n[_krome_ice] = (ice_data[i] / m_ice) * rho_i
            if self._call_back is not None:
                opt = { kw : arg[i] for (kw, arg) in kwargs.items() }
            else:
                opt = None

            params.append([dt, T_i, eps_i, n, self._call_back, opt])
        
        n_new = self._pool.map(_krome_update, params)
        
        for i in range(len(T)):
            n = n_new[i]
            
            rho_i = rho[i] / (1 - dust_frac[i])
            n[_krome_gas] *= m_gas / rho_i                              
            n[_krome_ice] *= m_ice / rho_i

            if self._renormalize:
                n /= n.sum()

            gas_data[i] = n[_krome_gas] 
            ice_data[i] = n[_krome_ice] 


    def explore_rates(self, T, rho, dust_frac, chem, cells, xvar, 
                      process_rates=None,
                      **kwargs):
        """Wrapper for the krome_explore_rates function"""

        # Load the reaction names
        reaction_names = []
        
        reactionfile = os.path.join(KROME_PATH, 'reactions_verbatim.dat')
        with open(reactionfile) as f:
            for l in range(_krome.krome_nrea):
                reaction_names.append(f.readline().strip())

        m_gas = chem.gas.masses * m_H
        m_ice = chem.ice.masses * m_H

        n = np.empty(_nmols + _ngrain, dtype='f8')

        gas_data = chem.gas.data.T
        ice_data = chem.ice.data.T
        
        rate_data = []
        for i in cells:
            T_i, rho_i, eps_i = T[i], rho[i], dust_frac[i]
            
            rho_i /= 1 - eps_i

            # Compute the number density
            n[_krome_gas] = (gas_data[i] / m_gas) * rho_i
            n[_krome_ice] = (ice_data[i] / m_ice) * rho_i

            if self._call_back is not None:
                opt = { kw : arg[i] for (kw, arg) in kwargs.items() }
                self._call_back(_krome, T_i, n, eps_i, **opt)

            # Create space for the result
            flux = np.zeros(_krome.krome_nrea, dtype='f8')

            # Do not send dummy grain species.
            _krome.lib.krome_get_flux(n[:-_ngrain], T_i, flux)
 
            if process_rates:
                flux = process_rates(flux, reaction_names)

            #Create the result
            maxflux = flux.max()
            sumflux = flux.sum()

            format = '{:8d} ' +'{:17E} '*5 + '{:3} {:50}\n'
            result = ''
            for j in range(_krome.krome_nrea):
                result += format.format(j+1, xvar, T_i, flux[j], 
                                        flux[j]/maxflux, flux[j]/sumflux,
                                        " ", reaction_names[j])
            rate_data.append(result)
        return rate_data

def main():
    import matplotlib.pyplot as plt
    from ..eos import LocallyIsothermalEOS
    from ..star import SimpleStar
    from ..grid import Grid
    from ..constants import Msun, AU
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

    grid = Grid(Rmin, Rmax, Ncell, spacing='log')
    R = grid.Rc

    eos = LocallyIsothermalEOS(SimpleStar(), cs0, q, alpha)
    eos.set_grid(grid)

    Sigma = (Mdot / (3 * np.pi * eos.nu)) * np.exp(-R/Rd)
    rho = Sigma / (np.sqrt(2 * np.pi) * eos.H * AU)

    T = np.minimum(eos.T, 1500.)
    dust_frac = np.ones_like(Sigma) * d2g / (1. + d2g)

    gas = KromeGasAbund(Ncell)
    ice = KromeIceAbund(Ncell)

    abund = KromeMolecularIceAbund(gas,ice)

    abund.gas.data[:] = 0
    abund.ice.data[:] = 0

    abund.gas.set_number_abund('H2',  1.0)
    abund.gas.set_number_abund('H',   9.10e-5)
    abund.gas.set_number_abund('HE',  9.80e-2)
    abund.gas.set_number_abund('H2O', 3.00e-4)
    abund.gas.set_number_abund('CO',  6.00e-5)
    abund.gas.set_number_abund('CO2', 6.00e-5)
    abund.gas.set_number_abund('CH4', 1.80e-5)
    abund.gas.set_number_abund('N2',  2.10e-5)
    abund.gas.set_number_abund('NH3', 2.10e-5)
    abund.gas.set_number_abund('NA',  3.50e-5)
    abund.gas.set_number_abund('H2S', 6.00e-6)

    abund.gas.data[:] *= (1-dust_frac) / abund.gas.total_abund

    abund.ice["grain"].data[:] = dust_frac

    print("Gas / Total Mean Mol. Weight:", abund.gas.mu()[0], abund.mu()[0])

    times = np.array([1e0,1e1,1e2,1e3,1e4,1e5,1e6])*2*np.pi

    KC = KromeChem(call_back=UserDust2GasCallBack())

    f, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, sharex=True)
    
    ax1.loglog(R, Sigma)
    ax1.set_xlabel('R [au]')
    ax1.set_ylabel('Sigma [g cm^-2]')

    ax2.loglog(R, T)
    ax2.set_xlabel('R [au]')
    ax2.set_ylabel('T [K]')

    ax3.set_xlabel('R [au]')
    ax3.set_ylabel('[CO]')

    ax4.set_xlabel('R [au]')
    ax4.set_ylabel('[H2O]')


    l, = ax3.loglog(R, abund.gas.number_abund('CO'), ls='-',
                    label=str(0.) + 'yr')
    ax3.loglog(R, abund.ice.number_abund('CO'), ls='--', c=l.get_color())

    t = 0.
    tStart = time.time()
    for ti in times:        
        dt = ti - t
        KC.update(dt, T, rho*(1-dust_frac), dust_frac, abund)
        dust_frac = abund.ice.total_abund
        t = ti
        
        tEnd = time.time()
        print ('Time {} ({} min)'.format(t,(tEnd-tStart)/60.))
        tStart = tEnd

        l, = ax3.loglog(R, abund.gas.number_abund('CO'), ls='-',
                        label=str(round(t/(2*np.pi),2)) + 'yr')
        ax3.loglog(R, abund.ice.number_abund('CO'), ls='--',
                   c=l.get_color())

        l, = ax3.loglog(R, abund.gas.number_abund('CO'), ls='-',
                        label=str(round(t/(2*np.pi),2)) + 'yr')
        ax3.loglog(R, abund.ice.number_abund('CO'), ls='--',
                   c=l.get_color())

        l, = ax4.loglog(R, abund.gas.number_abund('H2O'), ls='-')
        ax4.loglog(R, abund.ice.number_abund('H2O'), ls='--', c=l.get_color())



    print("Gas / Total Mean Mol. Weight:", abund.gas.mu()[0], abund.mu()[0])


    ax3.legend()
    plt.show()


if __name__ == "__main__":
    main()
