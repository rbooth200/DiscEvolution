import copy
import numpy as np
from ..constants import *

################################################################################
# Wrapper for chemistry data
################################################################################
class ChemicalAbund(object):
    """Simple wrapper class to hold chemical species data.

    Holds the mass abundance (g) of the chemical species relative to Hydrogen.

    args:
        species : list, maps species name to location in data array
        masses  : array, molecular masses in atomic mass units
        size    : Number of data points to hold chemistry for
    """
    def __init__(self, species, masses,*sizes):
        # Sizes is dimension zero if not specified
        sizes = sizes if sizes else (0,)
        if len(masses) != len(species):
            raise AttributeError("Number of masses must match the number of"
                                 "species")

        self._indexes = dict([(name, i) for i, name in enumerate(species)])
        self._names   = species
        self._mass    = masses
        self._Nspec = len(self._names)

        self._data = np.zeros((self.Nspec,) + sizes, dtype='f8')

    def __getitem__(self, k):
        return self._data[self._indexes[k]]

    def __setitem__(self, k, val):
        self._data[self._indexes[k]] = val

    def __iadd__(self, other):
        if self.names != other.names:
            raise AttributeError("Chemical species must be the same")

        self._data += other._data    
        return self

    def __iter__(self):
        return iter(self._names)

    def copy(self):
        return copy.deepcopy(self)
            
    def number_abund(self, k):
        """Number abundance of species k, n_k= rho_k / m_k"""
        return self[k]/self.mass(k)

    @property
    def total_abund(self):
        return self._data.sum(0)
    
    def set_number_abund(self, k, n_k):
        """Set the mass abundance from the number abundance"""
        self[k] = n_k * self.mass(k)

    def to_array(self):
        """Get the raw data"""
        return self.data

    def from_array(self, data):
        """Set the raw data"""
        if data.shape[0] == self.Nspec:
            raise AttributeError("Error: shape must be [Nspec, *]")
        self._data = data

   #def resize(self, n):
    #    '''Resize the data array, keeping any elements that we already have'''
    #    dn = n - self.size
    #    if dn < 0:
    #        self._data = self._data[:,:n].copy()
    #    else:
    #        self._data = np.concatenate([self._data,
    #                                     np.empty([self.Nspec,dn],dtype='f8')])
    def resize(self, shape):
        '''Resize the data array, keeping any elements that we already have'''
        try:
            shape = (self.Nspec, ) + tuple(shape)
        except TypeError:
            if type(shape) != type(1):
                raise TypeError("shape must be int, or array of int")
            shape = (self.Nspec, shape)

        new_data = np.zeros(shape, dtype='f8')
        
        idx = [ slice(0, min(ni)) for ni in zip(shape, self._data.shape)]

        # Copy accross the old data
        new_data[idx] = self._data[idx]

        self._data = new_data

    def append(self, other):
        """Append chemistry data from another container"""
        if self.names != other.names:
            raise  AttributeError("Chemical species must be the same")

        self._data = np.append(self._data, other.data)
            
    def mass(self, k):
        """Mass of species in atomic mass units"""
        return self._mass[self._indexes[k]]

    def mu(self):
        '''Mean molecular weight'''
        return self._data.sum(0) / (self._data.T / self._mass).sum(1)

    @property
    def masses(self):
        """Masses of all species in amu"""
        return self._mass

    @property
    def names(self):
        """Names of the species in order"""
        return self._names

    @property
    def Nspec(self):
        return self._Nspec

    @property
    def species(self):
        """Names of the chemical species held."""
        return self._indexes.keys()

    @property
    def data(self):
        return self._data

    @property
    def size(self):
        return self._data.shape[1]
    
    
################################################################################
# Wrapper for combined gas/ice phase data
################################################################################
class MolecularIceAbund(object):
    """Wrapper for holding the fraction of species on/off the grains"""
    def __init__(self, gas=None, ice=None):
        if type(gas) != type(ice):
            raise AttributeError("Both gas and ice must be of the same type")
        self.gas = gas
        self.ice = ice

    def mass(self, k):
        """Get the molecular mass in amu"""
        return self.gas.mass(k)

    def __iter__(self):
        """Iterate over species names"""
        names = list(self.gas.names)
        for name in self.ice.names:
            if name not in self.gas.names:
                names.append(name)
        return iter(names)

    def __len__(self):
        '''Total number of unique species'''
        return len(set(list(self.gas.names) + list(self.ice.names)))
    
        

################################################################################
# Simple models of time-independent C/N/O chemistry
################################################################################
class SimpleChemBase(object):
    """Tabulated time independent C/O and O/H chemistry.

    This model works with the atomic abundances for C, O and Si, computing
    molecular abundances for CO, CH4, CO2, H20, N2, NH3, C-grains and Silicate
    grains.

    args:
        fix_ratios : if True molecular ratios will be assumed to be constant
                     when the ice / gas fractions are calculated
    """
    def __init__(self, fix_ratios=True):

        self._fix_ratios=fix_ratios
        
        # Condensation thresholds
        # CO, CH4, C02, H20, C-grains, silicate grains
        self._T_cond  = { 'CO'  : 20,
                          'CH4' : 30,
                          'CO2' : 70,
                          'H2O' : 150,
                          'N2'  : 18,
                          'NH3' : 68,
                          'C-grain'  : 500,
                          'Si-grain' : 1500,
                          }

    def ASCII_header(self):
        """ASCII_header string"""
        return ('# {} fix_ratios: {}'.format(self.__class__.__name__, 
                                             self._fix_ratios))

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        return self.__class__.__name__, { "fix_ratios" : self._fix_ratios }

    def equilibrium_chem(self, T, rho, dust_frac, abund):
        """Compute the equilibrium chemistry"""

        ice = self.molecular_abundance(T, rho, dust_frac, abund)
        gas = ice.copy()

        for spec in ice.species:
            ice[spec] = self._equilibrium_ice_abund(T, rho, dust_frac,
                                                    spec, ice)
            gas[spec] = np.maximum(gas[spec] - ice[spec], 0)
            
        return MolecularIceAbund(gas=gas, ice=ice)


    def update(self, dt, T, rho, dust_frac, chem, **kwargs):

        if not self._fix_ratios:
            mol_abund  = chem.gas.copy()
            mol_abund += chem.ice

            chem.ice = self.molecular_abundance(T, rho, dust_frac, 
                                                mol_abund=mol_abund)
            chem.gas.data[:] = 0
        else:
            chem.ice += chem.gas
            chem.gas.data[:] = 0

        for spec in chem.ice.species:
            mtot = chem.ice[spec] + chem.gas[spec]

            ice = self._equilibrium_ice_abund(T, rho,  dust_frac,
                                              spec, chem.ice)
            chem.ice[spec] = ice
            chem.gas[spec] = np.maximum(mtot - ice, 0)

            
class StaticChem(SimpleChemBase):
    """Tabulated time independent C/O and O/H chemistry.

    This model works with the atomic abundances for C, O and Si, computing
    molecular abundances for CO, CH4, CO2, H20, N2, NH3, C-grains and Silicate
    grains.

    args:
        fix_ratios : if True molecular ratios will be assumed to be constant
                     when the ice / gas fractions are calculated
    """
    def __init__(self, fix_ratios=True):
        super(StaticChem, self).__init__(fix_ratios)
    
    def _equilibrium_ice_abund(self, T, rho, dust_frac, species, mol_abund):
        """Equilibrium ice fracion"""
        return mol_abund[species] * (T < self._T_cond[species])


class ThermalChem(object):
    """Computes grain thermal adsorption/desorption rates.

    Mixin class, to be used with TimeDependentChem and EquilibriumChem.

    args:
        sig_b   : Number of binding sites, cm^-2.               default = 1.5e15
        rho_s   : Grain density, g cm^-3.                       default = 1
        a       : Mean grain size by area, cm.                  default = 1e-5
        f_bind  : Fraction of grain covered by binding sites.   default = 1
        f_stick : Sticking probability.                         default = 1
        muH     : Mean Atomic weight, in m_H.                   default = 1.28
    """
    def __init__(self, sig_b=1.5e15, rho_s=1., a=1e-5,
                 f_bind=1.0, f_stick=1.0, mu=1.28):
        self._Tbind = { 
            # From KIDA database
            'CO' : 1150., 'CH4' : 1300.,
            'CO2' : 2575., 'H2O' : 5700.,
            # Fayolle+ (2016), Martin-Domenech+ (2014)
            'N2'  : 1266,  'NH3' : 2965,
            }
                        
        # Dry CO and N2
        self._Tbind['CO'] = 850
        self._Tbind['N2'] = 770 # Fayolle+ (2016)

        # Number of dust grains per nucleus, eta:
        m_g = 4*np.pi * rho_s * a**3 / 3
        eta = mu*m_H / m_g
        
        # X_max = (d2g) * eta * Nbind
        #      When X_ice > X_max all first layer binding sites on the
        #      grain are covered. Thus the desorption rate is limited to be
        #      proportional to min(X_ice, X_max)
        N_bind = sig_b * 4*np.pi * a**2 * f_bind
        self._etaNbind = eta*N_bind
        
        # Cache the adsorpsion/desorption coefficients
        self._nu0 = np.sqrt(2 * sig_b * k_B / (m_H *np.pi**2))
        self._v0  = np.sqrt(8 * k_B / (m_H * np.pi))
        
        self._f_des = (1/Omega0) 
        self._f_ads = (1/Omega0) * np.pi*a**2 * f_stick*eta
        
        self._mu = mu

        
        head = ('sig_b: {} cm^-2, rho_s: {} g cm^-1, a: {} cm, '
                'f_bind: {}, f_stick: {}, muH: {}')
        self._head = head.format(sig_b, rho_s, a, f_bind, f_stick, mu)

    def ASCII_header(self):
        """Time dependent chem header"""
        return (super(ThermalChem, self).ASCII_header() +
                ', {}'.format(self._head))

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        __, head = super(ThermalChem, self).HDF5_attributes()

        def fmt(item): return [x.strip() for x in item.split(":")]

        head.update(dict([ fmt(item) for item in self._head.split(',') ]))

        return self.__class__.__name__, head
                     
    def _nu_i(self, Tbind, m_mol):
        """Desorbtion rate per ice molecule"""
        return self._nu0 * np.sqrt(Tbind/m_mol) 

    def _v_therm(self, T, m_mol):
        """Thermal velocity of the species in the gas"""
        return self._v0 * np.sqrt(T/m_mol)

    def _equilibrium_ice_abund(self, T, rho, dust_frac, spec, tot_abund):

        if 'grain' in spec:
            return tot_abund[spec]
        
        Tbind = self._Tbind[spec]
        m_mol = tot_abund.mass(spec)
        mu = self._mu

        n = rho / (mu*m_H)
        X_t = tot_abund[spec] * mu / m_mol

        # Adsorption & desorption rate per molecule
        Sa = self._f_ads * self._v_therm(T, m_mol)  * dust_frac * n
        Sd = self._f_des * self._nu_i(Tbind, m_mol) * np.exp(-Tbind/T) 

        X_max = self._etaNbind * dust_frac
        
        X_eq = X_t - np.minimum(X_t   * Sd/(Sa + Sd + 1e-300),
                                X_max * Sd/(Sa + 1e-300))
        return X_eq * m_mol / mu
        
    
    def _update_ice_balance(self, dt, T, rho, dust_frac, spec, abund):

        if 'grain' in spec:
            # Smooth at the freeze out temperature
            # th = T / self._T_cond[spec]
            # f = 0.5*(1 + np.tanh(20*(th-1)))
            f = 0
            X_t = abund.ice[spec] + abund.gas[spec]

            abund.gas[spec] = X_t * f
            abund.ice[spec] = X_t * (1-f)

            return
                      
        Tbind = self._Tbind[spec]
        m_mol = abund.mass(spec)
        mu = self._mu

        n = rho / (mu*m_H)

        m_t = abund.gas[spec] + abund.ice[spec]
        X_t = m_t * mu / m_mol

        X_s = (abund.ice[spec] / m_mol)
        X_max = self._etaNbind * dust_frac
        
        #Ad/De-sorpstion rate per gas-phase molecule
        Sa = self._f_ads * self._v_therm(T, m_mol)  * dust_frac * n
        Sd = self._f_des * self._nu_i(Tbind, m_mol) * np.exp(-Tbind/T)

        # Rates in each phase:
        S0 = Sa + np.where(X_s > X_max, Sd, 0)
        S1 = Sa + np.where(X_s < X_max, Sd, 0)

        # Time of transition between 1st order/0th order phases
        X1 = X_t * Sa / (Sa + Sd + 1e-300)
        Xm = X_max * np.ones_like(X_s)
        Xm_1 = Xm * Sd / (Sa + 1e-300)
        Xm_2 = Xm * (Sd + Sa) / (Sa + 1e-300)

        tt = np.zeros_like(X_s)

        idx = (X_s < Xm)  & (X1 > Xm)
        tt[idx] = np.log((X_s[idx]-X1[idx])/(Xm[idx]-X1[idx]))
        
        idx = (X_s > Xm) & (X1 < Xm)
        term = (X_t[idx]-X_s[idx]-Xm_1[idx])/(X_t[idx]-Xm_2[idx])
        tt[idx] = np.log(term) / (Sa[idx]+1e-300)

        # Time integrated in each phase
        dt1 = np.maximum(dt - tt, 0)
        dt0 = dt - dt1

        eta = S0*dt0 + S1*dt1

        X_eq = X_t - np.minimum(X_t   * Sd/(Sa + Sd + 1e-300),
                                X_max * Sd/(Sa + 1e-300))

        X_d = np.minimum(X_s * np.exp(-eta) - X_eq * np.expm1(-eta), X_t)
        X_g = np.maximum(X_t - X_d, 0)
        
        abund.ice[spec] = X_d * m_mol / mu
        abund.gas[spec] = X_g * m_mol / mu



class TimeDependentChem(ThermalChem,SimpleChemBase):
    """Time dependent model of molecular absorbtion/desorption due to thermal
    processes.
    """
    def __init__(self, **kwargs):
        ThermalChem.__init__(self, **kwargs)
        SimpleChemBase.__init__(self)

    def update(self, dt, T, rho, dust_frac, chem, **kwargs):
        """Update the gas/ice abundances"""
        for spec in chem:
            self._update_ice_balance(dt, T, rho, dust_frac, spec, chem)

        
class EquilibriumChem(ThermalChem,SimpleChemBase):
    """Equilibrium chemistry, computed as equilibrium of time dependent model.
    """
    def __init__(self, fix_ratios=True, fix_grains=True, **kwargs):
        ThermalChem.__init__(self, **kwargs)
        SimpleChemBase.__init__(self, fix_ratios)
