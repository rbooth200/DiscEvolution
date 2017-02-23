import copy
import numpy as np
from constants import *


################################################################################
# Wrapper for chemistry data
################################################################################
class ChemicalAbund(object):
    '''Simple wrapper class to hold chemical species data.

    Holds the mass abundance (g) of the chemical species relative to Hydrogen.

    args:
        species : list, maps species name to location in data array
        masses  : array, molecular masses in atomic mass units
        size    : Number of data points to hold chemistry for
    '''
    def __init__(self, species, masses,size=0):
        if len(masses) != len(species):
            raise AttributeError("Number of masses must match the number of"
                                 "species")

        self._indexes = dict([(name, i) for i, name in enumerate(species)])
        self._names   = species
        self._mass    = masses
        self._Nspec = len(species)

        self._data = np.zeros([self.Nspec, size], dtype='f8')

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
        '''Number abundance of species k, n_k= rho_k / m_k'''
        return self[k]/self.mass(k)

    @property
    def total_abund(self):
        return self._data.sum(0)
    
    def set_number_abund(self, k, n_k):
        '''Set the mass abundance from the number abundance'''
        self[k] = n_k * self.mass(k)

    def to_array(self):
        '''Get the raw data'''
        return self.data

    def from_array(self, data):
        '''Set the raw data'''
        if data.shape[0] == self.Nspec:
            raise AttributeError("Error: shape must be [Nspec, *]")
        self._data = data

    def resize(self, n):
        '''Resize the data array, keeping any elements that we already have'''
        dn = n - self.size
        if dn < 0:
            self._data = self._data[:,:n].copy()
        else:
            self._data = np.concatenate([self._data,
                                         np.empty([self.Nspec,dn], dtype='f8')])

    def append(self, other):
        '''Append chemistry data from another container'''
        if self.names != other.names:
            raise  AttributeError("Chemical species must be the same")

        self._data = np.append(self._data, other.data)
            
    def mass(self, k):
        '''Mass of species in atomic mass units'''
        return self._mass[self._indexes[k]]

    @property
    def masses(self):
        '''Masses of all species in amu'''
        return self._mass

    @property
    def names(self):
        '''Names of the species in order'''
        return self._names

    @property
    def Nspec(self):
        return self._Nspec

    @property
    def species(self):
        '''Names of the chemical species held.'''
        return self._indexes.keys()

    @property
    def data(self):
        return self._data

    @property
    def size(self):
        return self._data.shape[1]
    
    
################################################################################
# Simple CO Chemistry wrappers
################################################################################
class SimpleCOAtomAbund(ChemicalAbund):
    '''Class to hold the raw atomic abundaces of C/O/Si for the CO chemistry'''
    def __init__(self, size=0):
        atom_ids = ['C', 'O', 'Si']
        masses   = [ 12., 16., 28. ]
        
        super(SimpleCOAtomAbund, self).__init__(atom_ids, masses, size)


    def set_solar_abundances(self, muH=1.28):
        '''Solar mass fractions of C, O and Si.
        
        args:
            muH : mean atomic mass, default = 1.28
        '''
        self._data[:] = np.outer(np.array([12*2.7e-4, 16*4.9e-4, 28*3.2e-5]),
                                 np.ones(self.size)) / muH

    
class SimpleCOMolAbund(ChemicalAbund):
    '''Class that holds the abundances of molecules needed for C/O chemistry'''
    def __init__(self, size=0):
        mol_ids  = [ 'CO', 'CH4', 'CO2', 'H2O', 'C-grain', 'Si-grain']
        mol_mass = [  28.,  16.,    44.,  18.,   12.,       100.]

        super(SimpleCOMolAbund, self).__init__(mol_ids, mol_mass, size)

        # Atomic make up of the molecules:
        self._n_spec = { 'CO'       : { 'C' : 1, 'O' : 1,         },
                         'CH4'      : { 'C' : 1,                  },
                         'CO2'      : { 'C' : 1, 'O' : 2,         },
                         'H2O'      : {          'O' : 1,         },
                         'C-grain'  : { 'C' : 1,                  },
                         'Si-grain' : {          'O' : 3, 'Si' : 1},
                         }
        
    def atomic_abundance(self):
        '''Compute the mass abundances of atomic species in the molecules'''

        atomic_abund = SimpleCOAtomAbund(self.data.shape[1])
        for mol in self.species:
            nspec = self._n_spec[mol]
            for atom in nspec:
                n_atom = (self[mol]/self.mass(mol)) * nspec[atom]
                atomic_abund[atom] += n_atom * atomic_abund.mass(atom)
                
        return atomic_abund

class MolecularIceAbund(object):
    '''Wrapper for holding the fraction of species on/off the grains'''
    def __init__(self, gas=None, ice=None):
        if type(gas) != type(ice):
            raise AttributeError("Both gas and ice must be of the same type")
        self.gas = gas
        self.ice = ice

    def mass(self, k):
        '''Get the molecular mass in amu'''
        return self.gas.mass(k)

    def __iter__(self):
        '''Iterate over species names'''
        return iter(self.gas)
    
        
################################################################################
# Simple models of time-independent C/O and O/H chemistry
################################################################################
class SimpleCOChemBase(object):
    '''Tabulated time independent C/O and O/H chemistry.

    This model works with the atomic abundances for C, O and Si, computing
    molecular abundances for CO, CH4, CO2, H20, C-grains and Silicate grains.

    args:
        fix_ratios : if True molecular ratios will be assumed to be constant 
                     when the ice / gas fractions are calculated
        fix_grains : if fix_ratio is False and fix_grains is true the molecular
                     abundances will be updated, but not the grain abundance.
    '''
    def __init__(self, fix_ratios=True, fix_grains=True):

        self._fix_ratios=fix_ratios
        self._fix_grains=fix_grains
        
        # Condensation thresholds
        # CO, CH4, C02, H20, C-grains, silicate grains
        self._T_cond  = { 'CO'  : 20,
                          'CH4' : 30,
                          'CO2' : 70,
                          'H2O' : 150,
                          'C-grain'  : 500,
                          'Si-grain' : 1500,
                          }

    def header(self):
        '''Header string'''
        return ('# {} fix_ratios: {}, '
                'fix_grains: {}'.format(self.__class__.__name__,
                                        self._fix_ratios, self._fix_grains))

    def equilibrium_chem(self, T, rho, dust_frac, abund):
        '''Compute the equilibrium chemistry'''

        ice = self.molecular_abundance(T, rho, dust_frac, abund)
        gas = ice.copy()

        for spec in ice.species:
            ice[spec] = self._equilibrium_ice_abund(T, rho, dust_frac,
                                                    spec, ice)
            gas[spec] = np.maximum(gas[spec] - ice[spec], 0)
            
        return MolecularIceAbund(gas=gas, ice=ice)


    def update(self, dt, T, rho, dust_frac, chem):

        if not self._fix_ratios:
            mol_abund  = chem.gas.copy()
            mol_abund += chem.ice
            
            grain_abund=None
            if self._fix_grains:
                grain_abund = mol_abund.copy()
                for spec in grain_abund:
                    if not spec.endswith('grain'):
                        grain_abund[spec] = 0
                grain_abund = grain_abund.atomic_abundance() 

            chem.ice = self.molecular_abundance(T, rho, dust_frac, 
                                                mol_abund.atomic_abundance(),
                                                grain_abund)
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

            
class StaticCOChem(SimpleCOChemBase):
    '''Tabulated time independent C/O and O/H chemistry.

    This model works with the atomic abundances for C, O and Si, computing
    molecular abundances for CO, CH4, CO2, H20, C-grains and Silicate grains.

    args:
        fix_ratios : if True molecular ratios will be assumed to be constant 
                     when the ice / gas fractions are calculated
    '''
    def __init__(self, fix_ratios=True):
        super(StaticCOChem, self).__init__(fix_ratios)
    
    def _equilibrium_ice_abund(self, T, rho, dust_frac, species, mol_abund):
        '''Equilibrium ice fracion'''
        return mol_abund[species] * (T < self._T_cond[species])


class ThermalCOChem(object):
    '''Computes grain thermal adsorption/desorption rates. 

    Mixin class, to be used with TimeDependentCOChem and EquilibriumCOChem.

    args:
        sig_b   : Number of binding sites, cm^-2.               default = 1.5e15
        rho_s   : Grain density, g cm^-3.                       default = 1
        a       : Mean grain size by area, cm.                  default = 1e-5
        f_bind  : Fraction of grain covered by binding sites.   default = 1
        f_stick : Sticking probability.                         default = 1
        muH     : Mean Atomic weight, in m_H.                   default = 1.28
    '''
    def __init__(self, sig_b=1.5e15, rho_s=1., a=1e-5,
                 f_bind=1.0, f_stick=1.0, mu=1.28):
        # From KIDA database
        self._Tbind = { 'CO' : 1150., 'CH4' : 1300.,
                        'CO2' : 2575., 'H2O' : 5700. }
                        
        # KIDA with Aikawa+ (1996) for CO
        self._Tbind['CO'] = 850

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

        
        head = ('sig_b: {}cm^-2, rho_s: {}g cm^-1, a: {} cm, '
                'f_bind: {}, f_stick: {}, muH: {}')
        self._head = head.format(sig_b, rho_s, a, f_bind, f_stick, mu)

    def header(self):
        '''Time dependent chem header'''
        return super(ThermalCOChem, self).header() + ', {}'.format(self._head)
                     
    def _nu_i(self, Tbind, m_mol):
        '''Desorbtion rate per ice molecule'''
        return self._nu0 * np.sqrt(Tbind/m_mol) 

    def _v_therm(self, T, m_mol):
        '''Thermal velocity of the species in the gas'''
        return self._v0 * np.sqrt(T/m_mol)

    def _equilibrium_ice_abund(self, T, rho, dust_frac, spec, tot_abund):

        if 'grain' in spec:
            return tot_abund[spec]
        
        Tbind = self._Tbind[spec]
        m_mol = tot_abund.mass(spec)
        mu = self._mu

        n = rho / (mu*m_H)
        X_t = tot_abund[spec] * mu / (m_mol)

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
        X_t = m_t * mu / (m_mol)

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


class TimeDependentCOChem(ThermalCOChem,SimpleCOChemBase):
    '''Time dependent model of molecular absorbtion/desorption due to thermal 
    processes.
    '''
    def __init__(self, **kwargs):
        ThermalCOChem.__init__(self, **kwargs)
        SimpleCOChemBase.__init__(self)

    def update(self, dt, T, rho, dust_frac, chem):
        '''Update the gas/ice abundances'''
        for spec in chem:
            self._update_ice_balance(dt, T, rho, dust_frac, spec, chem)

        
class EquilibriumCOChem(ThermalCOChem,SimpleCOChemBase):
    '''Equilibrium chemistry, computed as equilibrium of time dependent model.
    '''
    def __init__(self, fix_ratios=True, fix_grains=True, **kwargs):
        ThermalCOChem.__init__(self, **kwargs)
        SimpleCOChemBase.__init__(self, fix_ratios, fix_grains)

        
###############################################################################
# Specific Chemical models
###############################################################################
class COChemOberg(object):
    '''Chemical ratios from Oberg+ (2011)
    '''
    def molecular_abundance(self, T, rho, dust_frac, atomic_abund, grain_abund=None):
        '''Compute the fractions of species present given total abundances
        
        args: 
             T            : array(N)   temperature (K)
             atomic_abund : atomic abundaces, SimpleCOAtomAbund object
        
        returns:
            nmol : array(3, N) molecular mass-densities
        '''
        C  = atomic_abund.number_abund('C')
        O  = atomic_abund.number_abund('O')
        Si = atomic_abund.number_abund('Si')


        # Set up the number-density abundances
        mol_abund = SimpleCOMolAbund(atomic_abund.size)

        # If grain abundance provided use that value, otherwise set
        # the grain abundance
        if grain_abund is None:
            mol_abund['C-grain']  = 0.2*C
            mol_abund['Si-grain'] = Si
            C -= 0.2*C
            O -= 3*Si
        else:
            mol_abund['C-grain']  = grain_abund.number_abund('C')
            mol_abund['Si-grain'] = grain_abund.number_abund('Si')
            C -= grain_abund.number_abund('C')
            O -= 3*grain_abund.number_abund('Si')


        # From the amount of O available work out how much CO/CO_2 we can
        # have
        fCO2 = 0.15 / (0.65 + 0.15)
        mol_abund['CO2'] = np.minimum(C*fCO2, O - C)
        mol_abund['CO']  = C - mol_abund['CO2']
        
        # Put the remaining O in water (if any)
        O -= mol_abund['CO'] + 2*mol_abund['CO2']
        mol_abund['CH4'] = 0
        mol_abund['H2O'] = np.maximum(O, 0)

        #  Convert to mass abundances
        for spec in mol_abund.species:
            mol_abund[spec] *= mol_abund.mass(spec)
        
        return mol_abund

class COChemMadhu(object):
    '''Chemical ratios from Madhusudhan+ (2014c)
    '''
    def molecular_abundance(self, T, rho, dust_frac, atomic_abund, grain_abund=None):
        '''Compute the fractions of species present given total abundances
        
        args: 
             T            : array(N)   temperature (K)
             atomic_abund : atomic abundaces, SimpleCOAtomAbund object
        
        returns:
            nmol : array(3, N) molecular mass-densities
        '''
        C  = atomic_abund.number_abund('C')
        O  = atomic_abund.number_abund('O')
        Si = atomic_abund.number_abund('Si')

        mol_abund = SimpleCOMolAbund(atomic_abund.size)

        # First do the grain abundances:
        if grain_abund is None:
            mol_abund['Si-grain'] = Si
            mol_abund['C-grain']  = 0
            O -= 3*Si
        else:
            mol_abund['C-grain']  = grain_abund.number_abund('C')
            mol_abund['Si-grain'] = grain_abund.number_abund('Si')
            C -=   grain_abund.number_abund('C')
            O -= 3*grain_abund.number_abund('Si')


        # Compute the CO2 gas phase fraction (approximately)
        mol_abund['CO2'] = m_tCO2 = 0.1*C*mol_abund.mass('CO2')
        m_sCO2 = self._equilibrium_ice_abund(T, rho, dust_frac, 'CO2',
                                             mol_abund)
        xCO2 = m_tCO2
        args = m_tCO2 > 0
        xCO2[args] = 0.5 * (1 - m_sCO2[args]/m_tCO2[args])

        # Make sure we have enough O for this CO abundance. If not, increase 
        # CH4 abundance.
        xCO2 = np.maximum(xCO2, 1-O/(np.minimum(C,1e-300)))

        # Using the CH4 abundance, now update the maximum amount of CO2 
        # available
        nCO2 = np.maximum(np.minimum((O - C*(1 - xCO2)) / (1 + xCO2), 0.1*C), 0)

        # Set up the number-density abundaces for the molecules
        mol_abund['CO']  = (C-nCO2)*(1-xCO2)
        mol_abund['CH4'] = (C-nCO2)*xCO2
        mol_abund['CO2'] = nCO2

        # Put the remaining oxygen in water
        O -= mol_abund['CO'] + 2*mol_abund['CO2']
        mol_abund['H2O'] = np.maximum(O,0)

        #  Convert to mass abundances
        for spec in mol_abund.species:
            mol_abund[spec] *= mol_abund.mass(spec)
        
        return mol_abund


###############################################################################
# Combined Models
###############################################################################
class SimpleCOChemOberg(COChemOberg, StaticCOChem):
    def __init__(self, **kwargs):
        COChemOberg.__init__(self)
        StaticCOChem.__init__(self, **kwargs)

class SimpleCOChemMadhu(COChemMadhu, StaticCOChem):
    def __init__(self, **kwargs):
        COChemMadhu.__init__(self)
        StaticCOChem.__init__(self, **kwargs)

class TimeDepCOChemOberg(COChemOberg, TimeDependentCOChem):
    def __init__(self, **kwargs):
        COChemOberg.__init__(self)
        TimeDependentCOChem.__init__(self, **kwargs)

class EquilibriumCOChemOberg(COChemOberg, EquilibriumCOChem):
    def __init__(self, fix_ratios=False, fix_grains=True, **kwargs):
        COChemOberg.__init__(self)
        EquilibriumCOChem.__init__(self, 
                                   fix_ratios=fix_ratios, 
                                   fix_grains=fix_grains,
                                   **kwargs)

class EquilibriumCOChemMadhu(COChemMadhu, EquilibriumCOChem):
    def __init__(self, fix_ratios=False, fix_grains=True, **kwargs):
        COChemMadhu.__init__(self)
        EquilibriumCOChem.__init__(self,
                                   fix_ratios=fix_ratios, 
                                   fix_grains=fix_grains,
                                   **kwargs)

###############################################################################
# Tests
############################################################################### 
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from eos import LocallyIsothermalEOS
    from star import SimpleStar
    from grid import Grid

    for Chem in [ SimpleCOChemMadhu(), SimpleCOChemOberg(),
                  EquilibriumCOChemMadhu(), EquilibriumCOChemOberg(),]:

        T = np.logspace(0.5, 3, 6)

        Xi = SimpleCOAtomAbund(len(T))
        Xi.set_solar_abundances()
        
        mol = Chem.equilibrium_chem(T, 1e-10, 0.01, Xi)
        
        T *= 3
        Chem.update(0, T, 1e-10, 0.01, mol)

        atom  = mol.gas.atomic_abundance()
        atom += mol.ice.atomic_abundance()
        print Chem.__class__.__name__
        for X in atom:
            print X
            print mol.gas.atomic_abundance()[X] / atom[X]
            assert(np.allclose(Xi[X], atom[X], rtol=1e-12))
        print

    # Compare equilibrium chem with equilibrium of TD chem:

    # DISC model
    GM = 1.
    cs0 = (1/30.) 
    q = -0.25
    Mdot = 1e-8
    alpha = 1e-3

    Mdot *= Msun / (2*np.pi)
    Mdot /= AU**2

    Rin = 0.01
    Rout = 5e2
    Rd = 100.

    t0 = (2*np.pi)

    d2g = 0.01
    
    muH = 1.28

    grid = Grid(0.01, 1000, 1000, spacing='log')
    R = grid.Rc
    eos = LocallyIsothermalEOS(SimpleStar(), cs0, q, alpha)
    eos.set_grid(grid)
    Sigma =  (Mdot / (3 * np.pi * eos.nu))*np.exp(-R/Rd)
    rho = Sigma / (np.sqrt(2*np.pi)*eos.H*AU)
    
    T =  eos.T
    n = Sigma / (2.4*m_H)

    
    EQ_chem = SimpleCOChemOberg()
    TD_chem = TimeDepCOChemOberg(a=1e-5)

    X_solar = SimpleCOAtomAbund(n.shape[0])
    X_solar.set_solar_abundances()


    # Simple chemistry of Madhu:
    plt.subplot(211)
    S_chem  = SimpleCOChemMadhu()
    EQ_chem = EquilibriumCOChemMadhu()

    S_mol  = S_chem.equilibrium_chem(T, rho, d2g, X_solar)
    EQ_mol = EQ_chem.equilibrium_chem(T, rho, d2g, X_solar)
    
    S_atom = S_mol.gas.atomic_abundance()
    EQ_atom = EQ_mol.gas.atomic_abundance()
    plt.semilogx(R, S_atom.number_abund('C')  * 1e4*muH, 'r-')
    plt.semilogx(R, S_atom.number_abund('O')  * 1e4*muH, 'b-')
    plt.semilogx(R, EQ_atom.number_abund('C') * 1e4*muH, 'r:')
    plt.semilogx(R, EQ_atom.number_abund('O') * 1e4*muH, 'b:')
    plt.ylabel(r'$[X/H]\,(\times 10^4)$')

    plt.subplot(212)
    S_chem  = SimpleCOChemOberg()
    EQ_chem = EquilibriumCOChemOberg()

    S_mol  = S_chem.equilibrium_chem(T, rho, d2g, X_solar)
    EQ_mol = EQ_chem.equilibrium_chem(T, rho, d2g, X_solar)
    
    S_atom  = S_mol.gas.atomic_abundance()
    EQ_atom = EQ_mol.gas.atomic_abundance()
    plt.semilogx(R, S_atom.number_abund('C')  * 1e4*muH, 'r-')
    plt.semilogx(R, S_atom.number_abund('O')  * 1e4*muH, 'b-')
    plt.semilogx(R, EQ_atom.number_abund('C') * 1e4*muH, 'r:')
    plt.semilogx(R, EQ_atom.number_abund('O') * 1e4*muH, 'b:')
    plt.ylabel(r'$[X/H]\,(\times 10^4)$')
    plt.xlabel('$R\,[\mathrm{au}]$')


    
    mol_solar = S_chem.molecular_abundance(T, rho, d2g, X_solar)
    
    # Test the time-evolution
    plt.figure()
    times   = np.array([1e0, 1e2, 1e4, 1e6, 1e7])*t0
    H_eff   = np.sqrt(2*np.pi)*eos.H*AU
 

    chem = MolecularIceAbund(gas=mol_solar.copy(), ice=mol_solar.copy())
    if 1:
        for spec in chem:
            chem.ice[spec] = 0
    else:
        for spec in chem:
            chem.gas[spec] = 0
        
    t = 0.
    for ti in times:        
        dt = ti - t
        TD_chem.update(dt, T, rho, d2g, chem)
        
        t = ti
        
        l, = plt.semilogx(R, chem.gas['H2O']/mol_solar['H2O'], '-')
        plt.semilogx(R, chem.ice['H2O']/mol_solar['H2O'], l.get_color()+'--')

    plt.semilogx(R, EQ_mol.gas['H2O']/mol_solar['H2O'], 'k-')
    plt.semilogx(R, EQ_mol.ice['H2O']/mol_solar['H2O'], 'k:')
    plt.xlabel('$R\,[\mathrm{au}}$')
        
    plt.show()
