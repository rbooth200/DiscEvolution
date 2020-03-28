from __future__ import print_function
from operator import xor
import numpy as np
from ..constants import *
from .base_chem import ChemicalAbund, MolecularIceAbund
from .base_chem import SimpleChemBase, StaticChem, ThermalChem
from .base_chem import TimeDependentChem, EquilibriumChem

################################################################################
# Simple CO Chemistry wrappers
################################################################################
class SimpleCOAtomAbund(ChemicalAbund):
    """Class to hold the raw atomic abundaces of C/O/Si for the CO chemistry"""
    def __init__(self, *sizes):
        atom_ids = ['C', 'O', 'Si']
        masses   = [ 12., 16., 28. ]
        
        super(SimpleCOAtomAbund, self).__init__(atom_ids, masses, *sizes)


    def set_solar_abundances(self, muH=1.28):
        """Solar mass fractions of C, O and Si.

        args:
            muH : mean atomic mass, default = 1.28
        """
        self._data[:] = np.outer(np.array([12*2.7e-4, 16*4.9e-4, 28*3.2e-5]),
                                 np.ones(self.size)) / muH

    
class SimpleCOMolAbund(ChemicalAbund):
    """Class that holds the abundances of molecules needed for C/O chemistry"""
    def __init__(self, *sizes):
        mol_ids  = [ 'CO', 'CH4', 'CO2', 'H2O', 'C-grain', 'Si-grain']
        mol_mass = [  28.,  16.,    44.,  18.,   12.,       100.]

        super(SimpleCOMolAbund, self).__init__(mol_ids, mol_mass, *sizes)

        # Atomic make up of the molecules:
        self._n_spec = { 'CO'       : { 'C' : 1, 'O' : 1,         },
                         'CH4'      : { 'C' : 1,                  },
                         'CO2'      : { 'C' : 1, 'O' : 2,         },
                         'H2O'      : {          'O' : 1,         },
                         'C-grain'  : { 'C' : 1,                  },
                         'Si-grain' : {          'O' : 3, 'Si' : 1},
                         }
        
    def atomic_abundance(self):
        """Compute the mass abundances of atomic species in the molecules"""

        atomic_abund = SimpleCOAtomAbund(self.data.shape[1])
        for mol in self.species:
            nspec = self._n_spec[mol]
            for atom in nspec:
                n_atom = (self[mol]/self.mass(mol)) * nspec[atom]
                atomic_abund[atom] += n_atom * atomic_abund.mass(atom)
                
        return atomic_abund
        
        
###############################################################################
# Specific Chemical models
###############################################################################
class COChemOberg(object):
    """Chemical ratios from Oberg+ (2011)

    args:
        fix_grains : Whether to fix the dust grain abundances when recomputing
                     the molecular abundances
    """
    def __init__(self,fix_grains=True):
        self._fix_grains = fix_grains

    def ASCII_header(self):
        """CO Oberg chem header"""
        return (super(COChemOberg, self).ASCII_header() +
                ', fix_grains: {}'.format(self._fix_grains))

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        __, header = super(COChemOberg, self).HDF5_attributes()
        header['fix_grains'] = "{}".format(self._fix_grains)

        return self.__class__.__name__, header

    def molecular_abundance(self, T, rho, dust_frac,
                            atomic_abund=None, mol_abund=None):
        """Compute the fractions of species present given total abundances

        args:
             T            : array(N)   temperature (K)
             atomic_abund : atomic abundaces, SimpleCOAtomAbund object

        returns:
            nmol : array(3, N) molecular mass-densities
        """
        assert(xor(atomic_abund is None, mol_abund is None))
        if atomic_abund is None:
            atomic_abund = mol_abund.atomic_abundance()

        C  = atomic_abund.number_abund('C')
        O  = atomic_abund.number_abund('O')
        Si = atomic_abund.number_abund('Si')


        # Set up the number-density abundances
        initial_abund = False
        if mol_abund is None:
            initial_abund = True
            mol_abund = SimpleCOMolAbund(atomic_abund.size)  
        else:
            #  Convert to number abundances
            for spec in mol_abund.species:
                mol_abund[spec] = mol_abund.number_abund(spec)

        # If grain abundance provided use that value, otherwise set
        # the grain abundance
        if initial_abund or not self._fix_grains:
            mol_abund['C-grain']  = 0.2*C
            mol_abund['Si-grain'] = Si
            C -= 0.2*C
            O -= 3*Si
        else:
            C -=   mol_abund['C-grain']
            O -= 3*mol_abund['Si-grain']

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
    """Chemical ratios from Madhusudhan+ (2014c)

    args:
        fix_grains : Whether to fix the dust grain abundances when recomputing
                     the molecular abundances
    """
    def __init__(self,fix_grains=True):
        self._fix_grains = fix_grains

    def ASCII_header(self):
        """CO Madhu chem header"""
        return (super(COChemMadhu, self).ASCII_header() +
                ', fix_grains: {}'.format(self._fix_grains))

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        _, header = super(COChemMadhu, self).HDF5_attributes()
        header['fix_grains'] = "{}".format(self._fix_grains)

        return self.__class__.__name__, header

    def molecular_abundance(self, T, rho, dust_frac,
                            atomic_abund=None, mol_abund=None):
        """Compute the fractions of species present given total abundances

        args:
             T            : array(N)   temperature (K)
             atomic_abund : atomic abundaces, SimpleCOAtomAbund object

        returns:
            nmol : array(3, N) molecular mass-densities
        """
        assert(xor(atomic_abund is None, mol_abund is None))
        if atomic_abund is None:
            atomic_abund = mol_abund.atomic_abundance()
            
        C  = atomic_abund.number_abund('C')
        O  = atomic_abund.number_abund('O')
        Si = atomic_abund.number_abund('Si')

        # First do the grain abundances:
        initial_abund = False
        if mol_abund is None:
            initial_abund = True
            mol_abund = SimpleCOMolAbund(atomic_abund.size)
        else:
            #  Convert to number abundances
            for spec in mol_abund.species:
                mol_abund[spec] = mol_abund.number_abund(spec)

        if initial_abund or not self._fix_grains:
            mol_abund['Si-grain'] = Si
            mol_abund['C-grain']  = 0
            O -= 3*Si
        else:
            C -=   mol_abund['C-grain']
            O -= 3*mol_abund['Si-grain']
            
        # Compute the CO2 gas phase fraction (approximately)
        mol_abund['CO2'] = m_tCO2 = 0.1*C*mol_abund.mass('CO2')
        m_sCO2 = self._equilibrium_ice_abund(T, rho, dust_frac, 'CO2',
                                             mol_abund)
        xCO2 = m_tCO2
        args = m_tCO2 > 0
        xCO2[args] = 0.5 * (1 - m_sCO2[args]/m_tCO2[args])

        # Make sure we have enough O for this CO abundance. If not, increase 
        # CH4 abundance.
        xCO2 = np.maximum(xCO2, 1-O/np.maximum(C,1e-300))

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
class SimpleCOChemOberg(COChemOberg, StaticChem):
    def __init__(self, **kwargs):
        COChemOberg.__init__(self)
        StaticChem.__init__(self, **kwargs)

class SimpleCOChemMadhu(COChemMadhu, StaticChem):
    def __init__(self, **kwargs):
        COChemMadhu.__init__(self)
        StaticChem.__init__(self, **kwargs)

class TimeDepCOChemOberg(COChemOberg, TimeDependentChem):
    def __init__(self, **kwargs):
        COChemOberg.__init__(self)
        TimeDependentChem.__init__(self, **kwargs)

class EquilibriumCOChemOberg(COChemOberg, EquilibriumChem):
    def __init__(self, fix_ratios=False, fix_grains=True, **kwargs):
        COChemOberg.__init__(self, fix_grains)
        EquilibriumChem.__init__(self, 
                                 fix_ratios=fix_ratios, 
                                 **kwargs)

class EquilibriumCOChemMadhu(COChemMadhu, EquilibriumChem):
    def __init__(self, fix_ratios=False, fix_grains=True, **kwargs):
        COChemMadhu.__init__(self, fix_grains)
        EquilibriumChem.__init__(self,
                                 fix_ratios=fix_ratios, 
                                 **kwargs)

###############################################################################
# Tests
############################################################################### 
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ..eos import LocallyIsothermalEOS
    from ..star import SimpleStar
    from ..grid import Grid

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
        plt.semilogx(R, chem.ice['H2O']/mol_solar['H2O'],'--', c=l.get_color())

    plt.semilogx(R, EQ_mol.gas['H2O']/mol_solar['H2O'], 'k-')
    plt.semilogx(R, EQ_mol.ice['H2O']/mol_solar['H2O'], 'k:')
    plt.xlabel('$R\,[\mathrm{au}}$')
        
    plt.show()
