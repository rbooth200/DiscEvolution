from __future__ import print_function
from operator import xor
import numpy as np
from ..constants import *
from .base_chem import ChemicalAbund, MolecularIceAbund
from .base_chem import SimpleChemBase, StaticChem, ThermalChem
from .base_chem import TimeDependentChem, EquilibriumChem


################################################################################
# Simple CNO Chemistry wrappers
################################################################################
class SimpleCNOAtomAbund(ChemicalAbund):
    """Class to hold the raw atomic abundaces of C/O/Si for the CNO chemistry"""

    def __init__(self, *sizes):
        atom_ids = ['C', 'N', 'O', 'Si']
        masses = [12., 14., 16., 28.]

        super(SimpleCNOAtomAbund, self).__init__(atom_ids, masses, *sizes)

    def set_solar_abundances(self, muH=1.28):
        """Solar mass fractions of C, O and Si.

        args:
            muH : mean atomic mass, default = 1.28
        """
        m_abund = np.array([12 * 2.7e-4, 14 * 6.8e-5, 16 * 4.9e-4, 28 * 3.2e-5])
        self._data[:] = np.outer(m_abund, np.ones(self.size)) / muH


class SimpleCNOMolAbund(ChemicalAbund):
    """Class that holds the abundances of molecules needed for C/O chemistry"""

    def __init__(self, *sizes):
        mol_ids = ['CO', 'CH4', 'CO2', 'H2O', 'N2', 'NH3',
                   'C-grain', 'Si-grain']
        mol_mass = [28., 16., 44., 18., 28., 17.,
                    12., 100.]

        super(SimpleCNOMolAbund, self).__init__(mol_ids, mol_mass, *sizes)

        # Atomic make up of the molecules:
        self._n_spec = {'CO': {'C': 1, 'O': 1, },
                        'CH4': {'C': 1, },
                        'CO2': {'C': 1, 'O': 2, },
                        'H2O': {'O': 1, },
                        'N2': {'N': 2, },
                        'NH3': {'N': 1, },
                        'C-grain': {'C': 1, },
                        'Si-grain': {'O': 3, 'Si': 1},
                        }

    def atomic_abundance(self):
        """Compute the mass abundances of atomic species in the molecules"""

        atomic_abund = SimpleCNOAtomAbund(self.data.shape[1])
        for mol in self.species:
            nspec = self._n_spec[mol]
            for atom in nspec:
                n_atom = (self[mol] / self.mass(mol)) * nspec[atom]
                atomic_abund[atom] += n_atom * atomic_abund.mass(atom)

        return atomic_abund


###############################################################################
# Specific Chemical models
###############################################################################
class CNOChemOberg(object):
    """Chemical ratios from Oberg+ (2011)

    args:
        fNH3 : Fraction of nitrogen in NH_3
        fix_grains : Whether to fix the dust grain abundances when recomputing
                     the molecular abundances
        fix_NH3    : Whether to fix the nitrogen abundance when recomputing the
                     molecular abundances
    """

    def __init__(self, fNH3=None, fix_grains=True, fix_N=False):
        if fNH3 is None: fNH3 = 0.07
        self._fNH3 = fNH3
        self._fix_grains = fix_grains
        self._fix_N = fix_N

    def ASCII_header(self):
        """CNO Oberg chem header"""
        return (super(CNOChemOberg, self).ASCII_header() +
                ', f_NH3: {}, fix_grains: {}, fix_N: {}'
                ''.format(self._fNH3, self._fix_grains, self._fix_N))

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        __, header = super(CNOChemOberg, self).HDF5_attributes()
        header['f_NH3'] = '{}'.format(self._fNH3)
        header['fix_grains'] = "{}".format(self._fix_grains)
        header['fix_N'] = "{}".format(self._fix_N)

        return self.__class__.__name__, header

    def molecular_abundance(self, T, rho, dust_frac,
                            atomic_abund=None, mol_abund=None):
        """Compute the fractions of species present given total abundances

        args:
             T            : array(N)   temperature (K)
             atomic_abund : atomic abundaces, SimpleCNOAtomAbund object

        returns:
            nmol : array(3, N) molecular mass-densities
        """
        assert (xor(atomic_abund is None, mol_abund is None))
        if atomic_abund is None:
            atomic_abund = mol_abund.atomic_abundance()

        C = atomic_abund.number_abund('C')
        N = atomic_abund.number_abund('N')
        O = atomic_abund.number_abund('O')
        Si = atomic_abund.number_abund('Si')

        # Set up the number-density abundances
        initial_abund = False
        if mol_abund is None:
            initial_abund = True
            mol_abund = SimpleCNOMolAbund(atomic_abund.size)
        else:
            #  Convert to number abundances
            for spec in mol_abund.species:
                mol_abund[spec] = mol_abund.number_abund(spec)

        # If grain abundance provided use that value, otherwise set
        # the grain abundance
        if initial_abund or not self._fix_grains:
            mol_abund['C-grain'] = 0.2 * C
            mol_abund['Si-grain'] = Si
            C -= 0.2 * C
            O -= 3 * Si
        else:
            C -= mol_abund['C-grain']
            O -= 3 * mol_abund['Si-grain']

        # From the amount of O available work out how much CO/CO_2 we can
        # have
        fCO2 = 0.15 / (0.65 + 0.15)
        mol_abund['CO2'] = np.minimum(C * fCO2, O - C)
        mol_abund['CO'] = C - mol_abund['CO2']

        # Put the remaining O in water (if any)
        O -= mol_abund['CO'] + 2 * mol_abund['CO2']
        mol_abund['CH4'] = 0
        mol_abund['H2O'] = np.maximum(O, 0)

        # Nitrogen
        if initial_abund or not self._fix_N:
            mol_abund['NH3'] = self._fNH3 * N
            mol_abund['N2'] = 0.5 * (N - 1 * mol_abund['NH3'])

        #  Convert to mass abundances
        for spec in mol_abund.species:
            mol_abund[spec] *= mol_abund.mass(spec)

        return mol_abund


class CNOChemMadhu(object):
    """Chemical ratios from Madhusudhan+ (2014c)

    args:
        fNH3 : Fraction of nitrogen in NH_3
        fix_grains : Whether to fix the dust grain abundances when recomputing
                     the molecular abundances
        fix_NH3    : Whether to fix the nitrogen abundance when recomputing the
                     molecular abundances
    """

    def __init__(self, fNH3=None, fix_grains=True, fix_N=False):
        if fNH3 is None: fNH3 = 0.07
        self._fNH3 = fNH3
        self._fix_grains = fix_grains
        self._fix_N = fix_N

    def ASCII_header(self):
        """CNO Madhu chem header"""
        return (super(CNOChemMadhu, self).ASCII_header() +
                ', f_NH3: {}, fix_grains: {}, fix_N: {}'
                ''.format(self._fNH3, self._fix_grains, self._fix_N))

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        __, header = super(CNOChemMadhu, self).HDF5_attributes()
        header['f_NH3'] = '{}'.format(self._fNH3)
        header['fix_grains'] = "{}".format(self._fix_grains)
        header['fix_N'] = "{}".format(self._fix_N)

        return self.__class__.__name__, header

    def molecular_abundance(self, T, rho, dust_frac,
                            atomic_abund=None, mol_abund=None):
        """Compute the fractions of species present given total abundances

        args:
             T            : array(N)   temperature (K)
             atomic_abund : atomic abundaces, SimpleCNOAtomAbund object

        returns:
            nmol : array(3, N) molecular mass-densities
        """
        assert (xor(atomic_abund is None, mol_abund is None))
        if atomic_abund is None:
            atomic_abund = mol_abund.atomic_abundance()

        C = atomic_abund.number_abund('C')
        N = atomic_abund.number_abund('N')
        O = atomic_abund.number_abund('O')
        Si = atomic_abund.number_abund('Si')

        # First do the grain abundances:
        initial_abund = False
        if mol_abund is None:
            initial_abund = True
            mol_abund = SimpleCNOMolAbund(atomic_abund.size)
        else:
            #  Convert to number abundances
            for spec in mol_abund.species:
                mol_abund[spec] = mol_abund.number_abund(spec)

        if initial_abund or not self._fix_grains:
            mol_abund['Si-grain'] = Si
            mol_abund['C-grain'] = 0
            O -= 3 * Si
        else:
            C -= mol_abund['C-grain']
            O -= 3 * mol_abund['Si-grain']

        # Compute the CO2 gas phase fraction (approximately)
        mol_abund['CO2'] = m_tCO2 = 0.1 * C * mol_abund.mass('CO2')
        m_sCO2 = self._equilibrium_ice_abund(T, rho, dust_frac, 'CO2',
                                             mol_abund)
        xCO2 = m_tCO2
        args = m_tCO2 > 0
        xCO2[args] = 0.5 * (1 - m_sCO2[args] / m_tCO2[args])

        # Make sure we have enough O for this CO abundance. If not, increase 
        # CH4 abundance.
        xCO2 = np.maximum(xCO2, 1 - O / np.maximum(C, 1e-300))

        # Using the CH4 abundance, now update the maximum amount of CO2 
        # available
        nCO2 = np.maximum(np.minimum((O - C * (1 - xCO2)) / (1 + xCO2), 0.1 * C), 0)        

        # Set up the number-density abundances for the molecules
        mol_abund['CO'] = (C - nCO2) * (1 - xCO2)
        mol_abund['CH4'] = (C - nCO2) * xCO2
        mol_abund['CO2'] = nCO2

        # Put the remaining oxygen in water
        O -= mol_abund['CO'] + 2 * mol_abund['CO2']
        mol_abund['H2O'] = np.maximum(O, 0)

        #print(mol_abund.names[:4])
        #print(mol_abund.data[:4, 41])
        
        # Nitrogen
        if initial_abund or not self._fix_N:
            mol_abund['NH3'] = self._fNH3 * N
            mol_abund['N2'] = 0.5 * (N - 1 * mol_abund['NH3'])

        #  Convert to mass abundances
        for spec in mol_abund.species:
            mol_abund[spec] *= mol_abund.mass(spec)

        return mol_abund


###############################################################################
# Combined Models
###############################################################################
class SimpleCNOChemOberg(CNOChemOberg, StaticChem):
    def __init__(self, fNH3=None, **kwargs):
        CNOChemOberg.__init__(self, fNH3)
        StaticChem.__init__(self, **kwargs)


class SimpleCNOChemMadhu(CNOChemMadhu, StaticChem):
    def __init__(self, fNH3=None, **kwargs):
        CNOChemMadhu.__init__(self, fNH3)
        StaticChem.__init__(self, **kwargs)


class TimeDepCNOChemOberg(CNOChemOberg, TimeDependentChem):
    def __init__(self, fNH3=None, **kwargs):
        CNOChemOberg.__init__(self, fNH3)
        TimeDependentChem.__init__(self, **kwargs)


class EquilibriumCNOChemOberg(CNOChemOberg, EquilibriumChem):
    def __init__(self, fNH3=None, fix_ratios=False, fix_grains=True,
                 fix_N=False, **kwargs):
        CNOChemOberg.__init__(self, fNH3, fix_grains, fix_N)
        EquilibriumChem.__init__(self,
                                 fix_ratios=fix_ratios,
                                 **kwargs)


class EquilibriumCNOChemMadhu(CNOChemMadhu, EquilibriumChem):
    def __init__(self, fNH3=None, fix_ratios=False, fix_grains=True,
                 fix_N=False, **kwargs):
        CNOChemMadhu.__init__(self, fNH3, fix_grains, fix_N)
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
    cs0 = (1 / 30.)
    q = -0.25
    Mdot = 1e-8
    alpha = 1e-3

    Mdot *= Msun / (2 * np.pi)
    Mdot /= AU ** 2

    Rin = 0.01
    Rout = 5e2
    Rd = 100.

    t0 = (2 * np.pi)

    d2g = 0.01

    muH = 1.28

    grid = Grid(0.01, 1000, 1000, spacing='log')
    R = grid.Rc
    eos = LocallyIsothermalEOS(SimpleStar(), cs0, q, alpha)
    eos.set_grid(grid)
    Sigma = (Mdot / (3 * np.pi * eos.nu)) * np.exp(-R / Rd)
    rho = Sigma / (np.sqrt(2 * np.pi) * eos.H * AU)

    T = eos.T
    n = Sigma / (2.4 * m_H)

    EQ_chem = SimpleCNOChemOberg()
    TD_chem = TimeDepCNOChemOberg(a=1e-5)

    X_solar = SimpleCNOAtomAbund(n.shape[0])
    X_solar.set_solar_abundances()

    # Simple chemistry of Madhu:
    plt.subplot(211)
    S_chem = SimpleCNOChemMadhu()
    EQ_chem = EquilibriumCNOChemMadhu()

    S_mol = S_chem.equilibrium_chem(T, rho, d2g, X_solar)
    EQ_mol = EQ_chem.equilibrium_chem(T, rho, d2g, X_solar)

    S_atom = S_mol.gas.atomic_abundance()
    EQ_atom = EQ_mol.gas.atomic_abundance()
    plt.semilogx(R, S_atom.number_abund('C') * 1e4 * muH, 'r-')
    plt.semilogx(R, S_atom.number_abund('N') * 1e4 * muH, 'g-')
    plt.semilogx(R, S_atom.number_abund('O') * 1e4 * muH, 'b-')
    plt.semilogx(R, EQ_atom.number_abund('C') * 1e4 * muH, 'r:')
    plt.semilogx(R, EQ_atom.number_abund('N') * 1e4 * muH, 'g:')
    plt.semilogx(R, EQ_atom.number_abund('O') * 1e4 * muH, 'b:')
    plt.ylabel(r'$[X/H]\,(\times 10^4)$')

    plt.subplot(212)
    S_chem = SimpleCNOChemOberg()
    EQ_chem = EquilibriumCNOChemOberg()

    S_mol = S_chem.equilibrium_chem(T, rho, d2g, X_solar)
    EQ_mol = EQ_chem.equilibrium_chem(T, rho, d2g, X_solar)

    S_atom = S_mol.gas.atomic_abundance()
    EQ_atom = EQ_mol.gas.atomic_abundance()

    plt.semilogx(R, S_atom.number_abund('C') * 1e4 * muH, 'r-')
    plt.semilogx(R, S_atom.number_abund('N') * 1e4 * muH, 'g-')
    plt.semilogx(R, S_atom.number_abund('O') * 1e4 * muH, 'b-')
    plt.semilogx(R, EQ_atom.number_abund('C') * 1e4 * muH, 'r:')
    plt.semilogx(R, EQ_atom.number_abund('N') * 1e4 * muH, 'g:')
    plt.semilogx(R, EQ_atom.number_abund('O') * 1e4 * muH, 'b:')
    plt.ylabel(r'$[X/H]\,(\times 10^4)$')
    plt.xlabel('$R\,[\mathrm{au}]$')

    mol_solar = S_chem.molecular_abundance(T, rho, d2g, X_solar)

    # Test the time-evolution
    plt.figure()
    times = np.array([1e0, 1e2, 1e4, 1e6, 1e7]) * t0
    H_eff = np.sqrt(2 * np.pi) * eos.H * AU

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

        l, = plt.semilogx(R, chem.gas['H2O'] / mol_solar['H2O'], '-')
        plt.semilogx(R, chem.ice['H2O'] / mol_solar['H2O'], '--', c=l.get_color())

    plt.semilogx(R, EQ_mol.gas['H2O'] / mol_solar['H2O'], 'k-')
    plt.semilogx(R, EQ_mol.ice['H2O'] / mol_solar['H2O'], 'k:')
    plt.xlabel('$R\,[\mathrm{au}}$')

    plt.show()
