# test_simple_chem.py
#
# Author: R. Booth
# Date: 22 - May - 2018
#
# Check behaviour of simple chemistry functions
###############################################################################
import numpy as np

# CO chemistry
from DiscEvolution.chemistry import (
    SimpleCOAtomAbund,
    SimpleCOChemMadhu, SimpleCOChemOberg,
    EquilibriumCOChemMadhu, EquilibriumCOChemOberg,
    TimeDepCOChemOberg
)

# CNO chemistry
from DiscEvolution.chemistry import (
    SimpleCNOAtomAbund,
    SimpleCNOChemMadhu, SimpleCNOChemOberg,
    EquilibriumCNOChemMadhu, EquilibriumCNOChemOberg
)

def _test_chemical_model(ChemModel, Abundances):
    T = np.logspace(0.5, 3, 6)
    
    Xi = Abundances(len(T))
    Xi.set_solar_abundances()

    Chem = ChemModel()
    
    mol = Chem.equilibrium_chem(T, 1e-10, 0.01, Xi)
        
    T *= 3
    Chem.update(0, T, 1e-10, 0.01, mol)

    atom  = mol.gas.atomic_abundance()
    atom += mol.ice.atomic_abundance()
    for X in atom:
        assert(np.allclose(Xi[X], atom[X], rtol=1e-12))


def test_CO_chemistry():
    for Chem in [ SimpleCOChemMadhu, SimpleCOChemOberg,
                  EquilibriumCOChemMadhu, EquilibriumCOChemOberg,]:
        _test_chemical_model(Chem, SimpleCOAtomAbund)


def test_CNO_chemistry():
    for Chem in [ SimpleCNOChemMadhu, SimpleCNOChemOberg,
                  EquilibriumCNOChemMadhu, EquilibriumCNOChemOberg,]:
        _test_chemical_model(Chem, SimpleCNOAtomAbund)


def test_grain_cross_section():
    # The dust's cross-section replaces the fixed grain size: grains of
    # 10 micron are the same as a=1e-3 in the chemistry
    T = np.logspace(0.5, 3, 30)
    Xi = SimpleCOAtomAbund(len(T))
    Xi.set_solar_abundances()
    kappa = 3 / (4*1e-3)

    for Chem in [TimeDepCOChemOberg, EquilibriumCOChemOberg]:
        small, big = Chem(a=1e-5), Chem(a=1e-3)

        mol = small.equilibrium_chem(T, 1e-12, 0.01, Xi,
                                     grain_cross_section=kappa)
        ref = big.equilibrium_chem(T, 1e-12, 0.01, Xi)
        assert np.allclose(mol.ice.data, ref.ice.data, rtol=1e-12, atol=1e-15)

        # Start out of equilibrium, so the rates matter
        for m in (mol, ref):
            m.ice.data[:] = m.ice.data[:, ::-1]
        small.update(3e3, T, 1e-12, 0.01, mol, grain_cross_section=kappa)
        big.update(3e3, T, 1e-12, 0.01, ref)
        assert np.allclose(mol.ice.data, ref.ice.data, rtol=1e-12, atol=1e-15)
        assert np.allclose(mol.gas.data, ref.gas.data, rtol=1e-12, atol=1e-15)

