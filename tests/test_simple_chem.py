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
    EquilibriumCOChemMadhu, EquilibriumCOChemOberg
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
