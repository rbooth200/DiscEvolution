# test_atomic_data.py
#
# Author: R. Booth
# Date: 5 - June - 2018
#
# Test input / output routines
###############################################################################
from DiscEvolution.chemistry.atomic_data import atomic_composition
from DiscEvolution.chemistry.atomic_data import molecular_mass
from DiscEvolution.chemistry.atomic_data import atomic_abundances
from DiscEvolution.chemistry.base_chem import ChemicalAbund

import numpy as np

def test_composition_converter():
    neutrals = {
        'H'      : { 'H'  : 1 },
        'H2'     : { 'H'  : 2 },
        'He'     : { 'He' : 1 },
        'HE'     : { 'He' : 1 },
        'H2SO4'  : { 'H'  : 2, 'S'  : 1, 'O' : 4},
        'HCl'    : { 'H'  : 1, 'Cl' : 1 },
        'H3COH'  : { 'H'  : 4, 'C'  : 1, 'O' : 1},
        'CaCO3'  : { 'Ca' : 1, 'C'  : 1, 'O' : 3},
        'HCOOH'  : { 'H'  : 2, 'C'  : 1, 'O' : 2},
        'HCOCH3' : { 'H'  : 4, 'C'  : 2, 'O' : 1},
        'C2H+'   : { 'H'  : 1, 'C'  : 2, }, # Ignore charge
    }

    for mol in neutrals:
        assert atomic_composition(mol) == neutrals[mol], mol

    ions = {
        'C2H+' : { 'H'  : 1, 'C' : 2, 'charge' :  1},
        'H-'   : { 'H'  : 1, 'charge' : -1},
        'Ca++' : { 'Ca' : 1, 'charge' : 2},
        'Ca+2' : { 'Ca' : 1, 'charge' : 2},
        }

    for ion in ions:
        assert atomic_composition(ion, True) == ions[ion], ion


def test_atomic_composition():

    mols = { 'H'      : np.array([1., 2.]),
             'H2'     : np.array([1., 0.]),
             'CO'     : np.array([3., 3.]),
             'He'     : np.array([1., 0.]),
             'H3COH'  : np.array([1., 0.]),
             'CaCO3'  : np.array([0., 5.]),
             'HCOOH'  : np.array([2., 3.]),
             'HCOCH3' : np.array([2., 4.]),
             'C2H+'   : np.array([2., 1.])
             }

    atoms = { 'H'  : np.array([21., 25.]),
              'He' : np.array([ 1.,  0.]),
              'C'  : np.array([14., 21.]),
              'O'  : np.array([10., 28.]),
              'Ca' : np.array([ 0.,  5.]),
              'E'  : np.array([-2., -1.])
              }


    masses = np.array([molecular_mass(m) for m in mols])
    mol_abund = ChemicalAbund(mols.keys(), masses, 2)
    for mol, abund in mols.items():
        mol_abund.set_number_abund(mol, abund)
              
    atom_abund = atomic_abundances(mol_abund, True)

    for atom in atom_abund:
        assert np.allclose(atom_abund.number_abund(atom), atoms[atom])

    # Check we've not missed any mass
    assert np.allclose(atom_abund.total_abund, mol_abund.total_abund)
