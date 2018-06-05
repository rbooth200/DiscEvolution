# test_atomic_data.py
#
# Author: R. Booth
# Date: 5 - June - 2018
#
# Test input / output routines
###############################################################################
from DiscEvolution.chemistry.atomic_data import atomic_composition


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
