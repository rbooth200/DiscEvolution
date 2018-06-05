from collections import defaultdict
import re

from ..constants import m_H, m_n


ATOMIC_MASSES = {
    'H'  :    m_H,          'He' :  2*(m_H + m_n),
    
    'C'  :  6*(m_H + m_n),  'N'  :  7*(m_H + m_n),  'O'  :  8*(m_H + m_n),
    'Ne' : 10*(m_H + m_n),

    'Na' : 11*m_H + 12*m_n, 'Mg' : 12*(m_H + m_n),  'Al' : 13*m_H + 14*m_n,
    'Si' : 14*(m_H + m_n),  'P'  : 15*m_H + 16*m_n, 'S'  : 16*(m_H + m_n),
    'Cl' : 17*m_H + 18*m_n, 'Ar' : 18*(m_H + m_n),

    'K'  : 19*m_H + 20*m_n, 'Ca' : 20*(m_H + m_n),  'Fe' : 26*m_H + 29*m_n,    
    }


def atomic_composition(mol, charge=False):
    '''Compute the atomic composition of a molecule

    args:
        mol : string,
            Molecule to compute the composition of.
        charge : bool, default = False
            Whether to compute and store the molecular charge in the charge
    '''
    keys = reversed(sorted(ATOMIC_MASSES.keys()))

    atoms = defaultdict(int)
    for atom in keys:
        for chunk in re.findall(atom + '\d*', mol, re.IGNORECASE):
            if len(chunk) > len(atom):
                N = int(chunk[len(atom):])
            else:
                N = 1
            atoms[atom] += N
            mol = mol.replace(chunk, "", 1)

    for c in ['\+', '\-']:
        for chunk in re.findall(c + '\d*', mol, re.IGNORECASE):
            if charge:
                if len(chunk) > 1:
                    N = int(chunk[1:])
                else:
                    N = 1
                    
                if '+' in c:
                    atoms['charge'] += N
                else:
                    atoms['charge'] -= N
            mol = mol.replace(chunk, "", 1)

    assert mol == '', "Unknown component in molecule: {}".format(mol)
            
    return atoms
                               
                                  
def molecular_mass(molecule):
    """Compute the mass of a molecule, in hydrogen masses"""
    atoms = atomic_composition(molecule, charge=True)

    mass = 0
    for atom in atoms:
        if 'charge' in atom:
            mass += m_e * atoms['charge']
        else:
            mass += ATOMIC_MASSES[atom] * atoms[atom]

    return mass / m_H
