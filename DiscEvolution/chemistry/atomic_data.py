from collections import defaultdict
import re
import numpy as np

from ..constants import m_H, m_n, m_e
from .base_chem import ChemicalAbund

__all__ = [ "atomic_mass", "molecular_mass", "atomic_composition",
            "atomic_abundances" ]

ATOMIC_MASSES = {
    'H'  :    m_H,          'He' :  2*(m_H + m_n),
    
    'C'  :  6*(m_H + m_n),  'N'  :  7*(m_H + m_n),  'O'  :  8*(m_H + m_n),
    'Ne' : 10*(m_H + m_n),

    'Na' : 11*m_H + 12*m_n, 'Mg' : 12*(m_H + m_n),  'Al' : 13*m_H + 14*m_n,
    'Si' : 14*(m_H + m_n),  'P'  : 15*m_H + 16*m_n, 'S'  : 16*(m_H + m_n),
    'Cl' : 17*m_H + 18*m_n, 'Ar' : 18*(m_H + m_n),

    'K'  : 19*m_H + 20*m_n, 'Ca' : 20*(m_H + m_n),  'Fe' : 26*m_H + 29*m_n,    
    }

def atomic_mass(atom):
    '''Mass of the atom in hydrogen masses'''
    if atom == 'E':
        return m_e / m_H
    else:
        return ATOMIC_MASSES[atom] / m_H
                                  
def molecular_mass(molecule):
    """Compute the mass of a molecule, in hydrogen masses"""
    atoms = atomic_composition(molecule, charge=True)

    mass = 0
    for atom in atoms:
        if 'charge' in atom:
            mass += m_e * atoms['charge']
        else:
            mass += atomic_mass(atom) * atoms[atom]

    return mass 

def atomic_composition(mol, charge=False):
    '''Compute the atomic composition of a molecule

    args:
        mol : string,
            Molecule to compute the composition of.
        charge : bool, default = False
            Whether to compute and store the molecular charge in the charge
    '''
    keys = reversed(sorted(ATOMIC_MASSES.keys()))

    if mol == "Si-grain" or mol == "grain":
        return { 'Si' : 1, 'O' : 3 }
    if mol == "C-grain":
        return { 'C' : 1 }

    atoms = defaultdict(int)
    for atom in keys:
        if mol == '': break
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

def atomic_abundances(mol_abund, charge=False, ignore_grains=True):
    """Converts the molecular abundances to atomic abundances.

    Uses the list of molecular species to compute the atoms present and breaks
    down each molecule into its atomic components.

    args:
        mol_abund : ChemicalAbund object
           Abundance of the molecules to convert to atomic abundances
        charge : bool, optional
           Whether to track the net charge, i.e. the abundance of electrons
        ignore_grains : bool, optional
           Whether to ignore the contribution from dust species. 
    
    returns:
       atom_abund : ChemicalAbund object
           Abundance of the constituent atomic species
    """
    # Break down each molecule into atoms
    atoms = {}
    for mol in mol_abund.species:
        if mol.endswith('grain') and ignore_grains: continue
        composition = atomic_composition(mol, charge)

        for atom, count in composition.items():
            nmol = mol_abund.number_abund(mol)
            if atom not in atoms: atoms[atom] = np.zeros_like(nmol)
            atoms[atom] += nmol * count
            
    # Rename charge to E:
    try:
        atoms['E'] = -atoms['charge']
        del atoms['charge']
    except KeyError:
        pass

    # Create the ChemicalAbund object
    species = np.array(list(atoms.keys()))
    masses  = np.array([atomic_mass(s) for s in species])

    atom_abund = ChemicalAbund(species, masses, len(atoms[species[0]]))

    for atom, abund in atoms.items():
        atom_abund.set_number_abund(atom, abund)
    
    return atom_abund
