# utils.py
#
# Author: R. Booth
# Date: 5 - June - 2018
#
# Utilities for chemistry data
###############################################################################

from .base_chem import ChemicalAbund

from .CO_chem  import SimpleCOMolAbund, SimpleCOAtomAbund
from .CNO_chem import SimpleCNOAtomAbund, SimpleCNOMolAbund

_derived_abundance_types = ( 
    SimpleCOMolAbund, SimpleCOAtomAbund,
    SimpleCNOAtomAbund, SimpleCNOMolAbund, 
)


__all__ = [ "create_abundances" ]

def _map_grains(names):
    names = ["C-grain" if x == "Cgrain" else x for x in names]
    names = ["Si-grain" if x == "Sigrain" else x for x in names]
    return names

def _determine_chemistry_type(names):
    """Work out the type of chemistry included from the names"""
    names = sorted(_map_grains(names))
    for chem in _derived_abundance_types:
        if names == sorted(chem().species):
            return chem
    else:
        raise ValueError("Unknown set of chemical species")
    
class PrefixMap(object):
    """Map between names with and without prefix"""
    def __init__(self, prefix):
        self._prefix = prefix

    def add_prefix(self, name):
        return self._prefix + name
    
    def remove_prefix(self, name):
        if self._prefix:
            if name.startswith(self._prefix):
                return name[len(self._prefix):]
            else:
                raise ValueError("Name does not begin with prefix")
        else:
            return name

def create_abundances(names, data, masses=None, grain_prefix=''):
    """Helper function for creating abundance class from data arrays.

    args:
        names : array of strings
            List of names of chemical species
        data : Record array of float
            Chemical abundances of the species
        masses : array of float, optional
            List of masses of each chemical species. When provided we will 
            directly construct a base chemistry wrapper. If not provided,
            we will do our best to work out what is present and use the
            derived class.
        grain_prefix: string, optional
            If provided the names will be checked for a prefix, which will 
            be removed. Used to denote species present as ices on grains.
    """
    mapper = PrefixMap(grain_prefix)
    
    mapped_names = map(mapper.remove_prefix, names)
    mapped_names = _map_grains(mapped_names)

    if masses is None:
        chem_type = _determine_chemistry_type(mapped_names)
        chem = chem_type(data.shape[0])
    else:
        if len(names) != len(masses):
            raise AttributeError("Number of masses provided must match "
                                 "the number of names.")
        chem = ChemicalAbund(names, masses, *(data.shape))

    # Store the data
    for name, mapped in zip(names, mapped_names):
        chem[mapped] = data[name] 

    return chem
            



