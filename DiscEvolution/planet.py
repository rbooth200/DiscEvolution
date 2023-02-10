# planet.py
#
# Author: R.Booth
# Date: 10 - Feb - 2023
#
# Routines for including a live planet that interacts with the disc in the
# simulation.
#
################################################################################
from __future__ import print_function
import numpy as np

from DiscEvolution.constants import *

class Planet(object):
    """A simple planet model with a gap profile based on Duffel (2019)
    
    Parameters
    ----------
    Mp : float, units = MJup
        Planet mass in Jupiter masses
    ap : float, unit = au
        Planet semi-major axis
    """
    def __init__(self, Mp, ap):
        self._Mp = Mp
        self._ap = ap

    def update(self, Mp = None, ap = None):
        """Update the planet mass and/or semi-major axis.
        
        Parameters
        ----------
        Mp : float, optional. units = MJup
            New planet mass in Jupiter masses
        ap : float, optional. unit = au
            New planet semi-major axis
        """
        if Mp is not None:
            self._Mp = Mp
        if ap is not None:
            self._ap = ap

    def gap_profile(self, disc):
        """Compute the gap profile. I.e. the fractional change in surface 
            density.

        Parameters
        ----------
        disc : Disc object
            The disc in which the planet is embedded.
        """

        q = self._Mp * Mjup / (disc.star.M * Msun)
        h = disc.interp(self._ap, disc.H) / self._ap
        a = disc.interp(self._ap, disc.alpha * np.ones_like(disc.R))

        # D^3 and \tilde{q} in Duffel (2019)
        x = (disc.R / self._ap)
        D3 = 343 / (h**4.5 * a**0.75)
        qt = q / (1 + D3*(x**(1/6.) - 1)**6)**(1/3.)

        # delta in Duffel (2019)
        qNL = 1.04*h**3
        qw = 34*qNL*(a/h)**0.5
        delta = np.minimum(np.sqrt(qNL/qt), 1) + (qt/qw)**3

        # Term in denominator of gap expression:
        term = (0.15/np.pi) * delta * qt**2 / (a*h**5)

        return 1 / (1 + term)
        
    def update(self, dt, disc):
        pass

    def ASCII_header(self):
        """Write header information about the planet"""
        head  = 'Planet: Mp={}, ap={}'.format(self.Mp, self.ap)
        return head

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        return self.__class__.__name__, {
            'Mp' : self.Mp, 'ap' : self.ap 
        }

    @property
    def Mp(self):
        """Mass in MJup"""
        return self._Mp

    @property
    def ap(self):
        """Semi-major axis in au"""
        return self._ap


class PlanetList(list):
    """Container for a set of planets"""
    def __init__(self, *planets):
        super(PlanetList, self).__init__(*planets)

                
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from .disc import AccretionDisc
    from .grid import Grid
    from .constants import AU, Msun
    from .eos import LocallyIsothermalEOS
    from .star import SimpleStar

    # Create a disc
    alpha = 1e-2

    M = 1e-2 * Msun
    Rd = 30.
    T0 = (2 * np.pi)

    grid = Grid(0.1, 1000, 1000, spacing='natural')
    star = SimpleStar()
    eos = LocallyIsothermalEOS(star, 0.05, -0.5, alpha)
    eos.set_grid(grid)

    nud = np.interp(Rd, grid.Rc, eos.nu)

    Sigma = 100 / grid.Rc

    disc = AccretionDisc(grid, star, eos, Sigma)

    p = Planet(0.5, 30)

    #plt.loglog(disc.R, disc.Sigma, c='k', ls='-')
    plt.plot(disc.R, p.gap_profile(disc), c='k', ls='-')
    plt.xlabel('Radius [au]')
    plt.ylabel(r'Gap profile, $\Sigma/\Sigma_0$')
    plt.xlim(xmin=0, xmax=100)
    plt.tight_layout()
    plt.show()

