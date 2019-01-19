# disc.py
#
# Author: R. Booth
# Date: 8 - Nov - 2016
#
# Base class for an accretion disc. Used on its own, the AccretionDisc class
# represents a dust-free accretion disc, in which the total density is just the
# gas density.
#
################################################################################
import numpy as np
from .constants import AU, sig_H2, m_H

class AccretionDisc(object):

    def __init__(self, grid, star, eos, Sigma=None):
        self._grid = grid
        self._star = star
        self._eos  = eos
        self.UV = 0.0
        if Sigma is None:
            Sigma = np.zeros_like(self.R)
        self._Sigma = Sigma
        
        # Extra properties for dealing with half empty cells in timescale approach
        self.mass_lost = 0.0
        self.tot_mass_lost = 0.0
        self.i_edge = -1

        # Global, time dependent properties
        self._threshold = np.amin(self.Sigma)
        self._Rout = np.array([self.R[-1]])
        self._Mtot = np.array([])
        self.Mtot()

    def ASCII_header(self):
        """Write header information about the disc"""
        head  = ''
        head += self._grid.ASCII_header() + '\n'
        head += self._star.ASCII_header() + '\n'
        head += self._eos.ASCII_header()
        return head

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        return self.__class__.__name__, dict([ self._grid.HDF5_attributes(),
                                               self._star.HDF5_attributes(),
                                               self._eos.HDF5_attributes() ])

    def set_UV(self,UV):
        self.UV = UV

    @property
    def star(self):
        return self._star

    @property
    def R(self):
        """Cell centre radii"""
        return self._grid.Rc

    @property
    def R_edge(self):
        """Cell edge radii"""
        return self._grid.Re

    @property
    def grid(self):
        return self._grid

    @property
    def Sigma(self):
        """Surface density"""
        return self._Sigma

    @property
    def Sigma_G(self):
        """Gas surface density"""
        return self.Sigma 

    @property
    def cs(self):
        """Sound speed"""
        return self._eos.cs

    @property
    def T(self):
        """Temperature"""
        return self._eos.T

    @property
    def mu(self):
        return self._eos.mu
    
    @property
    def H(self):
        """Scale-height"""
        return self._eos.H

    @property
    def P(self):
        return self.midplane_gas_density * self.cs**2 

    @property
    def midplane_gas_density(self):
        return self.Sigma_G / (np.sqrt(2*np.pi) * self.H * AU)

    @property
    def midplane_density(self):
        return self.Sigma / (np.sqrt(2*np.pi) * self.H * AU)

    @property
    def Ncells(self):
        return self._grid.Ncells

    @property
    def alpha(self):
        return self._eos.alpha

    @property
    def nu(self):
        return self._eos.nu
    
    @property
    def Re(self):
        """Reynolds number"""
        return (self.alpha*self.Sigma_G*sig_H2) / (2*self._eos.mu*m_H)

    @property
    def Pr(self):
        """Prandtl number"""
        return self._eos.Pr

    @property
    def Omega_k(self):
        return self._star.Omega_k(self.R)

    def Rout(self):
        notempty = self.Sigma > self._threshold
        notempty_cells = self.R[notempty]
        if np.size(notempty_cells>0):
            R_outer = notempty_cells[-1]
        else:
            R_outer = 0.0
        self._Rout = np.append(self._Rout,[R_outer])
        return self._Rout[-1]

    def Rot(self,photoevap):
        # Get the photo-evaporation rates at each cell as if it were the edge USING GAS SIGMA
        not_empty = (self.Sigma_G > 0)
        Mdot = photoevap.mass_loss_rate(self,not_empty)
        # Find the maximum, corresponding to optically thin/thick boundary
        i_max = np.size(Mdot) - np.argmax(Mdot[::-1]) - 1
        self._Rout = np.append(self._Rout,[self.R[i_max]])
        return self._Rout[-1]

    def Mtot(self):
        Re = self.R_edge * AU
        dA = np.pi * (Re[1:] ** 2 - Re[:-1] ** 2)
        dM_tot = self.Sigma * dA
        self._Mtot = np.append(self._Mtot,[np.sum(dM_tot)])
        return self._Mtot[-1]

    def set_surface_density(self, Sigma):
        self._Sigma[:] = Sigma

    def update(self, dt):
        """Update the disc properites and age"""

        new_age = self._star.age + dt/(2*np.pi)
        self._star.evolve(new_age)
        self._eos.update(dt, self.Sigma, self._star)

    def interp(self, R, data):
        """Interpolate disc data to new radii

        args:
            R    : new radii
            data : data defined at grid locations
        """
        return self.grid.interp_centre(R, data)
