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
from scipy import optimize
from .constants import AU, sig_H2, m_H, yr, Msun

def LBP_profile(R,R_C,Sigma_C):
    """Defined for profile fitting"""
    x = R/R_C
    return np.log(Sigma_C) - np.log(x)-x

class AccretionDisc(object):

    def __init__(self, grid, star, eos, Sigma=None):
        self._grid = grid
        self._star = star
        self._eos  = eos
        self._FUV = 0.0
        if Sigma is None:
            Sigma = np.zeros_like(self.R)
        self._Sigma = Sigma
        
        """ Extra properties for dealing with half empty cells in timescale approach """
        self.mass_lost = 0.0
        self.tot_mass_lost = 0.0
        self.i_edge = -1

        self._gap_profile = np.ones_like(Sigma)

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

    def set_FUV(self,FUV):
        """Update the external FUV flux irradiating the disc"""
        self._FUV = FUV

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
    def h(self):
        """Aspect ratio"""
        return self.H/self.R

    @property
    def H_edge(self):
        """Scale-height at cell edge"""
        return self._eos.H_edge

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
    def column_density(self):
        n = self.midplane_gas_density / (self._eos._mu * m_H)
        Re = self.R_edge * AU
        ndR = n * (Re[1:] - Re[:-1])
        N = np.cumsum(ndR)
        return N

    @property
    def Ncells(self):
        return self._grid.Ncells

    @property
    def alpha(self):
        return self._eos.alpha

    @property
    def nu(self):
        return self._eos.nu / self._gap_profile
    
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

    @property
    def FUV(self):
        return self._FUV
    
    @property
    def gap_profile(self):
        return self._gap_profile

    """Methods to determine global properties of a viscous accretion disc"""
    def Rout(self, thresh=1e-5):
        """Determine the outer radius via density threshold"""
        notempty = self.Sigma_G > thresh
        notempty_cells = self.R_edge[1:][notempty]
        if np.size(notempty_cells>0):
            R_outer = notempty_cells[-1]
        else:
            R_outer = 0.0

        return R_outer

    def RC(self):
        """Fit an LBP profile to the disc and return the scale radius"""
        not_empty = (self.R < self.Rout())
        popt,pcov = optimize.curve_fit(LBP_profile,self.R[not_empty],np.log(self.Sigma_G[not_empty]),p0=[100,0.01],maxfev=5000)
        return popt[0]

    def Mtot(self):
        """Determine the total mass"""
        Re = self.R_edge * AU
        dA = np.pi * (Re[1:] ** 2 - Re[:-1] ** 2)
        dM_tot = self.Sigma * dA
        M_tot = np.sum(dM_tot)
        return M_tot

    def Mdot(self,viscous_velocity):
        """Determine the viscous accretion rate"""
        M_visc_out = 2*np.pi * self.R[0] * self.Sigma[0] * viscous_velocity * (AU**2)
        Mdot = -M_visc_out*(yr/Msun)
        return Mdot

    """Other methods"""
    def set_surface_density(self, Sigma):
        self._Sigma[:] = Sigma

    def set_gap_profile(self, gap):
        """Set a profile to be used to generate a gap."""
        self._gap_profile[:] = gap

    def update(self, dt):
        """Update the disc properties and age"""

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
