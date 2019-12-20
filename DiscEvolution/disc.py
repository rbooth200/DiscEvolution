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
from .history import history

def LBP_profile(R,R_C,Sigma_C):
    """Defined for profile fitting"""
    x = R/R_C
    return np.log10(Sigma_C) - np.log10(x)-x

class AccretionDisc(object):

    def __init__(self, grid, star, eos, Sigma=None):
        self._grid = grid
        self._star = star
        self._eos  = eos
        self._FUV = 0.0
        if Sigma is None:
            Sigma = np.zeros_like(self.R)
        self._Sigma = Sigma
        
        # Extra properties for dealing with half empty cells in timescale approach
        self.mass_lost = 0.0
        self.tot_mass_lost = 0.0
        self.i_edge = -1

        # Global, time dependent properties stored as history
        """self._threshold = 1e-5          # Threshold for defining edge by density
        self._Rout = np.array([])       # Outer radius of the disc (density), updated internally
        self._Rc_t = np.array([])       # Radius of current best fit scale radius, updated internally
        self._Rot = np.array([])        # Radius where Mdot maximum ie where becomes optically thick, updated internally
        self._Mtot = np.array([])       # Total mass, updated internally
        self._Mdot_acc = np.array([])   # Accretion rate, updated with velocity passed"""
        self.history = history()

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
    def column_density_est(self):
        N2 = 0.8 / np.sqrt(2*np.pi) * 1 / (self._eos._mu * self._eos._cs0 * self.R[0]**(1/4)) * self.Sigma_G[0] / m_H
        return N2

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

    @property
    def FUV(self):
        return self._FUV

    def Rout(self, fit_LBP=False, Track=False):
        """Determine the outer radius (density threshold) and add to history"""
        notempty = self.Sigma_G > self.history._threshold
        notempty_cells = self.R[notempty]
        if np.size(notempty_cells>0):
            R_outer = notempty_cells[-1]
        else:
            R_outer = 0.0

        if Track:
            self.history._Rout = np.append(self.history._Rout,[R_outer])
            """May also fit an LBP profile to the disc when testing viscous evolution"""
            if fit_LBP:
                not_empty = (self.R < R_outer)
                popt,pcov = optimize.curve_fit(LBP_profile,self.R[not_empty],np.log10(self.Sigma_G[not_empty]),p0=[100,0.01],maxfev=5000)
                self.history._Rc_t = np.append(self.history._Rc_t, [popt[0]])
        else:
            return R_outer

    @property
    def Rot(self):
        return self.history._Rot[-1]

    def Mtot(self, Track=False):
        """Determine the total mass (and add to history)"""
        Re = self.R_edge * AU
        dA = np.pi * (Re[1:] ** 2 - Re[:-1] ** 2)
        dM_tot = self.Sigma * dA
        M_tot = np.sum(dM_tot)
        if Track:
            self.history._Mtot = np.append(self.history._Mtot,[M_tot])
        else:
            return M_tot

    def Mdot(self,viscous_velocity, Track=False):
        """Determine the viscous accretion rate (and add to history)"""
        M_visc_out = 2*np.pi * self.R[0] * self.Sigma[0] * viscous_velocity * (AU**2)
        Mdot = -M_visc_out*(yr/Msun)
        if Track:
            self.history._Mdot_acc = np.append(self.history._Mdot_acc,[Mdot])
        else:
            return Mdot

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
