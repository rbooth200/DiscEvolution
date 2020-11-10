 # dust.py
#
# Author: R. Booth
# Date : 10 - Nov - 2016
#
# Classes extending accretion disc objects to include dust models.
################################################################################
from __future__ import print_function
import numpy as np
from .constants import *
from .disc import AccretionDisc
from .reconstruction import DonorCell, VanLeer

class DustyDisc(AccretionDisc):
    """Dusty accretion disc. Base class for an accretion disc that also
    includes one or more dust species.

    args:
        grid     : Disc gridding object
        star     : Stellar object
        eos      : Equation of state
        Sigma    : Initial surface density distribution
        rho_s    : solid density, default=1
        Sc       : Schmidt number, default=1
        feedback : When False, the dust mass is considered to be a negligible
                   fraction of the total mass.
    """
    def __init__(self, grid, star, eos, Sigma=None, rho_s=1., Sc=1.,
                 feedback=True):

        super(DustyDisc, self).__init__(grid, star, eos, Sigma)

        self._rho_s = rho_s
        self._Kdrag = (np.pi * rho_s) / 2.

        self._Sc = Sc
        self._feedback = feedback

    def Stokes(self, Sigma=None, size=None):
        """Stokes number of the particle"""
        if size is None:
            size = self.grain_size
        if Sigma is None:
            Sigma = self.Sigma_G
            
        return self._Kdrag * size / (Sigma + 1e-300)

    def mass(self):
        """Grain mass"""
        return (4*np.pi/3) * self._rho_s * self.grain_size**3

    @property
    def integ_dust_frac(self):
        """Total dust to gas ratio, or zero if not including dust feedback"""
        if self._feedback:
            return self.dust_frac.sum(0)
        else:
            return 0

    @property
    def dust_frac(self):
        """Dust mass fraction"""
        return self._eps

    @property
    def grain_size(self):
        """Grain size in cm"""
        return self._a

    @property
    def feedback(self):
        """True if drag from the dust on the gas is to be included"""
        return self._feedback

    @property
    def area(self):
        """Mean area of grains"""
        return self._area

    @property
    def Sc(self):
        """Schmidt number, Sc = nu/D"""
        return self._Sc

    # Overload Accretion disc densities to make it dusty
    @property
    def Sigma_G(self):
        return self.Sigma * (1-self.integ_dust_frac)

    @property
    def Sigma_D(self):
        return self.Sigma * self.dust_frac
    
    @property
    def midplane_dust_density(self):
        return self.Sigma_D / (np.sqrt(2*np.pi) * self.Hp * AU)
    
    @property
    def midplane_density(self):
        return self.midplane_gas_density + self.midplane_dust_density.sum(0)
    
    @property
    def Hp(self):
        """Dust scale height"""

        St = self.Stokes()
        a  = self.alpha/self.Sc
        eta = 1 - 1. / (2 + 1./St)

        return self.H * np.sqrt(eta * a / (a + St))

    """Methods to determine global properties of a dust disc"""
    def Rdust(self, thresholds=[0.68]):
        """Determine the dust radius by mass"""
        Re = self.R_edge * AU
        dA = np.pi * (Re[1:] ** 2 - Re[:-1] ** 2)
        dM_dust = self.Sigma_D.sum(0) * dA
        M_cum = np.cumsum(dM_dust)
        radii = {}
        for thresh in thresholds:
            outside = M_cum > (M_cum[-1] * thresh)
            R_outer = self.R[outside][0]
            radii[thresh] = R_outer
        return radii

    def Mdust(self):
        """Determine the dust mass"""
        Re = self.R_edge * AU
        dA = np.pi * (Re[1:] ** 2 - Re[:-1] ** 2)
        dM_dust = self.Sigma_D.sum(0) * dA
        M_dust = np.sum(dM_dust)
        return M_dust
    
    def update(self, dt):
        """Update the disc properites and age"""

        new_age = self._star.age + dt/(2*np.pi)
        self._star.evolve(new_age)
        self._eos.update(dt, self.Sigma,
                         amax=self.grain_size[-1], star=self._star)
    
    def update_ices(self, chem):
        """Update ice fractions"""
        pass

    def ASCII_header(self):
        """Dusty disc header"""
        head = super(DustyDisc, self).ASCII_header() + '\n'
        head += '# {} feedback: {}, rho_s: {}g cm^-3'
        return head.format(self.__class__.__name__,
                           self.feedback, self._rho_s)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        _, head = super(DustyDisc, self).HDF5_attributes()

        head["feedback"] = "{}".format(bool(self.feedback))
        head["rho_s"] = "{} g cm^-3".format(self._rho_s)

        return self.__class__.__name__, head


################################################################################
# Growth model
################################################################################
class FixedSizeDust(DustyDisc):
    """Simple model for dust of a fixed size

    args:
        grid     : Disc gridding object
        star     : Stellar object
        eos      : Equation of state
        eps      : Initial dust fraction (must broadcast to [size.shape, Ncell])
        size     : size, cm (float or 1-d array of sizes)
        Sigma    : Initial surface density distribution
        rhos     : solid density, default=1 g / cm^3
        feedback : default=True
    """
    def __init__(self, grid, star, eos, eps, size, Sigma=None, rhos=1, feedback=True):

        super(FixedSizeDust, self).__init__(grid, star, eos, Sigma, rhos, feedback)

        shape = np.atleast_1d(size).shape + (self.Ncells,)
        self._eps  = np.empty(shape, dtype='f8')
        self._a    = np.empty(shape, dtype='f8')
        self._eps.T[:] = np.atleast_1d(eps).T
        self._a.T[:]   = size

        self._area = np.pi * self._a**2

class DustGrowthTwoPop(DustyDisc):
    """Two-population dust growth model of Birnstiel (2011).

    This model computes the flux of two dust populations. The smallest size
    particles are assumed to always be well coupled to the gas. For the larger
    particles we solve their growth up to the most stringent limit set by
    radial drift and fragmentation.

    Any dust tracers are assumed to have the same mass distribution as the dust
    particles themselves.

    args:
        grid      : Disc gridding object
        star      : Stellar object
        eos       : Equation of state
        eps       : Initital dust fraction
        Sigma     : Initial surface density distribution
        rho_s     : solid density, default=1
        Sc        : Schmidt number, default=1
        rhos      : Grain solid density, default=1.
        uf_0      : Fragmentation velocity (default = 100 (cm/s))
        uf_ice    : Fragmentation velocity of icy grains (default = 1000 (cm/s))
        f_ice     : Ice fraction, default=1
        thresh    : Threshold ice fraction for switchng between icy/non icy
                    fragmentation velocity, default=0.1
        f_grow    : Growth time-scale factor, default=1.
        a0        : Initial particle size (default = 1e-5, 0.1 micron)
        amin      : Minimum particle size (default = 0.0)
        f_drift   : Drift fitting factor. Reduce by a factor ~10 to model the
                    role of bouncing (default=0.55).
        f_frag    : Fragmentation boundary fitting factor (default=0.37).
        feedback  : Whether to include feedback from dust on gas
        start_small:Whether to start at monomer size (True, default) or equilibrium (False)
        distribution_slope:
                    The slope d ln n(a) / d ln a of the number distribution with size (3.5 for MRN)
    """
    def __init__(self, grid, star, eos, eps, Sigma=None,
                 rho_s=1., Sc=1., uf_0=100., uf_ice=1e3, f_ice=1, thresh=0.1,
                 f_grow=1.0, a0=1e-5, amin=1e-5, f_drift=0.55, f_frag=0.37, feedback=True, start_small=True, distribution_slope=3.5):
        super(DustGrowthTwoPop, self).__init__(grid, star, eos, Sigma, rho_s, Sc, feedback)
        
        self._uf_0   = uf_0 / (AU * Omega0)
        self._uf_ice = uf_ice / (AU * Omega0)

        # Fitting factors
        self._fgrow  = f_grow 
        self._ffrag  = f_frag * (2/(3*np.pi)) 
        self._fdrift = f_drift * (2/np.pi) / f_grow 
        self._fmass  = np.array([0.97, 0.75])

        # Initialize the dust distribution
        Ncells = self.Ncells
        self._fm    = np.zeros(Ncells, dtype='f8')
        self._a0    = 0 # Force well-coupled limit
        self._eps   = np.empty([2, Ncells], dtype='f8')
        self._a     = np.empty([2, Ncells], dtype='f8')
        self._eps[0] = eps
        self._eps[1] = 0
        self._a[0]   = amin
        self._a[1]   = a0
        
        self._amin = amin 
        
        self._ice_threshold = thresh
        self._uf = self._frag_velocity(f_ice)
        self._area = np.pi * a0*a0
        self._start_small = start_small         # Whether to start at monomer size (True, default) or equilibrium (False)
        self._p = distribution_slope            # The slope d ln n(a) / d ln a of the number distribution with size (3.5 for MRN)

        self._head = (', uf_0: {}cm s^-1, uf_ice: {}cm s^-1, thresh: {}'
                      ', f_grow: {}, a0: {}cm'.format(uf_0, uf_ice, thresh,
                                                      f_grow, a0))

        self.update(0)

    def ASCII_header(self):
        """Dust growth header"""
        return super(DustGrowthTwoPop, self).ASCII_header() + self._head

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        name, head = super(DustGrowthTwoPop, self).HDF5_attributes()

        tmp = dict([x.strip().split(":") for x in self._head.split(",") if x])

        head.update(tmp)

        return self.__class__.__name__, head

    def _frag_velocity(self, f_ice):
        """Fragmentation velocity"""
        # Interplate between the icy/ice free region
        f_ice = np.minimum(f_ice/self._ice_threshold, 1)
        f_ice = f_ice*f_ice*f_ice*(10-f_ice*(15-6*f_ice))
        #f_ice = f_ice*f_ice*(3-2*f_ice)
        return self._uf_0 + (self._uf_ice - self._uf_0) * f_ice
        
    def _frag_limit(self):
        """Maximum particle size before fragmentation kicks in"""
        alpha = self.alpha/self.Sc
        af = (self.Sigma_G/(self._rho_s*alpha)) * (self._uf/self.cs)**2
        return self._ffrag * af

    def a_BT(self, eps_tot=None):
        """Size at transition between Brownian motion and turbulence dominated
        collision velocities"""
        if eps_tot is None:
            eps_tot = self.dust_frac.sum(0)

        alpha = self.alpha/self.Sc

        a0  = 8 * self.Sigma / (np.pi * self._rho_s) * self.Re**-0.25
        a0 *= np.sqrt(self.mu*m_H/(self._rho_s*alpha)) / (2*np.pi)
        return a0**0.4
        
    def _gammaP(self):
        """Dimensionless pressure gradient"""
        P = self.P
        R = self.R
        gamma = np.empty_like(P)
        gamma[1:-1] = abs((P[2:] - P[:-2])/(R[2:] - R[:-2]))
        gamma[ 0]   = abs((P[ 1] - P[  0])/(R[ 1] - R[ 0]))
        gamma[-1]   = abs((P[-1] - P[ -2])/(R[-1] - R[-2]))
        gamma *= R/(P+1e-300)

        return gamma
        
    def _drift_limit(self, eps_tot):
        """Maximum size due to drift limit or drift driven fragmentation"""
        gamma = self._gammaP()
        
        Sigma_D = self.Sigma * eps_tot
        Sigma_G = self.Sigma_G
            
        # Radial drift time-scale limit
        h = self.H / self.R
        ad = self._fdrift * (Sigma_D/self._rho_s) / (gamma * h**2+1e-300)

        # Radial drift-driven fragmentation:
        cs = self.cs
        St_d = 2 * (self._uf/cs) / (gamma*h + 1e-300)
        af = St_d * (2/np.pi) * (Sigma_G / self._rho_s)

        return ad, af

    def _t_grow(self, eps):
        return self._fgrow / (self.Omega_k * eps)

    def do_grain_growth(self, dt):
        """Apply the grain growth"""

        # Size and total gas fraction
        a = self._a[1]        
        eps_tot = self.dust_frac.sum(0)
                
        afrag_t = self._frag_limit()
        adrift, afrag_d =  self._drift_limit(eps_tot)
        t_grow = self._t_grow(eps_tot)
        
        afrag = np.minimum(afrag_t, afrag_d)
        a0    = np.minimum(afrag, adrift)       # a0 is the lower of the maximum sizes

        # Update the particle distribution
        #   Maximum size due to growth:
        if self._start_small:
            amax = np.minimum(a0, a*np.exp(dt/t_grow))  # If dust grains start small (default) first have to grow)
        else:
            amax = a0                                   # Ignore possibility of being in growth phase
        #   Reduce size due to erosion / fragmentation if grains have grown
        #   above this due to ice condensation
        # amin = a + np.minimum(0, afrag-a)*np.expm1(-dt/t_grow)
        # ignore empty cells:
        ids = eps_tot > 0
        self._a[1, ids] = np.maximum(amax[ids], self._amin)
        
        # Update the mass-fractions in each population
        fm   = self._fmass[1*(afrag < adrift)]
        self._fm[ids] = fm[ids]
        
        self._eps[0][ids] = ((1-fm)*eps_tot)[ids]
        self._eps[1][ids] = (   fm *eps_tot)[ids]

        # Set the average area:
        #self._area = np.pi * self.a_BT(eps_tot)**2

    def update_ices(self, grains):
        """Update the grain size due to a change in bulk ice abundance"""
        eps_new = grains.total_abund
            
        #f = eps_new / (self.integ_dust_frac + 1e-300)
        #self._a[1] = np.maximum(self._a0, self._a[1]*f**(1/3.))

        self._eps[0] = eps_new*(1-self._fm)
        self._eps[1] = eps_new*   self._fm

        # Update the ice fraction
        f_ice = 0
        for spec in grains:
            if 'grain' not in spec:
                f_ice += grains[spec]
        f_ice /= (eps_new + 1e-300)

        self._uf = self._frag_velocity(f_ice)

    def initialize_dust_density(self, dust_frac):
        """Set the initial dust density"""
        self._eps[0] = dust_frac


    def update(self, dt):
        """Do the standard disc update, and apply grain growth"""
        super(DustGrowthTwoPop, self).update(dt)
        self.do_grain_growth(dt)

################################################################################
# Radial drift
################################################################################
class SingleFluidDrift(object):
    """Radial Drift in the single fluid approximation with the short friction
    time limit.

    This class computes the single-fluid update of the dust fraction,
        d(eps_i)/dt = - (1/Sigma) grad [Sigma eps_i (Delta v_i - eps Delta v)],
    which is a vertically integrated version of equation (98) of Laibe & Price
    (2014). Note that the time-derivative on the LHS is the Lagrangian
    derivative in centre of mass frame. If an Eulerian (fixed) grid is used the
    advection step must be handled seperately.

    The dust-gas relative velocity, Delta v_i, is calculated following
    Tanaka+ (2005).

    Note:
        This currently neglects the viscous velocity, which can be important
        for small grains.

    args:
        diffusion : Diffusion algorithm, default=None
        settling  : Include settling in the velocity calculation, default=False
        van_leer  : Use 2nd-order Van-Leer reconstruction, default=False
    """
    def __init__(self, diffusion=None, settling=False, van_leer=False):
        self._diffuse = diffusion
        self._settling = settling
        self._van_leer = van_leer

    def ASCII_header(self):
        """Radial drift header"""
        head = ''
        if self._diffuse:
            head += self._diffuse.ASCII_header() + '\n'
        head += ('# {} diffusion: {} settling: {} van-leer: {}'
                 ''.format(self.__class__.__name__,
                           self._diffuse is not None,
                           self._settling,
                           self._van_leer))
        return head

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        head = { "diffusion" : "{}".format(self._diffuse is not None),
                 "settling"  : "{}".format(self._settling),
                 "van-leer"  : "{}".format(self._van_leer),
                 }
        if self._diffuse is not None:
            head.update(dict([self._diffuse.HDF5_attributes()]))
        return self.__class__.__name__ , head

    def max_timestep(self, disc, v_visc=None):
        step = np.inf
        Cou = 0.5       # Courant number
        
        dV = self._compute_deltaV(disc, v_visc)
        dVout = np.empty((dV.shape[0],dV.shape[-1]+2))
        dVout[:,1:-1] = dV
        dVout[:, 0] = dVout[:, 1]
        dVout[:,-1] = dVout[:,-2]
        dVtot = np.abs(dVout[:,1:]) + np.abs(dVout[:,:-1])  # Potentially a cell can lose dust in both directions, both should be included to ensure stability
        return Cou * (disc.grid.dRe / dVtot).min()
    
    def _donor_flux(self, Ree, deltaV_i, Sigma, eps_i):
        """Compute flux using Donor-Cell method"""
        # Add boundary cells        
        shape_v   = (eps_i.shape[-1]+1,)
        shape_rho = eps_i.shape[:-1] + (eps_i.shape[-1]+2,)
        
        dV_i = np.empty(shape_v, dtype='f8')
        dV_i[1:-1] = deltaV_i - self._epsDeltaV
        dV_i[ 0] = dV_i[ 1] 
        dV_i[-1] = dV_i[-2] 
            
        Sig_eps = np.zeros(shape_rho, dtype='f8')
        Sig_eps[...,1:-1] = Sigma*eps_i
            
        # Upwind the density
        dc = DonorCell(Ree[1:-1], 1)
        Sig_eps = dc(dV_i, Sig_eps)

        # Compute the fluxes
        flux = Sig_eps * dV_i
        return flux
        
    def _van_leer_flux(self, Ree, deltaV_i, Sigma, eps_i, dt):
        """Compute flux using Van-Leer reconstruction"""
        # Add boundary cells
        shape_v   = (eps_i.shape[-1]+3,)
        shape_rho = eps_i.shape[:-1] + (eps_i.shape[-1]+4,)
        
        dV_i = np.empty(shape_v, dtype='f8')
        dV_i[2:-2] = deltaV_i - self._epsDeltaV
        dV_i[  :2] = dV_i[2]
        dV_i[-2: ] = dV_i[-3] 
            
        Sig_eps = np.zeros(shape_rho, dtype='f8')
        Sig_eps[...,2:-2] = Sigma*eps_i

        Sig_eps[..., 1] = np.where(dV_i[ 1] < 0, Sig_eps[..., 2], 0)
        Sig_eps[...,-2] = np.where(dV_i[-2] > 0, Sig_eps[...,-3], 0)
        
        # Upwind the density
        vl = VanLeer(Ree, 1)
        Sig_eps = vl(dV_i, Sig_eps, dt)

        # Compute the fluxes
        flux = Sig_eps * dV_i[1:-1]
        return flux
        
    def _fluxes(self, disc, eps_i, deltaV_i, St_i, dt=0):
        """Update a quantity that moves with the gas/dust"""

        Sigma = disc.Sigma
        grid = disc.grid

        if self._van_leer:
            flux = self._van_leer_flux(grid.Ree, deltaV_i, Sigma, eps_i, dt)
        else:
            flux = self._donor_flux(grid.Ree, deltaV_i, Sigma, eps_i)
            
        # Do the update
        deps = - np.diff(flux*grid.Re) / ((Sigma+1e-300) * 0.5*grid.dRe2)

        if self._diffuse:
            St2 = St_i**2
            try:
                Sc = disc.Sc
            except ValueError:
                Sc = self._diffuse.Sc
            Sc = Sc * (0.5625/(1 + 4*St2) + 0.4375 + 0.25*St2)
            depsdiff = self._diffuse(disc, eps_i, Sc)
            deps += depsdiff

        return deps

    def _compute_deltaV(self, disc, v_visc=None):
        """Compute the total dust-gas background velocity"""

        Sigma  = disc.Sigma
        SigmaD = disc.Sigma_D
        Om_k   = disc.Omega_k
        a      = disc.grain_size
        R      = disc.R

        # Average to cell edges:        
        Om_kav  = 0.5*(Om_k      [1:] + Om_k      [:-1])
        Sig_av  = 0.5*(Sigma     [1:] + Sigma     [:-1]) + 1e-300
        SigD_av = 0.5*(SigmaD[...,1:] + SigmaD[...,:-1])
        a_av    = 0.5*(a    [..., 1:] + a     [...,:-1])
        R_av    = 0.5*(R         [1:] + R         [:-1])

        # Compute the density factors needed for the effect of feedback on
        # the radial drift velocity.
        eps_av = 0.
        eps_g = 1.
        SigG_av = Sig_av
        if disc.feedback:
            # By default, use the surface density
            eps_av = SigD_av / Sig_av
            eps_g = np.maximum(1 - eps_av.sum(0), 1e-300)
                
            SigG_av = Sig_av * eps_g
            # Use the midplane density instead
            if self._settling:
                rhoD = disc.midplane_dust_density
                rhoG = disc.midplane_gas_density
                rhoD_av = 0.5 * (rhoD[...,1:] + rhoD[...,:-1])
                rhoG_av = 0.5 * (rhoG    [1:] + rhoG    [:-1])
                rho_av = rhoD_av.sum(0) + rhoG_av + 1e-300

                eps_av = rhoD_av / rho_av
                eps_g  = np.maximum(rhoG_av / rho_av, 1e-300)
                
        # Compute the Stokes number        
        St_av = disc.Stokes(SigG_av, a_av+1e-300)

        # Compute the lambda factors
        # DON'T Use lambda * eps_g instead of lambda to avoid 0/0 in D_1 when eps_g -> 0.
        la0, la1 = 0, 0 
        St_1 = 1 / (1 + St_av**2)
        if disc.feedback:
            la0 = (eps_av/eps_g / (1     + St_av** 2)).sum(0)
            la1 = (eps_av/eps_g / (St_av + St_av**-1)).sum(0)

        # Compute the gas velocities due to pressure (with feedback):
        rho = disc.midplane_gas_density
        dPdr = np.diff(disc.P) / disc.grid.dRc
        eta = - dPdr / (0.5*(rho[1:] + rho[:-1] + 1e-300)*Om_kav)

        D_1 = 1 / ((1 + la0)**2 + la1**2)
        u_gas =            la1  * eta * D_1
        v_gas = - 0.5*(1 + la0) * eta * D_1

        # Compute the gas velocities due to viscosity per Dipierro+18 (with feedback):
        if v_visc is not None:
            u_gas += (1 + la0) * v_visc * D_1 / eps_g
            v_gas += 0.5 * la1 * v_visc * D_1 / eps_g

        # Dust-gas relative velocities:
        DeltaV = (2*v_gas / (St_av + St_av**-1) 
                  - u_gas / (1     + St_av**-2))

        # epsDeltaV = v_COM - v_gas (= 0 if dust mass is neglected)
        if disc.feedback:
            self._epsDeltaV = (eps_av * DeltaV).sum(0)
        else:
            self._epsDeltaV = 0

        return DeltaV

    def __call__(self, dt, disc, gas_tracers=None, dust_tracers=None, v_visc=None):
        """Apply the update for radial drift over time-step dt"""
        eps = disc.dust_frac
        a   = disc.grain_size
        eps_inv = 1. / (eps.sum(0) + np.finfo(eps.dtype).tiny)

        # Compute the dust-gas relative velocity
        DeltaV = self._compute_deltaV(disc, v_visc)
        
        # Compute and apply the fluxes
        if gas_tracers is not None:
            gas_tracers[:] += dt * self._fluxes(disc, gas_tracers, 0, 0, dt)

        # Update the dust fraction, size and tracers
        d_tr = 0
        for eps_k, dV_k, a_k, St_k in zip(disc.dust_frac, DeltaV,
                                          disc.grain_size, disc.Stokes()):

            if dust_tracers is not None:
                t_k = dust_tracers * eps_k * eps_inv
                d_tr  += dt*self._fluxes(disc, t_k, dV_k, St_k, dt)
                
            # multiply a_k by the dust-to-gas ratio, so that constant functions
            # are advected perfectly
            #eps_a = a_k * eps_k
            #eps_a +=  dt*self._fluxes(disc, eps_a, dV_k, St_k, dt)
            
            eps_k[:] += dt*self._fluxes(disc, eps_k, dV_k, St_k, dt)

            #a_k[:] = eps_a / (eps_k + 1e-300)

        if dust_tracers is not None:
            dust_tracers[:] += d_tr
            
    def radial_drift_velocity(self, disc):
        """Compute the radial drift velocity for the disc"""
        DeltaV = self._compute_deltaV(disc)
        return DeltaV - self._epsDeltaV
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from .grid import Grid
    from .eos import LocallyIsothermalEOS
    from .star import SimpleStar
    
    Mdot = 1e-8
    alpha = 1e-3

    Mdot *= Msun / (2*np.pi)
    Mdot /= AU**2
    Rd = 100.

    grid = Grid(0.1, 1000, 1000, spacing='log')
    star = SimpleStar()
    eos = LocallyIsothermalEOS(star, 1/30., -0.25, alpha)
    eos.set_grid(grid)
    Sigma =  (Mdot / (3 * np.pi * eos.nu))*np.exp(-grid.Rc/Rd)
    
    settling = True
    
    T0 = (2*np.pi)

    d2g = 0.01
    dust     = DustGrowthTwoPop(grid, star, eos, d2g, Sigma=Sigma)
    dust_ice = DustGrowthTwoPop(grid, star, eos, d2g, Sigma=Sigma)
    ices = {'H2O' : 0.9*d2g*(eos.T < 150), 'grains' : 0.1*d2g}

    class ices(dict):
        def __init__(self, init=None):
            if init is None: init = {}
            dict.__init__(self, init)


    I = np.ones_like(eos.T)
    ices = ices({'H2O' : 0.9*d2g*(eos.T < 150), 'grains' : 0.1*d2g*I})
    ices.total_abund = np.atleast_2d([ices[x] for x in ices]).sum(0)
    dust_ice.update_ices(ices)

    # Integrate the dust sizes at fixed radial location:
    times = np.array([0, 1e2, 1e3, 1e4, 1e5, 1e6, 3e6]) * T0

    t = 0
    for ti in times:
        dust.do_grain_growth(ti-t)
        dust_ice.do_grain_growth(ti-t)
        t = ti
        Sigma = dust.Sigma
        plt.subplot(211)
        l, = plt.loglog(grid.Rc, dust.Stokes(Sigma)[1])
        l, = plt.loglog(grid.Rc, dust_ice.Stokes(Sigma)[1],'--',c=l.get_color())
        plt.subplot(212)
        l, = plt.loglog(grid.Rc, dust.grain_size[1])
        l, = plt.loglog(grid.Rc, dust_ice.grain_size[1], '--', c=l.get_color())

    plt.subplot(211)
    plt.xlabel('$R\,[\mathrm{au}]$')
    plt.ylabel('Stokes number')

    plt.subplot(212)
    plt.loglog(grid.Rc, dust.a_BT(), 'k:')
    plt.xlabel('$R\,[\mathrm{au}]$')
    plt.ylabel('$a\,[\mathrm{cm}]$')
    # Test the radial drift code
    plt.figure()
    dust = FixedSizeDust(grid, star, eos, 0.01, [0.01, 0.1], Sigma=Sigma)
    drift = SingleFluidDrift(settling=settling)
      
    times = np.array([0, 1e2, 1e3, 1e4, 1e5, 1e6, 3e6]) * 2*np.pi
    
    t = 0
    n = 0
    for ti in times:
        while t < ti:
            dt = 0.5*drift.max_timestep(dust)
            dti = min(ti-t, dt)

            drift(dt, dust)
            t = np.minimum(t + dt, ti)
            n += 1

            if (n % 1000) == 0:
                print('Nstep: {}'.format(n))
                print('Time: {} yr'.format(t / (2 * np.pi)))
                print('dt: {} yr'.format(dt / (2 * np.pi)))

        print('Nstep: {}'.format(n))
        print('Time: {} yr'.format(t / (2 * np.pi)))
        l, = plt.loglog(grid.Rc, dust.Sigma_D[1])
        plt.loglog(grid.Rc, dust.Sigma_D[0], '-.', c=l.get_color())
        

    plt.loglog(grid.Rc, dust.Sigma_G, 'k:')
    plt.loglog(grid.Rc, dust.Sigma, 'k--')
    
    plt.xlabel('$R\,[\mathrm{au}]$')
    plt.ylabel('$\Sigma_{\mathrm{D,G}}$')
    plt.ylim(ymin=1e-10)
    plt.show()
    
