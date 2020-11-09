# porous_dust.py
#
# Author: R. Booth
# Date : 27 - July - 2018
#
# Classes for dusty discs include grain porosity
################################################################################
from __future__ import print_function
import numpy as np
from .constants import *
from .disc import AccretionDisc

class DustyDisc(AccretionDisc):
    """Dusty accretion disc. Base class for an accretion disc that also
    includes one or more dust species.

    args:
        grid     : Disc gridding object
        star     : Stellar object
        eos      : Equation of state
        Sc       : Schmidt number, default=1
        feedback : When False, the dust mass is considered to be a negligable
                   fraction of the total mass.
    """
    def __init__(self, grid, star, eos, Sigma=None, Sc=1., rho_s=1.,
                 feedback=True):

        super(DustyDisc, self).__init__(grid, star, eos, Sigma)

        self._Kdrag = (np.pi * rho_s) / 2.
        self._rho_s = rho_s
        self._l_mfp0 = (2.4*m_H/2e-15)

        self._Sc = Sc
        self._feedback = feedback

    def Stokes(self, v=0):
        a = self._a
        m = self._m
        Sigma = self.Sigma_G
        
        phi_a = (3*m) / (4*np.pi*self._rho_s*a**2)
        St_eps = self._Kdrag * phi_a / (Sigma + 1e-300)

        # Compute the Stokes regime correction factor
        rho_g = self.midplane_gas_density
        x = 4*a*rho_g/(9*self._l_mfp0)

        vth = self.cs*np.sqrt(8/np.pi)*AU*Omega0
        Re = 9*x*v/vth
        
        f = np.maximum(1, np.maximum(Re**0.4, 0.01833*Re))
        
        return St_eps * np.where(x < 1, 1, x/f)

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
    def grain_mass(self):
        return self._m

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
class DustGrowthPorous(DustyDisc):
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
        a0        : Initial particle size (default = 1e-5, 0.1 micron)
        amin      : Minimum particle size (default = 0.0)
        f_drift   : Drift fitting factor. Reduce by a factor ~10 to model the
                    role of bouncing (default=0.55).
        f_frag    : Fragmentation boundary fitting factor (default=0.37).
        feedback  : Whether to include feedback from dust on gas
    """
    def __init__(self, grid, star, eos, eps, Sigma=None,
                 rho_s=1., Sc=1., uf_0=100., uf_ice=1e3, f_ice=1, thresh=0.1,
                 a0=1e-5, amin=0., f_drift=0.55, f_frag=0.37, feedback=True):
        super(DustGrowthPorous, self).__init__(grid, star, eos,
                                               Sigma, rho_s, Sc, feedback)

        
        self._uf_0   = uf_0 / (AU * Omega0)
        self._uf_ice = uf_ice / (AU * Omega0)

        # Grain density
        self._rho_s = rho_s

        # Fitting factors
        self._ffrag  = f_frag * (2/(3*np.pi)) 
        self._fdrift = f_drift * (2/np.pi)
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

        self._head = (', uf_0: {}cm s^-1, uf_ice: {}cm s^-1, thresh: {}'
                      ', a0: {}cm'.format(uf_0, uf_ice, thresh, a0))


        self.update(0)

    def ASCII_header(self):
        """Dust growth header"""
        return super(DustGrowthPorous, self).ASCII_header() + self._head

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        name, head = super(DustGrowthPorous, self).HDF5_attributes()

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
            eps_tot = self.integ_dust_frac

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
        return 1 / (self.Omega_k * eps)

    def do_grain_growth(self, dt):
        """Apply the grain growth"""

        # Size and total gas fraction
        a = self._a[1]        
        eps_tot = self.dust_frac.sum(0)
        
        afrag_t = self._frag_limit()
        adrift, afrag_d =  self._drift_limit(eps_tot)
        t_grow = self._t_grow(eps_tot)
        
        afrag = np.minimum(afrag_t, afrag_d)
        a0    = np.minimum(afrag, adrift)

        # Update the particle distribution
        #   Maximum size due to growth:
        amax = np.minimum(a0, a*np.exp(dt/t_grow))
        #   Reduce size due to erosion / fragmentation if grains have grown
        #   above this due to ice condensation
        # amin = a + np.minimum(0, afrag-a)*np.expm1(-dt/t_grow)
        # ignore empty cells:
        ids = eps_tot > 0
        self._a[1, ids] = np.maximum(amax[ids], self._amin)

        # set the mass
        self._m = (4*np.pi/3) * self._rho_s * self._a**3

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
        super(DustGrowthPorous, self).update(dt)
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
    """
    def __init__(self, diffusion=None, settling=False):
        self._diffuse = diffusion
        self._settling = settling

    def ASCII_header(self):
        """Radial drift header"""
        head = ''
        if self._diffuse:
            head += self._diffuse.ASCII_header() + '\n'
        head += ('# {} diffusion: {} settling: {}'
                 ''.format(self.__class__.__name__,
                           self._diffuse is not None,
                           self._settling))
        return head

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        head = { "diffusion" : "{}".format(self._diffuse is not None),
                 "settling"  : "{}".format(self._settling)
                 }
        if self._diffuse is not None:
            head.update(dict([self._diffuse.HDF5_attributes()]))
        return self.__class__.__name__ , head

    def max_timestep(self, disc):
        step = np.inf
        
        dV = abs(self._compute_deltaV(disc))
        return 0.5 * (disc.grid.dRc / dV).min()
    
    def _fluxes(self, disc, eps_i, deltaV_i, St_i):
        """Update a quantity that moves with the gas/dust"""

        Sigma = disc.Sigma
        grid = disc.grid

        # Add boundary cells
        shape_v   = eps_i.shape[:-1] + (eps_i.shape[-1]+1,)
        shape_rho = eps_i.shape[:-1] + (eps_i.shape[-1]+2,)
        
        dV_i = np.empty(shape_v, dtype='f8')
        dV_i[...,1:-1] = deltaV_i - self._epsDeltaV
        dV_i[..., 0] = dV_i[..., 1] 
        dV_i[...,-1] = dV_i[...,-2] 

        Sig = np.zeros(shape_rho[-1], dtype='f8')
        eps = np.zeros(shape_rho,     dtype='f8')
        Sig[    1:-1] = Sigma
        eps[...,1:-1] = eps_i
        
        # Upwind the density
        Sig = np.where(dV_i > 0, Sig[    :-1], Sig[    1:])
        eps = np.where(dV_i > 0, eps[...,:-1], eps[...,1:])
        
        # Compute the fluxes
        flux = Sig*eps * dV_i

        # Do the update
        deps = - np.diff(flux*grid.Re) / ((Sigma+1e-300) * 0.5*grid.dRe2)
        if self._diffuse:
            St2 = St_i**2
            try:
                Sc = disc.Sc
            except ValueError:
                Sc = self._diffuse.Sc
            Sc = Sc * (0.5625/(1 + 4*St2) + 0.4375 + 0.25*St2)

            deps += self._diffuse(disc, eps_i, Sc)

        return deps

    def _compute_deltaV(self, disc):
        """Compute the total dust-gas background velocity"""

        Sigma  = disc.Sigma
        SigmaD = disc.Sigma_D
        Om_k   = disc.Omega_k
        St     = disc.Stokes()

        # Average to cell edges:        
        Om_kav  = 0.5*(Om_k      [1:] + Om_k      [:-1])
        Sig_av  = 0.5*(Sigma     [1:] + Sigma     [:-1]) + 1e-300
        SigD_av = 0.5*(SigmaD[...,1:] + SigmaD[...,:-1])
        St_av   = 0.5*(St    [...,1:] + St    [...,:-1])

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
                

        # Compute the lambda factors
        #   Use lambda * eps_g instead of lambda to avoid 0/0 when eps_g -> 0.
        la0, la1 = 0, 0 
        St_1 = 1 / (1 + St_av**2)
        if disc.feedback:
            la0 = (eps_av / (1     + St_av** 2)).sum(0)
            la1 = (eps_av / (St_av + St_av**-1)).sum(0)

        # Compute the gas velocities:
        rho = disc.midplane_gas_density
        dPdr = np.diff(disc.P) / disc.grid.dRc
        eta = - dPdr / (0.5*(rho[1:] + rho[:-1] + 1e-300)*Om_kav)

        D_1 = eps_g / ((eps_g + la0)**2 + la1**2)
        u_gas =                la1  * eta * D_1
        v_gas = - 0.5*(eps_g + la0) * eta * D_1

        # Dust-gas relative velocities:
        DeltaV = (2*v_gas / (St_av + St_av**-1) 
                  - u_gas / (1     + St_av**-2))

        # epsDeltaV = v_COM - v_gas (= 0 if dust mass is neglected)
        if disc.feedback:
            self._epsDeltaV = (eps_av * DeltaV).sum(0)
        else:
            self._epsDeltaV = 0

        return DeltaV

    def __call__(self, dt, disc, gas_tracers=None, dust_tracers=None):
        """Apply the update for radial drift over time-step dt"""
        eps = disc.dust_frac
        a   = disc.grain_size   
        eps_inv = 1. / (disc.integ_dust_frac + np.finfo(eps.dtype).tiny)

        # Compute the dust-gas relative velocity
        DeltaV = self._compute_deltaV(disc)
        
        # Compute and apply the fluxes
        if gas_tracers is not None:
            gas_tracers[:] += dt * self._fluxes(disc, gas_tracers, 0, 0)


        # Update the dust fraction, size and tracers
        d_tr = 0
        for eps_k, dV_k, a_k, St_k in zip(disc.dust_frac, DeltaV,
                                          disc.grain_size, disc.Stokes()):
            if dust_tracers is not None:
                t_k = dust_tracers * eps_k * eps_inv
                d_tr  += dt*self._fluxes(disc, t_k, dV_k, St_k)
                
            # multiply a_k by the dust-to-gas ratio, so that constant functions
            # are advected perfectly
            eps_a = a_k * eps_k
            eps_a +=  dt*self._fluxes(disc, eps_a, dV_k, St_k)
            
            eps_k[:] += dt*self._fluxes(disc, eps_k, dV_k, St_k)

            a_k[:] = eps_a / (eps_k + 1e-300)

        if dust_tracers is not None:
            dust_tracers[:] += d_tr
            
    def radial_drift_velocity(self, disc):
        """Compute the radial drift velocity for the disc"""
        eps = disc.dust_frac
        a   = disc.grain_size   
        eps_inv = 1. / (disc.integ_dust_frac + np.finfo(eps.dtype).tiny)

        # Compute the dust-gas relative velocity
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
    dust     = DustGrowthPorous(grid, star, eos, d2g, Sigma=Sigma)
    dust_ice = DustGrowthPorous(grid, star, eos, d2g, Sigma=Sigma)
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
    plt.show()
    
