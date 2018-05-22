from __future__ import print_function
import numpy as np
from .brent import brentq
from .constants import GasConst, sig_SB, AU, Omega0
from . import opacity
################################################################################
# Thermodynamics classes
################################################################################
class EOS_Table(object):
    """Base class for equation of state evaluated at certain locations.

    Stores pre-computed temperatures, viscosities etc. Derived classes need to
    provide the funcitons called by set_grid.
    """
    def __init__(self):
        self._gamma = 1.0
        self._mu    = 2.4
    
    def set_grid(self, grid):
        self._R = grid.Rc
        self._set_arrays()

    def _set_arrays(self):
        R = self._R
        self._cs    = self._f_cs(R)
        self._H     = self._f_H(R)
        self._nu    = self._f_nu(R)
        self._alpha = self._f_alpha(R)

    @property
    def cs(self):
        return self._cs

    @property
    def H(self):
        return self._H

    @property
    def nu(self):
        return self._nu

    @property
    def alpha(self):
        return self._alpha

    @property
    def gamma(self):
        return self._gamma

    @property
    def mu(self):
        return self._mu

    def update(self, dt, Sigma,star=None):
        """Update the eos"""
        pass

    def ASCII_header(self):
        """Print eos header"""
        head = '# {} gamma: {}, mu: {}'
        return head.format(self.__class__.__name__,
                           self.gamma, self.mu)
    
class LocallyIsothermalEOS(EOS_Table):
    """Simple locally isothermal power law equation of state:

    args:
        cs0     : sound-speed at 1AU
        q       : power-law index of sound-speed
        alpha_t : turbulent alpha parameter
        star    : stellar properties
        mu      : mean molecular weight, default=2.4
    """
    def __init__(self, star, cs0, q, alpha_t, mu=2.4):
        super(LocallyIsothermalEOS, self).__init__()
        
        self._cs0 = cs0
        self._q = q
        self._alpha_t = alpha_t
        self._H0 = cs0 * star.M**-0.5
        self._T0 = (AU*Omega0)**2 * mu / GasConst
        self._mu = mu
        
    def _f_cs(self, R):
        return self._cs0 * R**self._q

    def _f_H(self, R):
        return self._H0 * R**(1.5+self._q)
    
    def _f_nu(self, R):
        return self._alpha_t * self._f_cs(R) * self._f_H(R)

    def _f_alpha(self, R):
        return self._alpha_t

    @property
    def T(self):
        return self._T0 * self.cs**2

    @property
    def Pr(self):
        return np.zeros_like(self._R)

    def ASCII_header(self):
        """LocallyIsothermalEOS header string"""
        head = super(LocallyIsothermalEOS, self).ASCII_header()
        head += ', cs0: {}, q: {}, alpha: {}'
        return head.format(self._cs0, self._q, self._alpha_t)

    @staticmethod
    def from_file(filename):
        raise NotImplementedError('')

    @property
    def star(self):
        return self._star

_sqrt2pi = np.sqrt(2*np.pi)
class IrradiatedEOS(EOS_Table):
    """Model for an active irradiated disc.

    From Nakamoto & Nakagawa (1994), Hueso & Guillot (2005).

    args:
        star    : Stellar properties
        alpha_t : Viscous alpha parameter
        Tc      : External irradiation temperature (nebular), default=10
        Tmax    : Maximum temperature allowed in the disc, default=1500
        mu      : Mean molecular weight, default = 2.4
        gamma   : Ratio of specific heats
        accrete : Whether to include heating due to accretion,
                  default=True
    """
    def __init__(self, star, alpha_t, Tc=10, Tmax=1500., mu=2.4, gamma=1.4,
                 accrete=True, tol=None): # tol is no longer used
        super(IrradiatedEOS, self).__init__()

        self._star = star
        
        self._dlogHdlogRm1 = 2/7.

        self._alpha_t = alpha_t
        
        self._Tc = Tc
        self._Tmax = Tmax
        self._mu = mu

        self._accrete = accrete
        self._gamma = gamma
        
        self._kappa = opacity.Zhu2012
        
        self._T = None

        self._compute_constants()

    def _compute_constants(self):
        self._sigTc4 = sig_SB*self._Tc**4
        self._H0 = (Omega0**-1/AU) * (GasConst / (self._mu*self._star.M))**0.5


    def update(self, dt, Sigma, star=None):
        if star:
            self._star = star
            self._compute_constants()
        star = self._star
            
        # Temperature/gensity independent quantities:
        R = self._R
        Om_k = Omega0 * star.Omega_k(R)

        X = star.Rau/R
        f_flat  = (2/(3*np.pi)) * X**3
        f_flare = 0.5 * self._dlogHdlogRm1 * X**2
        
        # Heat capacity
        mu = self._mu
        #C_V = (k_B / (self._gamma - 1)) * (1 / (mu * m_H))
        
        alpha = self._alpha_t
        if not self._accrete:
            alpha = 0.

        # Local references 
        max_heat = sig_SB * (self._Tmax*self._Tmax)*(self._Tmax*self._Tmax)
        star_heat = sig_SB * star.T_eff**4
        sqrt2pi = np.sqrt(2*np.pi)            
        def balance(Tm):
            """Thermal balance"""
            cs = np.sqrt(GasConst * Tm / mu)
            H = cs / Om_k

            kappa = self._kappa(Sigma / (sqrt2pi * H), Tm)
            tau = 0.5 * Sigma * kappa
            H /= AU

            # External irradiation
            dEdt = self._sigTc4
            
            # Compute the heating from stellar irradiation
            dEdt += star_heat * (f_flat + f_flare * (H/R))

            # Viscous Heating
            visc_heat = 1.125*alpha*cs*cs * Om_k
            dEdt += visc_heat*(0.375*tau*Sigma + 1./kappa)
            
            # Prevent heating above the temperature cap:
            dEdt = np.minimum(dEdt, max_heat)

            # Cooling
            Tm2 = Tm*Tm
            dEdt -= sig_SB * Tm2*Tm2

            # Change in temperature
            return (dEdt/Omega0) # / (C_V*Sigma)

        # Solve the balance using brent's method (needs ~ 20 iterations)
        T0 = self._Tc
        T1 = self._Tmax
        if self._T is not None:
            dedt = balance(self._T)
            T0 = np.where(dedt > 0, self._T, T0)
            T1 = np.where(dedt < 0, self._T, T1)

        self._T =  brentq(balance, T0, T1)
        self._Sigma = Sigma
        self._set_arrays()


    def set_grid(self, grid):
        self._R = grid.Rc
        self._T = None

    def _set_arrays(self):
        super(IrradiatedEOS,self)._set_arrays()
        self._Pr = self._f_Pr()
    
    def __H(self, R, T):
        return self._H0 * np.sqrt(T * R*R*R)

    def _f_cs(self, R):
        return self._H0 * self._T**0.5

    def _f_H(self, R):
        return self.__H(R, self._T)
    
    def _f_nu(self, R):
        return self._alpha_t * self._f_cs(R) * self._f_H(R)

    def _f_alpha(self, R):
        return self._alpha_t

    def _f_Pr(self):
        rho = self._Sigma / ((2*np.pi)**0.5 * self.H * AU)
        kappa = self._kappa(rho, self._T)
        tau = 0.5 * self._Sigma * kappa
        f_esc = 1 + 2/(3*tau*tau)
        Pr_1 =  2.25 * self._gamma * (self._gamma - 1) * f_esc
        return 1. / Pr_1

    @property
    def T(self):
        return self._T

    @property
    def Pr(self):
        return self._Pr

    @property
    def star(self):
        return self._star

    def ASCII_header(self):
        """IrradiatedEOS header"""
        head = super(IrradiatedEOS, self).ASCII_header()
        head += ', opacity: {}, T_extern: {}K, accrete: {}, alpha: {}'
        head += ', Tmax: {}K'
        return head.format(self._kappa.__class__.__name__,
                           self._Tc, self._accrete, self._alpha_t,
                           self._Tmax)

    @staticmethod
    def from_file(filename):
        import star

        star = star.from_file(filename)
        alpha = None

        with open(filename) as f:
            for line in f:
                if not line.startswith('#'):
                    raise AttributeError("Error: EOS type not found in header")
                elif "IrradiatedEOS" in line:
                    string = line 
                    break 
                else:
                    continue

        kwargs = {}
        for item in string.split(','):    
            key, val = [ x.strip() for x in item.split(':')]

            if   key == 'gamma' or key == 'mu':
                kwargs[key] = float(val.strip())
            elif key == 'alpha':
                alpha = float(val.strip())
            elif key == 'accrete':
                kwargs[key] = bool(val.strip())
            elif key == 'T_extern':
                kwargs['Tc'] = float(val.replace('K','').strip())

        return IrradiatedEOS(star, alpha, **kwargs)

def from_file(filename):
    with open(filename) as f:
        for line in f:
            if not line.startswith('#'):
                raise AttributeError("Error: EOS type not found in header")
            elif "IrradiatedEOS" in line:
                return IrradiatedEOS.from_file(filename)      
            elif "LocallyIsothermalEOS" in line:
                return LocallyIsothermalEOS.from_file(filename)
            else:
                continue


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from star import SimpleStar
    from grid import Grid

    alpha = 1e-3
    star = SimpleStar(M=1.0, R=3.0, T_eff=4280.)

    active  = IrradiatedEOS(star, alpha)
    passive = IrradiatedEOS(star, alpha, accrete=False)

    powerlaw = LocallyIsothermalEOS(star, 1/30., -0.25, alpha)

    grid = Grid(0.1, 500, 1000, spacing='log')
    
    Sigma = 2.2e3 / grid.Rc**1.5

    c  = { 'active' : 'r', 'passive' : 'b', 'isothermal' : 'g' }
    ls = { 0 : '-', 1 : '--' }
    for i in range(2):
        for eos, name in [[active, 'active'],
                          [passive, 'passive'],
                          [powerlaw, 'isothermal']]:
            eos.set_grid(grid)
            eos.update(0, Sigma)

            label = None
            if ls[i] == '-':
                label = name
                
            plt.loglog(grid.Rc, eos.T, c[name] + ls[i], label=label)
        Sigma /= 10
    plt.legend()
    plt.show()
    
                    
