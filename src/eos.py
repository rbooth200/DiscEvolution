import numpy as np
from scipy.optimize import newton, brentq
from constants import GasConst, sig_SB, m_H, AU, Omega0, G_CGS
import opacity
################################################################################
# Thermodynamics classes
################################################################################
class EOS_Table(object):
    '''Base class for equation of state evaluated at certain locations.
    
    Stores pre-computed temperatures, viscosities etc. Derived classes need to
    provide the funcitons called by set_grid.
    '''
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
        '''Update the eos'''
        pass

    def header(self):
        '''Print eos header'''
        head = '# {} gamma: {}, mu: {}'
        return head.format(self.__class__.__name__,
                           self.gamma, self.mu)
    
class LocallyIsothermalEOS(EOS_Table):
    '''Simple locally isothermal power law equation of state:

    args:
        cs0     : sound-speed at 1AU
        q       : power-law index of sound-speed
        alpha_t : turbulent alpha parameter
        star    : stellar properties
        mu      : mean molecular weight, default=2.4
    '''
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

    def header(self):
        '''LocallyIsothermalEOS header string'''
        head = super(LocallyIsothermalEOS, self).header()
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
    '''Model for an active irradiated disc. 

    From Nakamoto & Nakagawa (1994), Hueso & Guillot (2005).

    The gravito-turbulent model is the model recommended by Rafikov (2015),
    except the optical depth function f(tau) from Hueso & Guillot (2005) is 
    used.

    args:
        star    : Stellar properties
        alpha_t : Viscous alpha parameter
        Tc      : External irradiation temperature (nebular), default=10
        mu      : Mean molecular weight, default = 2.4
        gamma   : Ratio of specific heats
        accrete : Whether to include heating due to accretion, 
                  default=True
        GI      : Include a local model of viscosity due to gravito-turbulence,
                  default= False
        tol     : Tolerence for change in density before recomputing T, 
                  default=0.01
    '''
    def __init__(self, star, alpha_t, Tc=10, mu=2.4, gamma=1.4,
                 accrete=True, GI=False, tol=0.01):

        super(IrradiatedEOS, self).__init__()

        self._star = star
        
        self._dlogHdlogRm1 = 2/7.

        self._alpha_t = alpha_t
        
        self._Tc = Tc
        self._sigTc4 = sig_SB*Tc**4

        self._H0 = (Omega0**-1/AU) * (GasConst / (mu*star.M))**0.5
        self._mu = mu

        self._accrete = accrete
        self._gamma = gamma
        
        self._kappa = opacity.Zhu2012
        
        self._T = None
        self._GI = GI
        self._tol = tol

        
    def __H(self, R, T):
        return self._H0 * np.sqrt(T * R*R*R)

    def _T_GI(self, Sigma, Om_k):
        '''Temperature in gravito-turbulent equilibrium'''
        Q0 = 1.0
        cs_Q = np.pi * G_CGS * Sigma * Q0 / Om_k
        T_Q = (self._mu * cs_Q*cs_Q / GasConst)
        
        return T_Q

    def _escape(self, tau):
        '''Escape probability function

        args:
            tau = 0.5 Sigma Kappa
        '''
        return (0.375*tau + 0.5/tau)

    def _irradiate(self, R, H):
        '''Heating from irradiation'''
        # External irradiation
        dEdt = self._sigTc4
        
        # Compute the heating from stellar irradiation
        X = (self._star.Rau/R)
        f_flat = (2/(3*np.pi)) * X*X*X
        f_flare = 0.5 * self._dlogHdlogRm1 * (X*X)*(H/R)
        dEdt += sig_SB * self._star.T_eff**4 * (f_flat + f_flare)

        return dEdt

    def _thermal_balance(self, Tm, R, Sigma, Om_k):
        cs = np.sqrt(GasConst * Tm / self._mu)
        H = cs / Om_k
        
        rho = Sigma / (_sqrt2pi * H)
        tau = 0.5 * Sigma * self._kappa(rho, Tm)
        H /= AU
        
        # Heating due to irradiation
        dEdt = self._irradiate(R, H)

        # Add in the heating due to gravito-turbulence
        if self._GI and self._accrete:
            sig_TGI4 = sig_SB * self._T_GI(Sigma, Om_k)**4
            x =  np.maximum(sig_TGI4 - dEdt, 0)
            dEdt += np.maximum(sig_TGI4 - dEdt, 0)
        
        # Viscous heating:
        heat = 1.125*self._alpha_t*cs*cs * Om_k * Sigma
        dEdt += heat*self._escape(tau)

        return dEdt - sig_SB*Tm**4

    def _compute_critical_density(self, R):
        '''Compute the minimum surface density for a significant contribution
        to the heating rate by viscosity'''
        # Save the alpha viscosity coefficient
        alpha, self._alpha_t = self._alpha_t, 0

        self._Tmin    = np.empty_like(R)
        self._Sig_min = np.empty_like(R)
        self._Sig_GI  = np.empty_like(R)
        T0, T1 = 1e-5, 2*self._star.T_eff
        for i, R_i in enumerate(R):
            # First compute the temperature without viscous heating
            Om_k = Omega0*self._star.Omega_k(R_i)
            self._Tmin[i] = Tm = self._solve_equilibrium_T(R_i, 1e-10,
                                                           Om_k, T0, T1)

            if not self._accrete: continue

            # Now compute the minimum surface density
            cs2 = (GasConst * Tm / self._mu)
            heat = 1.125* alpha *cs2 * Omega0*self._star.Omega_k(R_i)
            H = self.__H(R_i, Tm) * (2*np.pi)**0.5 * AU
            def f_heat(Sig):
                tau = 0.5 * Sig * self._kappa(Sig/H, Tm)
                return heat*(0.375*tau + 0.5/tau)*Sig - self._tol*sig_SB*Tm**4
            # Bound the temparatures
            Sig_min = 0.1 * self._tol * sig_SB * Tm**4 / heat
            # Sometimes the viscous heating rate is always important, even in
            # the optically thin limit
            if f_heat(Sig_min) > 0: 
                self._Sig_min[i] = 0
                continue
            Sig_max = 10*Sig_min
            while f_heat(Sig_max) < 0: Sig_max *= 2

            self._Sig_min[i] = brentq(f_heat, Sig_min, Sig_max)

            if self._GI:
                # Find the minimum density at which GI needs to be included
                def f_GI(Sig):
                    return self._T_GI(Sig, Om_k) - Tm
                self._Sig_GI[i] = newton(f_GI, cs2**0.5 * Om_k/(np.pi*G_CGS))
            
        # Reset the alpha viscosity coefficient    
        self._alpha_t = alpha

        # Unset temperature array to force recompute of temperature:
        self._T = None
        
    def _solve_equilibrium_T(self, R, Sigma, Om_k, T0, T1):
        return brentq(self._thermal_balance, T0, T1, args=(R,Sigma,Om_k))

    def update(self, dt, Sigma, star=None, update_all=False):
        '''Compute the equilibrium temperature, given the surface density 
        args:
            Sigma : array surface density in c.g.s
        '''
        if star:
            if any([self._star.M != star.M,
                    self._star.Rs != star.Rs,
                    self._star.T_eff != star.T_eff]):
                self._compute_critical_density(self._R)
            self._star = star            
        # No need to iterate if accretion is not included
        if not self._accrete:
            self._T = self._Tmin
            self._Sigma = Sigma
            self._set_arrays()
            return
        
        if self._T is None:
            self._T = np.empty_like(self._R)
            self._Sigma = np.zeros_like(self._R)

        # Set regions with negligible viscous heating to the pure irradiation
        # temperature, except if they are dense enough for GI to be relevant.
        i_min = Sigma < self._Sig_min
        if self._GI: i_min[Sigma > self._Sig_GI] = 0
        self._T[i_min] = self._Tmin[i_min]
        self._Sigma[i_min] = Sigma[i_min]

        # Update those in need of a new temperature
        need_update = (abs(Sigma-self._Sigma) > self._tol*Sigma) | update_all
        ids = np.where((~i_min) & need_update)[0]
        for i in ids:
            R_i, Sig_i = self._R[i], Sigma[i]
            try:
                T0 = max(self._Tmin[i],1e-5)
                T1 = 1500
                #if Sig_i < self._Sigma[i]: T1 = self._T[i]

                Om_k = Omega0*self._star.Omega_k(R_i)
                Ti = self._solve_equilibrium_T(R_i, Sig_i, Om_k, T0, T1)
            except ValueError:
                Ti = 1500

            self._T[i] = min(Ti, 1500)
            self._Sigma[i] = Sig_i

        self._set_arrays()
        
    def set_grid(self, grid):
        self._R = grid.Rc
        self._T = None
        self._compute_critical_density(self._R)

    def _set_arrays(self):
        super(IrradiatedEOS,self)._set_arrays()
        self._Pr = self._f_Pr()

        
    def _f_cs(self, R):
        return self._H0 * self._T**0.5

    def _f_H(self, R):
        return self.__H(R, self._T)
    
    def _f_nu(self, R):
        return self._f_alpha(R) * self._f_cs(R) * self._f_H(R)

    def _f_alpha(self, R):
        alpha_GI = 0
        if self._GI and self._accrete:
            cs = np.sqrt(GasConst * self._T / self._mu)
            Om_k = Omega0 * self._star.Omega_k(R)
            H = cs / Om_k
            
            Sig = self._Sigma
            rho = Sig / (_sqrt2pi * H)

            kappa = np.array([self._kappa(d, T) for d, T in zip(rho, self._T)])
            tau = 0.5 * Sig * kappa

            H /= AU

            GI_heat = np.maximum(
                sig_SB*self._T_GI(Sig, Om_k)**4 - self._irradiate(R, H),
                0)

            alpha_GI = GI_heat/(1.125*Om_k*Sig*cs*cs * self._escape(tau))
        
        return self._alpha_t + alpha_GI

    def _f_Pr(self):
        rho = self._Sigma / ((2*np.pi)**0.5 * self.H * AU)
        kappa = np.array([self._kappa(d, T) for d, T in zip(rho, self._T)])
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

    def header(self):
        '''IrradiatedEOS header'''
        head = super(IrradiatedEOS, self).header()
        head += ', opacity: {}, T_extern: {}K, accrete: {}, alpha: {}'
        return head.format(self._kappa.__class__.__name__,
                           self._Tc, self._accrete, self._alpha_t)

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
    GI      = IrradiatedEOS(star, alpha, GI=True)

    powerlaw = LocallyIsothermalEOS(star, 1/30., -0.25, alpha)

    grid = Grid(0.1, 500, 1000, spacing='log')
    
    Sigma = 2.2e3 / grid.Rc**1

    c  = { 'active' : 'r', 'passive' : 'b', 'isothermal' : 'g', 'GI' : 'k'}
    ls = { 0 : '-', 1 : '--' }
    for i in range(2):
        for eos, name in [[active, 'active'],
                          [passive, 'passive'],
                          [GI, 'GI'],
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
    
                    
