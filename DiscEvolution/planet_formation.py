from __future__ import print_function
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ispline
from scipy.interpolate import UnivariateSpline as spline
from scipy.integrate import ode
from .constants import *
from .disc_utils import make_ASCII_header

################################################################################
# Planet collections class
################################################################################
class Planets(object):
    """Data for growing planets.

    Holds the location, core & envelope mass, and composition of growing
    planets.

    args:
        Nchem : number of chemical species to track, default = None
    """
    def __init__(self, Nchem=None):
        self.R  = np.array([], dtype='f4')
        self.M_core = np.array([], dtype='f4')
        self.M_env  = np.array([], dtype='f4')
        self.t_form = np.array([], dtype='f4')

        self._N = 0

        if Nchem:
            self.X_core = np.array([[] for _ in range(Nchem)], dtype='f4')
            self.X_env  = np.array([[] for _ in range(Nchem)], dtype='f4')
        else:
            self.X_core = None
            self.X_env  = None
        self._Nchem = Nchem

    def add_planet(self, t, R, Mcore, Menv, X_core=None, X_env=None):
        """Add a new planet"""
        if self._Nchem:
            self.X_core = np.c_[self.X_core, X_core]
            self.X_env  = np.c_[self.X_env, X_env]

        self.R      = np.append(self.R, R)
        self.M_core = np.append(self.M_core, Mcore)
        self.M_env  = np.append(self.M_env, Menv)
        
        self.t_form = np.append(self.t_form, np.ones_like(Menv)*t)

        self._N += 1

    def append(self, planets):
        """Add a list of planets from another planet object"""
        self.add_planet(planets.t_form, planets.R,
                        planets.M_core, planets.M_env,
                        planets.X_core, planets.X_env)

    @property
    def M(self):
        return self.M_core + self.M_env

    @property
    def N(self):
        """Number of planets"""
        return self._N

    @property
    def chem(self):
        return self._Nchem > 0

    def __getitem__(self, idx):
        """Get a sub-set of the planets"""
        sub = Planets(self._Nchem)      

        sub.R      = self.R[idx]
        sub.M_core = self.M_core[idx]
        sub.M_env  = self.M_env[idx]
        sub.t_form = self.t_form[idx]
        if self.chem:
            sub.X_core = self.X_core[...,idx]
            sub.X_env  = self.X_env[...,idx]

        try:
            sub._N = len(sub.R)
        except TypeError:
            sub._N = 1

        return sub

    def __iter__(self):
        for i in range(self.N):
            yield self[i]
    
################################################################################
# Accretion
################################################################################
class GasAccretion(object):
    """Gas giant accretion model of Bitsch et al (2015).

    Combines models from Piso & Youdin (2014) for accretion onto low mass
    envelopes and Machida et al (2010) for accretion onto massive envelopes.

    args:
        General:
           disc  : Accretion disc
           f_max : maximum accretion rate relative to disc accretion rate,
                   default=0.8

        Piso & Youdin parameters:
           f_py      : accretion rate fitting factor, default=0.2
           kappa_env : envelope opacity [cm^2/g], default=0.06
           rho_core  : core density [g cm^-3], default=5.5
    """
    def __init__(self, disc, f_max=0.8,
                 f_py=0.2, kappa_env=0.05, rho_core=5.5):

        # General properties
        self._fmax = f_max
        self._disc = disc

        # Piso & Youdin parameters
        self._fPiso = 0.1 * 1.75e-3 / f_py**2
        self._fPiso /= kappa_env * (rho_core/5.5)**(1/6.)
        # Convert Mearth / M_yr to M_E Omega0**-1
        self._fPiso *= 1e-6 / (2*np.pi)

        head = {"f_max"     : "{}".format(f_max),
                "f_py"      : "{}".format(f_py),
                "kappa_env" : "{} cm^2 g^-1".format(kappa_env),
                "rho_core"  : "{} g cm^-1".format(rho_core),
                }
        self._head = (self.__class__.__name__, head)

    def ASCII_header(self):
        """Get header details"""
        return make_ASCII_header(self.HDF5_attributes())

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        return self._head

    def set_disc(self, disc):
        self._disc = disc
        self.update()

    def computeMdot(self, Rp, M_core, M_env):
        """Compute gas accretion rate.

        args:
            Rp     : radius, AU
            M_core : Core mass, Mearth
            M_env  : Envelope mass, Mearth

        returns:
            Mdot : accretion rate in Mearth per Omega0**-1
        """
        # Cache data:
        Mp = M_core + M_env
        
        # Piso & Youdin (2014) accretion rate:
        T81 = self._disc.interp(Rp, self._disc.T)/81
        Mdot_PY = self._fPiso * T81**-0.5 * M_core**(11/3.) / M_env
        
        # Machida+ (2010) accretion rate
        star = self._disc.star
        rH = star.r_Hill(Rp, Mp*Mearth/Msun)

        Sig = self._disc.interp(Rp, self._disc.Sigma_G)
        H   = self._disc.interp(Rp, self._disc.H)
        nu  = self._disc.interp(Rp, self._disc.nu)

        Om_k = star.Omega_k(Rp)
        
        # Accretion rate is the minimum of two branches, meeting at
        # rH/H ~ 0.3
        f = np.minimum(0.83 * (rH/H)**4.5, 0.14)

        
        # Convert to Mearth / AU**2
        Sig /= Mearth/AU**2

        Mdot_Machida = f * Om_k * Sig * H*H

        Mdot = np.where(M_core > M_env, Mdot_PY, Mdot_Machida)

        return np.minimum(Mdot, self._fmax * 3*np.pi*Sig*nu)

    def __call__(self, planets):
        """Compute gas accretion onto planets

        args:
             planets : planets object.

        returns:
            Mdot : accretion rate in Mearth per Omega0**-1
        """
        return self.computeMdot(planets.R, planets.M_core, planets.M_env)


    def update(self):
        """Update internal quantities after the disc has evolved"""
        pass
    
class PebbleAccretionHill(object):
    """Pebble accretion model of Bitsch+ (2015).

    See also, Lambrechts & Johansen (2012), Morbidelli+ (2015)
    """
    def __init__(self, disc):
        self.set_disc(disc)

    def ASCII_header(self):
        """Get header details"""
        return '# {}'.format(self.__class__.__name__)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        return self.__class__.__name__, {}

    def set_disc(self, disc):
        self._disc = disc
        self.update()
        
    def M_iso(self, R):
        """Pebble isolation mass."""
        h = self._disc.interp(R, self._disc.H) / R
        return 20. * (h/0.05)**3

    def M_transition(self, R):
        """Compute lowest mass for the hill accretion branch used by this model.

        args:
            R : radius, AU

        returns:
            M_t : transition mass, Mearth
        """
        h = self._disc.interp(R, self._disc.H) / R

        # Use a safe, noise free approximation here:
        #eta = - 0.5 * h*h * self._dlgP(np.log(R))
        eta = - 0.5 * h*h * (-2.75)
        Om_k = self._disc.star.Omega_k(R)
        v_k = Om_k * R
        
        M_t = (1/3.)**0.5 * (eta*v_k)**3 / (G * Om_k) * Msun / Mearth
        return M_t
        
        
    def computeMdot(self, Rp, Mp):
        """Compute the pebble accretion rate

        args :
             Rp : radius of planet in AU
             Mp : planet mass in M_earth
        """
        # Cache local varibales
        disc = self._disc
        star = disc.star
        
        # Interpolate disc properites to planet location
        Hp    = disc.interp(Rp, disc.Hp[-1])
        St    = disc.interp(Rp, disc.Stokes()[-1])
        Sig_p = disc.interp(Rp, disc.Sigma_D[-1])

        rH   = star.r_Hill(Rp, Mp*Mearth/Msun)
        Om_k = star.Omega_k(Rp)
        r_eff = rH * (St/0.1)**(1/3.)

        Sig_p /= Mearth / AU**2
        
        # Accretion rate in the limit Hp << rH
        Mdot = 2*np.minimum(rH*rH, r_eff*r_eff) * Om_k*Sig_p

        # 3D correction for Hp >~ r_H:
        # Replaces Sigma_p -> np.pi * rho_p * r_eff
        Mdot *= np.minimum(1, r_eff *(np.pi/8)**0.5 / Hp)

        return (Mp < self.M_iso(Rp)) * Mdot

    def __call__(self, planets):
        """Compute pebble accretion rate"""
        return self.computeMdot(planets.R, planets.M)

    def update(self):
        """Update internal quantities after the disc has evolved"""
        
        lgP = spline(np.log(self._disc.R), np.log(self._disc.P))
        self._dlgP = lgP.derivative(1)


################################################################################
# Migration
################################################################################

def _GK(p):
    gk0 = 16/25.

    f1 = gk0*p**1.5
    f2 = 1 - (1-gk0)*p**-(8/3.)

    return np.where(p < 1, f1, f2)


def _F(p):
    return 1 / (1 + (p/1.3)**2)




# Linblad torque
def _linblad(alpha, beta):
    return -2.5 - 1.7*beta + 0.1*alpha

# Linear co-rotation torques
def _cr_baro(alpha):
    return 0.7 * (1.5 - alpha)

def _cr_entr(alpha, beta, gamma):
    return (2.2 - 1.4/gamma) * (beta - (gamma-1)*alpha)

# Non-linear horse-shoe drag torques
def _hs_baro(alpha):
    return 1.1 * (1.5 - alpha)

def _hs_entr(alpha, beta, gamma):
    return 7.9 *(beta - (gamma-1)*alpha) / gamma

_k0 = np.sqrt(28 / (45 * np.pi))
def _K(p):
    return _GK(p/_k0)

_g0 = np.sqrt(8 / (45 * np.pi))
def _G(p):
    return _GK(p/_g0)


class TypeIMigration(object):
    """Type 1 Migration model of Paardekooper et al (2011)

    Only implemented for sofenting the default sofetning parameter b/h=0.4

    args:
        disc  : accretion disc model
        gamma : ratio of specific heats, default=1.4
        M     : central mass, default = 1
    """
    def __init__(self, disc, gamma=1.4):
        self._gamma = gamma

        #Tabulate gamma_eff to avoid underflow/overflow
        self._Q_tab = np.logspace(-2, 2, 100)
        self._gamma_eff_tab = self._gamma_eff(self._Q_tab)

        self.set_disc(disc)

    def ASCII_header(self):
        return '# {} gamma: {}'.format(self.__class__.__name__,
                                       self._gamma)
    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        return self.__class__.__name__, { "gamma" : "{}".format(self._gamma) }

    def set_disc(self, disc):
        self._disc = disc
        self.update()

    def update(self):
        """Update internal quantities after the disc has evolved"""
        disc = self._disc
        
        lgR = np.log(disc.R)
        # Horibble hack to smooth out artifacts?
        _lgSig = ispline(lgR, np.log(disc.Sigma))
        _lgT   = ispline(lgR, np.log(disc.T))

        self._dlgSig = _lgSig.derivative(1)
        self._dlgT   = _lgT.derivative(1)

    # Fitting functions

    def _gamma_eff(self, Q):
        """Effective adiabatic index"""
        Qg = Q*self._gamma
        Qg2 = Qg*Qg
        gm1 = self._gamma-1
        
        f1 = 2*np.sqrt((Qg2 + 1)**2 - 16*Q*Q*gm1)
        f2 = 2*Qg2-2

        return 2*Qg / (Qg + 0.5*np.sqrt(f1 + f2))

    def gamma_eff_tab(self, Q):
        """Effective adiabatic index, tabulated"""
        return np.interp(Q, self._Q_tab, self._gamma_eff_tab)
        
    def compute_torque(self, Rp, Mp):
        """Compute the torques acting on a planet driving Type I migration"""
        disc = self._disc
        star = disc.star
        
        # Interpolate the disc properties
        lgR = np.log(Rp)
        alpha = -self._dlgSig(lgR)
        beta  = -self._dlgT(lgR)

        h     = disc.interp(Rp, disc.H) / Rp
        Sigma = disc.interp(Rp, disc.Sigma)
        nu    = disc.interp(Rp, disc.nu)
        Pr    = disc.interp(Rp, disc.Pr)

        Om_k = star.Omega_k(Rp)
        
        Xi = nu/Pr
        Q = 2*Xi/(3*h*h*h*Rp*Rp*Om_k)
        g_eff = self.gamma_eff_tab(Q)
        
        q_h = (Mp*Mearth/(star.M*Msun)) / h

        jp = Om_k*Rp*Rp
        Om_kr_2 = jp*jp

        # Convert from g cm^-2 AU**4 Omega0**2 to Mearth AU**2 Omega0**2
        norm  = q_h*q_h*Sigma*Om_kr_2 / g_eff
        norm *= AU**2/Mearth
        
        # Compute the scaling factors
        k = jp / (2*np.pi * nu)
        x = (1.1 / g_eff**0.25) * np.sqrt(q_h)

        pnu = 2*np.sqrt(k*x*x*x)/3
        pXi = 3*pnu*np.sqrt(Pr)/2

        Fnu, Gnu, Knu = _F(pnu), _G(pnu), _K(pnu)
        FXi, GXi, KXi = _F(pXi), _G(pXi), _K(pXi)
        
        torque = (_linblad(alpha, beta) +
                  _hs_baro(alpha) * Fnu * Gnu +
                  _cr_baro(alpha) * (1 - Knu) +
                  _hs_entr(alpha, beta, g_eff) * Fnu * FXi * np.sqrt(Gnu*GXi) +
                  _cr_entr(alpha, beta, g_eff) * np.sqrt((1-Knu)*(1-KXi)))


        return norm*torque

    def migration_rate(self, Rp, Mp):
        """Migration rate, dRdt, of the planet"""
        J = Mp*Rp*self._disc.star.v_k(Rp)
        return 2 * (Rp/J) * self.compute_torque(Rp, Mp)
    
    def __call__(self, planets):
        """Migration rate, dRdt, of the planet"""
        return self.migration_rate(planets.R, planets.M)
    

    
class TypeIIMigration(object):
    """Giant planet migration. Uses relation of Baruteau et al (2014)
    """
    def __init__(self, disc):
        self._disc = disc

    def ASCII_header(self):
        """Generate ASCII header string"""
        return '# {}'.format(self.__class__.__name__)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        return self.__class__.__name__, {}

    def set_disc(self, disc):
        self._disc = disc
        self.update()

    def migration_rate(self, Rp, Mp):
        """Migration rate, dR/dt, of the planet"""
        disc = self._disc
        
        Sigma = disc.interp(Rp, disc.Sigma)
        nu    = disc.interp(Rp, disc.nu)

        Sigma *= AU**2/Mearth

        t_mig = Rp*Rp/nu * np.maximum(Mp /(4*np.pi*Sigma*Rp*Rp), 1)

        return - Rp / t_mig
        
    def __call__(self, planets):
        """Migration rate, dRdt, of the planet"""
        return self.migration_rate(planets.R, planets.M)

    def update(self):
        """Update internal quantities after the disc has evolved"""
        pass

################################################################################
# Combined models
################################################################################
    
class CridaMigration(object):
    """Migration by Type I and Type II with a switch based on the Crida &
    Morbidelli (2007) gap depth criterion.

    args:
        disc  : accretion disc model
        gamma : ratio of specific heats, default=1.4
    """
    def __init__(self, disc, gamma=1.4):
        self._typeI  = TypeIMigration(disc, gamma=gamma)
        self._typeII = TypeIIMigration(disc)
        self._disc = disc

    def ASCII_header(self):
        head = '# {} \n#\t{}\n#\t{}'.format(self.__class__.__name__,
                                            self._typeI.ASCII_header()[1:],
                                            self._typeII.ASCII_header()[1:])
        return head

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        return self.__class__.__name__, dict([self._typeI.HDF5_attributes(),
                                              self._typeII.HDF5_attributes()])

    def set_disc(self, disc):
        self._typeI.set_disc(disc)
        self._typeII.set_disc(disc)

        self._disc = disc


    def migration_rate(self, Rp, Mp):
        """Compute migration rate"""
        disc = self._disc
        star = disc.star
        
        vr_I  = self._typeI.migration_rate(Rp, Mp)
        vr_II = self._typeII.migration_rate(Rp, Mp)

        Me = Mp*Mearth/Msun
        q = Me / star.M
        rH = star.r_Hill(Rp, Mp)
        nu = disc.interp(Rp, disc.nu)
        H  = disc.interp(Rp, disc.H)

        Re = Rp * star.v_k(Rp) / nu

        P = np.maximum(0.75*H/rH + 50/(q*Re), 0.541)

        fP = np.where(P < 2.4646, 0.25*(P-0.541), 1 - np.exp(-P**0.75/3))

        return fP*vr_I + (1-fP)*vr_II


    def __call__(self, planets):
        """Compute migration rate"""
        return self.migration_rate(planets.R, planets.M)

    def update(self):
        """Update internal quantities after the disc has evolved"""
        self._typeI.update()
        self._typeII.update()
        
    
        
class Bitsch2015Model(object):
    """Pebble accretion + Gas accretion planet formation model based on
    Bisch et al (2015).

    The model is composed of the Hill branch pebble accretion along with
    gas envelope accretion.

    args:
        disc     : accretion disc model
        pb_gas_f : fraction of pebble accretion rate that arrives as gas,
                   default=0.1
        migrate  : Whether to include migration, default=True
        **kwargs : arguments passed to GasAccretion object
    """
    def __init__(self, disc, pb_gas_f=0.1, migrate=True, **kwargs):

        self._f_gas = pb_gas_f
        
        self._gas_acc = GasAccretion(disc, **kwargs)
        self._peb_acc = PebbleAccretionHill(disc)
        self._disc = disc

        self._migrate = None
        if migrate:
            self._migrate = CridaMigration(disc)

    def ASCII_header(self):
        """header"""
        head ='# {} pb_gas_f: {}, migrate: {}\n'.format(self.__class__.__name__,
                                                        self._f_gas,
                                                        bool(self._migrate))
        head += self._gas_acc.ASCII_header() + '\n' + self._peb_acc.ASCII_header()
        if self._migrate:
            head += '\n' + self._migrate.ASCII_header()
        return head

        def HDF5_attributes(self):
            """Class information for HDF5 headers"""
            head = dict([("pb_gas_f",  "{}".format(self._f_gas)),
                         ("migrate", "{}".format(self._migrate)),
                         self._gas_acc.HDF5_attributes(),
                         self._peb_acc.HDF5_attirbutes()])
            if self._migrate:
                head.update(dict(self._migrate.HDF5_attributes()))

            return self.__class__.__name__, head

    def set_disc(self, disc):
        """Set up the current disc model"""
        self._gas_acc.set_disc(r, Sigma_G, eos)
        self._peb_acc.set_disc(r, Sigma_G, Sigma_p, St, eos)

        if self._migrate:
            self._migrate.set_disc(r, Sigma_G, eos)

        self._disc = disc
            
    def update(self):
        """Update internal quantities after the disc has evolved"""
        self._gas_acc.update()
        self._peb_acc.update()
        if self._migrate:
            self._migrate.update()
        
    def insert_new_planet(self, t, R, planets):
        """Set the initial mass of the planets

        args:
            t : current time
            R : AU, formation locations
            planets : planets object to add planets to
        """
        M0 = self._peb_acc.M_transition(R)

        Mc, Me = M0 * (1-self._f_gas), M0*self._f_gas

        
        if planets.chem:
            Xc, Xe = self._compute_chem(R)
        else:
            Xc, Xe = None, None
            
        planets.add_planet(t, R, Mc, Me, Xc, Xe)


    def _compute_chem(self, R_p):
        disc = self._disc
        chem = disc.chem
        
        Xs = []
        Xg = []
        eps = np.maximum(disc.interp(R_p, disc.integ_dust_frac), 1e-300)
        for spec in chem:
            Xs_i, Xg_i = chem.ice[spec], chem.gas[spec]
            Xs.append(disc.interp(R_p, Xs_i) / eps)
            Xg.append(disc.interp(R_p, Xg_i))

        return np.array(Xs), np.array(Xg)

    def integrate(self, dt, planets):
        """Update the planet masses and radii:

        args:
            dt      : Time to integrate for
            planets : Planets container
        """
        if planets.N == 0: return
        self.update()
        
        chem = False
        if planets.chem:
            chem=True

        f = self._f_gas
        def dMdt(R_p, M_core, M_env):
            Mdot_s = self._peb_acc.computeMdot(R_p, M_core + M_env)
            Mdot_g = self._gas_acc.computeMdot(R_p, M_core, M_env)

            return Mdot_s*(1-f), Mdot_g + Mdot_s*f

        def dRdt(R_p, M_core, M_env):
            if self._migrate:
                return self._migrate.migration_rate(R_p, M_core + M_env)
            else:
                return np.zeros_like(R_p)

        N = planets.N
        Rmin = self._disc.R[0]
        def f_integ(_, y):
            R_p    = y[   :  N]
            M_core = y[N  :2*N]
            M_env  = y[2*N:3*N]

            Rdot = dRdt(R_p, M_core, M_env)
            Mcdot, Medot = dMdt(R_p, M_core, M_env)


            accreted = R_p <= Rmin
            Rdot[accreted] = Mcdot[accreted] = Medot[accreted] = 0
            
            dydt = np.empty_like(y)
            dydt[:N]    = Rdot
            dydt[N:2*N]  = Mcdot
            dydt[2*N:3*N] = Medot

            if chem:
                Xs, Xg =  self._compute_chem(R_p)

                #Ms = Mcdot * f / (1-f)
                Ms = 0
                Mg = np.maximum(Medot - Mcdot,0)
                Nspec = Xs.shape[0]
                dydt[ 3       *N:(3+  Nspec)*N] = (Mcdot*Xs).ravel()
                dydt[(3+Nspec)*N:(3+2*Nspec)*N] = (Ms*Xs + Mg*Xg).ravel()
            
            return dydt
            
        integ = ode(f_integ).set_integrator('dopri5', rtol=1e-5, atol=1e-5)

        if chem:
            Chem_core = (planets.M_core * planets.X_core).flat
            Chem_env  = (planets.M_env  * planets.X_env).flat
            X0 = np.concatenate([planets.R, planets.M_core, planets.M_env,
                                 Chem_core, Chem_env])
        else:
            X0 = np.concatenate([planets.R, planets.M_core, planets.M_env])
        integ.set_initial_value(X0, 0)

        integ.integrate(dt)

        # Compute the fraction of the core / envelope that was accreted in
        # solids

        planets.R = integ.y[:N]
        planets.M_core = integ.y[N:2*N]
        planets.M_env  = integ.y[2*N:3*N]
        
        if chem:
            Ns = np.prod(planets.X_core.shape)
            Xc = integ.y[3*N   :3*N  +Ns].reshape(-1, N)
            Xe = integ.y[3*N+Ns:3*N+2*Ns].reshape(-1, N)
            planets.X_core = Xc / np.maximum(planets.M_core, 1e-300)
            planets.X_env  = Xe / np.maximum(planets.M_env, 1e-300)           
        

    def dump(self, filename, time, planets):
        """Write out the planet info"""

        # First get the header info.
        with open(filename, 'w') as f:
            head = self.ASCII_header()
            f.write(head+'\n')
            print('# time: {}yr\n'.format(time / (2 * np.pi)))

            head = '# R M_core M_env t_form'
            if planets.chem:
                chem = self._disc.chem
                for k in chem.gas:
                    head += ' c{}'.format(k)
                for k in chem.ice:
                    head += ' e{}'.format(k)
            f.write(head+'\n')

            for p in planets:
                f.write('{} {} {} {}'.format(p.R, p.M_core, p.M_env, 
                                             p.t_form / (2 * np.pi)))
                if planets.chem:
                    for Xi in p.X_core:
                        f.write(' {}'.format(Xi))
                    for Xi in p.X_env:
                        f.write(' {}'.format(Xi))
                f.write('\n')
                        

            
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from .eos import LocallyIsothermalEOS, IrradiatedEOS
    from .star import SimpleStar
    from .grid import Grid
    from .dust import FixedSizeDust

    GM = 1.
    cs0 = (1/30.) 
    q = -0.25
    Mdot = 1e-9
    alpha = 1e-3

    Mdot *= Msun / (2*np.pi)
    Mdot /= AU**2

    Rin = 0.01
    Rout = 5e2
    Rd = 100.

    t0 = (2*np.pi)

    star = SimpleStar()
    

    grid = Grid(0.01, 1000, 1000, spacing='log')
    eos = LocallyIsothermalEOS(star, cs0, q, alpha)
    eos.set_grid(grid)
    Sigma =  (Mdot / (3 * np.pi * eos.nu))*np.exp(-grid.Rc/Rd)
    if 1:
        eos = IrradiatedEOS(star, alpha, tol=1e-3, accrete=False)     
        eos.set_grid(grid)
        eos.update(0, Sigma)
        
        # Now do a new guess for the surface density and initial eos.
        Sigma = (Mdot / (3 * np.pi * eos.nu))*np.exp(-grid.Rc/Rd)

        eos = IrradiatedEOS(star, alpha, tol=1e-3)
        eos.set_grid(grid)
        eos.update(0, Sigma)
    disc = FixedSizeDust(grid, star, eos, 1e-2, 1, Sigma)
    R = disc.R
    
    
    #######
    # Test the migration rate calculation
    migI  = TypeIMigration(disc)
    migII = TypeIIMigration(disc)

    migCrida = CridaMigration(disc)

    Rp = [1,5,25,100]
    M_p = np.logspace(-3, 4.0, 100)
    
    planets = Planets()
    for Mi in M_p:
        planets.add_planet(0, 1, Mi, 0)
    
    plt.subplot(211)
    for Ri in Rp:
        planets.R[:] = Ri
        Ri = Ri * np.ones_like(M_p)
        l, = plt.loglog(M_p, -Ri/migCrida(planets)/t0)
        plt.loglog(M_p, -Ri/migI(planets)/t0,  c=l.get_color(), ls='--')
        plt.loglog(M_p,  Ri/migI(planets)/t0,  c=l.get_color(), ls='-.')
        plt.loglog(M_p, -Ri/migII(planets)/t0, c=l.get_color(), ls=':')

    plt.xlabel('$M\,[M_\oplus]$')
    plt.ylabel('$t_\mathrm{mig}\,[yr]$')

    Rp = np.logspace(-0.5,2,100)
    planets.R[:] = Rp
    plt.subplot(212)
    for Mi in [1, 3, 10, 30]:
        planets.M_core[:] = Mi
        l, =plt.loglog(Rp, -Rp/migCrida(planets)/t0)
        plt.loglog(Rp, -Rp/migI(planets)/t0,  c=l.get_color(), ls='--')
        plt.loglog(Rp,  Rp/migI(planets)/t0,  c=l.get_color(), ls='-.')
        plt.loglog(Rp, -Rp/migII(planets)/t0, c=l.get_color(), ls=':')
    plt.xlabel('$R\,[AU]$')
    plt.ylabel('$t_\mathrm{mig}\,[yr]$')
    #######
    # Test the growth models
    
    # Set up some planet mass / envelope ratios
    #M_p = planets.M
    planets.M_core = np.minimum(20, 0.9*M_p)
    planets.M_env  = M_p - planets.M_core

    
    #Sigma = 1700 * R**-1.5
    Rp = [0.5, 5., 50.]
    
    PebAcc = PebbleAccretionHill(disc)
    GasAcc = GasAccretion(disc)


    plt.figure()
    for Ri in Rp:
        planets.R[:] = Ri
        l, = plt.loglog(M_p, M_p/PebAcc(planets)/t0)
        plt.loglog(M_p, M_p/GasAcc(planets)/t0,
                   c=l.get_color(), ls='--')

    plt.xlabel('$M\,[M_\oplus]$')
    plt.ylabel('$t_\mathrm{grow}\,[yr]$')

    # Growth tracks
    plt.figure()

    planet_model = Bitsch2015Model(disc, pb_gas_f=0.0)

    times = np.logspace(0, 7, 200)
    Rp  = np.array(Rp)

    planets = Planets()
    for Ri in Rp:
        planet_model.insert_new_planet(0, Ri, planets)

    print(planets.R)
    print(planets.M_core)
    print(planets.M_env)
        
    Rs, Mcs, Mes, = [], [], []
    t = 0
    for ti in times:
        ti *= t0
        planet_model.integrate(ti-t, planets)
        Rs.append(planets.R.copy())
        Mcs.append(planets.M_core.copy())
        Mes.append(planets.M_env.copy())
        t = ti

    Rs, Mcs, Mes = [ np.array(X) for X in [Rs, Mcs, Mes]]
        
    ax =plt.subplot(311)
    plt.loglog(times, Mcs)
    plt.ylabel('$M_\mathrm{core}\,[M_\oplus]$')
    plt.ylim(ymax=1e3)

    plt.subplot(312, sharex=ax)
    plt.loglog(times, Mes/317.8)
    plt.ylabel('$M_\mathrm{env}\,[M_J]$')

    plt.subplot(313, sharex=ax)
    plt.loglog(times, Rs)
    plt.ylabel('$R\,[\mathrm{au}]$')
    plt.ylim(Rin, Rout)
    
    plt.xlabel('$t\,[yr]$')
    plt.show()
        
