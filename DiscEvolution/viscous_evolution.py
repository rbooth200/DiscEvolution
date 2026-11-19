# viscous_evolution.py
#
# Author: R. Booth
# Date: 16 - Nov - 2016
#
# Contains classes for solving the viscous evolution equations.
################################################################################
from __future__ import print_function
import numpy as np
import copy
import warnings


def _set_ghost_cells(S, grid, in_bound, bound):
    """Set the ghost cells, S[0] and S[-1], of S = nu Sigma R^1/2 (or a
    multiple of it) at the centres grid.Rce, for the boundary types of
    ViscousEvolutionFV. The extrapolations use the ghost cells' own centres,
    so they hold on any grid, not only a log-spaced one."""
    Rce = grid.Rce

    # Inner boundary
    if in_bound == 'Zero':            # Zero torque
        S[0] = 0
    elif in_bound == 'Mdot':          # Constant flux (appropriate for power law): S ~ R^1/2
        S[0] = S[1] * np.sqrt(Rce[0] / Rce[1])
    else:
        raise ValueError("Error boundary type not recognised")

    # Outer boundary
    if bound == 'Zero':               # Zero torque
        S[-1] = 0
    elif bound == 'power_law':        # Power law in R through the last two cells
        p = np.log(Rce[-1] / Rce[-2]) / np.log(Rce[-2] / Rce[-3])
        S[-1] = S[-2] * (S[-2] / S[-3]) ** p
    elif bound == 'Mdot_out':         # Constant flux (appropriate for tail of LBP): S ~ R^-1/2
        S[-1] = S[-2] * np.sqrt(Rce[-2] / Rce[-1])
    elif bound == 'Mdot_inn':         # Constant flux (appropriate for power law): S ~ R^1/2
        S[-1] = S[-2] * np.sqrt(Rce[-1] / Rce[-2])
    else:
        raise ValueError("Error boundary type not recognised")


class ViscousEvolution(object):
    """Solves the 1D viscous evolution equation.

    This class handles the inclusion of dust species in the one-fluid
    approximation. The total surface density (Sigma = Sigma_G + Sigma_D) is
    updated under the action of viscous forces, which are taken to act only
    on the gas phase species.

    Can optionally update tracer species.

    args:
       tol       : Ratio of the time-step to the maximum stable one.
                   Default = 0.5
       in_bound  : Type of internal boundary condition:
                     'Zero'      : Zero torque boundary
                     'Mdot'      : Constant Mdot (power law).
       boundary  : Type of external boundary condition:
                     'Zero'      : Zero torque boundary
                     'power_law' : Power-law extrapolation
                     'Mdot_inn'  : Constant Mdot, same as inner (power law).
                     'Mdot_out'  : Constant Mdot, for tail of LBP.

    Deprecated: use ViscousEvolutionFV (or rkl.ViscousEvolutionRKL). This
    scheme conserves mass exactly only on log-spaced grids; the finite-volume
    one does on any grid, with the same accuracy and cost.
    """

    def __init__(self, tol=0.5, boundary='power_law', in_bound='Mdot'):
        warnings.warn("ViscousEvolution is deprecated: use ViscousEvolutionFV "
                      "(or rkl.ViscousEvolutionRKL), which conserves mass "
                      "exactly on any grid", DeprecationWarning, stacklevel=2)
        self._tol = tol
        self._bound = boundary
        self._in_bound = in_bound

    def ASCII_header(self):
        """header"""
        return '# {} tol: {}'.format(self.__class__.__name__, self._tol)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        return self.__class__.__name__, { "tol" : str(self._tol)}



    def _setup_grid(self, grid):
        """Compute the grid factors"""
        self._X = 2 * np.sqrt(grid.Rc)
        self._dXe = 2 * np.diff(np.sqrt(grid.Re))
        self._dXc = 2 * np.diff(np.sqrt(grid.Rce))
        self._RXdXe = grid.Rc * self._X * self._dXe

    def _init_fluxes(self, disc, Sigma_G=None):
        """Cache the important variables needed for the fluxes (for the
        disc's gas surface density, or Sigma_G if given)"""
        nuX = disc.nu * self._X

        S = np.zeros(len(nuX) + 2, dtype='f8')
        S[1:-1] = (disc.Sigma_G if Sigma_G is None else Sigma_G) * nuX
        _set_ghost_cells(S, disc.grid, self._in_bound, self._bound)

        self._dS = np.diff(S) / self._dXc

    def _fluxes(self):
        """Compute the mass fluxes for the viscous evolution equations.

        Gas update from Bath & Pringle (1981)
        """
        return 3. * np.diff(self._dS) / self._RXdXe

    def _tracer_fluxes(self, tracers):
        """Compute fluxes of a tracer.

        Uses the viscous update to compute the flux of  Sigma*tracers,
        divide by the updated Sigma to get the new tracer value.
        """
        shape = tracers.shape[:-1] + (tracers.shape[-1] + 2,)
        s = np.zeros(shape, dtype='f8')
        s[..., 1:-1] = tracers
        s[..., 0] = s[..., 1]
        s[..., -1] = s[..., -2]

        # Upwind the tracer density 
        ds = self._dS * np.where(self._dS <= 0, s[..., :-1], s[..., 1:])

        # Compute the viscous update
        return 3. * np.diff(ds) / self._RXdXe

    def viscous_velocity(self, disc, Sigma=None):
        """Compute the radial velocity due to viscosity"""
        self._setup_grid(disc.grid)
        self._init_fluxes(disc)

        # Can accept other density specifications (e.g. gas only to match specification of dust drift by Dipierro+18)
        if Sigma is None:
            Sigma = disc.Sigma
               
        RS = Sigma * disc.R
        return np.nan_to_num(- 3 * self._dS[1:-1] / (RS[1:] + RS[:-1])) # Avoid nans when working with Sigma_G -> 0

    def max_timestep(self, disc):
        """Courant limited time-step"""
        grid = disc.grid
        nu = disc.nu

        dXe2 = np.diff(2 * np.sqrt(grid.Re)) ** 2

        tc = ((dXe2 * grid.Rc) / (2 * 3 * nu)).min()
        return self._tol * tc

    def __call__(self, dt, disc, tracers=[], adv=[]):
        """Compute one step of the viscous evolution equation
        args:
            dt      : time-step
            disc    : disc we are updating
            tracers : Tracer species to update. Should be a list of arrays with
                      shape = [*, disc.Ncells].
            adv     : Tracers that are advected but not removed due to mass-loss
        """
        self._setup_grid(disc.grid)
        self._init_fluxes(disc)

        f = self._fluxes()
        Sigma_new = disc.Sigma + dt * f
        
        for t in (tracers+adv):
            if t is None: continue
            tracer_density = t*disc.Sigma
            t[:] = (dt*self._tracer_fluxes(t) + tracer_density) / (Sigma_new + 1e-300)

        disc.Sigma[:] = Sigma_new

class ViscousEvolutionFV(object):
    """Solves the 1D viscous evolution equation via a finite-volume method

    This class handles the inclusion of dust species in the one-fluid
    approximation. The total surface density (Sigma = Sigma_G + Sigma_D) is
    updated under the action of viscous forces, which are taken to act only
    on the gas phase species.

    Can optionally update tracer species.

    args:
       tol       : Ratio of the time-step to the maximum stable one.
                   Default = 0.5
       in_bound  : Type of internal boundary condition:
                     'Zero'      : Zero torque boundary
                     'Mdot'      : Constant Mdot (power law).
       boundary  : Type of external boundary condition:
                     'Zero'      : Zero torque boundary
                     'power_law' : Power-law extrapolation
                     'Mdot_inn'  : Constant Mdot, same as inner (power law).
                     'Mdot_out'  : Constant Mdot, for tail of LBP.
    """

    def __init__(self, tol=0.5, boundary='power_law', in_bound='Mdot'):
        self._tol = tol
        self._bound = boundary
        self._in_bound = in_bound

    def ASCII_header(self):
        """header"""
        return '# {} tol: {}'.format(self.__class__.__name__, self._tol)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        return self.__class__.__name__, { "tol" : str(self._tol)}

    def _setup_grid(self, grid):
        """Compute the grid factors"""
        self._Rh  = np.sqrt(grid.Rc)
        self._dR3 = np.diff(np.sqrt(grid.Rce)**3)

        self._dA = grid.Re
        self._dV = 0.5 * np.diff(grid.Re**2)

    def _init_fluxes(self, disc, Sigma_G=None):
        """Cache the important variables needed for the fluxes (for the
        disc's gas surface density, or Sigma_G if given)"""
        nuRh = disc.nu * self._Rh

        S = np.zeros(len(nuRh) + 2, dtype='f8')
        S[1:-1] = (disc.Sigma_G if Sigma_G is None else Sigma_G) * nuRh
        _set_ghost_cells(S, disc.grid, self._in_bound, self._bound)

        self._dS = 4.5 * np.diff(S) / self._dR3

    def _fluxes(self):
        """Compute the mass fluxes for the viscous evolution equations.

        Gas update from Bath & Pringle (1981)
        """
        return np.diff(self._dA * self._dS) / self._dV

    def _tracer_fluxes(self, tracers):
        """Compute fluxes of a tracer.

        Uses the viscous update to compute the flux of  Sigma*tracers,
        divide by the updated Sigma to get the new tracer value.
        """
        shape = tracers.shape[:-1] + (tracers.shape[-1] + 2,)
        s = np.zeros(shape, dtype='f8')
        s[..., 1:-1] = tracers
        s[..., 0] = s[..., 1]
        s[..., -1] = s[..., -2]

        # Upwind the tracer density 
        ds = self._dS * np.where(self._dS <= 0, s[..., :-1], s[..., 1:])

        # Compute the viscous update
        return np.diff(self._dA * ds) / self._dV

    def viscous_velocity(self, disc, S = None):
        """Compute the radial velocity due to viscosity"""
        self._setup_grid(disc.grid)
        self._init_fluxes(disc)

        if S is None:
            S = disc.Sigma 
        return - 2.0 * self._dS[1:-1] / (S[1:] + S[:-1])

    def max_timestep(self, disc):
        """Courant limited time-step"""
        grid = disc.grid
        nu = disc.nu

        dXe2 = np.diff(2 * np.sqrt(grid.Re)) ** 2

        tc = ((dXe2 * grid.Rc) / (2 * 3 * nu)).min()
        return self._tol * tc

    def __call__(self, dt, disc, tracers=[], adv=[]):
        """Compute one step of the viscous evolution equation
        args:
            dt      : time-step
            disc    : disc we are updating
            tracers : Tracer species to update. Should be a list of arrays with
                      shape = [*, disc.Ncells].
            adv     : Tracers that are advected but not removed due to mass-loss
        """
        self._setup_grid(disc.grid)
        self._init_fluxes(disc)

        f = self._fluxes()
        Sigma_new = disc.Sigma + dt * f

        for t in (tracers + adv):
            if t is None: continue
            t[:] += dt*(self._tracer_fluxes(t) - t*f) / (Sigma_new + 1e-300)

        disc.Sigma[:] = Sigma_new


class HybridWindModel(object):
    """Finite volume model for combined viscous evolution + winds
    
    args:
       psi_DW    : Ratio of the wind-driven alpha to the turbulent (viscous)
                   one, psi = alpha_DW / alpha_SS (Tabone et al. 2022).
       lambda_DW : Wind lever arm parameter. Default=3
       tol       : Ratio of the time-step to the maximum stable one.
                   Default = 0.5
       in_bound  : Type of internal boundary condition:
                     'Zero'      : Zero torque boundary
                     'Mdot'      : Constant Mdot (power law).
       boundary  : Type of external boundary condition:
                     'Zero'      : Zero torque boundary
                     'power_law' : Power-law extrapolation
                     'Mdot_inn'  : Constant Mdot, same as inner (power law).
                     'Mdot_out'  : Constant Mdot, for tail of LBP.

    Notes :
       The viscosity provided by the disc model, disc.nu, is the turbulent
       (viscous) part only, nu = alpha_SS cs H. The wind-driven radial velocity
       is v_DW = 1.5 psi_DW nu / R, so the total effective viscosity is
       (1 + psi_DW) nu. Since disc.alpha is alpha_SS, the turbulent diffusion
       coefficient and collision velocities need no correction: the Schmidt
       number keeps its usual (turbulent) meaning.

       Tracers passed in `tracers` (gas species) are removed by the wind along
       with the gas. Tracers passed in `adv` (dust and ices) are advected but
       retained, so their concentration increases as gas is lost.
    """
    def __init__(self, psi_DW, lambda_DW=3, tol=0.5, boundary='power_law', in_bound='Mdot'):
        self._tol = tol
        self._bound = boundary
        self._in_bound = in_bound

        self._lambda = lambda_DW
        self._psi = psi_DW

    def ASCII_header(self):
        """header"""
        return '# {} tol: {}, psi_DW: {}, lambda_DW: {}'.format(self.__class__.__name__, 
            self._tol, self._psi, self._lambda)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        return self.__class__.__name__, { "tol" : str(self._tol),
                                          "psi_DW" : str(self._psi),
                                          "lambda_DW" : str(self._lambda) }
    def _setup_grid(self, grid):
        """Compute the grid factors"""
        self._Rh  = np.sqrt(grid.Rc)
        self._dR3 = np.diff(np.sqrt(grid.Rce)**3)

        self._dA = grid.Re
        self._dV = 0.5 * np.diff(grid.Re**2)


    def _init_fluxes_visc(self, disc):
        """Compute the flux due to viscosity"""
        nuRh = disc.nu *  self._Rh

        S = np.zeros(len(nuRh) + 2, dtype='f8')
        S[1:-1] = disc.Sigma_G * nuRh
        _set_ghost_cells(S, disc.grid, self._in_bound, self._bound)

        self._f_visc = 4.5 * np.diff(S) / self._dR3

    def _init_fluxes_wind(self, disc, dt=0):
        """Compute the flux and mass-loss rate due to the wind"""
        #Use first order Donor cell method:
        v_DW = 1.5 * (disc.nu/disc.R) * self._psi

        F = np.zeros(len(disc.Sigma_G) + 1, dtype='f8')
        F[:-1] = v_DW * disc.Sigma_G

        # Outer boundary
        if self._bound == 'Zero':
            F[-1] = 0
        else:
            F[-1] = F[-2] * (v_DW[-1]*disc.R[-1]) / (v_DW[-2]*disc.R[-2])

        # Compute the mass-loss rate:
        loss_rate = v_DW * disc.Sigma_G / (2*(self._lambda-1) * disc.R)


        self._f_wind = F
        self._s_wind = loss_rate

    def viscous_velocity(self, disc, S = None):
        """Compute the radial velocity due to viscosity + winds"""
        self._setup_grid(disc.grid)
        self._init_fluxes_visc(disc)
        self._init_fluxes_wind(disc)

        if S is None:
            S = disc.Sigma 
        return - 2.0 * (self._f_visc + self._f_wind)[1:-1] / (S[1:] + S[:-1])

    def _fluxes(self):
        """Compute the mass fluxes and loss term """
        return np.diff(self._dA * (self._f_visc + self._f_wind)) / self._dV

    def _tracer_fluxes(self, tracers):
        """Compute fluxes of a tracer.

        Uses the viscous update to compute the flux of  Sigma*tracers,
        divide by the updated Sigma to get the new tracer value.
        """
        shape = tracers.shape[:-1] + (tracers.shape[-1] + 2,)
        s = np.zeros(shape, dtype='f8')
        s[..., 1:-1] = tracers
        s[..., 0] = s[..., 1]
        s[..., -1] = s[..., -2]

        F = self._f_visc + self._f_wind

        # Upwind the tracer density 
        ds = F * np.where(F <= 0, s[..., :-1], s[..., 1:])

        # Compute the viscous update
        return np.diff(self._dA * ds) / self._dV

    def max_timestep(self, disc):
        """Courant limited time-step"""
        grid = disc.grid
        nu = disc.nu

        dXe2 = np.diff(2 * np.sqrt(grid.Re)) ** 2

        t_visc = ((dXe2 * grid.Rc) / (2 * 3 * nu)).min()

        v_DW   = 1.5 * (disc.nu/grid.Rc) * self._psi
        t_wind = (np.diff(0.5*grid.Re**2) / (grid.Rc * v_DW)).min()

        return self._tol * min(t_visc, t_wind)

    def __call__(self, dt, disc, tracers=[], adv=[]):
        """Compute one step of the evolution equation
        args:
            dt      : time-step
            disc    : disc we are updating
            tracers : Tracer species to update. Should be a list of arrays with
                      shape = [*, disc.Ncells].
            adv     : Tracers that are advected but not removed due to mass-loss
        """
        self._setup_grid(disc.grid)
        self._init_fluxes_visc(disc)
        self._init_fluxes_wind(disc, dt)

        f = self._fluxes()
        Sigma_new = disc.Sigma + dt * (f - self._s_wind)

        # The wind removes gas only. Gas tracers are mass fractions of the
        # total Sigma, so they are lost at a rate s_wind * t * Sigma/Sigma_G,
        # i.e. faster than s_wind * t by s_wind * t * eps/(1-eps).
        Sigma = disc.Sigma
        eps = np.where(Sigma > 0, 1 - disc.Sigma_G/np.where(Sigma > 0, Sigma, 1), 0)
        s_extra = self._s_wind * eps / np.maximum(1 - eps, 1e-300)

        for t in tracers: # Tracers advected + removed
            if t is None: continue
            t[:] += dt*(self._tracer_fluxes(t) - t*(f + s_extra)) / (Sigma_new + 1e-300)

        for t in adv: # Tracers advected only (retained when gas is lost)
            if t is None: continue
            t[:] += dt*(self._tracer_fluxes(t) - t*(f - self._s_wind)) / (Sigma_new + 1e-300)
        
        disc.Sigma[:] = Sigma_new

class LBP_Solution(object):
    """Analytical solution for the evolution of an accretion disc,

    Lynden-Bell & Pringle, (1974).

    args:
        M     : Disc mass
        rc    : Critical radius at t=0
        n_c   : viscosity at rc
        gamma : radial dependence of nu, default=1
    """

    def __init__(self, M, rc, nuc, gamma=1):
        self._rc = rc
        self._nuc = nuc
        self._tc = rc * rc / (3 * (2 - gamma) ** 2 * nuc)

        self._Sigma0 = M * (2 - gamma) / (2 * np.pi * rc ** 2)
        self._gamma = gamma

    def __call__(self, R, t):
        """Surface density at R and t"""
        tt = t / self._tc + 1
        X = R / self._rc
        Xg = X ** - self._gamma
        ft = tt ** ((-2.5 + self._gamma) / (2 - self._gamma))

        return self._Sigma0 * ft * Xg * np.exp(- Xg * X * X / tt)

class TaboneSolution(object):
    """Analytical solution for the evolution of a hybrid viscous and
    wind-driven accretion disc,

    Tabone et al. (2022)

    args:
        M      : Disc mass
        rc     : Critical radius at t=0
        nuc    : Total effective viscosity at rc, (1 + psi_DW) * nu_SS, where
                 nu_SS is the turbulent viscosity (disc.nu in HybridWindModel)
        psi_DW : Ratio of wind-driven to viscous alpha, alpha_DW / alpha_SS
        lambda_DW : Magnetic lever arm parameter (sets the mass-loss efficiency)
    """

    def __init__(self, M, rc, nuc, psi_DW, lambda_DW=3):
        from scipy.special import gamma as gamma_fun
        self._rc = rc
        self._nuc = nuc
        self._tc = rc * rc / (3 * nuc)

        self._psi = psi_DW
        
        C = 4*psi_DW/((lambda_DW-1)*(1+psi_DW)**2)
        self._xi = 0.25*(1 + psi_DW) * (np.sqrt(1 + C)-1) 
        self._Sigma0 = M / (2 * np.pi * rc ** 2 * gamma_fun(1 + self._xi))

    def __call__(self, R, t):
        """Surface density at R and t"""
        tt = t / ((1 + self._psi)*self._tc) + 1
        X = R / (self._rc*tt)
        Xg = X ** -(1-self._xi)
        ft = tt ** -(2.5 + self._xi + self._psi/2)

        return self._Sigma0 * ft * Xg * np.exp(- X)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from .disc import AccretionDisc
    from .grid import Grid
    from .constants import AU, Msun
    from .eos import LocallyIsothermalEOS
    from .star import SimpleStar

    alpha = 5e-3

    M = 1e-2 * Msun
    Rd = 30.
    T0 = (2 * np.pi)

    grid = Grid(0.1, 1000, 1000, spacing='natural')
    star = SimpleStar()
    eos = LocallyIsothermalEOS(star, 1 / 30., -0.25, alpha)
    eos.set_grid(grid)

    nud = np.interp(Rd, grid.Rc, eos.nu)
    sol = LBP_Solution(M, Rd, nud, 1)

    Sigma = sol(grid.Rc, 0.)

    disc = AccretionDisc(grid, star, eos, Sigma)

    visc = ViscousEvolutionFV()

    plt.figure()

    # Integrate to given times
    times = np.array([0, 1e4, 1e5, 1e6, 3e6]) * T0

    t = 0
    n = 0
    for ti in times:
        while t < ti:
            dt = visc.max_timestep(disc)
            dti = min(dt, ti - t)

            visc(dti, disc)

            t = min(t + dt, ti)
            n += 1

            if (n % 1000) == 0:
                print('Nstep: {}'.format(n))
                print('Time: {} yr'.format(t / (2 * np.pi)))
                print('dt: {} yr'.format(dt / (2 * np.pi)))

        l, = plt.loglog(grid.Rc, disc.Sigma / AU ** 2)
        l, = plt.loglog(grid.Rc, sol(grid.Rc, t) / AU ** 2,
                        c=l.get_color(), ls='--')

    plt.xlabel(r'$R\,[\mathrm{au}]$')
    plt.ylabel(r'$\Sigma\,[\mathrm{g\,cm}]^{-2}$')

    psi_DW = 1
    sol = TaboneSolution(M / AU**2, Rd, (1 + psi_DW)*nud, psi_DW=psi_DW)

    Sigma = sol(grid.Rc, 0.)

    disc = AccretionDisc(grid, star, eos, Sigma)

    visc = HybridWindModel(psi_DW=psi_DW)

    plt.figure()

    # Integrate to given times
    times = np.array([0, 1e4, 1e5, 1e6, 3e6]) * T0
    t = 0
    n = 0
    for ti in times:
        while t < ti:
            dt = visc.max_timestep(disc)
            dti = min(dt, ti - t)

            visc(dti, disc)

            t = min(t + dt, ti)
            n += 1

            if (n % 1000) == 0:
                print('Nstep: {}'.format(n))
                print('Time: {} yr'.format(t / (2 * np.pi)))
                print('dt: {} yr'.format(dt / (2 * np.pi)))

        l, = plt.loglog(grid.Rc, disc.Sigma)
        l, = plt.loglog(grid.Rc, sol(grid.Rc, t),
                        c=l.get_color(), ls='--')

    plt.xlabel(r'$R\,[\mathrm{au}]$')
    plt.ylabel(r'$\Sigma\,[\mathrm{g\,cm}]^{-2}$')
    plt.show()


    plt.show()
