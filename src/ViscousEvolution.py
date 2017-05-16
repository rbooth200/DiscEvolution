# ViscousEvolution.py
#
# Author: R. Booth
# Date: 16 - Nov - 2016
#
# Contains classes for solving the viscous evolution equations.
################################################################################
import numpy as np


class ViscousEvolution(object):
    '''Solves the 1D viscous evoluation equation.

    This class handles the inclusion of dust species in the one-fluid 
    approximation. The total surface density (Sigma = Sigma_G + Sigma_D) is 
    updated under the action of viscous forces, which are taken to act only
    on the gas phase species.

    Can optionally update tracer species.

    args:
       tol       : Ratio of the time-step to the maximum stable one. 
                   Default = 0.5
       boundary  : Type of external boundary condition:
                     'Zero'      : Zero torque boundary
                     'power_law' : Power-law extrapolation
                     'Mdot'      : Constant Mdot, same as inner.
    '''
    def __init__(self, tol=0.5, boundary='power_law'):
        self._tol = tol
        self._bound = boundary

    def header(self):
        '''header'''
        return '# {} tol: {}'.format(self.__class__.__name__, self._tol)

    def _setup_grid(self, grid):
        '''Compute the grid factors'''
        self._X   = 2 * np.sqrt(grid.Rc)
        self._dXe = 2 * np.diff(np.sqrt(grid.Re))
        self._dXc = 2 * np.diff(np.sqrt(grid.Rce))
        self._RXdXe = grid.Rc * self._X * self._dXe

    def _init_fluxes(self, disc):
        '''Cache the important variables needed for the fluxes'''
        nuX = disc.nu * self._X

        S = np.zeros(len(nuX)+2, dtype='f8')
        S[1:-1] = disc.Sigma_G * nuX
        
        S[0] = S[1] * self._X[0] / self._X[1]
        if self._bound == 'Zero':
            S[-1] = 0
        elif self._bound == 'power_law':
            S[-1] = S[-2]**2 / S[-3]
        elif self._bound == 'Mdot':
            S[-1] = S[-2] * self._X[-2] / self._X[-1]
        else:
            raise ValueError("Error boundary type not recognised")

        self._dS = np.diff(S) / self._dXc
        
    def _fluxes(self):
        '''Compute the mass fluxes for the viscous evolution equations.
        
        Gas update from Bath & Pringle (1981)
        '''
        return 3. * np.diff(self._dS) / self._RXdXe

    def _tracer_fluxes(self, tracers):
        '''Compute fluxes of a tracer.

        Uses the viscous update to compute the flux of  Sigma*tracers,
        divide by the updated Sigma to get the new tracer value.
        '''
        shape = tracers.shape[:-1] + (tracers.shape[-1]+2,)
        s = np.zeros(shape, dtype='f8')
        s[...,1:-1] = tracers
        s[...,0] = s[...,1] ; s[...,-1] = s[...,-2]

        # Upwind the tracer density 
        ds = self._dS * np.where(self._dS <= 0, s[...,:-1], s[...,1:])
        
        # Compute the viscous update
        return  3. * np.diff(ds) / self._RXdXe

    def max_timestep(self, disc):
        '''Courant limited time-step'''
        grid = disc.grid
        nu   = disc.nu
        
        dXe2 = np.diff(2*np.sqrt(grid.Re))**2
        
        tc = ((dXe2 * grid.Rc) / (2 * 3 * nu)).min()
        return self._tol * tc
        
    
    def __call__(self, dt, disc, tracers=[]):
        '''Compute one step of the viscous evolution equation
        args:
            dt      : time-step
            disc    : disc we are updating
            tracers : Tracer species to update. Should be a list of arrays with
                      shape = [*, disc.Ncells].
        '''
        self._setup_grid(disc.grid)
        self._init_fluxes(disc)

        Sigma_new = disc.Sigma + dt * self._fluxes()

        for t in tracers:
            if t is None: pass
            t[:] += dt * self._tracer_fluxes(t) / (Sigma_new + 1e-300)

        disc.Sigma[:] = Sigma_new



class LBP_Solution(object):
    '''Analytical solution for the evolution of an accretion disc,
    
    Lynden-Bell & Pringle, (1974).

    args:
        M     : Disc mass
        rc    : Critical radius at t=0
        n_c   : viscosity at rc
        gamma : radial dependence of nu, default=1
    '''
    def __init__(self, M, rc, nuc, gamma=1):
        self._rc  = rc
        self._nuc = nuc
        self._tc  = rc*rc / (3*(2-gamma)**2 * nuc)

        self._Sigma0 = M * (2-gamma) / (2 * np.pi * rc**2)
        self._gamma = gamma
        
    def __call__(self, R, t):
        '''Surface density at R and t'''
        tt = t / self._tc + 1
        X = R/self._rc
        Xg = X**- self._gamma
        ft = tt ** ((-2.5 + self._gamma)/(2-self._gamma))
        
        return self._Sigma0 * ft * Xg * np.exp( - Xg*X*X / tt)
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from disc import AccretionDisc
    from grid import Grid
    from constants import AU, Msun
    from eos import LocallyIsothermalEOS
    from star import SimpleStar

    alpha = 5e-3

    M = 1e-2 * Msun
    Rd = 30.
    T0 = (2*np.pi)

    grid = Grid(0.1, 1000, 1000, spacing='natural')
    star = SimpleStar()
    eos = LocallyIsothermalEOS(star, 1/30., -0.25, alpha)
    eos.set_grid(grid)

    nud = np.interp(Rd, grid.Rc, eos.nu)
    sol = LBP_Solution(M, Rd, nud, 1)
    
    Sigma = sol(grid.Rc, 0.)

    disc = AccretionDisc(grid, star, eos, Sigma)

    visc = ViscousEvolution()
    
    # Integrate to given times
    times = np.array([0, 1e4, 1e5, 1e6, 3e6 ]) * T0

    t = 0
    n = 0
    for ti in times:
        while t < ti:
            dt = visc.max_timestep(disc)
            dti = min(dt, ti-t)

            visc(dti, disc)

            t = min(t+dt, ti)
            n += 1

            if (n % 1000) == 0:
                print 'Nstep: {}'.format(n)
                print 'Time: {} yr'.format(t/(2*np.pi))
                print 'dt: {} yr'.format(dt / (2*np.pi))


        l, = plt.loglog(grid.Rc, disc.Sigma / AU**2)
        l, = plt.loglog(grid.Rc, sol(grid.Rc, t) / AU**2, l.get_color() + '--')

    plt.xlabel('$R\,[\mathrm{au}]$')
    plt.ylabel('$\Sigma\,[\mathrm{g\,cm}]^{-2}$')
    plt.show()
