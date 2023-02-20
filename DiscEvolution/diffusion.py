# diffusion.py
#
# Author: R. Booth
# Date : 14 - Nov - 2016
#
# Classes handling diffusion of different species
################################################################################
from __future__ import print_function
import numpy as np


class TracerDiffusion(object):
    """Diffusion of trace species in a turbulent background.
    args:
        Sc : Schmidt number, ratio of momentum diffusivity to mass diffusivity,
             default=1.
        limit : Whether to limit the diffusive velocity to the sound speed,
                default=False.
    """
    def __init__(self, Sc=1, limit=False):
        self._Sc = Sc
        self._limit = limit

    def ASCII_header(self):
        """Tracer diffusion header"""
        return '# {} Sc: {}, flux_limit: {}'.format(self.__class__.__name__,
                                                    self.Sc, self._limit)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        return (self.__class__.__name__,
                {"Sc" : "{}".format(self.Sc),
                 "flux_limit" : "{}".format(bool(self._limit))}
                )

    @staticmethod
    def _diffusive_flux(disc, eps_i, Sc):
        """Compute the diffusive flux for a tracer
        args:
            disc  : accretion disc object
            eps_i : Concentration of species that is diffusing
            Sc    : Schmidt number

        returns:
            flux_diffuse : diffusive flux at edges (between cells only)
        """
        # Use alpha c_s H because nu has gap structure encoded in it.
        D = disc.alpha * disc.cs * disc.H / Sc
        Sigma_G = disc.Sigma_G

        # Use geometric average to avoid problems at the edge of evaporating
        # regions., where Sigma_G = 0 (and eps_i is ill-defined)
        DSig = D*Sigma_G
        DSig = np.sqrt(np.maximum(DSig[...,1:]*DSig[...,:-1], 0))

        return - DSig * np.diff(eps_i) / disc.grid.dRc

    def _get_Schmidt(self, disc, Sc):
        if Sc is None:
            try:
                Sc = disc.Sc
            except AttributeError:
                Sc = self.Sc
        return Sc

    def max_timestep(self, disc, Sc=None):
        """Courant limited time-step"""
        grid = disc.grid
        # Use alpha c_s H because nu has gap structure encoded in it.
        D = disc.alpha * disc.cs * disc.H / self._get_Schmidt(disc, Sc)

        return (0.25 * np.diff(grid.Re)**2 / D).min()

    def __call__(self, disc, eps_i, Sc=None):
        """Compute the rate of change of the surface density due to diffusion.

        args:
            disc  : Disc object
            eps_i : Concentration of species that is diffusing
            Sc    : Schmidt number. If not provided the value provided by the
                    disc is tried, before falling back to self.Sc
        returns:
            dSigma_i/dt : Rate of change of density of tracer
        """
        Sc = self._get_Schmidt(disc, Sc)

        grid = disc.grid
        Sigma = disc.Sigma
        F = self._diffusive_flux(disc, eps_i, Sc)

        if self._limit:
            max_f = Sigma*eps_i*self._eos.cs+1e-300
            F /= 1 + abs(F)/(0.5*(max_f[1:] + max_f[:-1]))

        F *= grid.Re[1:-1]

        depsdt = np.zeros_like(eps_i)
        depsdt[...,1:]  += F / grid.dRe2[1:]
        depsdt[...,:-1] -= F / grid.dRe2[:-1]

        depsdt /= Sigma + 1e-300
        return depsdt

    @property
    def Sc(self):
        return self._Sc


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from .grid import Grid
    from .eos import LocallyIsothermalEOS
    from .star import SimpleStar
    from .disc import AccretionDisc
    from .constants import Msun, AU

    Mdot  = 1e-8
    alpha = 1e-3

    Mdot *= Msun / (2*np.pi)
    Mdot /= AU**2
    Rd = 100.

    grid = Grid(0.1, 1000, 250, spacing='log')
    star = SimpleStar()
    eos = LocallyIsothermalEOS(star, 1/30., -0.25, alpha)
    eos.set_grid(grid)
    Sigma =  (Mdot / (3 * np.pi * eos.nu))*np.exp(-grid.Rc/Rd)

    disc = AccretionDisc(grid, star, eos, Sigma)

    eps = np.empty([2, grid.Ncells], dtype='f4')
    eps[0] =  0.01 * (1 + np.sin(np.pi*np.log(grid.Rc)))
    eps[1] =  0.01 * (1 + np.cos(np.pi*np.log(grid.Rc)))


    times = np.array([0, 1e2, 1e3, 1e4, 1e5, 1e6, 3e6]) * 2*np.pi

    diffuse = TracerDiffusion()

    dt = 100.0
    t = 0
    n = 0
    for ti in times:
        while t < ti:
            dti = min(dt, ti-t)
            eps += dti * diffuse(disc, eps)

            t = min(t+dti, ti)
            n += 1

            if (n % 1000) == 0:
                print('Nstep: {}'.format(n))
                print('Time: {} yr'.format(t / (2 * np.pi)))
                print('dt: {} yr'.format(dt / (2 * np.pi)))

        print('Nstep: {}'.format(n))
        print('Time: {} yr'.format(t / (2 * np.pi)))
        l, = plt.loglog(grid.Rc, Sigma*eps[0])
        l, = plt.loglog(grid.Rc, Sigma*eps[1], c=l.get_color(), ls='--')

    plt.loglog(grid.Rc, 0.01*Sigma, 'k:')
    plt.xlabel('$R$')
    plt.ylabel('$\Sigma_i$')
    plt.show()
