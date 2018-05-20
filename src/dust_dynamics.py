# dust_dynamics.py
#
# Author: R. Booth
# Date: 17 - Nov - 2016
#
# Combined model for the evolution of gas, dust and chemical species in a
# viscously evolving disc.
################################################################################
from __future__ import print_function
import numpy as np
import os

from diffusion import TracerDiffusion
from dust import SingleFluidDrift
from ViscousEvolution import ViscousEvolution
from disc_utils import mkdir_p


class DustDynamicsModel(object):
    """
    """

    def __init__(self, disc,
                 diffusion=False, radial_drift=False, viscous_evo=False,
                 evaporation=False, settling=False,
                 Sc=1, t0=0):

        self._disc = disc

        self._visc = None
        if viscous_evo:
            bound = 'power_law'
            # Power law extrapolation fails with zero-density, use simple
            # boundary condition instead
            if evaporation:
                bound = 'Zero'

            self._visc = ViscousEvolution(boundary=bound)

        self._diffusion = None
        if diffusion:
            diffusion = TracerDiffusion(Sc)

        # Diffusion can be handled by the radial drift object, without dust we
        # include it ourself.
        self._radial_drift = None
        if radial_drift:
            self._radial_drift = SingleFluidDrift(diffusion, settling)
        else:
            self._diffusion = diffusion

        self._evaporation = False
        if evaporation:
            self._evaporation = evaporation

        self._t = t0

    def __call__(self, tmax):
        """Evolve the disc for a single timestep

        args:
            dtmax : Upper limit to time-step

        returns:
            dt : Time step taken
        """
        # Compute the maximum time-step
        dt = tmax - self.t
        if self._visc:
            dt = min(dt, self._visc.max_timestep(self._disc))
        if self._radial_drift:
            dt = min(dt, self._radial_drift.max_timestep(self._disc))

        disc = self._disc

        # Do Advection-diffusion update
        if self._visc:
            dust = None
            size = None
            try:
                dust = disc.dust_frac
                size = disc.grain_size
            except AttributeError:
                pass
            self._visc(dt, disc, [dust, size])

        if self._radial_drift:
            self._radial_drift(dt, disc)

        # Pin the values to >= 0:
        disc.Sigma[:] = np.maximum(disc.Sigma, 0)
        disc.dust_frac[:] = np.maximum(disc.dust_frac, 0)

        # Apply any photo-evaporation:
        if self._evaporation:
            self._evaporation(disc, dt)

        # Now we should update the auxillary properties, do grain growth etc
        disc.update(dt)

        self._t += dt
        return dt

    @property
    def disc(self):
        return self._disc

    @property
    def t(self):
        return self._t

    def dump(self, filename):
        """Write the current state to a file, including header information"""

        # Put together a header containing information about the physics
        # included
        head = self.disc.header() + '\n'
        if self._visc:
            head += self._visc.header() + '\n'
        if self._radial_drift:
            head += self._radial_drift.header() + '\n'
        if self._diffusion:
            head += self._diffusion.header() + '\n'
        try:
            head += self._chem.header() + '\n'
        except AttributeError:
            pass

        with open(filename, 'w') as f:
            print(head, '# time: {}yr'.format(self.t / (2 * np.pi)))

            # Construct the list of variables that we are going to print
            Ncell = self.disc.Ncells

            Ndust = 0
            try:
                Ndust = self.disc.dust_frac.shape[0]
            except AttributeError:
                pass

            head = '# R Sigma T'
            for i in range(Ndust):
                head += ' epsilon[{}]'.format(i)
            for i in range(Ndust):
                head += ' a[{}]'.format(i)
            chem = None
            try:
                chem = self.disc.chem
                for k in chem.gas:
                    head += ' {}'.format(k)
                for k in chem.ice:
                    head += ' s{}'.format(k)
            except AttributeError:
                pass

            print(head)

            R, Sig, T = self.disc.R, self.disc.Sigma, self.disc.T
            for i in range(Ncell):
                print(R[i], Sig[i], T[i])
                for j in range(Ndust):
                    print(self.disc.dust_frac[j, i])
                for j in range(Ndust):
                    print(self.disc.grain_size[j, i])
                if chem:
                    for k in chem.gas:
                        print(chem.gas[k][i])
                    for k in chem.ice:
                        print(chem.ice[k][i])
                print()


class IO_Controller(object):
    """Handles time and book-keeping for when to dump data to file / screen.

    args:
        t_print  : times to print to screen
        t_save   : times to save files
        t_inject : times to inject planets
    """

    def __init__(self, t_print=[], t_save=[], t_inject=[]):
        self._tprint = sorted(t_print)
        self._tsave = sorted(t_save)
        self._tinject = sorted(t_inject)

        self._nsave = 0
        self._nprint = 0

    @property
    def t_next(self):
        """Next time to print or save"""
        t_next = np.inf
        if self._tprint: t_next = min(t_next, self._tprint[0])
        if self._tsave: t_next = min(t_next, self._tsave[0])
        if self._tinject: t_next = min(t_next, self._tinject[0])

        return t_next

    def need_print(self, t):
        """Check whether we need to print to screen"""
        if self._tprint: return self._tprint[0] <= t
        return False

    def need_save(self, t):
        """Check whether we need to print to screen"""
        if self._tsave: return self._tsave[0] <= t
        return False

    def need_injection(self, t):
        """Check whether we need to inject planets"""
        if self._tinject: return self._tinject[0] <= t
        return False

    @property
    def nprint(self):
        return self._nprint

    @property
    def nsave(self):
        return self._nsave

    def pop_times(self, t):
        """Remove any elapsed times from save & print lists"""
        while self._tprint and self._tprint[0] <= t:
            self._tprint.pop(0)
            self._nprint += 1

        while self._tsave and self._tsave[0] <= t:
            self._tsave.pop(0)
            self._nsave += 1

        while self._tinject and self._tinject[0] <= t:
            self._tinject.pop(0)

    @property
    def finished(self):
        return not (self._tprint or self._tsave or self._tinject)


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    from grid import Grid
    from eos import LocallyIsothermalEOS, IrradiatedEOS
    from star import SimpleStar
    from dust import DustGrowthTwoPop
    from constants import Msun, AU, Omega0
    from photoevaporation import FixedExternalEvaportation

    np.seterr(all='ignore')
    np.seterr(invalid='raise')

    models = {}
    N = 1
    for Mdot in [1e-8, 1e-9]:
        alpha = 1e-3
        for Rc in [50, 100, 200]:
            model = {'alpha': alpha, 'R_d': Rc,
                     'Mdot': Mdot,
                     'name': 'Rc_{}'.format(Rc)}
            models['{}'.format(N)] = model
            N += 1
        Rc = 100
        for alpha in [5e-4, 1e-3, 5e-3, 1e-2]:
            model = {'alpha': alpha, 'R_d': Rc,
                     'Mdot': Mdot,
                     'name': 'alpha_{}'.format(alpha)}
            models['{}'.format(N)] = model
            N += 1

    try:
        model = models[sys.argv[1]]
    except IndexError:
        model = models['1']

    # Model values
    Mdot = model['Mdot']
    alpha = model['alpha']
    Rd = model['R_d']

    R_in = 0.1
    R_out = 500

    N_cell = 250

    T0 = 2 * np.pi

    Mdot *= Msun / (2 * np.pi)
    Mdot /= AU ** 2

    eos_type = 'irradiated'
    #eos_type = 'isothermal'

    DIR = os.path.join('temp', eos_type,
                       model['name'], 'Mdot_{}'.format(model['Mdot']))
    mkdir_p(DIR)

    with open(os.path.join(DIR, 'model.dat'), 'w') as f:
        for k in model:
            print(k, model[k])

    # Initialize the disc model
    grid = Grid(R_in, R_out, N_cell, spacing='natural')
    star = SimpleStar(M=1, R=2.5, T_eff=4000.)

    eos = LocallyIsothermalEOS(star, 1 / 30., -0.25, alpha)
    eos.set_grid(grid)
    Sigma = (Mdot / (3 * np.pi * eos.nu)) * np.exp(-grid.Rc / Rd)
    if eos_type != 'isothermal':
        # Use a non accreting model to guess the initial density
        eos = IrradiatedEOS(star, alpha, tol=1e-3, accrete=False)
        eos.set_grid(grid)
        eos.update(0, Sigma)

        # Now do a new guess for the surface density and initial eos.
        Sigma = (Mdot / (3 * np.pi * eos.nu)) * np.exp(-grid.Rc / Rd)

        eos = IrradiatedEOS(star, alpha, tol=1e-3)
        eos.set_grid(grid)
        eos.update(0, Sigma)

    # Initialize the complete disc object
    disc = DustGrowthTwoPop(grid, star, eos, 0.01, Sigma=Sigma, feedback=True)

    # Setup the dust-dynamical model
    evo = DustDynamicsModel(disc,
                            viscous_evo=True,
                            radial_drift=True,
                            diffusion=True,
                            evaporation=FixedExternalEvaportation(Mdot=1e-8))

    # Solve for the evolution
    print_times = np.array([0, 1e5, 5e5, 1e6, 3e6]) * T0
    output_times = np.arange(0, 3e6 + 1e3, 1e4) * T0

    IO = IO_Controller(t_print=print_times, t_save=output_times)

    n = 0
    while not IO.finished:
        ti = IO.t_next
        while evo.t < ti:
            dt = evo(ti)

            n += 1
            if (n % 1000) == 0:
                print('Nstep: {}'.format(n))
                print('Time: {} yr'.format(evo.t / (2 * np.pi)))
                print('dt: {} yr'.format(dt / (2 * np.pi)))

        if IO.need_save(evo.t) and False:
            evo.dump(os.path.join(DIR, 'disc_{:04d}.dat'.format(IO.nsave)))

        if IO.need_print(evo.t):
            err_state = np.seterr(all='warn')

            print('Nstep: {}'.format(n))
            print('Time: {} yr'.format(evo.t / (2 * np.pi)))
            plt.subplot(221)
            l, = plt.loglog(grid.Rc, evo.disc.Sigma_G)
            plt.loglog(grid.Rc, evo.disc.Sigma_D.sum(0), '--', c=l.get_color())
            plt.xlabel('$R$')
            plt.ylabel('$\Sigma_\mathrm{G, D}$')

            plt.subplot(222)
            l, = plt.loglog(grid.Rc, evo.disc.dust_frac.sum(0))
            plt.xlabel('$R$')
            plt.ylabel('$\epsilon$')
            plt.subplot(223)
            l, = plt.loglog(grid.Rc, evo.disc.Stokes()[1])
            plt.xlabel('$R$')
            plt.ylabel('$St$')
            plt.subplot(224)
            l, = plt.loglog(grid.Rc, evo.disc.grain_size[1])
            plt.xlabel('$R$')
            plt.ylabel('$a\,[\mathrm{cm}]$')

            np.seterr(**err_state)

        IO.pop_times(evo.t)

    plt.show()
