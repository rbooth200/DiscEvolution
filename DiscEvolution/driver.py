# driver.py
#
# Author: R. Booth
# Date: 17 - Nov - 2016
#
# Combined model for dust, gas and chemical evolution
################################################################################
from __future__ import print_function
import numpy as np
import os
import FRIED.photorate as photorate
from .photoevaporation import FixedExternalEvaporation
from .constants import yr
from .internal_photo import TransitionDisc

from . import io

class DiscEvolutionDriver(object):
    """Driver class for full evolution model.

    Required Arguments:
        disc : Disc model to update

    Optional Physics update:
        gas       : Update due to gas effects, i.e. Viscous evolution
        dust      : Update the dust, i.e. radial drift
        diffusion : Seperate diffusion update
        chemistry : Solver for the chemical evolution

    Note: Diffusion is usually handled in the dust dynamics module

    Other options:
        t0  : Starting time, default = 0.
    """
    def __init__(self, disc, gas=None, dust=None, diffusion=None, chemistry=None, ext_photoevaporation=None, int_photoevaporation=None, t0=0., t_out=None):

        self._disc = disc

        self._gas       = gas
        self._dust      = dust
        self._diffusion = diffusion
        self._chemistry = chemistry
        self.photoevap = ext_photoevaporation
        self._internal_photo = int_photoevaporation

        self._t = t0
        if t0>0.:
            self._output_times = list(t_out[t_out <= t0/yr])
        else:
            self._output_times = []
        self._nstep = 0
        self._nsincehole = 0

    def __call__(self, tmax):
        """Evolve the disc for a single timestep

        args:
            dtmax : Upper limit to time-step

        returns:
            dt : Time step taken
        """
        disc = self._disc

        # Compute the maximum time-step
        dt = tmax - self.t
        if self._gas:
            dt = min(dt, self._gas.max_timestep(self._disc))
        if self._dust:
            v_visc = self._gas.viscous_velocity(disc, Sigma = disc.Sigma_G)
            dt = min(dt, self._dust.max_timestep(self._disc, v_visc))
            if self._dust._diffuse:
                dt = min(dt, self._dust._diffuse.max_timestep(self._disc))
        if self._diffusion:
            dt = min(dt, self._diffusion.max_timestep(self._disc))
        
        '''# If we are not using the timescale method of removal, we need to limit the time step based on photoevaporation
        if (self.photoevap is not None and not isinstance(self.photoevap,FixedExternalEvaporation)): # For FRIED photoevaporation
            if (not isinstance(self.photoevap.FRIED_Rates,photorate.FRIED_2DM) and not isinstance(self.photoevap.FRIED_Rates,photorate.FRIED_2DM400M)): # For density determined photoevaporation
                (dM_dot, dM_gas) = self.photoevap.optically_thin_weighting(disc)
                Dt = dM_gas[(dM_dot>0)] / dM_dot[(dM_dot>0)]
                Dt_min = np.min(Dt)
                dt = min(dt,Dt_min)'''
        """if self._internal_photo:
            dt = min(dt, self._internal_photo.get_dt(self._disc, dt))"""
        
        gas_chem, ice_chem = None, None
        dust = None
        try:
            gas_chem = disc.chem.gas.data
            ice_chem = disc.chem.ice.data
        except AttributeError:
            pass
        
        if self._dust:
            self._dust(dt, disc,
                       gas_tracers=gas_chem,
                       dust_tracers=ice_chem, v_visc=v_visc)

        try:
            gas_chem = disc.chem.gas.data
            ice_chem = disc.chem.ice.data
        except AttributeError:
            pass
        try:
            dust = disc.dust_frac
        except AttributeError:
            pass

        # Do Advection-diffusion update
        if self._gas:
            self._gas(dt, disc, [dust, gas_chem, ice_chem])
        
        if self._diffusion:
            if gas_chem is not None:
                gas_chem[:] += dt * self._diffusion(disc, gas_chem)
            if ice_chem is not None:
                ice_chem[:] += dt * self._diffusion(disc, ice_chem)
            if dust is not None:
                dust[:] += dt * self._diffusion(disc, dust)

        # Do external photoevaporation
        if self.photoevap:
            self.photoevap(disc, dt, self.t)

        # Do internal photoevaporation
        if self._internal_photo:
            self._internal_photo(disc, dt/yr, self.photoevap)
        
        # Pin the values to >= 0:
        disc.Sigma[:] = np.maximum(disc.Sigma, 0)
        try:
            disc.dust_frac[:] = np.maximum(disc.dust_frac, 0)
            disc.dust_frac[:] /= np.maximum(disc.dust_frac.sum(0), 1.0)
        except AttributeError:
            pass
        try:
            disc.chem.gas.data[:] = np.maximum(disc.chem.gas.data, 0)
            disc.chem.ice.data[:] = np.maximum(disc.chem.ice.data, 0)
        except AttributeError:
            pass
            
        # Chemistry
        if self._chemistry:
            rho = disc.midplane_gas_density
            eps = disc.dust_frac.sum(0)
            grain_size = disc.grain_size[-1]
            T = disc.T

            self._chemistry.update(dt, T, rho, eps, disc.chem, 
                                   grain_size=grain_size)

            # If we have dust, we should update it now the ice fraction has
            # changed
            disc.update_ices(disc.chem.ice)

        # Now we should update the auxillary properties, do grain growth etc
        disc.update(dt)

        # Update the internal hole and check whether we need to switch the mass loss prescription
        if self._internal_photo:                    # If doing internal photoevaporation
            if not self._internal_photo._switch:      # If the inner disc is not already thin or Mdot low then continue
                if self._internal_photo._Hole:      # If there is a hole, update its properties 
                    R_hole, Sigma_hole, N_hole = self._internal_photo.get_Rhole(disc, self.photoevap)
                if self._internal_photo._switch:      # If the hole is large enough that inner disc thin or Mdot low, switch internal photoevaporation to TD
                    if (self._internal_photo._swiTyp == "Thin"):
                        print("Column density to hole has fallen to N = {} < 10^22 g cm^-2".format(N_hole))
                    elif (self._internal_photo._swiTyp == "loMd"):
                        print("Mass loss rate has fallen below that for a transition disc.")
                    self._internal_photo = TransitionDisc(disc, R_hole, Sigma_hole, N_hole)

        self._t += dt
        self._nstep += 1
        return dt

    @property
    def disc(self):
        return self._disc

    @property
    def t(self):
        return self._t

    @property
    def num_steps(self):
        return self._nstep

    @property
    def gas(self):
        return self._gas
    @property
    def dust(self):
        return self._dust
    @property
    def diffusion(self):
        return self._diffusion
    @property
    def chemistry(self):
        return self._chemistry
    @property
    def photoevaporation(self):
        return self.photoevap
    @property
    def photoevaporation_internal(self):
        return self._internal_photo

    def dump_ASCII(self, filename):
        """Write the current state to a file, including header information"""

        # Put together a header containing information about the physics
        # included
        head = ''
        if self._gas:
            head += self._gas.ASCII_header() + '\n'
        if self._dust:
            head += self._dust.ASCII_header() + '\n'
        if self._diffusion:
            head += self._diffusion.ASCII_header() + '\n'
        if self._chemistry:
            head += self._chemistry.ASCII_header() + '\n'
        if self.photoevap:
            head += self.photoevap.ASCII_header() + '\n'
        if self._internal_photo:
            head += self._internal_photo.ASCII_header() + '\n'

        # Write it all to disc
        io.dump_ASCII(filename, self._disc, self.t, head)

    def dump_hdf5(self, filename):
        """Write the current state in HDF5 format, with header information"""
        headers = []
        if self._gas:            headers.append(self._gas.HDF5_attributes())
        if self._dust:           headers.append(self._dust.HDF5_attributes())
        if self._diffusion:      headers.append(self._diffusion.HDF5_attributes())
        if self._chemistry:      headers.append(self._chemistry.HDF5_attributes())
        if self.photoevap:       headers.append(self.photoevap.HDF5_attributes())
        if self._internal_photo: headers.append(self._internal_photo.HDF5_attributes())

        io.dump_hdf5(filename, self._disc, self.t, headers)


if __name__ == "__main__":
    from .star import SimpleStar
    from .grid import Grid
    from .eos  import IrradiatedEOS
    from .viscous_evolution import ViscousEvolution
    from .dust import DustGrowthTwoPop, SingleFluidDrift
    from .opacity import Zhu2012, Tazzari2016
    from .diffusion import TracerDiffusion
    from .chemistry import TimeDepCOChemOberg, SimpleCOAtomAbund
    from .constants import Msun, AU
    from .disc_utils import mkdir_p


    import matplotlib.pyplot as plt


    alpha = 1e-3
    Mdot  = 1e-8
    Rd    = 100.

    #kappa = Zhu2012
    kappa = Tazzari2016()
    
    N_cell = 250
    R_in  = 0.1
    R_out = 500.

    yr = 2*np.pi

    output_dir = 'test_DiscEvo'
    output_times = np.arange(0, 4) * 1e6 * yr
    plot_times = np.array([0, 1e4, 1e5, 5e5, 1e6, 3e6])*yr

    # Setup the initial conditions
    Mdot *= (Msun / yr) / AU**2
    
    grid = Grid(R_in, R_out, N_cell, spacing='natural')
    star = SimpleStar(M=1, R=2.5, T_eff=4000.)

    # Initial guess for Sigma:
    R = grid.Rc
    Sigma = (Mdot / (0.1 * alpha * R**2 * star.Omega_k(R))) * np.exp(-R/Rd)

    # Iterate until constant Mdot
    eos = IrradiatedEOS(star, alpha, kappa=kappa)
    eos.set_grid(grid)
    eos.update(0, Sigma)
    for i in range(100):
        Sigma = 0.5 * (Sigma + (Mdot / (3 * np.pi * eos.nu)) * np.exp(-R/Rd))
        eos.update(0, Sigma)

    # Create the disc object
    disc = DustGrowthTwoPop(grid, star, eos, 0.01, Sigma=Sigma)

    # Setup the chemistry
    chemistry = TimeDepCOChemOberg(a=1e-5)
    
    # Setup the dust-to-gas ratio from the chemistry
    solar_abund = SimpleCOAtomAbund(N_cell)
    solar_abund.set_solar_abundances()

    # Iterate ice fractions to get the dust-to-gas ratio:
    for i in range(10):
        chem = chemistry.equilibrium_chem(disc.T,
                                          disc.midplane_gas_density,
                                          disc.dust_frac.sum(0),
                                          solar_abund)
        disc.initialize_dust_density(chem.ice.total_abund)
    disc.chem = chem

    # Setup the dynamics modules:
    gas  = ViscousEvolution()
    dust = SingleFluidDrift(TracerDiffusion())

    evo = DiscEvolutionDriver(disc, gas=gas, dust=dust, chemistry=chemistry)


    # Setup the IO controller
    IO = io.Event_Controller(save=output_times, plot=plot_times)

    # Run the model!
    while not IO.finished():
        ti = IO.next_event_time()
        while evo.t < ti:
            dt = evo(ti)

            if (evo.num_steps % 1000) == 0:
                print('Nstep: {}'.format(evo.num_steps))
                print('Time: {} yr'.format(evo.t / yr))
                print('dt: {} yr'.format(dt / yr))

        if IO.check_event(evo.t, 'save'):
            from .disc_utils import mkdir_p
            mkdir_p(output_dir)

            snap_name = 'disc_{:04d}.dat'.format(IO.event_number('save'))
            evo.dump_ASCII(os.path.join(output_dir, snap_name))

            snap_name = 'disc_{:04d}.h5'.format(IO.event_number('save'))
            evo.dump_hdf5(os.path.join(output_dir, snap_name))

        if IO.check_event(evo.t, 'plot'):
            err_state = np.seterr(all='warn')

            print('Nstep: {}'.format(evo.num_steps))
            print('Time: {} yr'.format(evo.t / (2 * np.pi)))
            plt.subplot(321)
            l, = plt.loglog(grid.Rc, evo.disc.Sigma_G)
            plt.loglog(grid.Rc, evo.disc.Sigma_D.sum(0), '--', c=l.get_color())
            plt.xlabel('$R$')
            plt.ylabel('$\Sigma_\mathrm{G, D}$')

            plt.subplot(322)
            plt.loglog(grid.Rc, evo.disc.dust_frac.sum(0))
            plt.xlabel('$R$')
            plt.ylabel('$\epsilon$')
            plt.subplot(323)
            plt.loglog(grid.Rc, evo.disc.Stokes()[1])
            plt.xlabel('$R$')
            plt.ylabel('$St$')
            plt.subplot(324)
            plt.loglog(grid.Rc, evo.disc.grain_size[1])
            plt.xlabel('$R$')
            plt.ylabel('$a\,[\mathrm{cm}]$')

            plt.subplot(325)
            gCO = evo.disc.chem.gas.atomic_abundance()
            sCO = evo.disc.chem.ice.atomic_abundance()
            gCO.data[:] /= solar_abund.data
            sCO.data[:] /= solar_abund.data
            c = l.get_color()
            plt.semilogx(grid.Rc, gCO['C'], '-', c=c, linewidth=1)
            plt.semilogx(grid.Rc, gCO['O'], '-', c=c, linewidth=2)
            plt.semilogx(grid.Rc, sCO['C'], ':', c=c, linewidth=1)
            plt.semilogx(grid.Rc, sCO['O'], ':', c=c, linewidth=2)
            plt.xlabel('$R\,[\mathrm{au}}$')
            plt.ylabel('$[X]_\mathrm{solar}$')

            plt.subplot(326)
            plt.semilogx(grid.Rc, gCO['C'] / gCO['O'], '-', c=c)
            plt.semilogx(grid.Rc, sCO['C'] / sCO['O'], ':', c=c)
            plt.xlabel('$R\,[\mathrm{au}}$')
            plt.ylabel('$[C/O]_\mathrm{solar}$')

            np.seterr(**err_state)

        IO.pop_events(evo.t)

    if len(plot_times) > 0:
        plt.show()


