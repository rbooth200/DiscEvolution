# photoevaportation.py
#
# Author: R. Booth
# Date: 24 - Feb - 2017
#
# Models for photo-evaporation of a disc
###############################################################################
import numpy as np
from .constants import AU, Msun
import FRIED.photorate as photorate

class ExternalPhotoevaporationBase(object):
    """Base class for handling the external photo-evaportation of discs

    Implementations of ExternalPhotoevaporation classes must provide the
    following methods:
        mass_loss_rate(disc)     : returns mass-loss rate from outer edge
                                   (Msun/yr).
        max_size_entrained(dist) : returns the maximum size of dust entrained in
                                   the flow (cm).
    """

    def get_timescale(self, disc, dt):
        """Calculate mass loss rates and mass loss timescales"""
        # Disc densities / masses
        Sigma_G = disc.Sigma_G
        #Sigma_D = disc.Sigma_D
        not_empty = (disc.Sigma_G > 0)
        #outer_cell = np.sum(not_empty) - 1

        # Get the photo-evaporation rates at each cell as if it were the edge:
        Mdot = self.mass_loss_rate(disc,not_empty)
        #amax = self.max_size_entrained(disc)

        # Convert Msun / yr to g / dynamical time and compute mass evaporated:
        dM_evap = Mdot * Msun / (2 * np.pi)
	
	# Mass in gas/dust in each annulus 
        Re = disc.R_edge * AU
        dA = np.pi * (Re[1:] ** 2 - Re[:-1] ** 2)
        dM_gas = Sigma_G * dA
        #dM_dust = Sigma_D * dA
        #print (dM_gas[outer_cell-1:outer_cell+2])
        #dM_gas[outer_cell] *= disc.depletion # Account for depletion in outer cell
        #print (dM_gas[outer_cell-1:outer_cell+2])
        #dM_gas[outer_cell] = disc.outer_mass
        #dM_gas[outer_cell] *= (disc.RD**2 - Re[outer_cell]**2) / (Re[outer_cell+1]**2 - Re[outer_cell]**2)
        dM_gas[disc.i_edge] -= disc.mass_lost      

        # Work out which cells we need to empty of gas & entrained dust
        #dM_tot = dM_gas + (dM_dust * (disc.grain_size <= amax)).sum(0)
        #empty = M_tot < dM_evap
        
        # Work out which cells we need to empty of gas & entrained dust
        dM_tot = dM_gas
        M_tot = np.cumsum(dM_tot[::-1])[::-1]
        Dt_R = dM_gas[not_empty] / dM_evap # Dynamical time for each annulus to be depleted of mass
        Dt_R = np.concatenate((Dt_R, np.zeros(np.size(disc.Sigma_G)-np.size(Dt_R)))) # Append 0 for empty annuli
        dM_evap = np.concatenate((dM_evap, np.zeros(np.size(disc.Sigma_G)-np.size(dM_evap)))) # Append 0 for empty annuli
        Dt = np.cumsum(Dt_R[::-1])[::-1] # Dynamical time to deplete each annulus and those exterior
        #print (Dt)	
        # Return mass loss rate, annulus mass, cumulative mass and cumulative timescale
        return (dM_evap, dM_tot, M_tot, Dt)

    def timescale_remove(self, disc, dt, age):
        (dM_evap, dM_tot, M_tot, Dt) = self.get_timescale(disc,dt)

        # Calculate which cells have times shorter than the timestep and empty
        excess_t = dt - Dt
        empty = (excess_t > 0)
        disc.tot_mass_lost += np.sum(dM_tot[empty])
        disc.Sigma_G[empty] = 0

        # Deal with marginal cell (first for which dt<Dt) as long as entire disc isn't removed
        if (np.sum(empty)<np.size(disc.Sigma_G)):
            half_empty = -(np.sum(empty) + 1) # ID (from end) of half depleted cell

            mass_left = -1.0*excess_t[half_empty] / (Dt[half_empty]-Dt[half_empty+1]) # Work out fraction left in cell
            #disc.outer_mass = dM_tot[half_empty] * depletion
            if (half_empty==disc.i_edge):
                disc.mass_lost += dM_tot[half_empty] * (1.0-mass_left)
            else:
                disc.mass_lost = dM_tot[half_empty] * (1.0-mass_left)
                disc.i_edge = half_empty
            disc.tot_mass_lost += dM_tot[half_empty] * (1.0-mass_left)
            #print (disc.i_edge)

            # Sigma depletion method (doesn't work well for short dt)
            #disc.Sigma_G[half_empty] *= disc.depletion # Adjust density to remaining value            

            # Explicit Edge tracking method
            #Re = disc.R_edge # In AU
            #Rout = Re[1:]
            #Rin = Re[:-1]
            #disc.RD = np.sqrt(Rin[half_empty]**2*(1-depletion) + depletion * Rout[half_empty])

        #disc.Sigma = disc.Sigma_G
        #disc.M519.append((age/(2*np.pi),dM_tot[-519]))

    def optically_thin_weighting(self, disc, dt):
        # Locate and select cells that aren't empty
        Sigma_G = disc.Sigma_G
        not_empty = (disc.Sigma_G > 0)

        # Get the photo-evaporation rates at each cell as if it were the edge
        Mdot = self.mass_loss_rate(disc,not_empty)
        # Find the maximum, corresponding to optically thin/thick boundary
        i_max = np.size(Mdot) - np.argmax(Mdot[::-1]) - 1

        # Weighting function
        ot_radii = (disc.grid.Rc >= disc.grid.Rc[i_max])
        Sigma_tot = np.sum(Sigma_G[ot_radii])
        Sigma_weight = Sigma_G/Sigma_tot
        Sigma_weight *= ot_radii # Set weight of all cells inside the maximum to zero.
        M_dot_tot = np.sum(Mdot * Sigma_weight[not_empty]) # Contribution of all non-empty cells to mass loss rate
        M_dot_eff = M_dot_tot * Sigma_weight # Effective mass loss rate

        # Convert Msun / yr to g / dynamical time
        dM_dot = M_dot_eff * Msun / (2 * np.pi)

        # Annulus masses
        Re = disc.R_edge * AU
        dA = np.pi * (Re[1:] ** 2 - Re[:-1] ** 2)
        dM_gas = Sigma_G * dA

        return (dM_dot, dM_gas)        

    def weighted_removal(self, disc, dt):
        (dM_dot, dM_gas) = self.optically_thin_weighting(disc,dt)

        dM_evap = dM_dot * dt
        deplete = np.ones_like(disc.Sigma_G)
        not_empty = (dM_gas>0)
        deplete[not_empty] = (dM_gas[not_empty] - dM_evap[not_empty])/ dM_gas[not_empty]
        disc.Sigma_G[:] *= deplete
        disc.Sigma[:] = disc.Sigma_G

    def __call__(self, disc, dt, age):
        """Removes gas and dust from the edge of a disc"""
        if (isinstance(self,FixedExternalEvaporation)):
            self.timescale_remove(disc, dt, age)
        else:
            self.weighted_removal(disc, dt)

        """
        # Remove gas / entrained dust from empty cells, set density and
        # dust fraction of remaining dust
        for a, dM_i in zip(disc.grain_size, dM_dust):
            dM_i[(a < amax) & empty] = 0

        disc.Sigma[empty] = (dM_dust / A)[..., empty].sum(0)
        disc.dust_frac[..., empty] = \
            dM_dust[..., empty] / (A * disc.Sigma + 1e-300)[empty]

        # Reduce the surface density of the one cell that is partially emptied
        cell_id = np.searchsorted(-M_tot, -dM_evap) - 1
        if cell_id < 0: return

        # Compute the remaining mass to remove
        try:
            dM_cell = dM_evap - M_tot[cell_id + 1]
        except IndexError:
            # Handle case where we only remove mass from the outer-most cell
            dM_cell = dM_evap

        # Compute new surface densities of final cell
        #    Assume dust fraction of species entrained does not change, while
        #    the mass of those those not entrained stays the same.
        not_entrained = disc.grain_size[..., cell_id] > amax
        M_left = (dM_tot[cell_id] - dM_cell +
                  dM_dust[not_entrained, cell_id].sum())

        disc.Sigma[cell_id] = M_left / A[cell_id]
        disc.dust_frac[not_entrained, cell_id] = \
            dM_dust[not_entrained, cell_id] / M_left

        ### And we're done"""


class FixedExternalEvaporation(ExternalPhotoevaporationBase):
    """External photoevaporation flow with a constant mass loss rate, which
    entrains dust below a fixed size.

    args:
        Mdot : mass-loss rate in Msun / yr,  default = 10^-8
        amax : maximum grain size entrained, default = 10 micron
    """

    def __init__(self, Mdot=1e-8, amax=1e-3):
        self._Mdot = Mdot
        self._amax = amax

    def mass_loss_rate(self, disc, not_empty):
        return self._Mdot*np.ones_like(disc.Sigma_G[not_empty])

    def max_size_entrained(self, disc):
        return self._amax

class FRIEDExternalEvaporationMS(ExternalPhotoevaporationBase):
    """External photoevaporation flow with a mass loss rate which
    is dependent on radius and surface density.
	Currently ignores dust by setting max size to 0

    args:
        Mdot : mass-loss rate in Msun / yr,  default = 10^-8
        amax : maximum grain size entrained, default = 0
    """

    def __init__(self, disc, Mdot=0, amax=0):
        self.FRIED_Rates = photorate.FRIED_3DMS(photorate.grid_parameters,photorate.grid_rate,disc.star.M)
        self._Mdot = Mdot
        self._amax = amax

    def mass_loss_rate(self, disc, not_empty):
        UV_field = disc.UV * np.ones_like(disc.Sigma_G) 
        calc_rates = self.FRIED_Rates.PE_rate(( UV_field[not_empty], disc.Sigma_G[not_empty], disc.R[not_empty] ))
        norate = np.isnan(calc_rates)
        final_rates = calc_rates
        final_rates[norate] = 1e-10
        return final_rates

    def max_size_entrained(self, disc):
        return self._amax

class FRIEDExternalEvaporationS(ExternalPhotoevaporationBase):
    """External photoevaporation flow with a mass loss rate which
    is dependent on radius and surface density.
	Currently ignores dust by setting max size to 0

    args:
        Mdot : mass-loss rate in Msun / yr,  default = 10^-8
        amax : maximum grain size entrained, default = 0
    """

    def __init__(self, disc, Mdot=0, amax=0):
        self.FRIED_Rates = photorate.FRIED_3DS(photorate.grid_parameters,photorate.grid_rate,disc.star.M)
        self._Mdot = Mdot
        self._amax = amax

    def mass_loss_rate(self, disc, not_empty):
        UV_field = disc.UV * np.ones_like(disc.Sigma_G) 
        calc_rates = self.FRIED_Rates.PE_rate(( UV_field[not_empty], disc.Sigma_G[not_empty], disc.R[not_empty] ))
        norate = np.isnan(calc_rates)
        final_rates = calc_rates
        final_rates[norate] = 1e-10
        return final_rates

    def max_size_entrained(self, disc):
        return self._amax

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from .grid import Grid
    from .eos import LocallyIsothermalEOS
    from .star import SimpleStar
    from .dust import FixedSizeDust

    # Set up accretion disc properties
    Mdot = 1e-8
    alpha = 1e-3

    Mdot *= Msun / (2 * np.pi)
    Mdot /= AU ** 2
    Rd = 100.

    grid = Grid(0.1, 1000, 1000, spacing='log')
    star = SimpleStar()
    eos = LocallyIsothermalEOS(star, 1 / 30., -0.25, alpha)
    eos.set_grid(grid)
    Sigma = (Mdot / (3 * np.pi * eos.nu)) * np.exp(-grid.Rc / Rd)

    # Setup a dusty disc model
    disc = FixedSizeDust(grid, star, eos, 0.01, [1e-4, 0.1], Sigma=Sigma)

    # Setup the photo-evaporation
    photoEvap = FixedExternalEvaporation()

    times = np.linspace(0, 1e7, 6) * 2 * np.pi

    dA = np.pi * np.diff((disc.R_edge * AU) ** 2) / Msun

    # Test the removal of gas / dust
    t, M = [], []
    tc = 0
    for ti in times:
        photoEvap(disc, ti - tc)

        tc = ti
        t.append(tc / (2 * np.pi))
        M.append((disc.Sigma * dA).sum())

        c = plt.plot(disc.R, disc.Sigma_G)[0].get_color()
        plt.loglog(disc.R, disc.Sigma_D[0], c + ':')
        plt.loglog(disc.R, 0.1 * disc.Sigma_D[1], c + '--')

    plt.xlabel('R')
    plt.ylabel('Sigma')

    t = np.array(t)
    plt.figure()
    plt.plot(t, M)
    plt.plot(t, M[0] - 1e-8 * t, 'k')
    plt.xlabel('t [yr]')
    plt.ylabel('M [Msun]')
    plt.show()
