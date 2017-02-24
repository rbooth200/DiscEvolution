# photoevaportation.py
#
# Author: R. Booth
# Date: 24 - Feb - 2017
#
# Models for photo-evaporation of a disc
###############################################################################
import numpy as np
from constants import AU, Msun

class ExternalPhotoevaporationBase(object):
    '''Base class for handling the external photo-evaportation of discs

    Implementations of ExternalPhotoevaporation classes must provide the 
    following methods:
        mass_loss_rate(disc)     : returns mass-loss rate from outer edge
                                   (Msun/yr).
        max_size_entrained(dist) : returns the maximum size of dust entrained in
                                   the flow (cm).        
    '''
    def __call__(self, disc, dt):
        '''Removes gas and dust from the edge of a disc given'''
        
        # Get the photo-evaporation data:
        Mdot = self.mass_loss_rate(disc)
        amax = self.max_size_entrained(disc)

        # Convert Msun / yr to g / dynamical time and compute mass evaporated:
        dM_evap = Mdot * dt * Msun / (2*np.pi)

        # Disc densities / masses
        Sigma_G = disc.Sigma_G
        Sigma_D = disc.Sigma_D
        
        Re = disc.R_edge * AU
        A = np.pi * (Re[1:]**2 -  Re[:-1]**2) 
        dM_gas  = Sigma_G * A 
        dM_dust = Sigma_D * A
    
        # Work out which cells we need to empty of gas & entrained dust
        dM_tot = dM_gas + (dM_dust*(disc.grain_size <= amax)).sum(0)
        M_tot = np.cumsum(dM_tot[::-1])[::-1]
        empty = M_tot < dM_evap

        # Remove gas / entrained dust from empty cells, set density and
        # dust fraction of remaining dust
        dM_entrained = dM_dust[disc.grain_size <= amax]
        dM_entrained[...,empty] = 0
        dM_dust[disc.grain_size <= amax] = dM_entrained

        disc.Sigma[empty] = (dM_dust/A)[...,empty].sum(0)
        disc.dust_frac[...,empty] = \
            dM_dust[...,empty] / (A*disc.Sigma)[empty]

        # Reduce the surface density of the one cell that is partially emptied
        cell_id = np.searchsorted(-M_tot, -dM_evap) - 1
        if cell_id < 0: return
        try:
            dM_cell = dM_evap - M_tot[cell_id+1]
        except IndexError: 
            # Handle case where we only remove mass from the outer-most cell
            dM_cell = dM_evap

        # Compute new surface densities of final cell
        #    Assume dust fraction of species entrained does not change, while
        #    the mass of those those not entrained stays the same.
        not_entrained = disc.grain_size[...,cell_id] > amax
        M_left = (dM_tot[cell_id] - dM_cell +  
                  dM_dust[not_entrained, cell_id].sum())

        disc.Sigma[cell_id] = M_left / A[cell_id]
        disc.dust_frac[not_entrained, cell_id] = \
            dM_dust[not_entrained, cell_id] / M_left

        ### And we're done



class FixedExternalEvaportation(ExternalPhotoevaporationBase):
    '''External photoevaporation flow with a constant mass loss rate, which 
    entrains dust below a fixed size.

    args:
        Mdot : mass-loss rate in Msun / yr,  default = 10^-8
        amax : maximum grain size entrained, default = 10 micron
    '''
    def __init__(self, Mdot=1e-8, amax=1e-3):
        self._Mdot = Mdot
        self._amax = amax

    def mass_loss_rate(self, disc):
        return self._Mdot

    def max_size_entrained(self, disc):
        return self._amax


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from grid import Grid
    from eos import LocallyIsothermalEOS
    from star import SimpleStar
    from dust import FixedSizeDust

    # Set up accretion disc properties
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

    # Setup a dusty disc model
    disc = FixedSizeDust(grid, star, eos, 0.01, [1e-4, 0.1], Sigma=Sigma)

    # Setup the photo-evaporation
    photoEvap = FixedExternalEvaportation()

    times = np.linspace(0, 1e7, 6) * 2*np.pi

    dA = np.pi * np.diff((disc.R_edge * AU)**2) / Msun

    t, M = [], []
    tc = 0
    for ti in times:
        photoEvap(disc, ti-tc)
    
        tc = ti
        t.append(tc / (2*np.pi))
        M.append((disc.Sigma*dA).sum())

        c = plt.plot(disc.R, disc.Sigma_G)[0].get_color()
        plt.loglog(disc.R, disc.Sigma_D[0], c +':')
        plt.loglog(disc.R, 0.1*disc.Sigma_D[1], c +'--')

    plt.xlabel('R')
    plt.ylabel('Sigma')

    t = np.array(t)
    plt.figure()
    plt.plot(t, M)
    plt.plot(t, M[0] - 1e-8*t, 'k')
    plt.xlabel('t [yr]')
    plt.ylabel('M [Msun]')
    plt.show()
