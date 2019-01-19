# photoevaportation.py
#
# Author: R. Booth
# Date: 24 - Feb - 2017
#
# Models for photo-evaporation of a disc
###############################################################################
import numpy as np
from .constants import AU, Msun, yr
import FRIED.photorate as photorate
from .dust import DustyDisc

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
        Dt_R = dM_gas[not_empty] / dM_evap[not_empty] # Dynamical time for each annulus to be depleted of mass
        Dt_R = np.concatenate((Dt_R, np.zeros(np.size(disc.Sigma_G)-np.size(Dt_R)))) # Append 0 for empty annuli
        dM_evap = np.concatenate((dM_evap, np.zeros(np.size(disc.Sigma_G)-np.size(dM_evap)))) # Append 0 for empty annuli
        Dt = np.cumsum(Dt_R[::-1])[::-1] # Dynamical time to deplete each annulus and those exterior
        
        # Return mass loss rate, annulus mass, cumulative mass and cumulative timescale
        return (dM_evap, dM_tot, M_tot, Dt)

    def timescale_remove(self, disc, dt, age):
        (dM_evap, dM_tot, M_tot, Dt) = self.get_timescale(disc,dt)

        # Calculate which cells have times shorter than the timestep and empty
        excess_t = dt - Dt
        empty = (excess_t > 0)
        disc.tot_mass_lost += np.sum(dM_tot[empty])
        disc.Sigma[empty] = 0

        # Deal with marginal cell (first for which dt<Dt) as long as entire disc isn't removed
        if (np.sum(empty)<np.size(disc.Sigma_G)):
            half_empty = -(np.sum(empty) + 1) # ID (from end) of half depleted cell

            mass_left = -1.0*excess_t[half_empty] / (Dt[half_empty]-Dt[half_empty+1]) # Work out fraction left in cell
            if (half_empty==disc.i_edge):
                disc.mass_lost += dM_tot[half_empty] * (1.0-mass_left)
            else:
                disc.mass_lost = dM_tot[half_empty] * (1.0-mass_left)
                disc.i_edge = half_empty
            disc.tot_mass_lost += dM_tot[half_empty] * (1.0-mass_left)
        
    def optically_thin_weighting(self, disc, dt):
        # Locate and select cells that aren't empty OF GAS
        Sigma_G = disc.Sigma_G
        not_empty = (disc.Sigma_G > 0)

        # Get the photo-evaporation rates at each cell as if it were the edge USING GAS SIGMA
        Mdot = self.mass_loss_rate(disc,not_empty)
        # Find the maximum, corresponding to optically thin/thick boundary
        i_max = np.size(Mdot) - np.argmax(Mdot[::-1]) - 1

        # Annulus GAS masses
        Re = disc.R_edge * AU
        dA = np.pi * (Re[1:] ** 2 - Re[:-1] ** 2)
        dM_gas = disc.Sigma_G * dA

        # Weighting function USING GAS MASS
        ot_radii = (disc.R >= disc.R[i_max])
        s = disc.R**(3/2) * disc.Sigma_G
        s_tot = np.sum(s[ot_radii])
        s_weight = s/s_tot
        s_weight *= ot_radii # Set weight of all cells inside the maximum to zero.
        M_dot_tot = np.sum(Mdot * s_weight) # Contribution of all cells to mass loss rate
        M_dot_eff = M_dot_tot * s_weight # Effective mass loss rate

        # Convert Msun / yr to g / dynamical time
        dM_dot = M_dot_eff * Msun / (2 * np.pi)

        #self._Mdot = dM_dot
        return (dM_dot, dM_gas)

    def weighted_removal(self, disc, dt):
        (dM_dot, dM_gas) = self.optically_thin_weighting(disc,dt)

        if (isinstance(disc,DustyDisc)):
            self._amax = self.Facchini_limit(disc,dM_dot *(yr/Msun))
            Sigma_D = disc.Sigma_D
            not_dustless = (Sigma_D.sum(0) > 0)
            f_m = np.zeros_like(disc.Sigma)
            f_m[not_dustless] = disc.dust_frac[1,:].flatten()[not_dustless]/disc.integ_dust_frac[not_dustless]

        # Annulus Areas
        Re = disc.R_edge * AU
        dA = np.pi * (Re[1:] ** 2 - Re[:-1] ** 2)

        dM_evap = dM_dot * dt
        deplete = np.zeros_like(disc.Sigma)
        disc._Sigma -= dM_evap / dA # This amount of mass is lost in GAS
        
        #(f_ent_s , f_ent_l) = self.dust_entrainment(disc)
        #print ((f_ent_s , f_ent_l))
        M_ent = self.dust_entrainment(disc)
        M_ent_w= np.zeros_like(M_ent)
        M_ent_w[(dM_gas > 0)] = M_ent[(dM_gas > 0)] * dM_evap[(dM_gas > 0)] / dM_gas[(dM_gas > 0)]
        #(f_ent_s , f_ent_l) = (np.zeros_like(disc.R), np.zeros_like(disc.R))
        disc._Sigma -= M_ent_w / dA
        #disc._Sigma[not_empty] -= disc.Sigma_D[0,:].flatten()[not_empty] * f_ent_s.flatten()[not_empty] # This fraction of mass is lost in SMALL DUST
        #disc._Sigma[not_empty] -= disc.Sigma_D[1,:].flatten()[not_empty] * f_ent_l.flatten()[not_empty] # This fraction of mass is lost in LARGE DUST

        # For now, no entrainment, Sigma_D is as before so all of above loss must be gas 
        # With entrainment, must also lower the dust densities
        if (isinstance(disc,DustyDisc)):
            #disc._eps[0][not_empty] = Sigma_D[0,:].flatten()[not_empty] * (1.0 - f_ent_s.flatten()[not_empty]) / disc.Sigma[not_empty]
            #disc._eps[1][not_empty] = Sigma_D[1,:].flatten()[not_empty] * (1.0 - f_ent_l.flatten()[not_empty]) / disc.Sigma[not_empty]
            not_empty = (disc.Sigma > 0)
            new_dust_frac = np.zeros_like(disc.Sigma)
            new_dust_frac[not_empty] = (Sigma_D.sum(0)[not_empty] - M_ent_w[not_empty] / dA[not_empty]) / disc._Sigma[not_empty]
            disc._eps[0][not_empty] = new_dust_frac[not_empty] * (1-f_m[not_empty])
            disc._eps[1][not_empty] = new_dust_frac[not_empty] * f_m[not_empty]
            #print (new_eps)

    def dust_entrainment(self, disc):
        # Representative sizes
        a_ent = self._amax
        St_eq = disc._eos.alpha/2
        a_eq = 2/np.pi * St_eq * disc.Sigma_G/disc._rho_s
        a = disc.grain_size
        amax = a[1,:].flatten()

        # Annulus DUST masses
        Re = disc.R_edge * AU
        dA = np.pi * (Re[1:] ** 2 - Re[:-1] ** 2)
        M_dust = disc.Sigma_D * dA

        #
        not_empty = (disc.Sigma_G>0)
        M_ent = np.zeros_like(disc.Sigma)
        M_ent_small = np.zeros_like(disc.Sigma)
        M_ent_large = np.zeros_like(disc.Sigma)
        f_ent_s = np.zeros_like(disc.Sigma)
        f_ent_l = np.zeros_like(disc.Sigma)

        # Calculate mass of each population that is entrained
        """print(np.shape(M_ent[not_empty]))
        print(np.shape(M_dust.sum(0)[not_empty]))
        print(np.shape(amax[not_empty]))
        print(np.shape(a_ent[not_empty]))
        print(np.shape(np.ones_like(amax)[not_empty]))
        print(np.shape(M_dust.sum(0)[not_empty] * np.minimum(np.ones_like(amax)[not_empty],[(a_ent[not_empty]/amax[not_empty])**(1/2)]).flatten()))"""
        M_ent[not_empty] = M_dust.sum(0)[not_empty] * np.minimum(np.ones_like(amax)[not_empty],[(a_ent[not_empty]/amax[not_empty])**(1/2)]).flatten() # Take as entrained lower of all dust mass, or the fraction from MRN
        M_ent_small[not_empty] = M_dust[0,:][not_empty] * np.minimum(np.ones_like(amax)[not_empty],[(a_ent[not_empty]/np.minimum(a_eq[not_empty],amax[not_empty]))**(1/2)]).flatten() # Take as entrained, lower of all small dust, or the fraction from MRN, which depends on if upper limit is set by becoming large or largest dust, taking the smaller of those two
        M_ent_large[not_empty] = M_ent[not_empty] - M_ent_small[not_empty] # Take as entrained the difference between the total and small
        """Look at numpy.seterr in header /// Just need to consider total mass"""
        """Consider whether dust left behind or drifts in"""

        # Return fraction of each population that is entrained
        #f_ent_s[not_empty] = M_ent_small[not_empty]/M_dust[0,:][not_empty]
        #f_ent_l[not_empty] = M_ent_large[not_empty]/M_dust[1,:][not_empty]
        #print ((f_ent_s, f_ent_l))
        #return (f_ent_s, f_ent_l)
        #f_ent_s[not_empty] = M_ent[not_empty]/M_dust.sum(0)[not_empty]
        return M_ent

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

    def Facchini_limit(self, disc, Mdot):
        # Equation 35 of Facchini et al (2016)
        # Note following definitions:
        # F = H / sqrt(H^2+R^2)
        # v_th = \sqrt(8/pi) C_S in AU / t_dyn
        # Mdot is in units of Msun yr^-1
        # G=1 in units AU^3 Msun^-1 yr^-2
        
        F = disc.H / np.sqrt(disc.H**2+disc.R**2)
        rho = disc._rho_s
        Mstar = disc.star.M # In Msun
        v_th = np.sqrt(8/np.pi) * disc.cs
        
        a_entr = (v_th * Mdot) / (Mstar * 4 * np.pi * F * rho)
        a_entr *= yr * Msun / AU**2
        return a_entr 

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
        return self._Mdot*np.ones_like(disc.Sigma[not_empty])

    def max_size_entrained(self, disc):
        return self._amax

class FRIEDExternalEvaporationMS(ExternalPhotoevaporationBase):
    """External photoevaporation flow with a mass loss rate which
    is dependent on radius and surface density.

    args:
        Mdot : mass-loss rate in Msun / yr,  default = 10^-10
        amax : maximum grain size entrained, default = 0
    """

    def __init__(self, disc, Mdot=1e-10, amax=0):
        self.FRIED_Rates = photorate.FRIED_3DMS(photorate.grid_parameters,photorate.grid_rate,disc.star.M)
        self._Mdot = Mdot * np.ones_like(disc.R)
        self._amax = amax * np.ones_like(disc.R)

    def mass_loss_rate(self, disc, not_empty):
        UV_field = disc.UV * np.ones_like(disc.R) 
        calc_rates = np.zeros_like(disc.R)
        calc_rates[not_empty] = self.FRIED_Rates.PE_rate(( UV_field[not_empty], disc.Sigma_G[not_empty], disc.R[not_empty] ))
        norate = np.isnan(calc_rates)
        final_rates = calc_rates
        final_rates[norate] = 1e-10
        #self._Mdot = final_rates
        return final_rates

    def max_size_entrained(self, disc):
        # Update maximum entrained size
        self._amax = self.Facchini_limit(disc,self._Mdot)
        return self._amax

class FRIEDExternalEvaporationS(ExternalPhotoevaporationBase):
    """External photoevaporation flow with a mass loss rate which
    is dependent on radius and surface density.

    args:
        Mdot : mass-loss rate in Msun / yr,  default = 10^-10
        amax : maximum grain size entrained, default = 0
    """

    def __init__(self, disc, Mdot=1e-10, amax=0):
        self.FRIED_Rates = photorate.FRIED_3DS(photorate.grid_parameters,photorate.grid_rate,disc.star.M)
        self._Mdot = Mdot * np.ones_like(disc.R)
        self._amax = amax * np.ones_like(disc.R)

    def mass_loss_rate(self, disc, not_empty):
        UV_field = disc.UV * np.ones_like(disc.R) 
        calc_rates = np.zeros_like(disc.R)
        calc_rates[not_empty] = self.FRIED_Rates.PE_rate(( UV_field[not_empty], disc.Sigma_G[not_empty], disc.R[not_empty] ))
        norate = np.isnan(calc_rates)
        final_rates = calc_rates
        final_rates[norate] = 1e-10
        self._Mdot = final_rates
        return final_rates

    def max_size_entrained(self, disc):
        # Update maximum entrained size
        self._amax = self.Facchini_limit(disc,self._Mdot)
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
