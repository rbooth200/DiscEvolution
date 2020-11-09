# photoevaportation.py
#
# Author: R. Booth
# Date: 24 - Feb - 2017
#
# Models for photo-evaporation of a disc
###############################################################################
import numpy as np
from .constants import AU, Msun, yr, Mjup
from .dust import DustyDisc
from .FRIED import photorate

class ExternalPhotoevaporationBase(object):
    """Base class for handling the external photo-evaporation of discs

    Implementations of ExternalPhotoevaporation classes must provide the
    following methods:
        mass_loss_rate(disc)     : returns mass-loss rate from outer edge
                                   (Msun/yr).
        max_size_entrained(dist) : returns the maximum size of dust entrained in
                                   the flow (cm).
    """

    def unweighted_rates(self, disc):
        """Calculates the raw mass loss rates for each annulus in code units"""
        # Locate and select cells that aren't empty OF GAS
        Sigma_G = disc.Sigma_G
        not_empty = (disc.Sigma_G > 0)

        # Annulus GAS masses
        Re = disc.R_edge * AU
        dA = np.pi * (Re[1:] ** 2 - Re[:-1] ** 2)
        dM_gas = disc.Sigma_G * dA

        # Get the photo-evaporation rates at each cell as if it were the edge USING GAS SIGMA
        Mdot = self.mass_loss_rate(disc,not_empty)

        # Convert Msun / yr to g / dynamical time
        dM_dot = Mdot * Msun / (2 * np.pi)

        return (dM_dot, dM_gas)

    def get_timescale(self, disc):
        """From mass loss rates, calculate mass loss timescales for each cell"""
        # Retrieve unweighted rates
        (dM_evap, dM_gas) = self.unweighted_rates(disc)     
        not_empty = (disc.Sigma_G > 0)

        # Work out which cells we need to empty of gas & entrained dust
        M_tot = np.cumsum(dM_gas[::-1])[::-1]
        Dt_R = np.zeros_like(dM_evap)
        Dt_R[not_empty] = dM_gas[not_empty] / dM_evap[not_empty] # Dynamical time for each annulus to be depleted of mass
        Dt = np.cumsum(Dt_R[::-1])[::-1] # Dynamical time to deplete each annulus and those exterior

        # Record mass loss
        if sum(not_empty)!=0:
            self._Rot  = disc.R[not_empty][-1]
            self._Mdot = dM_evap[not_empty][-1]*(yr/Msun)
        else:
            self._Rot  = 0
            self._Mdot = 0
            self._empty = True
        
        # Return mass loss rate, annulus mass, cumulative mass and cumulative timescale
        return (dM_evap, dM_gas, M_tot, Dt)

    def timescale_remove(self, disc, dt):
        """Remove gas from cells according to timescale implementation"""
        """Only implemented correctly for case with no dust"""
        # Retrieve timescales
        (dM_evap, dM_tot, M_tot, Dt) = self.get_timescale(disc)

        # Calculate which cells have times shorter than the timestep and empty
        excess_t = dt - Dt
        empty = (excess_t > 0)
        disc._Sigma[empty] = 0

        # Deal with marginal cell (first for which dt<Dt) as long as entire disc isn't removed
        if (np.sum(empty)<np.size(disc.Sigma_G)):
            half_empty = -(np.sum(empty) + 1) # ID (from end) of half depleted cell
            disc._Sigma[half_empty] *= -1.0*excess_t[half_empty] / (Dt[half_empty]-Dt[half_empty+1]) # Work out fraction left in cell 

        self._Mcum_gas  += dt*dM_evap[empty].sum()+dM_tot[half_empty]*-1.0*excess_t[half_empty] / (Dt[half_empty]-Dt[half_empty+1])  # Record mass loss
        
    def optically_thin_weighting(self, disc):
        """Identify optical thickness transition and weight raw mass loss rates exterior"""
        # Retrieve unweighted rates
        (Mdot, dM_gas) = self.unweighted_rates(disc)

        # Find the maximum, corresponding to optically thin/thick boundary
        i_max = np.size(Mdot) - np.argmax(Mdot[::-1]) - 1
        self._Rot = disc.R[i_max]

        # Weighting function USING GAS MASS
        ot_radii = (disc.R >= self._Rot)
        s = disc.R**(3/2) * disc.Sigma_G
        s_tot = np.sum(s[ot_radii])
        s_weight = s/s_tot
        s_weight *= ot_radii # Set weight of all cells inside the maximum to zero.
        M_dot_tot = np.sum(Mdot * s_weight) # Contribution of all cells to mass loss rate
        M_dot_eff = M_dot_tot * s_weight # Effective mass loss rate

        # Record mass loss
        self._Mdot = M_dot_tot*(yr/Msun)

        return (M_dot_eff, dM_gas)

    def weighted_removal(self, disc, dt):
        """Remove gas according to the weighted prescription"""
        (dM_dot, dM_gas) = self.optically_thin_weighting(disc)

        # Account for dust entrainment
        if (isinstance(disc,DustyDisc)):
            # First get initial dust conditions
            Sigma_D0 = disc.Sigma_D
            not_dustless = (Sigma_D0.sum(0) > 0)
            f_m = np.zeros_like(disc.Sigma)
            f_m[not_dustless] = disc.dust_frac[1,:].flatten()[not_dustless]/disc.integ_dust_frac[not_dustless]  # Large fraction

            # Update the maximum entrained size
            self.max_size_entrained(disc)
            # Work out the total mass in entrained dust
            M_ent = self.dust_entrainment(disc)

        # Annulus Areas
        Re = disc.R_edge * AU
        dA = np.pi * (Re[1:] ** 2 - Re[:-1] ** 2)

        dM_evap = dM_dot * dt
        disc._Sigma -= dM_evap / dA # This amount of mass is lost in GAS

        self._Mcum_gas  += dM_evap.sum()  # Record mass loss
        
        if (isinstance(disc,DustyDisc)):
            # Remove dust mass proportionally to gas loss 
            M_ent_w = np.zeros_like(M_ent)
            M_ent_w[(dM_gas > 0)] = M_ent[(dM_gas > 0)] * dM_evap[(dM_gas > 0)] / dM_gas[(dM_gas > 0)]
            disc._Sigma -= M_ent_w / dA

            # Update the dust mass fractions
            not_empty = (disc.Sigma > 0)
            new_dust_frac = np.zeros_like(disc.Sigma)
            new_Sigma_D = Sigma_D0.sum(0)[not_empty] - M_ent_w[not_empty] / dA[not_empty]
            new_dust_frac[not_empty] = new_Sigma_D / disc.Sigma[not_empty]
            disc._eps[0][not_empty] = new_dust_frac[not_empty] * (1.0-f_m[not_empty])
            disc._eps[1][not_empty] = new_dust_frac[not_empty] * f_m[not_empty]

            self._Mcum_dust += M_ent_w.sum() # Record mass loss

    def dust_entrainment(self, disc):
        # Representative sizes
        a_ent = self._amax
        a = disc.grain_size
        amax = a[1,:].flatten()

        # Annulus DUST masses
        Re = disc.R_edge * AU
        dA = np.pi * (Re[1:] ** 2 - Re[:-1] ** 2)
        dM_dust = disc.Sigma_D * dA

        # Select cells with gas
        not_empty = (disc.Sigma_G>0)
        M_ent = np.zeros_like(disc.Sigma)

        # Calculate total that is entrained
        f_ent = np.minimum(np.ones_like(amax)[not_empty],[(a_ent[not_empty]/amax[not_empty])**(4-disc._p)]).flatten() # Take as entrained lower of all dust mass, or the fraction from MRN
        M_ent[not_empty] = dM_dust.sum(0)[not_empty] * f_ent
        return M_ent

    def __call__(self, disc, dt, age):
        """Remove gas and dust from the edge of a disc"""
        raise NotImplementedError("Derived class must implement the choice of mass loss proceedure")

def Facchini_limit(disc, Mdot):
    """
    Equation 35 of Facchini et al (2016)
    Note following definitions:
    F = H / sqrt(H^2+R^2) (dimensionless)
    v_th = \sqrt(8/pi) C_S in AU / t_dyn
    Mdot is in units of Msun yr^-1
    G=1 in units AU^3 Msun^-1 t_dyn^-2
    """
    
    F = disc.H / np.sqrt(disc.H**2+disc.R**2)
    rho = disc._rho_s
    Mstar = disc.star.M # In Msun
    v_th = np.sqrt(8/np.pi) * disc.cs
    
    a_entr = (v_th * Mdot) / (Mstar * 4 * np.pi * F * rho)
    a_entr *= Msun / AU**2 / yr
    return a_entr

## Available instances of the photoevaporation module ##
## - Fixed (user defined rate)
## - FRIED_S (uses the surface density at the edge directly)
## - FRIED_MS (extrapolates from surface density at the edge to M400) 
## - FRIED_M  (extrapolates from disc mass to M400)

class FixedExternalEvaporation(ExternalPhotoevaporationBase):
    """External photoevaporation flow with a constant mass loss rate, which
    entrains dust below a fixed size.

    args:
        Mdot : mass-loss rate in Msun / yr,  default = 10^-8
        amax : maximum grain size entrained, default = 10 micron
    """

    def __init__(self, disc, Mdot=1e-8, amax=10):
        self._Mdot = Mdot
        self._amax = amax * np.ones_like(disc.R)
        self._empty = False

        self._Mcum_gas  = 0.0
        self._Mcum_dust = 0.0

    def __call__(self, disc, dt):
        if (self._Mdot > 0):
            self.timescale_remove(disc, dt)
    
    def mass_loss_rate(self, disc, not_empty):
        return self._Mdot * np.ones_like(disc.Sigma)

    def max_size_entrained(self, disc):
        return self._amax * np.ones_like(disc.Sigma)

    def ASCII_header(self):
        return ("# FixedExternalEvaportation, Mdot: {}, amax: {}"
                "".format(self._Mdot, self._amax))

    def HDF5_attributes(self):
        header = {}
        header['Mdot'] = '{}'.format(self._Mdot)
        header['amax'] = '{}'.format(self._amax)
        return self.__class__.__name__, header
    
class TimeExternalEvaporation(ExternalPhotoevaporationBase):
    """Mass loss via external evaporation at a constant mass-loss timescale,
    Almost certainly deprecated in this branch
        Mdot = pi R^2 Sigma / t_loss.

    args:
        time-scale : mass loss time-scale in years
        amax : maximum grain size entrained, default = 10 micron
    """

    def __init__(self, time=1e6, amax=1e-3):
        self._time = time
        self._amax = amax

        self._Mcum_gas  = 0.0
        self._Mcum_dust = 0.0

    def mass_loss_rate(self, disc):
        k = np.pi * AU**2 / Msun
        return k * disc.R**2 * disc.Sigma / self._time

    def max_size_entrained(self, disc):
        return self._amax * np.ones_like(disc.Sigma)

    def ASCII_header(self):
        return ("# TimeExternalEvaportation, time: {}, amax: {}"
                "".format(self._time, self._amax))

    def HDF5_attributes(self):
        header = {}
        header['time'] = '{}'.format(self._time)
        header['amax'] = '{}'.format(self._amax)
        return self.__class__.__name__, header

###### FRIED Variants

class FRIEDExternalEvaporationS(ExternalPhotoevaporationBase):
    """External photoevaporation flow with a mass loss rate which
    is dependent on radius and surface density.

    args:
        Mdot : mass-loss rate in Msun / yr,  default = 10^-10
        amax : maximum grain size entrained, default = 0
    """

    def __init__(self, disc, Mdot=1e-10, amax=0):
        self.FRIED_Rates = photorate.FRIED_2DS(photorate.grid_parameters,photorate.grid_rate,disc.star.M,disc.FUV)
        self._Mdot = 0
        self._amax = amax * np.ones_like(disc.R)
        self._FUV = disc.FUV
        self._Rot = max(disc.R)
        self._density = True

        self._Mcum_gas  = 0.0
        self._Mcum_dust = 0.0

    def __call__(self, disc, dt):
        self.weighted_removal(disc, dt) # For FRIED mass loss rates calculated with density, need to use optical depth method

    def mass_loss_rate(self, disc, not_empty):
        calc_rates = np.zeros_like(disc.R)
        calc_rates[not_empty] = self.FRIED_Rates.PE_rate(( disc.Sigma_G[not_empty], disc.R[not_empty] ))
        norate = np.isnan(calc_rates)
        calc_rates[norate] = 1e-10
        return calc_rates
        
    def max_size_entrained(self, disc):
        self._amax = Facchini_limit(disc,self._Mdot) # Update maximum entrained size
        return self._amax

    def ASCII_header(self):
        return ("# FRIEDExternalEvaporationS: {} G0".format(self._FUV))

    def HDF5_attributes(self):
        header = {}
        return self.__class__.__name__, header

class FRIEDExternalEvaporationMS(ExternalPhotoevaporationBase):
    """External photoevaporation flow with a mass loss rate which
    is dependent on radius and surface density.
    Calculated by converting to the mass within 400 AU (M400 ~ R Sigma)

    args:
        Mdot : mass-loss rate in Msun / yr,  default = 10^-10
        amax : maximum grain size entrained, default = 0
    """

    def __init__(self, disc, Mdot=1e-10, amax=0):
        self.FRIED_Rates = photorate.FRIED_2DM400S(photorate.grid_parameters,photorate.grid_rate,disc.star.M,disc.FUV)
        self._Mdot = 0
        self._amax = amax * np.ones_like(disc.R)
        self._FUV = disc.FUV
        self._Rot = max(disc.R)
        self._density = True

        self._Mcum_gas  = 0.0
        self._Mcum_dust = 0.0

    def __call__(self, disc, dt):
        self.weighted_removal(disc, dt) # For FRIED mass loss rates calculated with density, need to use optical depth method

    def mass_loss_rate(self, disc, not_empty):
        calc_rates = np.zeros_like(disc.R)
        calc_rates[not_empty] = self.FRIED_Rates.PE_rate(( disc.Sigma_G[not_empty], disc.R[not_empty] ))
        norate = np.isnan(calc_rates)
        calc_rates[norate] = 1e-10
        return calc_rates

    def max_size_entrained(self, disc):
        self._amax = Facchini_limit(disc,self._Mdot) # Update maximum entrained size
        return self._amax

    def ASCII_header(self):
        return ("# FRIEDExternalEvaporationMS: {} G0".format(self._FUV))

    def HDF5_attributes(self):
        header = {}
        return self.__class__.__name__, header

class FRIEDExternalEvaporationM(ExternalPhotoevaporationBase):
    """External photoevaporation flow with a mass loss rate which
    is dependent on radius and integrated mass interior.
    Calculated by converting to the mass within 400 AU (M400 ~ M / R)

    args:
        Mdot : mass-loss rate in Msun / yr,  default = 10^-10
        amax : maximum grain size entrained, default = 0
    """

    def __init__(self, disc, Mdot=1e-10, amax=0):
        self.FRIED_Rates = photorate.FRIED_2DM400M(photorate.grid_parameters,photorate.grid_rate,disc.star.M,disc.FUV)
        self._Mdot = 0
        self._amax = amax * np.ones_like(disc.R)
        self._FUV = disc.FUV
        self._Rot = max(disc.R)

        self._Mcum_gas  = 0.0
        self._Mcum_dust = 0.0

    def __call__(self, disc, dt):
        self.timescale_remove(disc, dt) # For FRIED mass loss rates calculated with total mass, can use timescale method (doesn't account for dust)

    def mass_loss_rate(self, disc, not_empty):
        Re = disc.R_edge * AU
        dA = np.pi * (Re[1:] ** 2 - Re[:-1] ** 2)
        dM_gas = disc.Sigma_G * dA
        integ_mass = np.cumsum(dM_gas)/ Mjup
        calc_rates = np.zeros_like(disc.R)
        calc_rates[not_empty] = self.FRIED_Rates.PE_rate(( integ_mass[not_empty], disc.R[not_empty] ))
        norate = np.isnan(calc_rates)
        calc_rates[norate] = 1e-10
        return calc_rates

    def max_size_entrained(self, disc):
        self._amax = Facchini_limit(disc,self._Mdot) # Update maximum entrained size
        return self._amax

    def ASCII_header(self):
        return ("# FRIEDExternalEvaporationM: {} G0".format(self._FUV))

    def HDF5_attributes(self):
        header = {}
        return self.__class__.__name__, header

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
