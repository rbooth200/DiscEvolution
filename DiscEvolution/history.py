# history.py
#
# Author: A. Sellek
# Date: 06 November 2020
#
# Classes for tracking evolution of global disc quantities
#
################################################################################
from __future__ import print_function
import numpy as np
from .constants import yr

class History(object):

    """Setup"""
    def __init__(self, dust, dthresh):
        self._threshold = 1e-5          # Threshold for defining edge by density
        self._dust = dust
        if self._dust:
            self._dthresholds = dthresh # Threshold percentiles of dust mass
        
        self._times = np.array([])      # Output times

        """Radii"""
        self._Rout  = np.array([])      # Outer radius of the disc (density)
        self._Rc    = np.array([])      # Radius of current best fit scale radius
        self._Ropt  = np.array([])      # Radius where Mdot maximum ie where becomes optically thick
        self._Rh    = np.array([])      # Outer radius of the transition disc hole
        if dust:
            self._Rdust = {}            # Radius containing user defined fraction of dust
            for threshold in self._dthresholds:
                self._Rdust[threshold] = np.array([])

        """Mass"""
        self._Mtot = np.array([])       # Total mass
        if dust:
            self._Mdust = np.array([])  # Mass of dust in disc

        """Mass Loss"""
        self._Mdot_acc = np.array([])   # Accretion rate, updated with velocity passed
        self._Mdot_ext = np.array([])   # External photoevaporation rate
        self._Mdot_int = np.array([])   # Internal photoevaporation rate
        self._Mcum_gas = np.array([])   # Total amount of gas lost to wind
        if dust:
            self._Mcum_dust = np.array([])  # Total amount of dust lost to wind

    """Methods to return values"""
    # Return times
    @property
    def times(self):
        return self._times

    # Return radii/masses/mass loss rates
    @property
    def radii(self):
        return self._Rout, self._Rc, self._Ropt, self._Rh
    @property
    def radii_dust(self):
        if self._dust:
            return self._Rdust
        else:
            print("No dust radii to return")
            return None

    @property
    def mass(self):
        return self._Mtot, self._Mcum_gas
    @property
    def mass_dust(self):
        if self._dust:
            return self._Mdust, self._Mcum_dust
        else:
            print("No dust masses to return")
            return None

    @property
    def mdot(self):
        return self._Mdot_acc, self._Mdot_ext, self._Mdot_int
    
    """When restarting, for any variable found in input file, import data"""
    def restart(self, filename, snap_number):
        restartdata = np.genfromtxt(filename, names=True, comments='#', max_rows=snap_number+1)
        
        self._times = restartdata['t']
        print("Restarting with times:\n",self.times)
        # No exception - must have times!

        try:
            self._Rout = restartdata['R_out']
        except ValueError:
            pass
        try:
            self._Rc   = restartdata['R_C']
        except ValueError:
            pass
        try:
            self._Ropt = restartdata['R_out']
        except ValueError:
            pass
        try:
            self._Rh   = restartdata['R_hole']
        except ValueError:
            if 'M_int' in restartdata.dtype.names:
                self._Rh = np.full_like(self._times,np.nan)
            else:
                pass
        try:
            self._Mtot = restartdata['M_D']
        except ValueError:
            pass
        try:
            self._Mdot_acc = restartdata['M_acc']
        except ValueError:
            pass
        try:
            self._Mdot_ext = restartdata['M_ext']
        except ValueError:
            pass
        try:
            self._Mdot_int = restartdata['M_int']
        except ValueError:
            pass
        try:
            self._Mcum_gas = restartdata['M_wg']
        except ValueError:
            pass

        # Dust-Only Parameters
        if self._dust:
            for threshold in self._dthresholds:
                try:
                    self._Rdust[threshold] = restartdata['R_{}'.format(int(float(threshold)*100))]
                except ValueError:
                    pass
            try:
                self._Mdust = restartdata['M_d']
            except ValueError:
                pass
            try:
                self._Mcum_dust = restartdata['M_wd']
            except ValueError:
                pass

    """Call"""
    def __call__(self, driver):
        self._times = np.append(self._times,[driver.t / yr])

        # 1 Radius: Density threshold / Scale radius of LBP / Optically Thin
        self._Rout = np.append(self._Rout,[driver.disc.Rout(thresh=self._threshold)])
        if not (driver.photoevaporation_external or driver.photoevaporation_internal):
            self._Rc = np.append(self._Rc,[driver.disc.RC()])
        # Optically thin still needed here!

        # 2 Disc Mass: Total
        self._Mtot = np.append(self._Mtot,[driver.disc.Mtot()])

        # 3 Dust radii and mass
        if self._dust:
            dust_radii = driver.disc.Rdust(thresholds=self._dthresholds) # Radius containing proportion of dust mass
            for thresh in self._dthresholds:
                self._Rdust[thresh] = np.append(self._Rdust[thresh],dust_radii[thresh])
            self._Mdust = np.append(self._Mdust,[driver.disc.Mdust()])   # Remaining dust mass

        # 4 Viscous accretion rate
        if driver.gas:            
            self._Mdot_acc = np.append(self._Mdot_acc,[driver.disc.Mdot(driver._gas.viscous_velocity(driver.disc)[0])])

        # 5 External photoevaporation mass loss
        if driver.photoevaporation_external:
            self._Mdot_ext  = np.append(self._Mdot_ext,[driver.photoevaporation_external._Mdot])
            self._Ropt      = np.append(self._Ropt,[driver.photoevaporation_external._Rot])
            self._Mcum_gas  = np.append(self._Mcum_gas,[driver.photoevaporation_external._Mcum_gas])       # Total mass of gas  lost in wind
            if self._dust:
                self._Mcum_dust = np.append(self._Mcum_dust,[driver.photoevaporation_external._Mcum_dust]) # Total mass of dust lost in wind

        # 6 Internal photoevaporation
        if driver.photoevaporation_internal:
            R_hole, N_hole = driver.photoevaporation_internal.get_Rhole(driver.disc)
            self._Rh = np.append(self._Rh,[R_hole])
            self._Mdot_int = np.append(self._Mdot_int, [driver.photoevaporation_internal._Mdot_true])

    """Remove hole radii if identification of hole later deemed to be wrong"""
    def clear_hole(self):
        self._R = np.full_like(self._Rh, np.nan)

    """Save the history so far"""
    def save(self, driver, save_directory):
        # 0 Select times of recording
        used_times = self.times

        # 1 Retrieve radii
        outer_radii, scale_radii, opt_radii, hole_radii = self.radii
        radii_select = {}
        if driver.photoevaporation_external:
            radii_select['R_out'] = opt_radii
        else:
            radii_select['R_out'] = outer_radii
        if np.isnan(scale_radii).sum() < len(scale_radii):
            radii_select['R_C'] = scale_radii
        if np.isnan(hole_radii).sum() < len(hole_radii):
            radii_select['R_hole'] = hole_radii

        # 2 Retrieve masses
        disc_masses, gas_wind = self.mass

        # 3 Dust
        if driver.dust:
            dust_radii = self.radii_dust
            dust_masses, dust_wind = self.mass_dust

        # 4 Accretion rates
        Macc, Mext, Mint = self.mdot

        # 5 Photoevaporation rates
        Mevap = {}
        if driver.photoevaporation_external:
            Mevap['M_ext'] = Mext
        if driver.photoevaporation_internal:
            Mevap['M_int'] = Mint

        # Save data
        outputdata = np.column_stack((used_times, disc_masses))
        head  = ['t','M_D']
        units = ['yr','g']
        if driver.dust:
            outputdata = np.column_stack((outputdata, dust_masses))
            head.append('M_d')
            units.append('g')

        for key, radii in radii_select.items():
            outputdata = np.column_stack((outputdata, radii))
            head.append(key)
            units.append('AU')
        if driver.dust:
            for key, radii in dust_radii.items():
                outputdata = np.column_stack((outputdata, radii))
                head.append('R_{}'.format(int(float(key)*100)))
                units.append('AU')

        if driver.gas:
            outputdata = np.column_stack((outputdata, Macc))
            head.append('M_acc')
            units.append('Msun/yr')
        for key, mdot in Mevap.items():
            outputdata = np.column_stack((outputdata, mdot))
            head.append(key)
            units.append('Msun/yr')
        
        if driver.photoevaporation_external:
            outputdata = np.column_stack((outputdata, gas_wind))
            head.append('M_extg')
            units.append('g')
        if driver.photoevaporation_external and driver.dust:
            outputdata = np.column_stack((outputdata, dust_wind))
            head.append('M_extd')
            units.append('g')

        head  = "\t".join(head)
        units = "\t".join(units)
        full_head = "\n".join([head,units])
            
        np.savetxt(str(save_directory)+"/discproperties.dat", outputdata, delimiter='\t', header=full_head)

        return outputdata               
