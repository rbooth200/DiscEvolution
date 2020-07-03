# history.py
#
# Author: A. Sellek
# Date: 30 Jun 2020
#
# Classes for tracking evolutino of global disc quantities
#
################################################################################
from __future__ import print_function
import numpy as np

class history(object):

    def __init__(self):
        self._threshold = 1e-5          # Threshold for defining edge by density

        self._times = np.array([])      # Output times

        """Radii"""
        self._Rout  = np.array([])      # Outer radius of the disc (density), updated through disc.py
        self._Rc_t  = np.array([])      # Radius of current best fit scale radius, updated through disc.py
        self._Rot   = np.array([])      # Radius where Mdot maximum ie where becomes optically thick, updated through photoevaproation.py
        self._Rh    = np.array([])      # Outer radius of the transition disc hole, updated through internal_photo.py

        """Mass"""
        self._Mtot = np.array([])       # Total mass, updated through disc.py

        """Mass Loss"""
        self._Mdot_acc = np.array([])   # Accretion rate, updated with velocity passed through disc.py
        self._Mdot_ext = np.array([])   # External photoevaporation rate, updated through photoevaproation.py
        self._Mdot_int = np.array([])   # Internal photoevaporation rate, updated through internal_photo.py

    # Return times
    def times(self):
        return self._times

    # Return radii/masses/mass loss rates
    def radii(self):
        return self._Rout, self._Rc_t, self._Rot, self._Rh

    def mass(self):
        return self._Mtot

    def mdot(self):
        return self._Mdot_acc, self._Mdot_ext, self._Mdot_int

    # When restarting, for any variable found in input file, import data
    def restart(self, restartdata, time):
        not_future = restartdata['t'] <= time

        try:
            self._times = restartdata['t'][not_future]
        except KeyError:
            pass
        try:
            self._Rout = restartdata['R_out'][not_future]
        except KeyError:
            pass
        try:
            self._Rc_t = restartdata['R_C'][not_future]
        except KeyError:
            pass
        try:
            self._Rot  = restartdata['R_out'][not_future]
        except KeyError:
            pass
        try:
            self._Rh   = restartdata['R_hole'][not_future]
        except KeyError:
            if 'M_int' in restartdata.keys():
                self._Rh = np.full_like(self._times,np.nan)
            else:
                pass                
        try:
            self._Mtot = restartdata['M_D'][not_future]
        except KeyError:
            pass
        try:
            self._Mdot_acc = restartdata['M_acc'][not_future]
        except KeyError:
            pass
        try:
            self._Mdot_ext = restartdata['M_ext'][not_future]
        except KeyError:
            pass
        try:
            self._Mdot_int = restartdata['M_int'][not_future]
        except KeyError:
            pass

        return not_future

# Extra handles for the dust
class dust_history(history):

    def __init__(self, thresholds):
        super().__init__()

        self._dthresholds = thresholds  # Threshold percentiles of dust mass
        
        """Radii"""
        self._Rdust = {}                # Radius containing user defined fraction of dust, updated through dust.py
        for threshold in self._dthresholds:
            self._Rdust[threshold] = np.array([])

        """Mass"""
        self._Mdust = np.array([])      # Mass of dust in disc, updated through dust.py
        self._Mwind = np.array([])      # Amount of dust lost to wind, updated through dust.py
        self._Mwind_cum  = 0.           # Amount of dust lost to wind, updated through photoevaporation.py

    # Return radii/masses/mass loss rates
    def radii_dust(self):
        return self._Rdust

    def mass_dust(self):
        return self._Mdust, self._Mwind

    # When restarting, for any variable found in input file, import data
    def restart(self, restartdata, time):
        not_future = super().restart(restartdata, time)

        try:
            self._Mdust = restartdata['M_d'][not_future]
        except KeyError:
            pass
        try:
            self._Mwind     = restartdata['M_wind'][not_future]
            self._Mwind_cum = restartdata['M_wind'][not_future][-1]
        except KeyError:
            pass

        for threshold in self._dthresholds:
            try:
                self._Rdust[threshold] = restartdata['R_{}'.format(int(float(threshold)*100))][not_future]
            except KeyError:
                pass

        return not_future

