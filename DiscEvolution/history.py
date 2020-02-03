# history.py
#
# Author: R. Booth
# Date: 17 - Nov - 2016
#
# Combined model for dust, gas and chemical evolution
################################################################################
from __future__ import print_function
import numpy as np

class history(object):

    def __init__(self):
        self._threshold = 1e-5          # Threshold for defining edge by density

        """Radii"""
        self._Rout  = np.array([])      # Outer radius of the disc (density), updated internally
        self._Rc_t  = np.array([])      # Radius of current best fit scale radius, updated internally
        self._Rot   = np.array([])      # Radius where Mdot maximum ie where becomes optically thick, updated internally
        self._Rh    = np.array([])      # Outer radius of the transition disc hole

        """Mass"""
        self._Mtot = np.array([])       # Total mass, updated internally

        """Mass Loss"""
        self._Mdot_acc = np.array([])   # Accretion rate, updated with velocity passed
        self._Mdot_ext = np.array([])   # External photoevaporation rate
        self._Mdot_int = np.array([])   # Internal photoevaporation rate

    def radii(self):
        return self._Rout, self._Rc_t, self._Rot, self._Rh

    def mass(self):
        return self._Mtot

    def mdot(self):
        return self._Mdot_acc, self._Mdot_ext, self._Mdot_int

    def restart(self, restartdata, time):
        not_future = restartdata['t'] <= time

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

class dust_history(history):

    def __init__(self, thresholds):
        super().__init__()

        self._dthresholds = thresholds  # Threshold percentiles of dust mass
        
        """Radii"""
        self._Rdust = {}                # Radius containing user defined fraction of dust
        for threshold in self._dthresholds:
            self._Rdust[threshold] = np.array([])

        """Mass"""
        self._Mdust = np.array([])      # Mass of dust in disc
        self._Mwind = np.array([])      # Amount of dust lost to wind
        self._Mwind_cum  = 0.           # Amount of dust lost to wind       # THIS IS UPDATED EVERY STEP BY PHOTOEVAPORATION.PY

    def mass_dust(self):
        return self._Mdust, self._Mwind

    def radii_dust(self):
        return self._Rdust

    def restart(self, restartdata, time):
        super().restart(restartdata, time)
        not_future = restartdata['t'] <= time

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

