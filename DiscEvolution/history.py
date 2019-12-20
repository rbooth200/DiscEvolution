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

class dust_history(history):

    def __init__(self):
        super().__init__()
        
        # Extra stuff for dust

    def mass(self):
        return self._Mtot
