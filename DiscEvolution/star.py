# star.py
#
# Author: R. Booth
# Date: 8 - Nov - 2016
#  
# Contains stellar properties classes
################################################################################
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from .constants import Msun, Rsun, AU

# Base class for all stars, implements general properties that should be
# common to all stars
class StarBase(object):
    """Wrapper class for stellar properties.

    args:
        M     : mass, Msun.                  default = 1
        R     : radius, Rsun.                default = 1
        T_eff : effective temperature, K.    default = 5770
        age   : Stellar age, yr.             default = 0
    """
    def __init__(self, M=1,R=1,T_eff=5770, age=0):
        self._M    = M
        self._Rs   = R
        self._Rau  = R*Rsun/AU
        self._Teff = T_eff
        self._age = age

    def Omega_k(self,  r):
        """Keplerian angular speed of a test particle.

        args:
            r : distance, AU
        returns:
           Omega : 2 Pi AU / yr
        """
        return np.sqrt(self._M / (r*r*r))

    def v_k(self,  r):
        """Keplerian velocity of a test particle.

        args:
            r : distance, AU
        returns:
           Omega : 2 Pi AU / yr
        """
        return np.sqrt(self._M / r)

    def r_Hill(self, R, M):
        """Compute the hill radius of a planet

        args:
            R : radius, AU
            M : planet mass
        """
        return R * (M / (3*self._M))**(1/3.)

    def evolve(self, age, M=None):
        """Update the stellar properties based on current age and mass

        args:
           age : stellar age in Omega(1)^-1
           M   : mass, Msun
        """
        raise AttributeError("StarBase::Evolve must be implemented in "
                             "class")
        
    @property
    def Rs(self):
        """Radius is Rsun"""
        return self._Rs

    @property
    def Rau(self):
        """Radius in AU"""
        return self._Rau

    @property
    def M(self):
        """Mass in Msun"""
        return self._M

    @property
    def T_eff(self):
        """Effective temperature in K"""
        return self._Teff

    @property
    def age(self):
        """Stellar age"""
        return self._age

    def ASCII_header(self):
        """Print stellar type header"""
        head = '# {} M: {}Msun, R: {}Rsun, T: {}K, age: {}yr'
        return head.format(self.__class__.__name__,
                           self.M, self.Rs, self.T_eff, self.age)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        return self.__class__.__name__, { "M"   : "{} Msun".format(self.M),
                                          "R"   : "{} Rsun".format(self._Rs),
                                           "T"   : "{} K".format(self.T_eff),
                                           "age" : "{} yr".format(self.age),
                                        }


# A simple non-evolving star class
class SimpleStar(StarBase):
    """Simple non-evolving star."""
    def __init__(self, **kwargs):
        super(SimpleStar, self).__init__(**kwargs)

    def evolve(self, age, M=None):
        """Update the stellar properties based on current age and mass

        args:
           age : stellar age in yr
           M   : mass, Msun
        """
        self._age = age
        if M is not None: self._M = M

    @staticmethod
    def from_string(string):
        """Read a simple star from a string"""
        string = string.replace('# SimpleStar', '').strip()
        kwargs = {}
        for item in string.split(','):
            key, val = [ x.strip() for x in item.split(':')]

            if   key == 'M':
                kwargs[key] = float(val.replace('Msun','').strip())
            elif key == 'R':
                kwargs[key] = float(val.replace('Rsun','').strip())
            elif key == 'T' or key == 'T_eff':
                kwargs['T_eff'] = float(val.replace('K','').strip())
            elif key == 'age':
                kwargs[key] = float(val.replace('yr', '').strip())
            else:
                raise AttributeError("Error: Attribute {} for SimpleStar not "
                                     "known".format(key))
        return SimpleStar(**kwargs)
                                     
# A star with a photoevaporating luminosity
class PhotoStar(SimpleStar):
    def __init__(self, LX=1e30, Phi=0, **kwargs):
        super().__init__(**kwargs)
        self._L_X = LX
        self._Phi = Phi

    @property
    def L_X(self):
        """X-ray Luminosity"""
        return self._L_X

    @property
    def Phi(self):
        """EUV Photon Luminosity"""
        return self._Phi


class MesaStar(PhotoStar):
    """Star with data read from MESA output"""
    def __init__(self, data_file, M, age, **kwargs):
        super(MesaStar, self).__init__(M=M, age=age, **kwargs)
        self._file = data_file 

        self._load_data(self._file)

    def _load_data(self, data_file):
        # Get the column names:

        cols = {}
        with open(data_file, 'r') as f:
            line = f.readline()
            if not line.startswith('#'):
                raise ValueError("Expected header line in MESA output file")
            names = line[1:-1].split(',')
            for col, name in enumerate(names):
                cols[name.strip()] = col

        data = np.genfromtxt(data_file).T
        age = data[cols['Age']]
        Teff = 10**data[cols['log Teff']]
        R = 10**data[cols['log R']]


        self._tab_Teff = InterpolatedUnivariateSpline(age, Teff, ext='const')
        self._tab_R = InterpolatedUnivariateSpline(age, R, ext='const')

        self.evolve(self.age)

    def evolve(self, age):
        self._age = age

        self._Teff = self._tab_Teff(self.age)
        self._Rs = self._tab_R(self.age)
        self._Rau = self._Rs*Rsun/AU 

    def ASCII_header(self):
        """Print stellar type header"""
        head = '# {} M: {}Msun, R: {}Rsun, T: {}K, age: {}yr, '
        head += 'L_X: {}erg/s, Phi: {}, MESA_file: {}'
        return head.format(self.__class__.__name__,
                           self.M, self.Rs, self.T_eff, self.age,
                           self.L_X, self.Phi, self._file)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        return self.__class__.__name__, { "M"   : "{} Msun".format(self.M),
                                          "R"   : "{} Rsun".format(self._Rs),
                                           "T"   : "{} K".format(self.T_eff),
                                           "age" : "{} yr".format(self.age),
                                           "LX" : "{} erg/s".format(self.L_X),
                                           "Phi" : "{}".format(self.Phi),
                                           "MESA_file" : self._file,
                                        }

    @staticmethod
    def from_string(string):
        """Read a simple star from a string"""
        string = string.replace('# MesaStar', '').strip()
        
        kwargs = {}
        for item in string.split(','):
            key, val = [ x.strip() for x in item.split(':')]

            if   key == 'M':
                M = float(val.replace('Msun','').strip())
            elif key == 'age':
                age = float(val.replace('yr', '').strip())
            elif key == 'MESA_file':
                filname = val.strip()
            elif key == "L_X":
                kwargs[key] = float(val.replace('erg/s', '').strip())
            elif key == "Phi":
                kwargs[key] = float(val)

        return MesaStar(filname, M, age, **kwargs)

def from_file(filename):
    with open(filename) as f:
        for line in f:
            if not line.startswith('#'):
                raise AttributeError("Error: Star type not found in header")
            elif "SimpleStar" in line:
                return SimpleStar.from_string(line)
            elif "MesaStar" in line:
                return MesaStar.from_string(line)
            else:
                continue
            
