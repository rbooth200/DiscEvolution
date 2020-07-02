# io.py
#
# Author: R. Booth
# Date: 21 - May - 2018
#
# Input/Ouput routines
###############################################################################
from __future__ import print_function
from six import string_types
import collections
import numpy as np
import re
import os

from . import constants
from .chemistry import create_abundances, MolecularIceAbund

__all__ = [ "Event_Controller", "dump_ASCII", "dump_hdf5", "DiscReader" ]
###############################################################################
# I/O Controller
###############################################################################
class Event_Controller(object):
    """Handles booking keeping for events that occur at the specified times.

    Event types are specified through key word arguments, e.g.
        Event_Controller(save=output_times, plot=plot_times)
    where the values passed must be iterable lists of times.
    """
    def __init__(self, **events):
        
        self._events  = {}
        self._event_number = {}
        for key in events:
            self._events[key] = sorted(events[key])
            self._event_number[key] = 0

    def event_types(self):
        return self._events.keys()

    def event_times(self, event_type):
        """Return the times of the specified event type"""
        return self._events[event_type]

    def next_event_time(self, event_type=None):
        """Time of next event. 

        If no event type is specified, the next event of any time is returned
        """
        if event_type is not None:
            return self._next_event_time(event_type)
        else:
            # All events:
            t_next = np.inf
            for event in self._events:
                t_next = min(t_next, self._next_event_time(event))
            return t_next

    def _next_event_time(self, event):
        try:
            return self._events[event][0]
        except IndexError:
            return np.inf

    def next_event(self):
        """The type of the next event"""
        t_next = self.next_event_time()
        for event in self._events:
            if self._next_event_time(event) == t_next:
                return event    
        return None

    def check_event(self, t, event):
        """Has the next occurance of a specified event occured?"""
        try:
            return self._events[event][0] <= t
        except IndexError:
            return False

    def events_passed(self, t):
        """Returns a list of event types that have passed since last pop"""
        return [ e for e in self._events if self.check_event(t, e) ]

    def event_number(self, key):
        """The number of times the specified event has occurred"""
        return self._event_number[key]

    def pop_events(self, t, event_type=None):
        """Remove events that have passed.

        If no event type is specified, pop all event types
        """
        if event_type is not None:
            self._pop(t, event_type)
        else:
            for event in self._events:
                self._pop(t, event)

    def _pop(self, t, event):        
        try:
            while self.check_event(t, event):
                self._events[event].pop(0)
                self._event_number[event] += 1
        except IndexError:
            pass
        
    def finished(self):
        """Returns True all events have been popped"""
        for event in self._events:
            if len(self._events[event]) > 0:
                return False
        else:
            return True


################################################################################
# Write data to an ASCII file
################################################################################
def dump_ASCII(filename, disc, time, header=None):
    """Write an ASCII dump of the disc data.

    args:
        filename : string
            Name of the new dump file
        disc     : disc object
            Disc that will be saved to disc.
        time     : float 
            Current time (in Omega0)
        header   : string, list of strings, or None
            Additional header data to write 
    """
    if header is None:
        header = []
    if isinstance(header, string_types):
        header = [header,]

    # Construct the header
    head = disc.ASCII_header() + '\n'
    for h in header:
        head += h
        if not h.endswith('\n'):
            head += '\n'

    with open(filename, 'w') as f:
        f.write(head)
        f.write('# time: {}yr\n'.format(time / constants.yr))

        # Construct the list of variables that we are going to print
        Ncell = disc.Ncells
        
        Ndust = 0
        try:
            Ndust = disc.dust_frac.shape[0]
        except AttributeError:
            pass

        head = '# R Sigma T'
        for i in range(Ndust):
            head += ' epsilon[{}]'.format(i)
        for i in range(Ndust):
            head += ' a[{}]'.format(i)
            
        chem = None
        try:
            chem = disc.chem
            for k in chem.gas:
                head += ' {}'.format(k)
            for k in chem.ice:
                head += ' s{}'.format(k)
        except AttributeError:
            pass

        f.write(head+'\n')
        
        R, Sig, T = disc.R, disc.Sigma, disc.T
        for i in range(Ncell):
            f.write('{} {} {}'.format(R[i], Sig[i], T[i]))
            for j in range(Ndust):
                f.write(' {}'.format(disc.dust_frac[j, i]))
            for j in range(Ndust):
                f.write(' {}'.format(disc.grain_size[j, i]))
            if chem:
                for k in chem.gas:
                    f.write(' {}'.format(chem.gas[k][i]))
                for k in chem.ice:
                    f.write(' {}'.format(chem.ice[k][i]))
            f.write('\n')

################################################################################
# Write data using HDF5
################################################################################
def _write_nested_hdf5_header(hdf_obj, header):
    """Write the header data as attributes in a nested fashion."""
    for key, value in header.items():
        if isinstance(value, collections.Mapping):
            group = hdf_obj.create_group(key)
            _write_nested_hdf5_header(group, value)
        else:
             hdf_obj.attrs[key] = value

def dump_hdf5(filename, disc, time, headers=None):
    """Write disc data to file.

    args:
        filename : string
            Name of the new dump file
        disc     : disc object
            Disc that will be saved to disc.
        time     : float 
            Current time (in Omega0)
        header   : string, list of strings, or None
            Additional header data to write. Each item should be a single 
    """
    try:
        import h5py  # Here to avoid forcing h5py depedency
    except ImportError:
        # Raise own more useful message
        msg = ("{}.{} {}".format("DiscEvolution.io", "dump_hdf5",
                                 "requires h5py, which could not be found."))
        raise ImportError(msg)

    if headers is None:
        headers = []

    with h5py.File(filename, "w") as f:
        # Write the header information for auxillary objects
        for name, attrs in headers:
            group = f.create_group(name)
            _write_nested_hdf5_header(group, attrs)

        # Generate and write the header information for the disc data
        name, attrs = disc.HDF5_attributes()
        _write_nested_hdf5_header(f.create_group(name), attrs)

        f.create_dataset("time", data=time/constants.yr).attrs["units"] = "yr"

        # Create a group for the physical data, store the disc type
        phys = f.create_group("data")
        phys.attrs["disc model"] = name

        # Store the physical data, along with units:
        R = phys.create_dataset("radius", data=disc.R)
        R.attrs["units"] = "au"

        Sigma = phys.create_dataset("surface density", data=disc.Sigma)
        Sigma.attrs["units"] = "g cm^-2"

        T = phys.create_dataset("temperature", data=disc.T)
        T.attrs["units"] = "K"

        # Dust properties, if present
        try:
            a, eps = disc.grain_size, disc.dust_frac

            Ndust = a.shape[0]

            dust = phys.create_group("dust")
            dust.attrs["number of dust species"] = Ndust

            size = dust.create_dataset("grain size", data=a)
            size.attrs["units"] = "cm"

            dust_frac = dust.create_dataset("mass fraction", data=eps)
            dust_frac.attrs["units"] = ""
        except  AttributeError:
            pass

        # Chemistry, if present
        try:
            gas, ice = disc.chem.gas, disc.chem.ice

            chem = phys.create_group("chemistry")

            for name, phase in zip( ["gas", "ice"], [gas, ice]):
                data = chem.create_group(name)
                data.attrs["number of species"] = phase.Nspec

                # Names / Masses of chemical species
                names = np.array(phase.names).astype('S') # Use ASCII strings
                data.create_dataset("names", data=names)
                masses = data.create_dataset("masses", data=phase.masses)
                masses.attrs["units"] = "proton masses"

                abund = data.create_dataset("mass fraction", data=phase.data)
                abund.attrs["units"] = ""
        except AttributeError:
            pass


################################################################################
# Snapshot data Readers
################################################################################
class DiscSnap(object):
    """Base class for disc data"""
    @property
    def photo_type(self):
        if hasattr(self,"_IPE"):
            return self._IPE        
        else:
            return None
    @property
    def time(self):
        return self._t
    @property
    def R(self):
        return self._R
    @property
    def Sigma(self):
        return self._Sigma
    @property
    def T(self):
        return self._T
    @property
    def dust_frac(self):
        return self._eps
    @property
    def grain_size(self):
        return self._a
    @property
    def chem(self):
        return self._chem


class AsciiDiscSnap(DiscSnap):
    """Reads data from ASCII dump file.
    
    args:
        filename : string
            Name of the file to read
    """
    def __init__(self, filename):
         self.read(filename)

    def read(self, filename):
        """Read disc data from file"""
        # read the header
        head = ''
        found_time  = False
        count = 0
        with open(filename) as f:
            for line in f:
                if not found_time:

                    if not (line.startswith('# time') or line.startswith('# InternalEvaporation')):
                        head += line
                    elif line.startswith('# InternalEvaporation'):
                        # Get internal photoevaporation type
                        self._IPE = line.split(',')[1].split(':')[-1].lstrip()
                        print(self._IPE)
                    elif line.startswith('# time'):
                        found_time = True
                        # Get the time
                        self._t = float(line.strip().split(':')[1][:-2])
                    count += 1
                    continue
                # Get data variables stored
                data = line[2:].split(' ')
                assert(len(data) % 2 == 1)

                # Get the number of dust species
                Ndust = len([x for x in data  if x.startswith('epsilon')])
                Nchem = (len(data) - 3 - 2*Ndust) // 2
                
                iChem = 2*Ndust + 3
                chem_spec = data[iChem:iChem + Nchem]
                break
            
        # Parse the actual data:
        data = np.genfromtxt(filename, skip_header=count, names=True)
        Ndata = data.shape[0]
        names = data.dtype.names
        self._R     = data['R']
        self._Sigma = data['Sigma']
        self._T     = data['T']

        self._eps = np.empty([Ndust, Ndata], dtype='f8')
        self._a   = np.empty([Ndust, Ndata], dtype='f8')
        for i in range(Ndust):
            self._eps[i] = data['epsilon{}'.format(i)]
            self._a[i]   = data['a{}'.format(i)]


        if Nchem:
            gas = create_abundances(names[iChem:iChem+Nchem], data)
            ice = create_abundances(names[iChem+Nchem:], data, grain_prefix='s')

            self._chem = MolecularIceAbund(gas, ice)


class Hdf5DiscSnap(DiscSnap):
    """Reads data from an HDF5 dump file.
    
    args:
        filename : string
            Name of the file to read
    """
    def __init__(self, filename):
        self.read(filename)

    def read(self, filename):
        try:
            import h5py  # Here to avoid forcing h5py depedency
        except ImportError:
            # Raise own more useful message
            msg = "{}.{} {}".format("DiscEvolution.io", "Hdf5DiscSnap", 
                                    "requires h5py, which could not be found.")
            raise ImportError(msg)


        self._h5File = f = h5py.File(filename, "r")
        
        self._t = float(np.array(f['time']))
        
        data = f['data']             
        self._R     = np.array(data['radius'])
        self._Sigma = np.array(data['surface density'])
        self._T     = np.array(data['temperature'])
            
        try:
            dust = data['dust']
            self._a   = np.array(dust['grain size'])
            self._eps = np.array(dust['mass fraction'])
        except KeyError:
            pass

        try:
            chem = data['chemistry']
            
            gas = chem['gas']
            names, masses = np.array(gas['names']), np.array(gas['masses'])
            names = np.array([n.decode() for n in names])                
            gas = create_abundances(names,
                                    np.rec.fromarrays(gas['mass fraction'],
                                                      names=tuple(names)),
                                    masses=masses)

            ice = chem['ice']
            names, masses = np.array(ice['names']), np.array(ice['masses'])
            names = np.array([n.decode() for n in names])
            ice = create_abundances(names,
                                    np.rec.fromarrays(ice['mass fraction'],
                                                      names=tuple(names)),
                                    masses=masses)

            self._chem = MolecularIceAbund(gas, ice)
        except KeyError:
            pass
                
    @property
    def h5File(self):
        return self._h5File


class Reader(object):
    """Base class for reading data from entire simulation"""
    def __init__(self, SnapType, DIR, base='*', tail='.dat'):
        self._SnapType = SnapType
        self._DIR = DIR

        m = re.compile(r'^'+base+r'_\d\d\d\d'+tail+'$')
        self._files = [ f for f in os.listdir(DIR) if m.findall(f)]
        
        snaps = {}
        Nmax = 0
        for f in self._files:
            n = int(f[-(4+len(tail)):-len(tail)])
            snaps[n] = os.path.join(self._DIR, f)
            Nmax = max(n, Nmax)
        self._snaps = snaps
        self._Nmax = Nmax
        
            
    def __getitem__(self, n):
        return self._SnapType(self._snaps[n])

    def filename(self, n):
        return self._snaps[n]

    @property
    def Num_Snaps(self):
        return self._Nmax

                    
class DiscReader(Reader):
    """Read disc snaphshots from file"""
    def __init__(self, DIR, base='disc', type='hdf5'):
        
        if type.lower() == 'ascii':
            SnapType = AsciiDiscSnap
            extension =  '.dat'
        else:
            SnapType = Hdf5DiscSnap
            extension = '.h5'

        super(DiscReader, self).__init__(SnapType, DIR, base, extension)
