# io.py
#
# Author: R. Booth
# Date: 21 - May - 2018
#
# Input/Ouput routines
###############################################################################
from __future__ import print_function
import numpy as np

__all__ = [ "Event_Controller", "dump_ASCII" ]

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


def _test_event_controller():
    
    t0 = [0, 2, 4, 6, 8 ]
    t1 = [0, 3, 6, 9]

    # Label for the next possible events
    next_event = [ 'et', 'e', 'e', 't', 'e', 'et', 'et', 'e', 'e', 't']

    EC = Event_Controller(evens=t0, threes=t1)
    
    assert(set(EC.event_types()) == set(['evens', 'threes']))

    try:
        EC.next_event_time('odds')
        failed = False
    except KeyError:
        failed = True
    assert(failed)

    for i in range(10):
        t_next = EC.next_event_time()    

        assert(EC.next_event()[0] in next_event[i])
        
        if (i % 2) == 0:
            assert(t_next == EC.next_event_time('evens'))
            assert(EC.event_number('evens') == (i/2))
        if (i % 3) == 0:
            assert(t_next == EC.next_event_time('threes'))
            assert(EC.event_number('threes') == (i/3))

        assert(not EC.finished())
        EC.pop_events(i)        

    assert(EC.finished())


################################################################################
# Write data to an ASCII file
################################################################################
def dump_ASCII(filename, disc, time, header=None):
    """Write an ASCII dump of the disc data.

    args:
        filename : name of the new dump file
        disc     : disc object
        time     : current time (in Omega0)
        header   : Additional header data to write (default = None)
    """

    head = disc.ASCII_header() + '\n'
    if header is not None:
        head += header
        if not head.endswith('\n'):
            head += '\n'

    with open(filename, 'w') as f:
        f.write(head)
        f.write('# time: {}yr\n'.format(time / (2 * np.pi)))

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

if __name__ == "__main__":

    _test_event_controller()
