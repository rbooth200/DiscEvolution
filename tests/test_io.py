# test_io.py
#
# Author: R. Booth
# Date: 22 - May - 2018
#
# Test input / output routines
###############################################################################
from DiscEvolution.io import  Event_Controller

def test_event_controller():
    """Tests that event controller correctly generates events"""
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

