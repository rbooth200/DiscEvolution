import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rcParams
rcParams['image.cmap'] = 'plasma'
from snap_reader import DiscReader, PlanetReader
from chemistry import SimpleCOMolAbund, SimpleCOAtomAbund



class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)

if __name__ ==  "__main__":
    import sys
    DIR = os.path.join('../planets/TimeDep/irradiated/Rc_100/Mdot_1e-08/')
    
    try:
        DIR = sys.argv[1]
    except IndexError:
        pass
    
    planets = PlanetReader(DIR, 'planets').compute_planet_evo()


    tf = {}
    for p in planets:
        if p.t_form[0] not in tf:
            tf[p.t_form[0]] = []
        tf[p.t_form[0]].append(p)

    N = len(tf)
    i = 0
    keys = sorted(tf.keys())
    plt.ion()
    while True:
        plt.clf()
        
        for p in tf[keys[i]]:
            plt.subplot(311)
            plt.loglog(p.time, p.M / 317.8)
            plt.subplot(312)
            plt.loglog(p.time, p.M_core)
            plt.subplot(313)
            plt.loglog(p.time, p.R)

        plt.subplot(311)
        plt.title('t_0 = {:g}yr'.format(p.t_form[0]))
        plt.ylabel('$M\,[M_J]$')
        plt.xlim(xmin=1e4)
        plt.subplot(312)
        plt.ylabel('$M_c\,[M_\oplus]$')
        plt.xlim(xmin=1e4)
        plt.subplot(313)
        plt.xlabel('$t\,[\mathrm{yr}]$')
        plt.ylabel('$R\,[\mathrm{au}]$')
        plt.xlim(xmin=1e4)


        plt.pause(1)
        i = (i+1) % N
        plt.show()
