import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rcParams
rcParams['image.cmap'] = 'plasma'
from snap_reader import DiscReader, PlanetReader
from chemistry import SimpleCOMolAbund, SimpleCOAtomAbund



def make_plot_planets(planets):
    for p in planets:
        plt.subplot(211)
        plt.loglog(p.R, p.M / 317.8)
        plt.subplot(212)
        plt.loglog(p.R, p.M_core)

    plt.subplot(211)
    plt.title('t_0 = {:g}yr'.format(p.t_form[0]))
    plt.ylabel('$M\,[M_J]$')
    plt.plot([0.1, 300], [1,1], 'k--')
    plt.xlim(0.1, 300)
    plt.subplot(212)
    plt.ylabel('$M_c\,[M_\oplus]$')
    plt.xlabel('$R\,[\mathrm{au}]$')
    plt.xlim(0.1, 300)

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


    try:
        tn = float(sys.argv[2])
        make_plot_planets(tf[tn])
        plt.show()
        exit()
    except IndexError:
        pass

    N = len(tf)
    i = 0
    keys = sorted(tf.keys())
    plt.ion()
    while True:
        plt.clf()
        
        make_plot_planets(tf[keys[i]])

        plt.show()
        plt.pause(1)
        i = (i+1) % N
