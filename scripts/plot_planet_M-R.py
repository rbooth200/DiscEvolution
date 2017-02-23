import os
import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rcParams
rcParams['image.cmap'] = 'magma'
from snap_reader import DiscReader, PlanetReader
from chemistry import SimpleCOMolAbund, SimpleCOAtomAbund

class Formatter(object):
    def __init__(self, x,y,z):
        self._x, self._y, self._z = x,y,z
        self._zi = interp2d(np.log(self._x), self._y, np.log(self._z),
                            bounds_error=True)
    def __call__(self, x, y):
        try:
            z = np.exp(self._zi(np.log(x), y)[0])
            return 'x={:5g}, y={:5g}, z={:5g}'.format(x, y, z)
        except ValueError:
            return 'x={:5g}, y={:5g}'.format(x, y)

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
    DIR = os.path.join('../planets/pb_gas_acc_f_0.0/TimeDep/'
                       'irradiated/Rc_200/Mdot_1e-09/')
    
    try:
        DIR = sys.argv[1]
    except IndexError:
        pass
    
    planets = PlanetReader(DIR, 'planets').compute_planet_evo()


    # Collect the planet data
    tf = {}
    for p in planets:
        if p.t_form[0] not in tf:
            tf[p.t_form[0]] = []
        tf[p.t_form[0]].append(p)
    
    t_form = np.array(sorted(tf.keys()), dtype='f8')
    R_form = np.array([p.R[0] for p in tf[t_form[0]]], dtype='f8')

    M_final = np.empty([t_form.shape[0], R_form.shape[0]], dtype='f8')
    R_final = np.empty([t_form.shape[0], R_form.shape[0]], dtype='f8')

    for i, ti in enumerate(t_form):
        for j, p in enumerate(tf[ti]):
            M_final[i,j] = p[-1].M
            R_final[i,j] = p[-1].R
    
    t_form /= 1e6
    MJ = 317.8
    im = plt.pcolormesh(R_form, t_form, M_final, shading='gouraud',
                        norm=LogNorm(), vmin=0.5, vmax=10*MJ)
    plt.gca().format_coord = Formatter(R_form, t_form, M_final)

    CS = plt.contour(R_form, t_form, R_final, colors='b', 
                levels=[0.5, 1, 2, 5, 10, 20, 50, 100])
    plt.colorbar(im, label='$M\,[\mathrm{M}_\oplus]$')
    plt.clabel(CS, CS.levels[1::2], inline=True, fmt='%.1f')
               #manual=[(21, 0.2), (20, 0.8), (63, 0.5)])
               


    plt.xscale('log')

    plt.xlabel('$R\,[\mathrm{au}]$')
    plt.ylabel('$t_\mathrm{form}\,[\mathrm{Myr}]$')


    plt.show()
