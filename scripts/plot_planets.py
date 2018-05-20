from __future__ import print_function
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
    DIR = os.path.join('..', 'planets', 'TimeDep', 'irradiated', 'Model43')
    
    try:
        DIR = sys.argv[1]
    except IndexError:
        pass

    print('Model params:')
    for l in open(os.path.join(DIR, 'model.dat')):
        print('\t', l.strip())
    print()

    planets = PlanetReader(DIR, 'planets').compute_planet_evo()
    
    solar = SimpleCOAtomAbund(1)
    solar.set_solar_abundances()

    f1, s1 = plt.subplots(3,1, sharex=True)
    f2 = plt.figure()
    s2 = [ plt.subplot(221), plt.subplot(222), plt.subplot(212)]
    

    Rs, Rss, C_Hs, C_Os, Ms, ts= [],[], [], [], [], []
    for ps in planets:
        p0 = ps[0]
        p = ps[-1]
        if not (0.5 < p.R <   50): continue
        if not (20  < p.M < 31780): continue

        s1[0].loglog(ps.time, ps.M / 317.8)
        s1[1].loglog(ps.time, ps.M_core)
        s1[2].loglog(ps.time, ps.R)

        mol_abund = SimpleCOMolAbund(2)
        mol_abund.data[:,0] = p.X_core
        mol_abund.data[:,1] = p.X_env

        atom = mol_abund.atomic_abundance()
        atom_tot = SimpleCOAtomAbund(1)
        for k in atom_tot:
            atom_tot[k] = (atom[k][0]*p.M_core + atom[k][1]*p.M_env)/p.M

            
        R = [p.R, p.R]
        C_H = [atom['C'][1]/solar['C'], atom_tot['C']/solar['C']]
        C_O = [atom['C'][1]/atom['O'][1], atom_tot['C']/atom_tot['O']]

        s2[0].quiver(p.R, C_H[0], 0, C_H[1] - C_H[0],
                     scale_units='xy', angles='xy', scale=1)
        s2[1].quiver(p.R, C_O[0], 0, C_O[1] - C_O[0],
                     scale_units='xy', angles='xy', scale=1)

        Rs.append(p.R)
        Rss.append(R)
        C_Hs.append(C_H)
        C_Os.append(C_O)
        Ms.append(p.M / 317.8)
        ts.append(p.t_form/1e6)

    im0 = s2[0].scatter(Rss, C_Hs, c=C_Os, s=25,
                        edgecolor='none',
                        vmin=0.0, vmax=1)

    im1 = s2[1].scatter(Rss, C_Os, c=C_Hs, s=25,
                        norm=LogNorm(),
                        edgecolor='none')

    im2 = s2[2].scatter(Rs, Ms, c=ts,
                        s=25, edgecolor='none')

    s1[0].set_ylabel('$M\,[M_J]$')
    s1[1].set_ylabel('$M_c\,[M_\oplus]$')
    s1[2].set_ylabel('$R\,[\mathrm{au}]$')
    s1[2].set_xlabel('$t\,[\mathrm{yr}]$')

    s2[0].set_ylabel('$[C/H]$')
    s2[0].set_xlabel('$R\,[\mathrm{au}]$')
    s2[0].set_xscale('log')
    s2[0].set_yscale('log')

    s2[1].set_ylabel('$[C/O]$')
    s2[1].set_xlabel('$R\,[\mathrm{au}]$')
    s2[1].set_xscale('log')

    s2[2].set_ylabel('$M\,[M_J]$')
    s2[2].set_xlabel('$R\,[\mathrm{au}]$')
    s2[2].set_xscale('log')
    

    cbar = f2.colorbar(im0,ax=s2[0])
    cbar.set_label('[C/O]')
    cbar = f2.colorbar(im1,ax=s2[1])
    cbar.set_label('[C/H]')
    cbar = f2.colorbar(im2,ax=s2[2])
    cbar.set_label('$t_0\,[10^6\,\mathrm{yr}]$')

    plt.show()
