import os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rcParams
import matplotlib.animation as animation

rcParams['image.cmap'] = 'plasma'
#mpl.rcParams['figure.figsize'] = 13.94, 7.96 # twice the mnras fig size
##                                              for a 2-column figure
mpl.rcParams['figure.figsize'] = 1.5*6.64, 1.5*4.98 # twice the mnras fig size
#                                             for a 1-column figure
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 16
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['legend.numpoints'] = 1
mpl.rcParams['legend.handlelength'] = 1
mpl.rcParams['legend.frameon'] = 0
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['lines.markeredgewidth'] = 2
mpl.rcParams['lines.markersize'] = 6
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
from DiscEvolution.io import DiscReader
from DiscEvolution.chemistry.atomic_data import atomic_abundances

if __name__ == "__main__":

    DIR = sys.argv[1]
    reader = DiscReader(DIR, 'disc')

    f, ax = plt.subplots(1)
    ax.set_xlabel('Radius [au]')
    ax.set_ylabel('Surface Density [g/cm$^2$]')
    ax.set_ylim(1e-5, 1e2)


    for n in sys.argv[2:]:
        n = int(n)



        disc = reader[n]

        R = disc.R
        a = disc.grain_size[:,0]

        sigma_d = disc.dust_frac * disc.Sigma
        l, = ax.loglog(R, disc.Sigma * (1 - disc.dust_frac.sum(0)),
                       label=str(n))

        ax.loglog(R, sigma_d.sum(0), c=l.get_color(), ls='--')

        if n > 0 or len(sys.argv) == 2:
            plt.figure()
            dloga = np.diff(np.log(a)).mean()

            a_St01 = 0.1 * 2*disc.Sigma/np.pi
            a_bar = (sigma_d*disc.grain_size).sum(0) / sigma_d.sum(0)

            plt.pcolormesh(R, a, sigma_d/dloga, norm=LogNorm(vmin=1e-5))
            plt.plot(R, a_St01, c='k')
            plt.plot(R, a_bar)


            plt.colorbar(label='Dust Surface Density [g/cm$^2$]')
            
            plt.xlabel('Radius [au]')
            plt.ylabel('Grain size [cm]')
            plt.xscale('log')
            plt.yscale('log')

    ax.legend()
    plt.show()
