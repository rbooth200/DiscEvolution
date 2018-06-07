import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def Zhu2012(rho, T, amax=None):
    """Opacity tables for PPD's from Zhu et al. (2012, [1]).

    An updated version of the Bell & Lin (1994, [2]) opacities (assuming ISM 
    grains).

    args:
        rho : float or array of float
            Density, c.g.s
        T : float or array of float
            Temperature, K
        amax : float or array of float, ignored
            Maximum grain size, cm       

    References:
        [1] Zhu, Z., Hartmann, L., Nelson, R. P., et al. 2012, ApJ, 746, 110.
        [2] Bell, K. R. & Lin, D. N.C. 1994, ApJ, 427, 987.
    """

    pre = rho * T * 8.314472e7 / 2.4
    xlp = np.log10(pre)
    xlt = np.log10(T)

    # Electron scattering
    xlop = -0.48

    # Water ice
    idx = xlt < 2.23567 + 0.01899 * (xlp - 5.)
    xlop = np.where(idx, 1.5 * (xlt - 1.16331) - 0.736364, xlop)
    done = idx

    # Ice Evaporation
    idx = (~done) & (xlt < 2.30713 + 0.01899 * (xlp - 5.))
    xlop = np.where(idx, -3.53154212 * xlt + 8.767726 - (7.24786 - 8.767726) * (xlp - 5.) / 16., xlop)
    done |= idx

    # Metal grains
    idx = (~done) & (xlt < 2.79055)
    xlop = np.where(idx, 1.5 * (xlt - 2.30713) + 0.62, xlop)
    done |= idx

    # Graphite Corrosion
    idx = (~done) & (xlt < 2.96931)
    xlop = np.where(idx, -5.832 * xlt + 17.7, xlop)
    done |= idx

    # Grain opacity
    idx = (~done) & (xlt < 3.29105 + (3.29105 - 3.07651) * (xlp - 5.) / 8.)
    xlop = np.where(idx, 2.129 * xlt - 5.9398, xlop)
    done |= idx

    # Silicate Evaporation
    idx = (~done) & (xlt < 3.08 + 0.028084 * (xlp + 4))
    xlop = np.where(idx, 129.88071 - 42.98075 * xlt + (142.996475 - 129.88071) * 0.1 * (xlp + 4), xlop)
    done |= idx

    # Water vapour
    idx = (~done) & (xlt < 3.28 + xlp / 4. * 0.12)
    xlop = np.where(idx, -15.0125 + 4.0625 * xlt, xlop)
    done |= idx

    # More water vapour
    idx = (~done) & (xlt < 3.41 + 0.03328 * xlp / 4.)
    xlop = np.where(idx, 58.9294 - 18.4808 * xlt + (61.6346 - 58.9294) * xlp / 4., xlop)
    done |= idx

    # Molecular opacity
    idx = (~done) & (xlt < 3.76 + (xlp - 4) / 2. * 0.03)
    xlop = np.where(idx, -12.002 + 2.90477 * xlt + (xlp - 4) / 4. * (13.9953 - 12.002), xlop)
    done |= idx

    # H scattering
    idx = (~done) & (xlt < 4.07 + (xlp - 4) / 2. * 0.08)
    xlop = np.where(idx, -39.4077 + 10.1935 * xlt + (xlp - 4) / 2. * (40.1719 - 39.4077), xlop)
    done |= idx

    # Bound-free, free-free
    idx = (~done) & (xlt < 5.3715 + (xlp - 6) / 2. * 0.5594)
    xlop = np.where(idx, 17.5935 - 3.3647 * xlt + (xlp - 6) / 2. * (17.5935 - 15.7376), xlop)
    done |= idx

    # Extra limits
    idx = (xlop < 3.586 * xlt - 16.85) & (xlt < 4.)
    xlop = np.where(idx, 3.586 * xlt - 16.85, xlop)

    return 10. ** xlop

class Tazzari2016(object):
    '''Tabulated opacities for icy grains.

    The grain composition is taken from Tazzari et al. (2016, [1]).

    args:
        dust_to_gas : float, default = 0.01
            Dust-to-gas mass ratio
        q : float, default = 3.
            Slope of the grain size distribution, n(a) da ~ a^-q da

    References:
        [1] Tazzari, M., Testi, L., Ercolano, B., et al. 2016, A&A, 588, A53.
    '''
    def __init__(self, dust_to_gas = 0.01, q=3.5):
        data_dir = os.path.join(os.path.dirname(__file__), 'data', 'opacity')


        kappa_table_abs = 'kappa_table_abs_q{}.npy'.format(q)
        kappa_table_sca = 'kappa_table_sca_q{}.npy'.format(q)
        
        a         = np.load(os.path.join(data_dir, 'amax.npy'))
        T         = np.load(os.path.join(data_dir, 'T.npy'))
        kappa_abs = np.load(os.path.join(data_dir, kappa_table_abs)).T
        kappa_sca = np.load(os.path.join(data_dir, kappa_table_sca)).T

        self._amin = a[0]
        self._amax = a[-1]
        self._Tmin = T[0]
        self._Tmax = T[-1]

        self._kappa = RegularGridInterpolator((np.log(a), np.log(T)),
                                              np.log(kappa_abs + kappa_sca).T)

        self._eps = dust_to_gas

        self._q = q

        # Setup the coefficients for extrapolation
        k0_30, kl_30 = 3.84368330e+00, 1.15351466e-02
        k0_35, kl_35 = 3.94169075e+00, 2.27306292e-02

        self._k0 = k0_30 + (k0_35-k0_30) * ((q-3)/0.5)
        self._kl = kl_30 + (kl_35-kl_35) * ((q-3)/0.5)

    def __call__(self, rho, T, a):
        """Evaluate the opacity

        args:
            rho : float or array of float, ignored
                Density, c.g.s
            T : float or array of float
                Temperature, K
            amax : float, or array of float
                Maximum grain size, cm       
        """
        # Clip the limits to the tabulated ranges
        #  T < Tmin should never be needed, and kappa(T) ~ const at high T.
        #  a < amin kappa ~ const. and a > amax, we scale

        a_clip = np.minimum(np.maximum(a, self._amin), self._amax)
        T_clip = np.minimum(np.maximum(T, self._Tmin), self._Tmax)

        kappa = np.exp(self._kappa((np.log(a_clip), np.log(T_clip))))

        # Scale for large grains
        amax = np.maximum(a, a_clip)
        kp = self._q - self._k0 + self._kl*np.log10(amax/100)
        kappa *= (amax/a_clip)**kp

        return kappa * (self._eps / 0.01)
        
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    T = np.logspace(0.5, 3.5, 300)
    rho = np.logspace(-10, 0, 301)

    T, rho = np.meshgrid(T, rho)

    vmax =  Zhu2012(rho, T).max()

    amax = np.logspace(-5, 3, 301)
    T = np.logspace(0.5, 3.5, 300)
    T, amax = np.meshgrid(T, amax)

    vmin =  Tazzari2016()(1., T, amax).min()


    plt.subplot(121)
    plt.title("Zhu2012")
    plt.pcolormesh(rho, T, Zhu2012(rho, T), norm=LogNorm(),
                   vmax=vmax, vmin=vmin)
    plt.colorbar(label='opacity [cm$^2$ g$^{-1}$]')
    plt.xlabel(r'density')
    plt.ylabel('T')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(T.min(), T.max())

    plt.subplot(122)
    plt.title("Tazzari2016")   
    plt.pcolormesh(amax, T, Tazzari2016()(1., T, amax), norm=LogNorm(),
                   vmax=vmax, vmin=vmin)
    plt.colorbar(label='opacity [cm$^2$ g$^{-1}$]')
    plt.xlabel('amax')
    plt.ylabel('T')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(T.min(), T.max())

    plt.tight_layout()

    plt.figure()
    
    rho  = 1e3 / (0.05*1.5e13)
    amax = 0.01

    T = np.logspace(0.5, 3.5, 1000)

    plt.loglog(T, Zhu2012(rho, T, amax), 'k', label='Zhu2012')
    plt.loglog(T, Tazzari2016()(rho, T, amax), 'k--', label='Tazzari2016')

    plt.legend()
    plt.xlabel(r'$T$ [K]')
    plt.ylabel(r'$\kappa$ [g cm$^2$]')


    plt.show()
