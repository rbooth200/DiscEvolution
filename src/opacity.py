import numpy as np
from scipy.interpolate import interp2d

def _Zhu2012(rho, T):
    '''Zhu, Hartmann, Nelson & Gammie (2012) opacity tables for PPD's
    
    args:
        rho : density, c.g.s
        T   : Temperature, K
    '''

    pre=rho*T*8.314472e7/2.4
    xlp = np.log10(pre)
    xlt = np.log10(T)

    if xlt < 2.23567+0.01899*(xlp-5.):
	xlop=1.5*(xlt-1.16331)-0.736364
    elif xlt < 2.30713+0.01899*(xlp-5.):
	xlop=-3.53154212*xlt+8.767726-(7.24786-8.767726)*(xlp-5.)/16.
    elif xlt<2.79055:
        xlop=1.5*(xlt-2.30713)+0.62	
    elif xlt<2.96931:
	xlop=-5.832*xlt+17.7
    elif xlt<3.29105+(3.29105-3.07651)*(xlp-5.)/8.:
	xlop=2.129*xlt-5.9398
    elif xlt<3.08+0.028084*(xlp+4):
        xlop=129.88071-42.98075*xlt+(142.996475-129.88071)*0.1*(xlp+4)
    elif xlt<3.28+xlp/4.*0.12:
        xlop=-15.0125+4.0625*xlt
    elif xlt<3.41+0.03328*xlp/4.:
        xlop=58.9294-18.4808*xlt+(61.6346-58.9294)*xlp/4.
    elif xlt<3.76+(xlp-4)/2.*0.03:
        xlop=-12.002+2.90477*xlt+(xlp-4)/4.*(13.9953-12.002)
    elif xlt<4.07+(xlp-4)/2.*0.08:
        xlop=-39.4077+10.1935*xlt+(xlp-4)/2.*(40.1719-39.4077)
    elif xlt<5.3715+(xlp-6)/2.*0.5594:
        xlop=17.5935-3.3647*xlt+(xlp-6)/2.*(17.5935-15.7376)
    else:
        xlop=-0.48
        
    if xlop < 3.586*xlt-16.85 and xlt<4.:
        xlop=3.586*xlt-16.85

    return 10.**xlop

Zhu2012 = _Zhu2012



class Zhu2012_tab(object):
    '''Tabulated form of zhu 2012'''
    def __init__(self, rho_min, rho_max, T_min, T_max,
                 Nrho, NT):

        lrho_i = np.linspace(np.log10(rho_min), np.log10(rho_max), Nrho)
        lT_i   = np.linspace(np.log10(T_min), np.log10(T_max), NT)

        grho_i, gT_i = np.meshgrid(lrho_i, lT_i)
        opac = np.vectorize(_Zhu2012)(10**grho_i, 10**gT_i)

        self._opac = interp2d(lrho_i, lT_i, np.log10(opac))
        self._rho = [rho_min, rho_max]
        self._T   = [T_min, T_max]

    def __call__(self, rho, T):
        if (self._rho[0]<=rho<=self._rho[1]) and (self._T[0]<=T<=self._T[1]):
            return 10**self._opac(np.log10(rho), np.log10(T))
        return Zhu2012(rho, T)
