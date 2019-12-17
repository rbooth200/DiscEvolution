import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from DiscEvolution.constants import *
from DiscEvolution.star import PhotoStar

DefaultModel = "DiscConfig_default.json"
plt.rcParams['text.usetex'] = "True"
plt.rcParams['font.family'] = "serif"

class PhotoBase():
    def __init__(self, R, star):
        self.mdot_X(star)
        self._Sigmadot = np.zeros_like(R)

    def mdot_X(self, star):
        self._MdotX = 0

    def Sigma_dot(self, R, star):
        self._Sigmadot = np.zeros_like(R)

    def scaled_R(self, R, star):
        raise AttributeError("PhotoBase::scaled_R must be implemented in subclass")

    def get_dt(self, disc):
        where_photoevap = (self.dSigmadt > 0)
        dt = disc.Sigma_G[where_photoevap] / self.dSigmadt[where_photoevap]
        return dt, min(dt)

    def remove_mass(self, disc, dt):
        _, t_w = self.get_dt(disc)
        if (dt > t_w):
            raise Exception("Timestep is too short - cells would be completely emptied")
        disc._Sigma -= self.dSigmadt * dt

    @property
    def Mdot(self):
        return self._MdotX

    @property
    def dSigmadt(self):
        return self._Sigmadot

"""
Primoridal Discs (Owen+12)
"""
class PrimordialDisc(PhotoBase):
    def __init__(self, R, star):
        super().__init__(R, star)
        self._a1 = 0.15138
        self._b1 = -1.2182
        self._c1 = 3.4046
        self._d1 = -3.5717
        self._e1 = -0.32762
        self._f1 = 3.6064
        self._g1 = -2.4918
        self.Sigma_dot(R, star)

    def mdot_X(self, star):
        # Equation B1
        self._MdotX = 6.25e-9 * star.M**(-0.068) * (star.L_X / 1e30)**(1.14) # In Msun/yr

    def scaled_R(self, R, star):
        # Equation B3
        # Where R in AU
        x = 0.85 * R / star.M
        return x

    def Sigma_dot(self, R, star):
        # Equation B2
        x = self.scaled_R(R,star)
        where_photoevap = (x > 0.7) # No mass loss close to star
        logx = np.log(x[where_photoevap])
        log10 = np.log(10)
        log10x = logx/log10

        # First term
        exponent = self._a1 * log10x**6 + self._b1 * log10x**5 + self._c1 * log10x**4 + self._d1 * log10x**3 + self._e1 * log10x**2 + self._f1 * log10x + self._g1
        t1 = 10**exponent

        # Second term
        terms = 6*self._a1*logx**5/log10**7 + 5*self._b1*logx**4/log10**6 + 4*self._c1*logx**3/log10**5 + 3*self._d1*logx**2/log10**4 + 2*self._e1*logx/log10**3 + self._f1/log10**2
        t2 = terms/x[where_photoevap]**2

        # Third term
        t3 = np.exp(-(x[where_photoevap]/100)**10)

        # Combine terms
        self._Sigmadot[where_photoevap] = t1 * t2 * t3

        # Work out total mass loss rate for normalisation
        M_dot = 2*np.pi * R * self._Sigmadot
        total = np.trapz(M_dot,R)

        # Normalise, convert to cgs and return
        self._Sigmadot *= self.Mdot / total * Msun / AU**2 # in g cm^-2 / yr

"""
Transition Discs (Owen+12)
"""
class TransitionDisc(PhotoBase):
    def __init__(self, R, star, R_hole):
        super().__init__(R, star)
        self._a2 = -0.438226
        self._b2 = -0.10658387
        self._c2 = 0.5699464
        self._d2 = 0.010732277
        self._e2 = -0.131809597
        self._f2 = -1.32285709
        self.__call__(R_hole, R, star)

    def mdot_X(self, star):
        # Equation B4
        self._MdotX = 4.8e-9 * star.M**(-0.148) * (star.L_X / 1e30)**(1.14) # In Msun/yr

    def scaled_R(self, R, star):
        # Equation B6
        # Where R in AU
        x = 0.95 * (R-self._Rhole) / star.M
        return x

    def Sigma_dot(self, R, star):
        # Equation B5
        x = self.scaled_R(R,star)
        where_photoevap = (x > 0.0) # No mass loss inside hole
        use_x = x[where_photoevap]

        # First term
        terms = self._a2*self._b2 * np.exp(self._b2*use_x) + self._c2*self._d2 * np.exp(self._d2*use_x) + self._e2*self._f2 * np.exp(self._f2*use_x)
        t1 = terms/R[where_photoevap]

        # Second term
        t2 = np.exp(-(x[where_photoevap]/57)**10)

        # Combine terms
        self._Sigmadot[where_photoevap] = t1 * t2

        # Work out total mass loss rate for normalisation
        M_dot = 2*np.pi * R * self._Sigmadot
        total = np.trapz(M_dot,R)

        # Normalise, convert to cgs and return
        self._Sigmadot *= self.Mdot / total * Msun / AU**2 # in g cm^-2 / yr

    def __call__(self, R_hole, R, star):
        # Update the hole radius and hence the mass-loss profile
        # Sigma_dot will update the profile stored such that it doesn't have to be called unless R_hole changes
        # Also returns the profile here immediately
        self._Rhole = R_hole
        self.Sigma_dot(R, star)
        return self.dSigmadt

"""
Run as Main
"""
class DummyDisc(object):
    def __init__(self, R):
        self._M = 10 * Mjup
        self._Sigma = self._M / (2 * np.pi * max(R) * R * AU**2)

    @property
    def Sigma(self):
        return self._Sigma

    @property
    def Sigma_G(self):
        return self._Sigma

def main():
    #Sigma_dot_plot()
    Test_Removal()

def Test_Removal():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default=DefaultModel)
    args = parser.parse_args()
    model = json.load(open(args.model, 'r'))

    star1 = PhotoStar(LX=1e30, M=model['star']['mass'], R=model['star']['radius'], T_eff=model['star']['T_eff'])
    R = np.linspace(0,200,2001)

    disc1 = DummyDisc(R)
    internal_photo = PrimordialDisc(R, star1)

    plt.figure()
    #plt.loglog(R, disc1.Sigma, label='{}'.format(0))
    for t in np.linspace(0,1e5,6):
        internal_photo.remove_mass(disc1, 2e4)
        plt.loglog(R, disc1.Sigma, label='{}'.format(t))
    plt.legend()
    plt.show()

def Sigma_dot_plot():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default=DefaultModel)
    args = parser.parse_args()
    model = json.load(open(args.model, 'r'))

    star1 = PhotoStar(LX=1e30, M=model['star']['mass'], R=model['star']['radius'], T_eff=model['star']['T_eff'])
    R = np.linspace(0,200,2001)

    plt.figure(figsize=(6,6))

    internal_photo = PrimordialDisc(R, star1)    
    Sigma_dot = internal_photo.dSigmadt
    plt.plot(R, Sigma_dot, label='Primordial Disc')

    Rhole = 10
    internal_photo2 = TransitionDisc(R, star1, Rhole)
    Sigma_dot = internal_photo2.dSigmadt
    plt.plot(R, Sigma_dot, label='Transition Disc ($R_{{\\rm hole}} = {}$)'.format(Rhole))

    plt.xlabel("R / AU")
    plt.ylabel("$\dot{\Sigma}_{\\rm w}$ / g cm$^{-2}$ s$^{-1}$")
    plt.xlim([0,40])
    plt.legend()
    plt.show()

    """
    # Make cumulative plot to compare with Fig 4 of Owen+11
    plt.figure()
    
    M_dot = 2*np.pi * R * Sigma_dot
    cum_Mds = []
    for r in R:
        select = (R < r)
        cum_M_dot = np.trapz(M_dot[select],R[select])
        cum_Mds.append(cum_M_dot)
    norm_cum_M_dot = np.array(cum_Mds) / cum_Mds[-1]
    plt.plot(R, norm_cum_M_dot)

    plt.xlim([0,80])
    plt.ylim([0,1])
    plt.show()
    """

if __name__ == "__main__":
    main()

