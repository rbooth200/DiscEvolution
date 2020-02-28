import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from DiscEvolution.constants import *
from DiscEvolution.star import PhotoStar
from scipy.signal import argrelmin

DefaultModel = "../test_internal_photo/DiscConfig_default.json"
plt.rcParams['text.usetex'] = "True"
plt.rcParams['font.family'] = "serif"

class PhotoBase():
    def __init__(self, disc):
        # Basic mass loss properties
        self.mdot_X(disc.star)
        self._Sigmadot = np.zeros_like(disc.R)

        # Evolutionary state flags
        self._Hole = False      # Has the hole started to open?
        self._Thin = False      # Is the hole exposed (ie low column density to star)? 
        self._loMd = False      # Is the mass loss below that of the TD?
        self._empty = False     # When no longer a valid hole radius
        self._switch = False    # Generic flag to be set to either Thin or loMd
        self._swiTyp = "Thin"   # Determine whether switch is on Thin or loMd
        # Equation B4, needed to assess when Mdot is low
        self._Mdot_TD = 4.8e-9 * disc.star.M**(-0.148) * (disc.star.L_X / 1e30)**(1.14) # In Msun/yr

    def update_switch(self):
        # Update the switch depending on the type used
        if (self._swiTyp == "Thin"):
            self._switch = self._Thin
        elif (self._swiTyp == "loMd"):
            self._switch = self._loMd
        else:
            raise AttributeError("Photobase::{} is no a valid switchtype".format(self._swiTyp))

    def mdot_X(self, star):
        # Without prescription, mass loss is 0
        self._MdotX     = 0
        self._Mdot_true = 0

    def Sigma_dot(self, R, star):
        # Without prescription, mass loss is 0
        self._Sigmadot = np.zeros_like(R)

    def scaled_R(self, R, star):
        # Prescriptions may rescale the radius variable 
        # Without prescription, radius is unscaled
        return R

    def get_dt(self, disc, dt, R_out):
        # Work out the timescale to clear cell
        where_photoevap = (self.dSigmadt > 0)
        t_w = np.full_like(disc.R,np.inf)
        t_w[where_photoevap] = disc.Sigma_G[where_photoevap] / self.dSigmadt[where_photoevap]

        # Return minimum value for cells inside outer edge 
        try:
            self._tw = min(t_w[(disc.R < R_out)])
            return self._tw, np.argmin(t_w[(disc.R < R_out)])
        except ValueError: # If above fails it is usually because R_out -> 0 < R_hole
            print("get_dt finds the disc empty")
            self._empty = True
            return 0, 0

    def remove_mass(self, disc, dt, photoevap=None):
        # Find disc "outer edge" so we can apply mass loss only inside
        try:
            R_out = photoevap._Rot
        except:
            R_out = disc.Rout()

        # Determine mass loss
        self.get_dt(disc, dt, R_out)
        dSigma = np.minimum(self.dSigmadt * dt, disc.Sigma_G)   # Limit mass loss to density of cell
        dSigma *= (disc.R < R_out)                              # Only apply mass loss inside disc outer edge

        # Apply, preserving the dust mass
        if hasattr(disc, 'Sigma_D'):
            Sigma_D = disc.Sigma_D                              # Save the dust density
        disc._Sigma -= dSigma
        if hasattr(disc, 'Sigma_D'):
            dusty = Sigma_D.sum(0)>0
            disc.dust_frac[:,dusty] = np.fmin(Sigma_D[:,dusty]/disc.Sigma[dusty],disc.dust_frac[:,dusty]/disc.dust_frac.sum(0)[dusty])
            disc.dust_frac[:] /= np.maximum(disc.dust_frac.sum(0), 1.0)           # Renormalise to 1 if it exceeds

        # Calculate actual mass loss given limit
        if dt>0:
            dM = 2*np.pi * disc.R * dSigma
            self._Mdot_true = np.trapz(dM,disc.R) / dt * AU**2 / Msun

    def get_Rhole(self, disc, photoevap=None, Track=False):
        # Deal with calls when there is no hole
        if not self._Hole:
            if Track:
                disc.history._Rh = np.append(disc.history._Rh, [np.nan])
                disc.history._Mdot_int = np.append(disc.history._Mdot_int, self._Mdot_true)
            else:
                print("No hole for which to get radius. Ignoring command.")
            return 0, 0

        # Otherwise continue on to find hole
        # First find outer edge of disc - hole must be inside this
        try:
            R_out = photoevap._Rot
        except:
            R_out = disc.Rout()
        empty_indisc = (disc.Sigma_G <= 0) * (disc.R < R_out)

        try:
            if (np.sum(empty_indisc) == 0):         # If none in disc are empty
                i_hole_out = argrelmin(disc.Sigma_G)[0][0]   # Position of hole is minimum density
            else:
                #i = np.nonzero(empty_indisc)[0][-1] # Position of hole is given by outermost empty cell inside the disc 
                # First find the inner edge of the innermost hole
                i_hole_in  = np.nonzero(empty_indisc)[0][0]
                # The hole cell is defined as the one inside the first non-empty cell outside the inner edge of the hole 
                i_hole_out = np.nonzero(~empty_indisc * (disc.R>disc.R_edge[i_hole_in]))[0][0] - 1  
        except IndexError:
            # No hole found, so switch it off again
            self._Hole = False
            return 0, 0

        # If everything worked, update hole properties
        self._R_hole = disc.R_edge[i_hole_out+1]
        self._N_hole = disc.column_density[i_hole_out]

        # Test whether Thin or loMd and update switch
        if (self._N_hole < 1e22):
            self._Thin = True
        if (self._Mdot_true < self._Mdot_TD):
            self._loMd = True
        self.update_switch()

        # Save state if tracking
        if Track:
            disc.history._Rh = np.append(disc.history._Rh,[self._R_hole])
            disc.history._Mdot_int = np.append(disc.history._Mdot_int, self._Mdot_true)
        else:
            return self._R_hole, self._N_hole
        
    @property
    def Mdot(self):
        return self._MdotX

    @property
    def dSigmadt(self):
        return self._Sigmadot

    def __call__(self, disc, dt, photoevap=None):
        self.remove_mass(disc,dt, photoevap)

    def ASCII_header(self):
        return ("InternalEvaporation, Type: {}, Mdot: {}"
                "".format(self._type,self._MdotX))

    def HDF5_attributes(self):
        header = {}
        header['Type'] = self._type
        header['Mdot'] = '{}'.format(self._MdotX)
        return self.__class__.__name__, header

"""
Primoridal Discs (Owen+12)
"""
class PrimordialDisc(PhotoBase):
    def __init__(self, disc):
        super().__init__(disc)
        self._type = 'Primordial'
        # Parameters for mass loss
        self._a1 = 0.15138
        self._b1 = -1.2182
        self._c1 = 3.4046
        self._d1 = -3.5717
        self._e1 = -0.32762
        self._f1 = 3.6064
        self._g1 = -2.4918
        # Run the mass loss rates to update the table
        #self.Sigma_dot(disc.R, disc.star)
        self.Sigma_dot(disc.R_edge, disc.star)

    def mdot_X(self, star):
        # Equation B1
        self._MdotX = 6.25e-9 * star.M**(-0.068) * (star.L_X / 1e30)**(1.14) # In Msun/yr
        self._Mdot_true = self._MdotX

    def scaled_R(self, R, star):
        # Equation B3
        # Where R in AU
        x = 0.85 * R / star.M
        return x

    def Sigma_dot(self, R, star):
        # Equation B2
        Sigmadot = np.zeros_like(R)
        x = self.scaled_R(R,star)
        where_photoevap = (x >= 0.7) # No mass loss close to star
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
        Sigmadot[where_photoevap] = t1 * t2 * t3

        # Work out total mass loss rate for normalisation
        M_dot = 2*np.pi * R * Sigmadot
        total = np.trapz(M_dot,R)

        # Normalise, convert to cgs
        Sigmadot *= self.Mdot / total * Msun / AU**2 # in g cm^-2 / yr

        # Store values as average of mass loss rate at cell edges
        self._Sigmadot = (Sigmadot[1:] + Sigmadot[:-1]) / 2

    def get_dt(self, disc, dt, R_out):
        t_w, i_hole = super().get_dt(disc, dt, R_out)
        if (dt > self._tw):         # If an entire cell can deplete
            if not self._Hole:
                print("Alert - hole will open after this timestep at {:.2f} AU".format(disc.R[i_hole]))
                print("Outer radius is currently {:.2f} AU".format(R_out))
            self._Hole = True       # Set hole flag
        return self._tw

"""
Transition Discs (Owen+12)
"""
class TransitionDisc(PhotoBase):
    def __init__(self, disc, R_hole, N_hole):
        super().__init__(disc)
        self._type = 'Transition'
        # Parameters for mass loss
        self._a2 = -0.438226
        self._b2 = -0.10658387
        self._c2 = 0.5699464
        self._d2 = 0.010732277
        self._e2 = -0.131809597
        self._f2 = -1.32285709
        # Parameters of hole
        self._R_hole = R_hole
        self._N_hole = N_hole
        # Run the mass loss rates to update the table
        #self.Sigma_dot(disc.R, disc.star)
        self.Sigma_dot(disc.R_edge, disc.star)
        # Update flags
        self._Hole = True        
        self._Thin = True   # Assuming thin switch
        self._loMd = True   # Assuming mass loss rate switch

    def mdot_X(self, star):
        # Equation B4
        self._MdotX = 4.8e-9 * star.M**(-0.148) * (star.L_X / 1e30)**(1.14) # In Msun/yr
        self._Mdot_true = self._MdotX

    def scaled_R(self, R, star):
        # Equation B6
        # Where R in AU
        x = 0.95 * (R-self._R_hole) / star.M
        return x

    def Sigma_dot(self, R, star):
        # Equation B5
        Sigmadot = np.zeros_like(R)
        x = self.scaled_R(R,star)
        where_photoevap = (x >= 0.0) # No mass loss inside hole
        use_x = x[where_photoevap]

        # First term
        terms = self._a2*self._b2 * np.exp(self._b2*use_x) + self._c2*self._d2 * np.exp(self._d2*use_x) + self._e2*self._f2 * np.exp(self._f2*use_x)
        t1 = terms/R[where_photoevap]

        # Second term
        t2 = np.exp(-(x[where_photoevap]/57)**10)

        # Combine terms
        Sigmadot[where_photoevap] = t1 * t2

        # Work out total mass loss rate for normalisation
        M_dot = 2*np.pi * R * Sigmadot
        total = np.trapz(M_dot,R)

        # Normalise, convert to cgs
        Sigmadot *= self.Mdot / total * Msun / AU**2 # in g cm^-2 / yr

        # Store values as average of mass loss rate at cell edges
        self._Sigmadot = (Sigmadot[1:] + Sigmadot[:-1]) / 2

    def __call__(self, disc, dt, photoevap=None):
        # Update the hole radius and hence the mass-loss profile
        self.get_Rhole(disc)
        #self.Sigma_dot(disc.R, disc.star) # Need to update as the normalisation changes based on R, not just x~R-Rhole
        self.Sigma_dot(disc.R_edge, disc.star) # Need to update as the normalisation changes based on R, not just x~R-Rhole
        super().__call__(disc, dt, photoevap)
        
"""
Run as Main
"""
class DummyDisc(object):
    def __init__(self, R, star):
        self._M = 10 * Mjup
        self._Sigma = self._M / (2 * np.pi * max(R) * R * AU**2)
        self.R = R
        self.star = star

    def Rout(self):
        return max(self.R)

    @property
    def Sigma(self):
        return self._Sigma

    @property
    def Sigma_G(self):
        return self._Sigma

def main():
    Sigma_dot_plot()
    #Test_Removal()
	#test_resolution()

def Test_Removal():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default=DefaultModel)
    args = parser.parse_args()
    model = json.load(open(args.model, 'r'))

    star1 = PhotoStar(LX=1e30, M=model['star']['mass'], R=model['star']['radius'], T_eff=model['star']['T_eff'])
    R = np.linspace(0,200,2001)
    disc1 = DummyDisc(R, star1)

    internal_photo = PrimordialDisc(disc1)

    plt.figure()
    #plt.loglog(R, disc1.Sigma, label='{}'.format(0))
    for t in np.linspace(0,1e5,6):
        internal_photo.remove_mass(disc1, 2e4)
        plt.loglog(R, disc1.Sigma, label='{}'.format(t))
    plt.legend()
    plt.show()

def Sigma_dot_plot():
    from control_scripts import run_model
    # Set up dummy model
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default=DefaultModel)
    args = parser.parse_args()
    model = json.load(open(args.model, 'r'))

    star1 = PhotoStar(LX=1e30, M=model['star']['mass'], R=model['star']['radius'], T_eff=model['star']['T_eff'])
    #R = np.linspace(0,100,2001)
    #disc1 = DummyDisc(R, star1)
    disc1 = run_model.setup_disc(model)
    R = disc1.R

    # Calculate rates
    internal_photo = PrimordialDisc(disc1)    
    Sigma_dot = internal_photo.dSigmadt
    photoevaporating = (Sigma_dot>0)
    t_w = disc1.Sigma[photoevaporating] / Sigma_dot[photoevaporating]
    print("Mdot maximum at R = {} AU".format(R[np.argmax(Sigma_dot)]))    
    print("Time minimum at R = {} AU".format(R[photoevaporating][np.argmin(t_w)]))
    frac_in = np.trapz((R*Sigma_dot)[R<R[photoevaporating][np.argmin(t_w)]],R[R<R[photoevaporating][np.argmin(t_w)]]) / np.trapz(R*Sigma_dot,R)
    print(frac_in)

    Rhole = R[photoevaporating][np.argmin(t_w)]
    internal_photo2 = TransitionDisc(disc1, Rhole, None)
    Sigma_dot2 = internal_photo2.dSigmadt

    # Plot mass loss rates
    plt.figure(figsize=(6,6))
    plt.plot(R, R*Sigma_dot, label='Primordial Disc')
    plt.plot(R, R*Sigma_dot2, label='Transition Disc ($R_{{\\rm hole}} = {}$)'.format(Rhole))
    plt.xlabel("R / AU")
    plt.ylabel("$\dot{\Sigma}_{\\rm w}$ / g cm$^{-2}$ s$^{-1}$")
    plt.xlim([0.1,1000])
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim([1e-8,1e-4])
    plt.legend()
    plt.show()

    # Plot depletion time
    plt.figure(figsize=(6,6))
    plt.semilogy(R[photoevaporating], t_w, label='Primordial Disc')
    plt.xlabel("R / AU")
    plt.ylabel("$t_w / yr$")
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

def test_resolution():
    #from DiscEvolution.disc import AccretionDisc
    from control_scripts import run_model
    # Set up dummy model
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", "-m", type=str, nargs='+', default=DefaultModel)
    args = parser.parse_args()

    #R = np.linspace(0,100,2001)
    plt.figure(figsize=(6,6))
    Rhole = 10

    for model_no in args.models:
        model = json.load(open('DiscConfig{}.json'.format(model_no), 'r'))

        disc1 = run_model.setup_disc(model)

        # Calculate rates
        internal_photo1 = PrimordialDisc(disc1)
        Sigma_dot1 = internal_photo1.dSigmadt
        M_dot = 2*np.pi * disc1.R * AU * Sigma_dot1 / Msun

        # Plot mass loss rates
        plt.subplot(221)
        plt.plot(disc1.R, M_dot, label='Primordial Disc ($N={}$)'.format(model['grid']['N']))

        # Cumulative
        plt.subplot(222)
        cum_Mds = []
        for r in disc1.R:
            select = (disc1.R < r)
            cum_M_dot = np.trapz(M_dot[select],disc1.R[select])
            cum_Mds.append(cum_M_dot*AU)
        plt.plot(disc1.R, cum_Mds)

        # Calculate rates
        internal_photo2 = TransitionDisc(disc1, Rhole, None)
        Sigma_dot2 = internal_photo2.dSigmadt
        M_dot = 2*np.pi * disc1.R * AU * Sigma_dot2 / Msun

        # Plot mass loss rates
        plt.subplot(223)
        plt.plot(disc1.R, M_dot, label='Transition Disc ($N={}$)'.format(model['grid']['N']))

        # Cumulative
        plt.subplot(224)
        cum_Mds = []
        for r in disc1.R:
            select = (disc1.R < r)
            cum_M_dot = np.trapz(M_dot[select],disc1.R[select])
            cum_Mds.append(cum_M_dot*AU)
        plt.plot(disc1.R, cum_Mds)

    global_xlims=[0,120]

    plt.subplot(221)
    plt.ylabel("$\\frac{{\\rm d} \dot{M}}{{\\rm d} R}$ / $M_\odot~{\\rm yr}^{-1}~{\\rm AU}^{-1}$")
    plt.xlim(global_xlims)
    plt.legend()

    plt.subplot(222)
    plt.ylabel("$\dot{M}(<R)$ / $M_\odot~{\\rm yr}^{-1}$")
    plt.xlim(global_xlims)
    plt.plot(global_xlims,[internal_photo1.Mdot,internal_photo1.Mdot],linestyle='--',color='darkslategray')

    plt.subplot(223)
    plt.xlabel("R / AU")
    plt.ylabel("$\\frac{{\\rm d} \dot{M}}{{\\rm d} R}$ / $M_\odot~{\\rm yr}^{-1}~{\\rm AU}^{-1}$")
    plt.xlim(global_xlims)
    plt.legend()

    plt.subplot(224)
    plt.xlabel("R / AU")
    plt.ylabel("$\dot{M}(<R)$ / $M_\odot~{\\rm yr}^{-1}$")
    plt.xlim(global_xlims)
    plt.plot(global_xlims,[internal_photo2.Mdot,internal_photo2.Mdot],linestyle='--',color='darkslategray')

    plt.show()

    """
    # Make cumulative plot to compare with Fig 4 of Owen+11


    plt.xlim([0,80])
    plt.ylim([0,1])
    plt.show()
    """


if __name__ == "__main__":
    main()

