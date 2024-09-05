# coagulation.py
#
# Author: R. Booth
# Date : 14 - Feb - 2023
#
# A dust module that handles coagulation using the Smoluchowski equation.
################################################################################
from __future__ import print_function
import multiprocessing
import numpy as np

from .dust import FixedSizeDust
from . import constants

# Locate the coagulation tooling
import sys, os
try:
    COAG_TOOLKIT = os.environ["COAG_TOOLKIT"]
    sys.path.append(COAG_TOOLKIT)
except KeyError:
    raise ImportError("coagulation module requires COAG_TOOLKIT environment "
                      "variable to be set")

from coag_toolkit import CoagSolver

class SmoluchowskiDust(FixedSizeDust):
    """Dust model including growth and radial drift.

    Parameters
    ----------
    grid : Disc gridding object
        Radius grid of the model
    star : Stellar object
        Central star
    eos : EOS object
        Equation of state
    amin, amax : float
        Min / max grain sizes in cm
    Nbins : int,
        Number of grain sizes.
    eps_t : float or array
        Initial total dust-to-gas ratio.
    gsd : string, optional
        Type of grain size distribution to use. If not provided eps_t will be
        interpretted as the per-size dust-to-gas ratio.
    gsd_params : dict, optional
        Parameters of the grain size distribution. For the options, see 
        generate_size_distrubtion.
    Sigma : 1D array, optional
        Surface density of the disc in g / cm^2.
    Schmidt : float, default=1.0
        Schmidt number of the turbulence.
    feedback : bool, default = True
        Whether to include feedback (backreaction) in the velocity calculation
    settling : bool, default = False
        Whether to include dust settling in the velocity calculation.
    rhos : float
        Solid density, default=1 g / cm^3
    mu : float, default=2.3
        Mean molecular weight
    u_frag : float, default=1000 cm/s
        Fragmentation threshold velocity. u_frag <= 0 means no fragmentation
    u_bounce : float, default= 0 cm/s
        Bouncing threshold velocity. u_frag <= 0 means no bouncing
    kernel_type : string, default='Birnstiel'
        Type of growth kernel used. Must be 'Birnstiel' or 'Garaud'
    Xi : float, default=1
        Erosion multiplier, multiple of projectile mass removed from target
        during erosion.
    eta : float, default=11/6
        Slope of the fragment distribution. MRN is 11/6
    num_threads : int, optional
        Number of threads to use for coagulation. If not provided it will be
        determined from either OMP_NUM_THREADS or the number of cores
        available.
    """
    def __init__(self, grid, star, eos, m_min, m_max, Nbins, 
                 eps_t, gsd='MRN', gsd_params={}, Sigma=None,
                 Schmidt=1.0, feedback=True, settling=False,
                 rho_s=1, mu=2.3, kernel_type='Birnstiel', 
                 u_frag=1000., u_bounce=0, Xi=1., eta=11/6., num_threads=None):

        self._m_min = m_min 
        self._m_max = m_max
        self._settling = settling

        # Setup the size distribution:
        m_edge = np.geomspace(m_min, m_max, Nbins+1)

        if num_threads is None:
            num_threads = int(os.environ.get('OMP_NUM_THREADS', '0'))
            if num_threads <= 0:
                num_threads = max(multiprocessing.cpu_count() // 2, 1)

        # Initialize the coagulation toolkit:
        self._coag = CoagSolver(m_edge, kernel_type=kernel_type, rho_grain=rho_s, mu=mu,
                                v_frag=u_frag, v_bounce=u_bounce, Xi=Xi, eta=eta,
                                num_threads=num_threads)
        
        self._me = m_edge
        self._mc = self._coag.masses

        # Initialize the sub-class.
        size = self._coag.grain_size(self._mc)
        super(SmoluchowskiDust, self).__init__(
            grid, star, eos, eps_t, size, Sigma=Sigma, rhos=rho_s, feedback=feedback)

        # Set the grain-size distribution
        if gsd is not None:
            self.generate_size_distrubtion(eps_t, gsd, **gsd_params)

        # Set parameter for optimizing coagulation:
        self._tnext = np.zeros(self.Ncells, dtype='f8')
        self._tlast = np.zeros(self.Ncells, dtype='f8')
        self._t = 0


    def generate_size_distrubtion(self, eps, gsd='MRN', **pars):
        """Generate a simple grain size distribution for the disc and set dust_frac.
        
        Parameters
        ----------
        eps_total : float, or 1D array of values for each cell.
            The integrated dust-to-gas ratio for the grain size distribtion
        gsd : String, default='MRN'
            Functional form of the grain size distribution. Must be either 'MRN' or
            'Gaussian'. For the parameters of these see the Notes.

        -----
        Notes
         - For the MRN distribution the maximum grain size is given by the parameter
           a_max, which defaults to 1e-4 cm (1 micron)
         - For the Gaussian distribution, the mean is taken to be zero and the standard
           deviation given by the parameter 'a_scale', which defaults to the minimum
           grain size in the model.
        """
        if gsd == 'MRN':
            a_max = pars.get('a_max', 1e-4)
            a = self.grain_size
            dm = np.diff(self._me)

            gsd = np.where(a < a_max, a**-0.5, 0) * dm.reshape(-1,1)

        elif gsd == 'Gaussian':
            a = self.grain_size
            a_scale = pars.get('a_scale', a[0])
            dm = np.diff(self._me)

            gsd = np.exp(-0.5*(a/a_scale)**2) * dm.reshape(-1,1)
        else:
            raise ValueError(f"gsd must be one of 'MRN' or 'Gaussian', not {gsd}")

        eps = np.atleast_1d(eps).reshape(1,-1)
        gsd /= gsd.sum(0)

        self.dust_frac[:] = eps * gsd

    def _head(self):
        Nbins = self.num_bins
        mu = self._coag.mean_mol_weight
        rhos = self._coag.monomer_density
        kernel = self._coag.kernel_type
        uf, ub = self._coag.fragmentation_velocity, self._coag.bouncing_velocity
        Xi, eta = self._coag.erosion_multiplier, self._coag.fragment_distribution

        return (f"m_min: {self._m_min}, m_max: {self._m_max}, Nbins: {Nbins}, "
                f"rho_s: {rhos}, mu:{mu}, settling: {self._settling}, " +
                f"kernel_type: {kernel}, " +
                f"u_frag: {uf}, u_bounce: {ub}, Xi: {Xi}, eta:{eta}")

    def ASCII_header(self):
        """Dust growth header"""
        return super(SmoluchowskiDust, self).ASCII_header() + self._head()

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        name, head = super(SmoluchowskiDust, self).HDF5_attributes()

        tmp = dict([x.strip().split(":") for x in self._head().split(",") if x])

        head.update(tmp)

        return self.__class__.__name__, head

    def _get_velocities(self):
        """Evaluate the radial and azimuthal dust velocities"""

        rhog = self.midplane_gas_density
        eta_vk = -0.5*np.gradient(self.P, self.R) / (rhog * self.Omega_k)

        St = self.Stokes()

        # Compute lambda co-efficients
        la0, la1 = 0, 0
        if self.feedback:
            if self._settling:
                eps = self.midplane_dust_density / rhog
            else:
                eps_g = np.maximum(1-self.integ_dust_frac, 1e-100)
                eps = self.dust_frac / eps_g
            
            la0 = (eps / (1  + St** 2)).sum(0)
            la1 = (eps / (St + St**-1)).sum(0)

        #Evaluate gas velocity:
        eta_vk = eta_vk / ((1 + la0)**2 + la1**2)
        vg_r   = -2*eta_vk * la1 
        vg_phi = +  eta_vk * (1 + la0)

        vd_r =  (2*vg_phi / (St + St**-1) 
                 - vg_r   / (1  + St**-2))

        vd_phi = (-0.5*vg_r  / (St + St**-1) 
                  +   vg_phi / (1  + St** 2))

        return vd_r, vd_phi


    def update(self, dt):
        """Evolve the dust fraction due to grain growth"""
        
        unit_vel = constants.Omega0 * constants.AU

        v_r, v_phi = self._get_velocities()
        c_s = self.cs

        v_r, v_phi, c_s = [x*unit_vel for x in (v_r, v_phi, c_s)]

        Omega_k = self.Omega_k * constants.Omega0
        Sigma_g = self.Sigma_G
        alpha = np.ones_like(Sigma_g) * self.alpha / self.Sc

        Sigma_dust = self.Sigma_D

        self._t += dt
        
        # determine cells that need to run:
        active = self._t >= self._tnext
        dt_step = (self._t - self._tlast[active]) / constants.Omega0
        
        def get_active(arr):
            return np.array(arr.T[active])

        coag_args = [get_active(x) for x in [Sigma_dust, v_r, v_phi, 
                                             c_s, Sigma_g, Omega_k, alpha]]

        dt_next = self._coag.integrate_parallel(dt_step, *coag_args)
        dt_next *= constants.Omega0

        self.dust_frac[:,active] = coag_args[0].T / self.Sigma[active]

        self._tlast[active] = self._t
        self._tnext[active] = self._t + dt_next
    
    def update_ices(self, *args):
        raise RuntimeError("SmoluchowskiDust does not yet work with chemistry")


    @property   
    def num_bins(self):
        return self._coag.num_bins

    @property
    def masses(self):
        return self._mc
