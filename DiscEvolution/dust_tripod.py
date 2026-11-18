# dust_tripod.py
#
# Author: R. Booth
# Date : 12 - Nov - 2026
#
# TriPoD's transport scheme (Pfeil, Birnstiel & Klahr 2024, A&A 691, A45) and
# two variants of it, TriMean and TriMax, on the DustGrowthTriMom disc and with
# the integrator of dust_trimom.MomentDrift, for comparison with the moment
# scheme. The three transport the two populations' dust fractions, eps0 and
# eps1, each with the velocity of its mass-averaged size, and differ in the
# third quantity:
#     TriPoDDrift  : eps1*amax, with the velocity of the large population
#                    and optional shrinkage of amax in gaps
#     TriMeanDrift : M_1 = eps_tot <a>, with the velocity of M_2/M_1
#     TriMaxDrift  : amax itself, advected with the velocity of amax
################################################################################
from __future__ import print_function
import numpy as np
from scipy.special import lambertw
from .dust_trimom import (MomentDrift, mean_size, population_sizes,
                          large_fraction, s_from_split, _lnE, _ln_exprel)
from .brent import brentq

__all__ = ["TriPoDDrift", "TriMeanDrift", "TriMaxDrift", "make_drift",
           "fit_mean_size", "density_at_amax"]


def density_at_amax(eps0, eps1, L):
    """Mass per unit ln(a) at amax, eps_tot exp(sL)/E(s, L), of the power law
    with population masses eps0, eps1 > 0 and L = ln(amax/amin) (TriPoD's
    closure, s from s_from_split)"""
    s = s_from_split(eps0, eps1, L)
    return (eps0 + eps1)*np.exp(s*L - _lnE(s, L))


def fit_mean_size(eps0, eps1, lnX, s_min=-np.inf):
    """s and L of the power law with population masses eps0, eps1 > 0 (split
    at sqrt(amin amax)) and mass-averaged size amin exp(lnX) > amin, as used
    by the 'trimean' scheme.

    Where s < s_min, s = s_min instead and L is the one giving the mean size
    amin exp(lnX) (keeping the mass and <a>; the population masses are then
    those of the returned s, L, large_fraction). It exists, below the
    unclipped L, since <a> rises with s at fixed L and tends to amin as L -> 0.

    With c = 2 ln(eps1/eps0) = s L, the mean size is
        <a>/amin = exprel(c + L) / exprel(c),
    which increases monotonically from 1 (L -> 0) to infinity, so L is unique:
    x = c + L solves exprel(x) = K = exprel(c) <a>/amin, i.e. e^x = 1 + K x,
        x = -1/K - W(-e^(-1/K)/K),
    on the branch of the Lambert W function that excludes the root x = 0
    (W_-1 for K > 1, W_0 for K < 1). Near K = 1, where W loses precision,
    the series inverse of ln exprel is used instead.

    returns:
        s, L
    """
    c = 2*np.log(eps1/eps0)
    y = lnX + _ln_exprel(c)                 # ln K
    near = np.abs(y) < 1e-2
    yn = np.where(near, y, 0.)
    x_series = yn*(2 + yn*(-1/3. + yn*(1/9. + yn*(-19/540. + yn*(17/1620. -
                                                  yn*13/4536.)))))
    yw = np.where(near, 1., y)
    iK = np.exp(-yw)
    x_W = -lambertw(-np.exp(-iK - yw), np.where(yw > 0, -1, 0)).real - iK
    L = np.where(near, x_series, x_W) - c
    s = c/L
    steep = s < s_min
    if steep.any():
        Lu = np.where(steep, L, 1.)
        lo = 1e-10*Lu
        g = lambda l: _ln_exprel((s_min + 1)*l) - _ln_exprel(s_min*l) - lnX
        # The root is bracketed except at round-off: s just below s_min
        # (root at Lu), or <a> = amin to round-off (root ~ 2 lnX -> 0)
        at_hi = g(Lu) <= 0
        at_lo = g(lo) >= 0
        ok = steep & ~at_hi & ~at_lo
        def f(l):
            return np.where(ok, g(l), l - 0.5*Lu)
        root = brentq(f, lo, Lu, xtol=1e-10, rtol=1e-10)
        root = np.where(at_hi, Lu, np.where(at_lo, np.maximum(2*lnX, lo), root))
        L = np.where(steep, root, L)
        s = np.where(steep, s_min, s)
    return s, L


class TriPoDDrift(MomentDrift):
    """TriPoD's transport scheme: eps0, eps1 and amax*eps1 transported with
    the velocities of f_drift*a0, f_drift*a1 (the mass-averaged sizes of the
    two populations), and optional shrinkage of amax in gaps (Pfeil et al.
    2024, Sec. 4.3). The closure is TriPoD's (s from eps1/eps0). The
    integrator, gas advection and diffusion are as MomentDrift's; the dust
    tracers are split between the populations.

    args:
        settling, van_leer : as MomentDrift
        f_drift : Factor multiplying grain size in drift velocities,
                  default=1 (0.8 is the TriPoD calibration)
        shrink  : TriPoD shrinkage, default=True
        f_crit  : TriPoD critical large-grain fraction, default=0.425
        a_lim   : TriPoD minimum size for shrinkage, default=1e-4 cm
    """
    _scheme = 'tripod'

    # Representative size (_sizes) of each transported quantity, in the drift
    # and in diffusion
    _species = [0, 1, 1]
    _diffusion_species = [0, 1, 1]

    def __init__(self, settling=False, van_leer=True, f_drift=1.0, shrink=True,
                 f_crit=0.425, a_lim=1e-4):
        super(TriPoDDrift, self).__init__(settling, van_leer, f_drift)
        self._shrink = shrink
        self._f_crit = f_crit
        self._a_lim = a_lim

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        name, head = super(TriPoDDrift, self).HDF5_attributes()
        head["shrink"] = "{}".format(self._shrink)
        head["f_crit"] = "{}".format(self._f_crit)
        head["a_lim"] = "{}".format(self._a_lim)
        return name, head

    ############################################################################
    # Transported quantities
    ############################################################################
    def _transported(self, disc):
        """Transported quantities per unit total surface density, [3, Ncells]"""
        return np.array([disc.dust_frac[0], disc.dust_frac[1],
                         disc.dust_frac[1]*disc.amax])

    def _reconstruct(self, disc, Q, shrink=True):
        """Set the disc's size distribution from the transported quantities
        (shrink: TriPoD's shrinkage, if enabled)"""
        amin = disc.amin
        eps0 = np.maximum(Q[0], 0)
        eps1 = np.maximum(Q[1], 0)
        amax = np.where(eps1 > 0, Q[2]/np.where(eps1 > 0, eps1, 1),
                        disc.amax)
        amax = np.maximum(amax, amin*np.exp(disc._L_min))
        disc.set_populations(eps0, eps1, amax)
        if self._shrink and shrink:
            self._tripod_shrink(disc)

    def _tripod_shrink(self, disc):
        """TriPoD's size reduction in gaps (Pfeil et al. 2024, Sec. 4.3, as
        implemented by enforce_f in TriPoDPy)"""
        eps0, eps1 = disc.dust_frac
        eps_tot = eps0 + eps1
        delta = np.maximum(0, self._f_crit*eps_tot - eps1)
        do = (delta > 0) & (disc.amax > self._a_lim) & (eps0 > 0) & (eps1 > 0)
        if not do.any():
            return
        _, s, L = disc.size_distribution()
        f1 = large_fraction(s, L)
        # d eps1 / d amax at fixed q and eps_tot
        deps1_da = eps_tot*0.5*s*f1*(1 - f1)/disc.amax
        da = np.where(do & (deps1_da != 0), delta/np.where(deps1_da != 0, deps1_da, 1), 0)
        amax = np.maximum(disc.amax + np.where(do, da, 0), self._a_lim)
        disc.set_populations(np.where(do, eps0 - delta, eps0),
                             np.where(do, eps1 + delta, eps1),
                             np.where(do, amax, disc.amax))

    ############################################################################
    # Velocities
    ############################################################################
    def _sizes(self, s, L, amin):
        """f_drift times the mass-averaged sizes of the two populations of
        power laws (s, L)"""
        return self._f_drift*np.array(population_sizes(s, L, amin))

    def _cell_sides(self, disc):
        """The cells to the left and right of each interior edge, as sides"""
        _, s, L = disc.size_distribution()
        Sd = disc.Sigma*disc.dust_frac
        return (Sd[..., :-1], s[:-1], L[:-1]), (Sd[..., 1:], s[1:], L[1:])

    def _side_velocities(self, disc, left, right, v_visc=None):
        """Velocities of the transported quantities at the interior edges,
        relative to the centre of mass, from the sides (Sd, s, L) to their
        left and right: those of the representative sizes (_sizes, _species).
        The populations give the feedback (any further size has no weight
        in it).

        returns:
            V_L, V_R : velocities from the left / right sides, [3, Ncells-1]
        """
        amin = disc.amin
        (SdL, sL, LL), (SdR, sR, LR) = left, right
        aL, aR = self._sizes(sL, LL, amin), self._sizes(sR, LR, amin)
        if len(aL) > 2:
            SdL = np.concatenate([SdL, 0*SdL[:1]])
            SdR = np.concatenate([SdR, 0*SdR[:1]])
        V_L, V_R, _ = self._node_velocities(disc, aL, SdL, aR, SdR, v_visc)
        return V_L[self._species], V_R[self._species]

    ############################################################################
    # Diffusion
    ############################################################################
    def diffusion_stokes(self, disc):
        """Stokes numbers with which diffuse() diffuses the transported
        quantities (pre_gas_transport) and the dust tracers.

        returns:
            St_Q [3, Ncells], and the mass-weighted mean of the populations'
            St [Ncells] (diffuse splits the tracers by population)
        """
        _, s, L = disc.size_distribution()
        a_rep = self._sizes(s, L, disc.amin)[self._diffusion_species]
        St_Q = disc.Stokes(disc.Sigma_G, a_rep)
        return St_Q, (self._population_fractions(disc)*St_Q[:2]).sum(0)

    @staticmethod
    def _population_fractions(disc):
        """eps0/eps_tot and eps1/eps_tot (zero where there is no dust)"""
        eps = disc.dust_frac
        return eps/(eps.sum(0) + np.finfo(eps.dtype).tiny)

    def diffuse(self, dt, disc, diffusion, Q, gas_tracers=None, dust_tracers=None):
        """Diffuse the transported quantities Q (from pre_gas_transport), the
        dust tracers and the gas tracers over dt, in one step of the diffusion
        object (TracerDiffusion, or rkl.RKLDiffusion), with the Schmidt
        numbers of its dust_schmidt: each quantity at its representative
        Stokes number, the tracers split between the populations, the gas
        tracers as the gas"""
        St_Q, _ = self.diffusion_stokes(disc)
        items = [(Q, diffusion.dust_schmidt(disc, St_Q))]
        if dust_tracers is not None:
            t_p = dust_tracers[None]*self._population_fractions(disc)[:, None]
            items.append((t_p, diffusion.dust_schmidt(disc, St_Q[:2])[:, None]))
        if gas_tracers is not None:
            items.append((gas_tracers, diffusion.gas_schmidt(disc)))
        diffusion.diffuse(dt, disc, items)
        if dust_tracers is not None:
            dust_tracers[:] = t_p.sum(0)

    ############################################################################
    # Fluxes and face states
    ############################################################################
    def _dust_mass(self, Q):
        """The dust mass per unit Sigma: eps0 + eps1"""
        return Q[0] + Q[1]

    def _dust_mass_fluxes(self, pos, neg):
        """The parts of the dust mass flux: those of the two populations"""
        return pos[:2].sum(0), neg[:2].sum(0)

    def _third_primitive(self, disc, s, L):
        """The third reconstructed variable: ln amax"""
        return np.log(disc.amax)

    def _primitives(self, disc):
        """The reconstructed variables of the cells, (Sigma_0, Sigma_1,
        ln amax)"""
        Sigma = disc.Sigma
        _, s, L = disc.size_distribution()
        eps = disc.dust_frac
        return np.array([Sigma*eps[0], Sigma*eps[1], self._third_primitive(disc, s, L)])

    @staticmethod
    def _dust_density(W):
        """Dust surface density of the primitives W"""
        return W[0] + W[1]

    @staticmethod
    def _population_faces(W):
        """The population surface densities of the face states W, and the
        same with floors that guard an exactly empty population (outside the
        models' domain), which only face states reach"""
        S0, S1 = np.maximum(W[0], 0), np.maximum(W[1], 0)
        floor = np.maximum(1e-150*(S0 + S1), 1e-300)
        return S0, S1, np.maximum(S0, floor), np.maximum(S1, floor)

    def _face_sides(self, disc, W):
        """Sides (Sd, s, L) of the face states W (_primitives) [3, ...]"""
        s_lo, s_hi = disc._s_bounds
        S0, S1, S0f, S1f = self._population_faces(W)
        L = np.log(np.maximum(np.exp(W[2])/disc.amin, np.exp(disc._L_min)))
        s = np.clip(s_from_split(S0f, S1f, L), max(s_lo, -30.), s_hi)
        return (np.array([S0, S1]), s, L)

    def _face_quantities(self, disc, W, W_cell, SQ_cell):
        """Sigma times the transported quantities of the face states W
        [3, Ncells+1] (W_cell, SQ_cell unused)"""
        S0, S1 = np.maximum(W[0], 0), np.maximum(W[1], 0)
        return {'Q': np.array([S0, S1, S1*np.exp(W[2])])}


class TriMeanDrift(TriPoDDrift):
    """TriPoD's variables with the mean size in place of amax: eps0, eps1 and
    M_1 = eps_tot*<a> transported with the velocities of f_drift times the
    mass-averaged size of each population, and of M_2/M_1 for M_1 (each flux
    is exact for St << 1). The power law is then refitted to eps0, eps1 and
    <a> in closed form (fit_mean_size); no mass moves between the populations
    and there is no shrinkage. The refit needs no bounds on q: use a disc
    with q_bounds=(-np.inf, np.inf), or the disc clips s.

    args:
        settling, van_leer, f_drift : as TriPoDDrift
        q_min : floor on q in the refit. Steeper cells take q = q_min with
                amax set to keep Sigma_d and <a>, moving mass between the
                populations (fit_mean_size). Default None (no floor)
    """
    _scheme = 'trimean'
    _species = [0, 1, 2]
    _diffusion_species = [0, 1, 2]

    def __init__(self, settling=False, van_leer=True, f_drift=1.0, q_min=None):
        super(TriMeanDrift, self).__init__(settling, van_leer, f_drift, shrink=False)
        self._s_min = -np.inf if q_min is None else q_min + 4

    def _transported(self, disc):
        """Transported quantities per unit total surface density, [3, Ncells]"""
        eps_tot, s, L = disc.size_distribution()
        return np.array([disc.dust_frac[0], disc.dust_frac[1],
                         eps_tot*mean_size(s, 0, L, disc.amin)])

    def _reconstruct(self, disc, Q, shrink=True):
        """Set the disc's size distribution from the transported quantities.
        Defined for eps0, eps1 > 0 and <a> > amin; other cells (empty or
        degenerate) keep their amax"""
        amin = disc.amin
        eps0 = np.maximum(Q[0], 0)
        eps1 = np.maximum(Q[1], 0)
        eps_tot = eps0 + eps1
        _, _, L0 = disc.size_distribution()
        ok = (eps0 > 0) & (eps1 > 0) & (Q[2] > amin*eps_tot)
        e = lambda x: np.where(ok, x, 1)
        s, L = fit_mean_size(e(eps0), e(eps1),
                             np.log(e(Q[2])/e(eps_tot)/amin), self._s_min)
        # Diagnostic: cells outside the domain, or with amax below the
        # disc's minimum (exp(L_min) amin)
        self.n_clamped += ((eps_tot > 0) & ~ok).sum() + \
            (ok & (L < disc._L_min)).sum()
        # Clipped cells (q floored): the populations of the clipped fit
        clip = ok & (s == self._s_min)
        if clip.any():
            f1 = large_fraction(np.where(clip, s, 0.), np.where(clip, L, 1.))
            eps0 = np.where(clip, eps_tot*(1 - f1), eps0)
            eps1 = np.where(clip, eps_tot*f1, eps1)
        disc.set_populations(eps0, eps1, amin*np.exp(np.where(ok, L, L0)))

    def _sizes(self, s, L, amin):
        """f_drift times the mass-averaged sizes of the two populations, and
        the M_1-weighted size M_2/M_1"""
        a = list(population_sizes(s, L, amin)) + [mean_size(s+1, 0, L, amin)]
        return self._f_drift*np.array(a)

    def _third_primitive(self, disc, s, L):
        """The third reconstructed variable: ln <a>"""
        return np.log(mean_size(s, 0, L, disc.amin))

    def _face_sides(self, disc, W):
        """Sides (Sd, s, L) of the face states W, (Sigma_0, Sigma_1, ln <a>)"""
        S0, S1, S0f, S1f = self._population_faces(W)
        s, L = fit_mean_size(S0f, S1f, np.log(np.maximum(np.exp(W[2])/disc.amin, 1 + 1e-12)),
                             self._s_min)
        return (np.array([S0, S1]), s, L)

    def _face_quantities(self, disc, W, W_cell, SQ_cell):
        """Sigma times the transported quantities of the face states W
        [3, Ncells+1] (W_cell, SQ_cell unused)"""
        S0, S1 = np.maximum(W[0], 0), np.maximum(W[1], 0)
        return {'Q': np.array([S0, S1, (S0 + S1)*np.exp(W[2])])}


class TriMaxDrift(TriPoDDrift):
    """TriPoD's variables, eps0 and eps1 transported with the velocities of
    f_drift times the mass-averaged size of each population, and amax
    advected (not conserved) with the velocity of f_drift*amax:
    d amax/dt + v d amax/dR = 0, upwinded, so that amax stays within its
    upwind values. No shrinkage; the closure is TriPoD's (s from eps1/eps0).
    With the gas and in diffusion amax is a tracer of eps1 (eps1*amax, as
    TriPoDDrift). Use a disc with no q bounds, as for TriMeanDrift.

    amax is carried along its characteristics together with a pseudo-density
    rho, the dust mass per unit ln(a) at amax (_trimax_rho, from the power law
    of the cell or edge values), with v(amax); amax is the ratio. The upwind
    fluxes make it a rho-weighted average of the cell's and upwind values, so
    it stays within them and is set by the grains at amax. (Carrying
    eps1*amax with v(amax) instead would add the spurious source
    -amax div(Sigma eps1 (v(amax) - v1)) and could overshoot.)

    args:
        settling, van_leer, f_drift : as TriPoDDrift
    """
    _scheme = 'trimax'
    _species = [0, 1, 2]
    _diffusion_species = [0, 1, 1]

    def __init__(self, settling=False, van_leer=True, f_drift=1.0):
        super(TriMaxDrift, self).__init__(settling, van_leer, f_drift, shrink=False)

    def _sizes(self, s, L, amin):
        """f_drift times the mass-averaged sizes of the two populations, and
        amax"""
        a = list(population_sizes(s, L, amin)) + [amin*np.exp(L)]
        return self._f_drift*np.array(a)

    def _trimax_rho(self, disc, SigD, amax):
        """The pseudo-density: dust mass per unit ln(a) at amax of the power
        law with population surface densities SigD [2, ...] (floors guard
        exactly empty populations, outside the domain)"""
        floor = np.maximum(1e-150*(SigD[0] + SigD[1]), 1e-300)
        L = np.log(np.maximum(amax/disc.amin, np.exp(disc._L_min)))
        return density_at_amax(np.maximum(SigD[0], floor),
                               np.maximum(SigD[1], floor), L)

    def _cell_state(self, disc, Q):
        """The cell values (Q and amax) at the start of the step, and their
        edge values from the cells"""
        amax = disc.amax.copy()
        return ({'Q': Q, 'amax': amax},
                {'Q': self._cell_faces(disc.Sigma*Q[:2]), 'A': self._cell_faces(amax)})

    def _advance(self, dt, disc, state, faces, V_L, V_R):
        """eps0, eps1 and amax after dt (as eps0, eps1, eps1*amax), and the
        parts of the populations' upwind fluxes"""
        Sigma = disc.Sigma
        Q, amax = state['Q'], state['amax']
        QL, QR = faces['Q']
        AL, AR = faces['A']
        pos, neg = self._flux_parts(QL, QR, V_L[:2], V_R[:2])
        eps = Q[:2] + dt*self._divergence(disc, pos + neg)
        VL, VR = V_L[2], V_R[2]
        rL, rR = self._trimax_rho(disc, QL, AL), self._trimax_rho(disc, QR, AR)
        r0 = self._trimax_rho(disc, Sigma*Q[:2], amax)
        r1 = r0 + Sigma*dt*self._divergence(disc, self._flux(rL, rR, VL, VR))
        rA = r0*amax + Sigma*dt*self._divergence(disc, self._flux(rL*AL, rR*AR, VL, VR))
        ok = r1 > 0
        amax = np.where(ok, rA/np.where(ok, r1, 1), amax)
        return np.array([eps[0], eps[1], np.maximum(eps[1], 0)*amax]), pos, neg

    def _face_quantities(self, disc, W, W_cell, SQ_cell):
        """Sigma eps0 and Sigma eps1 of the face states W [3, Ncells+1], and
        their amax"""
        S0, S1 = np.maximum(W[0], 0), np.maximum(W[1], 0)
        return {'Q': np.array([S0, S1]), 'A': np.exp(W[2])}


SCHEMES = {'moments': MomentDrift, 'tripod': TriPoDDrift,
           'trimean': TriMeanDrift, 'trimax': TriMaxDrift}


def make_drift(scheme='moments', **kwargs):
    """The drift class of a scheme ('moments', 'tripod', 'trimean' or
    'trimax'), constructed with kwargs"""
    if scheme not in SCHEMES:
        raise ValueError("Unknown scheme: {}".format(scheme))
    return SCHEMES[scheme](**kwargs)
