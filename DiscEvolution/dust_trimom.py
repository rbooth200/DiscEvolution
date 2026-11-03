# dust_trimom.py
#
# Author: R. Booth
# Date : 5 - Oct - 2026
#
# Moment-consistent version of the TriPoD dust evolution model (Pfeil,
# Birnstiel & Klahr 2024, A&A 691, A45).
#
# The dust size distribution is a truncated power law, n(a) ∝ a^q on
# [amin, amax], described by the TriPoD parameters (eps0, eps1, amax). The
# local coagulation/fragmentation physics is TriPoD's. The difference is in the
# transport: rather than transporting amax*eps1 as a tracer of eps1, three
# linear functionals of the size distribution (by default the moments
# [a^-1, 1, a], i.e. area, mass and size) are transported, each with its own
# exactly weighted velocity. The reduction of amax by filtering in pressure
# traps then follows from transport, without TriPoD's ad hoc shrinkage terms.
#
# This module is a drop-in replacement for dust.py: DustGrowthTwoPop and
# SingleFluidDrift are provided as aliases with compatible interfaces.
################################################################################
from __future__ import print_function
import numpy as np
from .diffusion import dust_schmidt
from scipy.special import exprel, expit, lambertw
from .constants import *
from .dust import DustyDisc, FixedSizeDust
from .reconstruction import VanLeer
from .brent import brentq

__all__ = ["DustyDisc", "FixedSizeDust", "DustGrowthTriMom", "MomentDrift",
           "DustGrowthTwoPop", "SingleFluidDrift"]

################################################################################
# Truncated power-law size distribution
################################################################################
# The mass distribution is eps(a) ∝ a^(q+3) on [amin, amax]. We use
#   s = q + 4,  L = ln(amax/amin),  l = ln(a/amin),
# so that eps(a) da ∝ exp(s l) dl. Integrals of the form
#   E(sig, L) = int_0^L exp(sig l) dl = (x^sig - 1)/sig,   x = amax/amin,
# are evaluated via exprel, which removes the special cases (q = -3, -4, -5)
# of Pfeil et al. (2024) eqs. 24-26.

def _ln_exprel(x):
    """ln((e^x - 1)/x), without overflow for large x"""
    big = x > 700
    return np.where(big, x - np.log(np.where(big, x, 1.)),
                    np.log(exprel(np.where(big, 0., x))))

def _lnE(sig, L):
    """ln E(sig, L), finite for all sig (including sig = 0)"""
    return np.log(L) + _ln_exprel(sig*L)

def _mean_l(sig, L):
    """Mean of l on [0, L] with weight exp(sig l), i.e. d lnE / d sig"""
    x = sig*L
    small = np.abs(x) < 1e-3
    xs = np.where(small, 1., x)
    g = np.where(small, 0.5 + x/12. - x**3/720.,
                 1/(-np.expm1(-xs)) - 1/xs)
    return L*g

def moment_ratio(k, s, L, amin):
    """M_k / M_0 of the (mass-weighted) truncated power law"""
    return amin**k * np.exp(_lnE(s+k, L) - _lnE(s, L))

def large_fraction(s, L):
    """eps1 / eps_tot, with the populations split at a_int = sqrt(amin amax)"""
    return expit(0.5*s*L)

def s_from_split(eps0, eps1, L):
    """s = q + 4 from the population masses (Pfeil et al. 2024, eq. 23)"""
    return 2*np.log(eps1/eps0) / L

def density_at_amax(eps0, eps1, L):
    """Mass per unit ln(a) at amax, eps_tot exp(sL)/E(s, L), of the power law
    with population masses eps0, eps1 > 0 and L = ln(amax/amin) (TriPoD's
    closure, s from s_from_split)"""
    s = s_from_split(eps0, eps1, L)
    return (eps0 + eps1)*np.exp(s*L - _lnE(s, L))

def mean_size(s, l0, l1, amin):
    """Mass-averaged size on [amin e^l0, amin e^l1] (Pfeil et al. 2024, eq. 26)"""
    dl = l1 - l0
    return amin*np.exp(l0 + _lnE(s+1, dl) - _lnE(s, dl))


class SizeQuadrature(object):
    """Composite Gauss-Legendre quadrature in ln(a) over [amin, amax].

    The nodes are split evenly between the two TriPoD populations, so that
    partial integrals over each population are exact sub-sums.

    The default (24 nodes) gives the drift velocities to ~1e-5 for
    amax/amin <~ 1e4, and 5e-4 in the worst case (q = -1, amax/amin ~ 1e4).

    args:
        n_panels : number of panels per population, default=2
        order    : Gauss-Legendre order per panel, default=6
    """
    def __init__(self, n_panels=2, order=6):
        x, w = np.polynomial.legendre.leggauss(order)
        t = (np.arange(n_panels)[:,None] + 0.5*(x[None,:] + 1)) / n_panels
        self._t = t.ravel()
        self._w = np.tile(0.5*w / n_panels, n_panels)
        n = len(self._t)
        self._pop = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])

    @property
    def population(self):
        """Population (0 or 1) that each node belongs to"""
        return self._pop

    def __call__(self, s, L, amin):
        """Nodes and mass weights of the distribution.

        returns:
            a : node sizes, shape [Nq, Ncells]
            w : fraction of the dust mass in each node, shape [Nq, Ncells]
        """
        half = 0.5*L
        l = np.concatenate([self._t[:,None]*half, half + self._t[:,None]*half])
        w = np.concatenate([self._w[:,None]*half]*2)
        w = w * np.exp(s*l - _lnE(s, L))
        return amin*np.exp(l), w

################################################################################
# Relative velocities (following cuDisc, Robinson, Booth & Owen 2024)
################################################################################
def vrel_sqd_OC07(St1, St2, sqrtRe_inv):
    """Ormel & Cuzzi (2007) turbulent relative velocity squared, in units of
    alpha cs^2. Uses cuDisc's closed-form approximation to St* (good to 2%).
    """
    a_ = 0.51505382481121010
    b_ = 0.12876345620280252

    St_l = np.maximum(St1, St2)
    St_c = np.minimum(St_l, 1)
    St_star = np.where(St_l > 1, 1.,
                       St_c*(1 - b_ + np.sqrt(b_*b_ + a_*(1-St_c)/(1+St_c))))
    Sts = np.maximum(St_star, sqrtRe_inv)

    # V1(St) = St (St/(Sts + St) - St/(1 + St)) and
    # V2(St) = (Sts - Re^-1/2) + St^2/(St + Sts) - St^2/(St + Re^-1/2),
    # each factorised: fewer operations, no cancellation in V1
    c1, c2 = 1 - Sts, Sts - sqrtRe_inv
    def V12(St):
        St2_s = St*St/(St + Sts)
        return c1*St2_s/(1 + St), c2*(1 - St2_s/(St + sqrtRe_inv))
    V11, V21 = V12(St1)
    V12_, V22 = V12(St2)

    dv2 = ((St1 - St2)/(St1 + St2 + 1e-300))*(V11 - V12_) + V21 + V22
    # Re <= 1: the turbulent viscosity is below the molecular one, so there is
    # no inertial range and no turbulence (dv2 -> 0 as Re -> 1; the expression
    # is invalid below)
    return np.where(sqrtRe_inv >= 1, 0., np.maximum(dv2, 0))

def _drift_factors(St):
    """St/(1+St^2) and St^2/(1+St^2), safe for St -> 0 and for St^2
    overflowing (absurd sizes in reconstructed states of empty cells)"""
    St2 = St*St
    D = 1 / (1 + St2)
    big = ~np.isfinite(St2)
    if np.any(big):
        return np.where(big, 1/np.where(big, St, 1.), St*D), np.where(big, 1., St2*D)
    return St*D, St2*D

################################################################################
# Fitting the power law to transported quantities
################################################################################
# Largest ln(amax/amin) allowed in the fits
_L_MAX = np.log(1e12)


def _fit_s_L(resid, ok, s0, L0, s_bounds, L_min, y1, tol=1e-12, max_iter=50):
    """Solve resid(s, ln L) = 0 by damped Newton iteration from (s0, L0).

    If the solution is not realizable (s at its bounds), keep the mass and the
    mass-flux moment, y1 = ln(M_1/M_0) - ln(amin), and solve for L instead.

    Each cell is iterated only until its update moves (s, ln L) by less than
    tol. A cell held at a bound counts as converged once ln L stops moving.

    args:
        resid : resid(s, l, idx) -> residuals and Jacobian [2, 2, len(idx)]
                for the cells idx (all cells if idx is None)

    returns:
        s, L : fitted parameters (s0, L0 where ok is False)
        bad  : cells where the fallback was used
    """
    s_lo, s_hi = s_bounds
    # Keep amax/amin in [exp(L_min), exp(_L_MAX)] so exprel(s L) cannot
    # overflow
    l_lo, l_hi = np.log(L_min), np.log(_L_MAX)

    s = s0.copy()
    l = np.log(L0)
    act = np.flatnonzero(ok)
    for _ in range(max_iter):
        if act.size == 0:
            break
        sa, la = s[act], l[act]
        r, J = resid(sa, la, act)
        det = J[0,0]*J[1,1] - J[0,1]*J[1,0]
        det = np.where(det == 0, 1e-300, det)
        ds = np.clip(-( J[1,1]*r[0] - J[0,1]*r[1]) / det, -0.5, 0.5)
        dl = np.clip(-(-J[1,0]*r[0] + J[0,0]*r[1]) / det, -0.5, 0.5)
        s_new = np.clip(sa + ds, s_lo, s_hi)
        l_new = np.clip(la + dl, l_lo, l_hi)
        s[act], l[act] = s_new, l_new
        act = act[(np.abs(s_new - sa) >= tol) | (np.abs(l_new - la) >= tol)]

    # Realizability: if s is at its bounds, keep the mass and the
    # mass-flux moment (k=1) and solve for L.
    r, _ = resid(s, l, None)
    bad = ok & (np.abs(r).max(0) > 1e-6)
    ib = np.flatnonzero(bad)
    if ib.size:
        sb, lb, yb = s[ib], l[ib], y1[ib]
        for _ in range(max_iter):
            L = np.exp(lb)
            f = _lnE(sb+1, L) - _lnE(sb, L) - yb
            df = L*(np.exp(-_lnE(-(sb+1), L)) - np.exp(-_lnE(-sb, L)))
            dl = np.clip(-f/np.where(df == 0, 1e-300, df), -0.5, 0.5)
            l_new = np.clip(lb + dl, l_lo, l_hi)
            done = (np.abs(l_new - lb) < tol).all()
            lb = l_new
            if done:
                break
        l[ib] = lb

    s = np.where(ok, s, s0)
    L = np.where(ok, np.exp(l), L0)
    return s, L, bad


def fit_moments(Q, amin, s0, L0, s_bounds, L_min):
    """Fit the truncated power law to the mass-weighted moments M_-1, M_0, M_1.

    args:
        Q        : moments int a^k eps(a) da for k = -1, 0, 1, shape [3, Ncells]
        amin     : minimum grain size
        s0, L0   : current parameters (starting guess, and the result where
                   the moments are not all positive)
        s_bounds : bounds on s = q + 4
        L_min    : minimum amax/amin

    returns:
        eps_tot, s, L : the fitted distribution
        bad           : cells where the realizability fallback was used
    """
    eps_new = np.maximum(Q[1], 0)
    ok = (eps_new > 0) & (Q[0] > 0) & (Q[2] > 0)
    safe = np.where(ok, eps_new, 1)
    ya = np.log(np.where(ok, Q[0], 1)/safe) + np.log(amin)
    yb = np.log(np.where(ok, Q[2], 1)/safe) - np.log(amin)

    def resid(s, l, idx=None):
        L = np.exp(l)
        lnE0 = _lnE(s, L)
        ya_i, yb_i = (ya, yb) if idx is None else (ya[idx], yb[idx])
        r = np.array([_lnE(s-1, L) - lnE0 - ya_i,
                      _lnE(s+1, L) - lnE0 - yb_i])
        m0, dL0 = _mean_l(s, L), np.exp(-_lnE(-s, L))
        J = np.empty((2, 2) + s.shape)
        for i, k in enumerate((-1, 1)):
            J[i, 0] = _mean_l(s+k, L) - m0
            J[i, 1] = L*(np.exp(-_lnE(-(s+k), L)) - dL0)
        return r, J

    # The fallback keeps the mass-flux moment, y = yb
    s, L, bad = _fit_s_L(resid, ok, s0, L0, s_bounds, L_min, yb)
    return eps_new, s, L, bad


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



################################################################################
# Tabulated inversion of the moments (M_-1, M_0, M_1)
################################################################################
# The shape of the power law is described by two functions of the moments,
#   X = ln(M_1 / (amin M_0))         (position: mass-weighted size),
#   D = ln(M_1 M_-1 / M_0^2) >= 0    (width, independent of amin).
# The member of the family with the same X and L -> inf has the largest width,
# D_inf(X) = X - ln(2 - e^-X), so T = D / D_inf(X) lies in (0, 1). The table
# is uniform in xi = ln X (with nodes at the corners of the parameter box) and
# in
#   v = (logit T - logit T_lo(X)) / (logit T_hi(X) - logit T_lo(X)),
# where T_lo, T_hi bound the image of s in [s_lo, s_hi], L in [L_min, L_max].
# These coordinates map the realizable moments onto the unit square: in
# (M_1/M_0, M_-1/M_0) they are a thin sliver along the diagonal, with the
# inverse singular on it, and logit T resolves both narrow distributions
# (T -> 0) and large L (T -> 1). Moments outside the box are projected by
# clipping v (and X), which keeps M_0 and M_1.

def moment_coords(s, L):
    """X = ln(M_1/(amin M_0)) and D = ln(M_1 M_-1/M_0^2) of the power law"""
    e0, ep, em = _lnE(s, L), _lnE(s+1, L), _lnE(s-1, L)
    return ep - e0, ep + em - 2*e0

def _D_inf(X):
    return X - np.log(2 - np.exp(-X))

def _logit_T(X, D):
    T = np.clip(D/_D_inf(X), 1e-15, 1 - 1e-15)
    return np.log(T/(1 - T))


class MomentTable(object):
    """Inverse of the map (s, L) -> (X, D), tabulated on the parameter box
    s in s_bounds, L in [L_min, L_max] (bilinear interpolation of s and ln L).
    With the default resolution the moments of the interpolated power law
    agree with the input to ~1e-4 in ln M_k.

    Use MomentTable.get to share tables between discs.

    args:
        s_bounds     : bounds on s = q + 4 (finite)
        L_min, L_max : bounds on L = ln(amax/amin)
        nx, nv       : table size
    """
    _tables = {}

    @classmethod
    def get(cls, s_bounds, L_min, L_max=_L_MAX, nx=512, nv=256):
        key = (tuple(s_bounds), L_min, L_max, nx, nv)
        if key not in cls._tables:
            cls._tables[key] = cls(s_bounds, L_min, L_max, nx, nv)
        return cls._tables[key]

    def __init__(self, s_bounds, L_min, L_max=_L_MAX, nx=512, nv=256):
        s_lo, s_hi = s_bounds
        if not (np.isfinite(s_lo) and np.isfinite(s_hi)):
            raise ValueError("MomentTable needs finite bounds on q")
        self._box = (s_lo, s_hi, L_min, L_max)

        # Boundaries of the image of the box, logit T against X. Lower: the
        # edge L = L_min, then s = s_hi. Upper: s = s_lo, then L = L_max.
        n = 4000
        sg = np.linspace(s_lo, s_hi, n)
        Lg = np.exp(np.linspace(np.log(L_min), np.log(L_max), n))
        def edge(s, L):
            X, D = moment_coords(s + 0*L, L + 0*s)
            return X, _logit_T(X, D)
        def join(a, b):
            X = np.concatenate([a[0], b[0][1:]])
            y = np.concatenate([a[1], b[1][1:]])
            keep = np.concatenate([[True], np.diff(X) > 1e-12*X[1:]])
            return X[keep], y[keep]
        lo = edge(sg, L_min), edge(s_hi, Lg)
        hi = edge(s_lo, Lg), edge(sg, L_max)
        self._lo, self._hi = join(*lo), join(*hi)
        corners = [lo[0][0][0], lo[0][0][-1], hi[0][0][-1], lo[1][0][-1]]
        self.X_min, self.X_max = lo[0][0][0], lo[1][0][-1]

        # Grid: uniform in ln X between the corners, uniform in v
        knots = np.unique(np.log(corners))
        seg = np.diff(knots)
        nseg = np.maximum(2, np.round(nx*seg/seg.sum()).astype(int))
        self._xi = np.concatenate(
            [np.linspace(knots[i], knots[i+1], nseg[i] + 1)[:-1]
             for i in range(len(seg))] + [knots[-1:]])
        self._nv = nv
        XI, V = np.meshgrid(self._xi, np.linspace(0, 1, nv), indexing='ij')
        X = np.exp(XI.ravel())
        lo_X, hi_X = self._bounds(X)
        lt = lo_X + V.ravel()*(hi_X - lo_X)
        D = expit(lt)*_D_inf(X)

        # Invert at the nodes by Newton iteration from the nearest of a set of
        # forward samples
        from scipy.spatial import cKDTree
        sf, Lf = np.meshgrid(np.linspace(s_lo, s_hi, 300),
                             np.exp(np.linspace(np.log(L_min), np.log(L_max),
                                                300)))
        sf, Lf = sf.ravel(), Lf.ravel()
        Xf, Df = moment_coords(sf, Lf)
        _, i = cKDTree(np.c_[np.log(Xf), 20*Df/_D_inf(Xf)]).query(
            np.c_[np.log(X), 20*D/_D_inf(X)])
        Q = np.array([np.exp(D - X), np.ones_like(X), np.exp(X)])
        _, s, L, _ = fit_moments(Q, 1.0, sf[i], Lf[i],
                                 (s_lo - 1e-9, s_hi + 1e-9), L_min*(1 - 1e-9))
        s, L = np.clip(s, s_lo, s_hi), np.clip(L, L_min, L_max)
        self._s = s.reshape(XI.shape)
        self._lnL = np.log(L).reshape(XI.shape)

    def _bounds(self, X):
        """logit T on the lower and upper boundaries at X"""
        return np.interp(X, *self._lo), np.interp(X, *self._hi)

    def __call__(self, X, D):
        """Project (X, D) onto the box and invert.

        returns:
            X, D  : the projected values (unchanged where realizable)
            s, L  : the power law
            moved : cells where the projection changed X or D
        """
        s_lo, s_hi, _, _ = self._box
        Xc = np.clip(X, self.X_min, self.X_max)
        lt = _logit_T(Xc, D)
        lo, hi = self._bounds(Xc)
        ltc = np.clip(lt, lo, hi)
        moved = (np.abs(Xc - X) > 1e-12*X) | (np.abs(ltc - lt) > 1e-9)
        Dc = np.where(moved, expit(ltc)*_D_inf(Xc), D)
        Xc = np.where(moved, Xc, X)

        # Bilinear interpolation in (ln X, v)
        xi = np.log(np.clip(Xc, self.X_min, self.X_max))
        v = (ltc - lo)/np.maximum(hi - lo, 1e-300)
        i = np.clip(np.searchsorted(self._xi, xi) - 1, 0, len(self._xi) - 2)
        fx = (xi - self._xi[i])/(self._xi[i+1] - self._xi[i])
        nv = self._nv - 1
        j = np.minimum((v*nv).astype(int), nv - 1)
        fv = v*nv - j
        def interp(t):
            return (1 - fx)*((1 - fv)*t[i, j] + fv*t[i, j+1]) + \
                fx*((1 - fv)*t[i+1, j] + fv*t[i+1, j+1])
        s = np.clip(interp(self._s), s_lo, s_hi)
        return Xc, Dc, s, np.exp(interp(self._lnL)), moved


_S_INNER = -1.0     # lower bound on s of the default table box (q = -5)


class NestedMomentTable(object):
    """Moment inversion for a lower bound on q below -5, keeping the default
    table's resolution where the states are: the q >= -5 table (MomentTable's
    default box and size) is used for every state inside its box, and a table
    for the full box only where a state lies outside it.

    A single table on the wider box keeps its number of nodes, so for narrow
    distributions the q >= -5 shapes are squeezed into a few per cent of its
    second coordinate and lose most of their resolution."""
    def __init__(self, s_bounds, L_min):
        self._inner = MomentTable.get((max(s_bounds[0], _S_INNER), s_bounds[1]), L_min)
        self._outer = MomentTable.get(s_bounds, L_min)

    def __call__(self, X, D):
        """As MomentTable.__call__"""
        X, D = np.asarray(X, dtype='f8'), np.asarray(D, dtype='f8')
        Xc, Dc, s, L, out = self._inner(X, D)
        moved = np.zeros_like(out)
        if out.any():
            Xo, Do, so, Lo, mo = self._outer(X[out], D[out])
            Xc, Dc, s, L = (np.array(a, dtype='f8') for a in (Xc, Dc, s, L))
            Xc[out], Dc[out], s[out], L[out] = Xo, Do, so, Lo
            moved[out] = mo
        return Xc, Dc, s, L, moved


################################################################################
# Growth model
################################################################################
class DustGrowthTriMom(DustyDisc):
    """Three-parameter (TriPoD) dust growth model with moment-consistent
    transport.

    The size distribution is a truncated power law described by the dust
    fractions of the small (eps0) and large (eps1) populations, split at
    a_int = sqrt(amin amax), and the maximum size amax. dust_frac = [eps0, eps1]
    and grain_size = [a0, amax], where a0 is the mass-averaged size of the
    small population.

    The state is the total dust fraction, M_0, and the scaled moments
    M_-1/M_0 and M_1/M_0. The power-law parameters are computed from them
    when needed, and growth changes only the scaled moments, so it conserves
    the dust mass exactly. Direct writes to dust_frac or grain_size are
    copied to the state.

    Growth of amax is integrated analytically at fixed growth time-scale,
    assuming dv_max is a power law in a during each step, with the local index
    estimated from the contributions to dv_max. Mass exchange between the
    populations (sweep-up and fragmentation) is integrated analytically at
    fixed coefficients.

    args:
        grid      : Disc gridding object
        star      : Stellar object
        eos       : Equation of state
        eps       : Initial dust fraction
        Sigma     : Initial surface density distribution
        rho_s     : Grain solid density, default=1
        Sc        : Schmidt number, default=1
        uf_0      : Fragmentation velocity (default = 100 (cm/s))
        uf_ice    : Fragmentation velocity of icy grains (default = 1000 (cm/s))
        f_ice     : Ice fraction, default=1
        thresh    : Threshold ice fraction for switchng between icy/non icy
                    fragmentation velocity, default=0.1
        f_grow    : Growth time-scale factor, default=1.2755 (calibrated)
        a0        : Initial maximum particle size (default = 1e-4, 1 micron)
        amin      : Minimum particle size (default = 1e-5, 0.1 micron)
        feedback  : Whether to include feedback from dust on gas
        distribution_slope:
                    Initial slope -d ln n(a) / d ln a of the number
                    distribution (3.5 for MRN)
        f_dv      : Size ratio for the relative velocities of the largest
                    grains, default=0.4892 (calibrated; TriPoD: 0.4)
        s_frag    : Steepness of the growth/fragmentation transition
                    (TriPoD calibration, default=3)

                    The defaults of the growth parameters are calibrated
                    against coag_toolkit (scripts/trimom/production/
                    calibration/data/calibration_final.json, notes.md there):
                    TriPoD's f_grow 1, f_dv 0.4, q_sweep -3, q_turb2 -3.75,
                    frag_width 5 and f_coll 1 reproduce TriPoD.
        q_sweep, q_turb1, q_turb2, q_drift_frag :
                    Size distribution exponents in the different regimes:
                    sweep-up, turbulent fragmentation with the largest grains
                    in the intermediate (turb1) or small-eddy (turb2) regime,
                    and drift-induced fragmentation. Defaults -2.55
                    (calibrated), -3.5, -3.5993 (calibrated), -3.75; TriPoDPy:
                    (-3, -3.5, -3.75, -3.75).
        growth_cfl: Maximum change in ln(amax) per (sub-)step away from
                    fragmentation equilibrium, default=0.5
        eq_factor : No growth time-step limit for growing cells whose amax
                    is within this factor of its equilibrium size (found by a
                    root search, _equilibrium_size), default=2. There amax
                    relaxes onto the equilibrium with the secant index.
                    Shrinking cells always take a single step (their
                    equilibrium is searched for down to 1.5 amin). Set to 1 to
                    always apply growth_cfl.
        max_substeps: Maximum number of growth sub-steps, default=1000
        vg2_factor: Turbulent gas velocity squared in units of alpha cs^2,
                    default=1.5 (Ormel & Cuzzi 2007, DustPy, coag_toolkit's
                    reference runs and TriPoD's calibration). cuDisc and
                    two-pop-py use 1.
        drift_switch: The size distribution follows the drift-fragmentation
                    exponent where the drift relative velocity exceeds
                    drift_switch times the turbulent one. Default 0.3, as in
                    TriPoDPy (pdriftfrag); Pfeil et al. (2024, App. C) and
                    Birnstiel et al. (2012) correspond to 1.
        turb_switch: The size distribution follows the intermediate-regime
                    (rather than the small-eddy) turbulent exponent where the
                    Stokes number of the largest grains exceeds turb_switch
                    times Re^-1/2. Default 5 (TriPoD).
        frag_width: Width parameter w of the transition from the sweep-up to
                    the fragmentation exponent, p_frag = exp(-(w (x - 1))^2)
                    with x = min(dv_max/v_frag, 1). Default 1.5
                    (calibrated; TriPoD: 5); smaller is a wider transition.
        f_coll    : Size, relative to amax, of the larger grain in the
                    collision velocity of the largest grains, dv_max =
                    dv(f_coll amax, f_dv f_coll amax). Below 1 the size
                    distribution reaches beyond a velocity jump (e.g. the
                    OC07 small-eddy transition) by 1/f_coll. Default 0.9
                    (calibrated; TriPoD: 1).
        q_bounds  : Bounds on the power-law index q, used in the moment fit
                    and when q is read from the population split,
                    default=(-5, -1). Use -np.inf to remove the lower bound.
        quadrature: SizeQuadrature object, default=SizeQuadrature()
        moment_table: Invert the moments with a precomputed table
                    (MomentTable), default=True. Needs a finite lower bound
                    on q; otherwise Newton iteration is used.

        f_frag, f_drift, start_small:
                    Accepted for compatibility with DustGrowthTwoPop, unused.
    """
    def __init__(self, grid, star, eos, eps, Sigma=None,
                 rho_s=1., Sc=1., uf_0=100., uf_ice=1e3, f_ice=1, thresh=0.1,
                 f_grow=1.2755, a0=1e-4, amin=1e-5, f_drift=None, f_frag=None,
                 feedback=True, start_small=True, distribution_slope=3.5,
                 f_dv=0.4892, s_frag=3.,
                 q_sweep=-2.55, q_turb1=-3.5, q_turb2=-3.5993, q_drift_frag=-3.75,
                 growth_cfl=0.5, eq_factor=2.0, max_substeps=1000,
                 vg2_factor=1.5,
                 q_bounds=(-5., -1.), quadrature=None, moment_table=True,
                 drift_switch=0.3, turb_switch=5.0, frag_width=1.5, f_coll=0.9):
        super(DustGrowthTriMom, self).__init__(grid, star, eos, Sigma, rho_s,
                                              Sc, feedback)
        # Memos of the growth step: equilibrium roots (_equilibrium_size),
        # cached parameters and gas velocities
        self._l_eq_cache = None
        self._params_memo = None
        self._gas_vel_memo = None

        if a0 <= 1.5*amin:
            raise ValueError("Initial size a0 must exceed 1.5*amin")

        self._uf_0   = uf_0 / (AU * Omega0)
        self._uf_ice = uf_ice / (AU * Omega0)
        self._ice_threshold = thresh
        self._uf = self._frag_velocity(f_ice)

        self._fgrow = f_grow
        self._f_dv = f_dv
        self._f_coll = f_coll
        self._s_frag = s_frag
        self._q_sweep = q_sweep
        self._q_turb1 = q_turb1
        self._q_turb2 = q_turb2
        self._q_drift_frag = q_drift_frag
        self._growth_cfl = growth_cfl
        self._eq_factor = eq_factor
        self._max_substeps = max_substeps
        self._vg2_factor = vg2_factor
        self._drift_switch = drift_switch
        self._turb_switch = turb_switch
        self._frag_width = frag_width

        if quadrature is None:
            quadrature = SizeQuadrature()
        self._quad = quadrature

        # Number distribution slope (used for dust entrainment in
        # photoevaporation.py).
        self._p = distribution_slope

        # Initialize the dust distribution
        Ncells = self.Ncells
        self._amin = amin
        self._L_min = np.log(1.5)
        self._s_bounds = (q_bounds[0] + 4, q_bounds[1] + 4)   # on s = q + 4

        if moment_table and np.isfinite(self._s_bounds[0]):
            if self._s_bounds[0] < _S_INNER:
                self._table = NestedMomentTable(self._s_bounds, self._L_min)
            else:
                self._table = MomentTable.get(self._s_bounds, self._L_min)
        else:
            self._table = None

        L = np.log(a0/amin) * np.ones(Ncells)
        s = (4 - distribution_slope) * np.ones(Ncells)

        # Cache of the power-law parameters
        self._eps = np.empty([2, Ncells], dtype='f8')
        self._a   = np.empty([2, Ncells], dtype='f8')
        self._s_last = s.copy()
        self._snap = None
        self.set_size_distribution(eps*np.ones(Ncells), s, L)

        # Diagnostics: realizability fallbacks in remove_dust_with_gas
        self.n_removal_clamped = 0

        # The ices last passed to update_ices, which growth mixes
        self._ices = None

        self._head = (', uf_0: {}cm s^-1, uf_ice: {}cm s^-1, thresh: {}'
                      ', f_grow: {}, a0: {}cm, amin: {}cm, f_dv: {}'
                      ', s_frag: {}'.format(
                          uf_0, uf_ice, thresh, f_grow, a0, amin,
                          f_dv, s_frag))

        self.update(0)

    def ASCII_header(self):
        """Dust growth header"""
        return super(DustGrowthTriMom, self).ASCII_header() + self._head

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        name, head = super(DustGrowthTriMom, self).HDF5_attributes()
        tmp = dict([x.strip().split(":") for x in self._head.split(",") if x])
        head.update(tmp)
        return self.__class__.__name__, head

    ############################################################################
    # Size distribution
    ############################################################################
    @property
    def amin(self):
        """Minimum grain size"""
        return self._amin

    @property
    def amax(self):
        """Maximum grain size"""
        self._params()
        return self._a[1]

    @property
    def dust_frac(self):
        """Dust fractions of the two populations, [eps0, eps1]"""
        self._params()
        return self._eps

    @property
    def grain_size(self):
        """Grain sizes, [a0, amax]"""
        self._params()
        return self._a

    @property
    def integ_dust_frac(self):
        """Total dust fraction, or zero if not including dust feedback"""
        if self._feedback:
            self._sync()
            return self._eps_tot.copy()
        return 0

    def size_distribution(self):
        """Parameters of the truncated power law.

        returns:
            eps_tot : total dust fraction
            s       : q + 4
            L       : ln(amax/amin)
        """
        self._params()
        _, s, L = self._cached_params()
        return self._eps_tot.copy(), s, L

    def set_size_distribution(self, eps_tot, s, L):
        """Set the state from the power-law parameters"""
        eps_tot = np.array(np.broadcast_to(eps_tot, (self.Ncells,)), dtype='f8')
        self._set_state(eps_tot, self._shape_moments(s, L))
        self._set_cache(eps_tot, s, L)

    def set_populations(self, eps0, eps1, amax):
        """Set the state from the TriPoD parameters, which are kept exactly"""
        self._eps[0] = eps0
        self._eps[1] = eps1
        self._a[1] = amax
        eps_tot, s, L = self._cached_params()
        self._set_state(eps_tot, self._shape_moments(s, L))
        self._s_last = s
        self._a[0] = mean_size(s, 0, 0.5*L, self._amin)
        self._cache_ok = True
        self._take_snapshot()

    def set_moments(self, Q):
        """Set the state from the moments (M_-1, M_0, M_1) per unit Sigma.

        Moments outside the range of the power law are projected onto it,
        keeping M_0 and M_1. Where M_0, M_-1 or M_1 is not positive the shape
        is kept.

        returns:
            the number of cells projected
        """
        self._sync()
        eps = np.maximum(Q[1], 0)
        ok = (eps > 0) & (Q[0] > 0) & (Q[2] > 0)
        e = np.where(ok, eps, 1)
        y = np.array([np.log(np.where(ok, Q[0], 1)/e*self._amin),
                      np.log(np.where(ok, Q[2], 1)/e/self._amin)])
        y = np.where(ok, y, self._y)
        if self._table is not None:
            X, D, s, L, moved = self._table(y[1], y[0] + y[1])
            moved &= ok
            y = np.where(moved, [D - X, X], y)
        else:
            s, L, moved = self._fit_newton(y)
            moved &= ok
            y = np.where(moved, self._shape_moments(s, L), y)
        self._set_state(eps, y)
        self._pending = (s, L)
        return moved.sum()

    def _update_a0(self):
        _, s, L = self._cached_params()
        self._a[0] = mean_size(s, 0, 0.5*L, self._amin)

    def moments(self, ks=(-1, 0, 1)):
        """Mass-weighted moments, int a^k eps(a) da, for each k in ks"""
        # From the state where possible, otherwise from the power law
        self._sync()
        y = {-1: self._y[0], 0: 0., 1: self._y[1]}
        if not set(ks) <= set(y):
            _, s, L = self.size_distribution()
        return np.array([self._eps_tot*(np.exp(y[k])*self._amin**k if k in y
                                        else moment_ratio(k, s, L, self._amin))
                         for k in ks])

    @property
    def area_size(self):
        """Area-equivalent grain size, a_A = M_0 / M_-1"""
        self._sync()
        return self._amin*np.exp(-self._y[0])

    @property
    def surface_area(self):
        """Total grain surface area per unit gas mass [cm^2 g^-1]"""
        self._sync()
        M_m1 = self._eps_tot*np.exp(self._y[0])/self._amin
        return 3*M_m1 / self._rho_s / np.maximum(1 - self._eps_tot, 1e-300)

    @property
    def grain_cross_section(self):
        """Geometric cross-section of the grains per unit dust mass,
        3 / (4 rho_s a_A) [cm^2 g^-1], a quarter of their surface area"""
        return 3 / (4*self._rho_s*self.area_size)

    ############################################################################
    # State and cache
    ############################################################################
    # The state is eps_tot = M_0 and y = [ln(amin M_-1/M_0), ln(M_1/(amin M_0))].
    # The cache holds the power-law parameters (_eps = [eps0, eps1],
    # _a = [a0, amax], and s), computed from the state when first needed.
    # A snapshot of the cache detects direct writes to it, which then define
    # the state. Such writes need an up-to-date cache: read dust_frac,
    # grain_size or size_distribution() first.
    def _shape_moments(self, s, L):
        """y = [ln(M_-1/M_0) + ln(amin), ln(M_1/M_0) - ln(amin)]"""
        e0 = _lnE(s, L)
        return np.array([_lnE(s-1, L) - e0, _lnE(s+1, L) - e0])

    def _set_state(self, eps_tot, y):
        self._eps_tot = np.array(eps_tot, dtype='f8')
        self._y = np.array(y, dtype='f8')
        self._cache_ok = False
        self._pending = None

    def _take_snapshot(self):
        self._snap = (self._eps.copy(), self._a[1].copy())

    def _cached_params(self):
        """(eps0 + eps1, s, L) of the cached parameters (memoised on them:
        the growth step reads them several times per state)"""
        key = (self._eps[0], self._eps[1], self._a[1], self._s_last,
               np.array(self._s_bounds, dtype='f8'), np.array(self._L_min, dtype='f8'))
        m = self._params_memo
        if m is not None and all(np.array_equal(k, km) for k, km in zip(key, m[0])):
            return m[1][0].copy(), m[1][1].copy(), m[1][2].copy()
        out = self._compute_cached_params()
        self._params_memo = (tuple(np.array(k, copy=True) for k in key), out)
        return out[0].copy(), out[1].copy(), out[2].copy()

    def _compute_cached_params(self):
        eps0 = np.maximum(self._eps[0], 0)
        eps1 = np.maximum(self._eps[1], 0)
        L = np.log(np.maximum(self._a[1]/self._amin, np.exp(self._L_min)))
        ok = (eps0 > 0) & (eps1 > 0)
        s = np.where(ok, s_from_split(np.where(ok, eps0, 1),
                                      np.where(ok, eps1, 1), L),
                     self._s_last)
        return eps0 + eps1, np.clip(s, *self._s_bounds), L

    def _params(self):
        """Make the state and the cached parameters up to date"""
        self._sync()
        if not self._cache_ok:
            if self._pending is not None:
                s, L = self._pending
            elif self._table is not None:
                _, _, s, L, _ = self._table(self._y[1], self._y[0] + self._y[1])
            else:
                s, L, _ = self._fit_newton(self._y)
            self._set_cache(self._eps_tot, s, L)

    def _set_cache(self, eps_tot, s, L):
        f1 = large_fraction(s, L)
        self._eps[0] = eps_tot*(1 - f1)
        self._eps[1] = eps_tot*f1
        self._a[1] = self._amin*np.exp(L)
        self._a[0] = mean_size(s, 0, 0.5*L, self._amin)
        self._s_last = np.array(s, dtype='f8')
        self._cache_ok = True
        self._pending = None
        self._take_snapshot()

    def _sync(self):
        """Copy direct writes to the cached parameters to the state"""
        if self._snap is None or not self._cache_ok:
            return
        eps_s, a_s = self._snap
        eps, amax = self._eps, self._a[1]
        changed = (eps != eps_s).any(0) | (amax != a_s)
        if not changed.any():
            return
        # Both populations rescaled by the same factor: only M_0 changes
        tot_s = eps_s.sum(0)
        scaled = changed & (amax == a_s) & (tot_s > 0) & \
            np.isclose(eps[0]*eps_s[1], eps[1]*eps_s[0], rtol=1e-12, atol=0)
        f = eps.sum(0)/np.where(tot_s > 0, tot_s, 1)
        eps_tot = np.where(scaled, self._eps_tot*f, self._eps_tot)
        y = self._y
        # Otherwise the parameters define the state
        other = changed & ~scaled
        if other.any():
            e, s, L = self._cached_params()
            eps_tot = np.where(other, e, eps_tot)
            y = np.where(other, self._shape_moments(s, L), y)
            self._s_last = np.where(other, s, self._s_last)
        self._eps_tot, self._y = eps_tot, y
        self._take_snapshot()

    def _fit_newton(self, y):
        """(s, L) of the scaled moments y by Newton iteration from the cached
        parameters, and the cells that needed the realizability fallback"""
        s0 = self._s_last
        L0 = np.log(np.maximum(self._a[1]/self._amin, np.exp(self._L_min)))
        Q = np.array([np.exp(y[0])/self._amin, np.ones_like(y[0]),
                      np.exp(y[1])*self._amin])
        _, s, L, bad = fit_moments(Q, self._amin, s0, L0,
                                   self._s_bounds, self._L_min)
        return s, L, bad

    @property
    def area(self):
        """Mean area of grains (by number)"""
        _, s, L = self.size_distribution()
        return np.pi*moment_ratio(-1, s, L, self._amin) / \
            moment_ratio(-3, s, L, self._amin)

    def quadrature(self):
        """Quadrature nodes and mass weights of the current distribution"""
        _, s, L = self.size_distribution()
        return self._quad(s, L, self._amin)

    ############################################################################
    # Velocities
    ############################################################################
    def dust_scale_height(self, St):
        """Dust scale height (as used in cuDisc)"""
        return self.H / np.sqrt(1 + St/(self.alpha/self._Sc))

    def _sqrtRe_inv(self):
        """1/sqrt(Re), with Re = alpha Sigma_g sigma_H2 / (2 mu m_p) as in
        DustPy and coag_toolkit (cuDisc's mean-free-path form has sqrt(2 pi)
        in place of 2)"""
        Re = 0.5*self.alpha*self.Sigma_G*sig_H2 / (self.mu*m_p)
        return 1/np.sqrt(np.maximum(Re, 1e-300))

    def _gas_velocities(self):
        """Gas velocities relative to Keplerian at cell centres, including
        feedback from the dust (Dipierro+18, without the viscous term).

        returns:
            u_gas, v_gas : radial and azimuthal gas velocities
        """
        if not self.feedback:
            # Gas only: memoised on the pressure
            P = self.P
            m = self._gas_vel_memo
            if m is not None and np.array_equal(P, m[0]):
                return m[1][0].copy(), m[1][1].copy()
        rho = self.midplane_gas_density
        eta = - np.gradient(self.P, self.R) / ((rho + 1e-300)*self.Omega_k)

        if not self.feedback:
            out = (0*eta, -0.5*eta)
            self._gas_vel_memo = (P.copy(), out)
            return out[0].copy(), out[1].copy()

        a, w = self.quadrature()
        eps_tot = self.integ_dust_frac
        eps_g = np.maximum(1 - eps_tot, 1e-300)
        St = self.Stokes(self.Sigma_G, a)
        f1, f2 = _drift_factors(St)
        la0 = (eps_tot/eps_g) * (w*(1 - f2)).sum(0)
        la1 = (eps_tot/eps_g) * (w*f1).sum(0)

        D_1 = 1 / ((1 + la0)**2 + la1**2)
        return la1*eta*D_1, -0.5*(1 + la0)*eta*D_1

    def _collision_gas(self, gas_vel=None):
        """The gas quantities of relative_velocity(_index), computed once per
        growth-rate evaluation: gas velocities, Sigma_G, cs, Re^-1/2, Omega_k,
        H and alpha/Sc"""
        if gas_vel is None:
            gas_vel = self._gas_velocities()
        return dict(vel=gas_vel, SigG=self.Sigma_G, cs=self.cs, alpha=self.alpha,
                    sqrtRe_inv=self._sqrtRe_inv(), Om=self.Omega_k, H=self.H,
                    aSc=self.alpha/self._Sc)

    def relative_velocity(self, a1, a2, gas_vel=None, components=False, gas=None):
        """Relative velocity of grains of size a1 and a2.

        Includes Brownian motion, turbulence (Ormel & Cuzzi 2007), radial and
        azimuthal drift, and differential settling, as in cuDisc.

        args:
            a1, a2     : grain sizes, broadcastable to [..., Ncells]
            gas_vel    : Gas velocities from _gas_velocities (computed if None)
            components : Also return the drift contributions
            gas        : The gas quantities from _collision_gas (computed,
                         with gas_vel, if None)

        returns:
            dv : Total relative velocity (code units)
            dv_rad, dv_az : Radial and azimuthal drift contributions (only if
                            components=True)
        """
        dv, dv_rad, dv_az, _ = self._relative_velocity(
            a1, a2, self._collision_gas(gas_vel) if gas is None else gas)
        if components:
            return dv, np.abs(dv_rad), np.abs(dv_az)
        return dv

    def relative_velocity_index(self, a1, a2, gas_vel=None, gas=None, components=False):
        """Relative velocity of grains of size a1 and a2, and its logarithmic
        derivative with respect to size when both sizes scale together,
        beta = d ln dv / d ln a.

        dv^2 is a sum of contributions, so beta is the dv_i^2-weighted mean of
        the indices of the contributions:
            Brownian motion: -3/2 (exact)
            Radial / azimuthal drift, settling: exact derivatives
            Turbulence: by regime, 1 (small eddies, St < Re^-1/2), 1/2
                        (intermediate) and -1/2 (St > 1), joined smoothly.

        gas, components: as relative_velocity.

        returns:
            dv, beta (and dv_rad, dv_az if components)
        """
        dv, dv_rad, dv_az, beta = self._relative_velocity(
            a1, a2, self._collision_gas(gas_vel) if gas is None else gas, index=True)
        if components:
            return dv, beta, np.abs(dv_rad), np.abs(dv_az)
        return dv, beta

    def _relative_velocity(self, a1, a2, g, index=False):
        """dv, its signed radial and azimuthal drift contributions, and (if
        index) beta, for gas quantities g (_collision_gas)"""
        u_gas, v_gas = g['vel']
        cs = g['cs']
        St1 = self.Stokes(g['SigG'], a1)
        St2 = self.Stokes(g['SigG'], a2)

        # Turbulence
        turb2 = self._vg2_factor*g['alpha']*cs*cs * \
            vrel_sqd_OC07(St1, St2, g['sqrtRe_inv'])

        # Brownian motion
        m1 = (4*np.pi/3)*self._rho_s*a1**3
        m2 = (4*np.pi/3)*self._rho_s*a2**3
        bm2 = (8*m_p/np.pi)*self.mu*(m1 + m2)/(m1*m2) * cs*cs

        # Radial and azimuthal drift: X(St) = A St/(1+St^2) + B St^2/(1+St^2)
        # with (A, B) = (2 v_gas, -u_gas) and (-u_gas/2, -v_gas)
        f11, f21 = _drift_factors(St1)
        f12, f22 = _drift_factors(St2)
        dv_rad = (2*v_gas*f11 - u_gas*f21) - (2*v_gas*f12 - u_gas*f22)
        dv_az = (-0.5*u_gas*f11 - v_gas*f21) - (-0.5*u_gas*f12 - v_gas*f22)

        # Differential settling: Y(St) = Omega H_d(St) min(St, 1/2), with
        # dust_scale_height
        Y1 = g['Om']*(g['H'] / np.sqrt(1 + St1/g['aSc']))*np.minimum(St1, 0.5)
        Y2 = g['Om']*(g['H'] / np.sqrt(1 + St2/g['aSc']))*np.minimum(St2, 0.5)
        d_set = Y1 - Y2

        dv2 = turb2 + bm2 + dv_rad*dv_rad + dv_az*dv_az + d_set*d_set
        if not index:
            return np.sqrt(dv2), dv_rad, dv_az, None

        # beta: each contribution Delta^2 adds Delta d Delta / d ln a
        St_L = np.maximum(St1, St2)
        p_small = 1/(1 + (St_L/g['sqrtRe_inv'])**4)
        p_large = St_L**4/(1 + St_L**4)
        beta_turb = p_small + (1 - p_small)*(0.5 - p_large)
        num = turb2*beta_turb - 1.5*bm2

        def dX(St, A, B):       # St dX/dSt
            D = 1/(1 + St*St)**2
            return A*St*(1 - St*St)*D + B*2*St*St*D
        num = num + dv_rad*(dX(St1, 2*v_gas, -u_gas) - dX(St2, 2*v_gas, -u_gas))
        num = num + dv_az*(dX(St1, -0.5*u_gas, -v_gas) - dX(St2, -0.5*u_gas, -v_gas))

        def dlnY(St):
            x = St/g['aSc']
            return (St < 0.5) - 0.5*x/(1 + x)
        num = num + d_set*(Y1*dlnY(St1) - Y2*dlnY(St2))

        return np.sqrt(dv2), dv_rad, dv_az, num/dv2

    ############################################################################
    # Growth / fragmentation
    ############################################################################
    def _frag_velocity(self, f_ice):
        """Fragmentation velocity"""
        # Interplate between the icy/ice free region
        f_ice = np.minimum(f_ice/self._ice_threshold, 1)
        f_ice = f_ice*f_ice*f_ice*(10-f_ice*(15-6*f_ice))
        return self._uf_0 + (self._uf_ice - self._uf_0) * f_ice

    def _q_equilibrium(self, dv_max, dv_rad, dv_az, St_max):
        """Target size distribution exponent (Pfeil et al. 2024, eqs. 36-38
        and App. C, following the TriPoDPy implementation)"""
        alpha = self.alpha
        sqrtRe_inv = self._sqrtRe_inv()

        # Transition between the two turbulent regimes: p_t1 -> 1 when the
        # largest grains are in the intermediate regime
        # (St_max > turb_switch Re^-1/2; TriPoD: 5)
        f_t1t2 = self._turb_switch*sqrtRe_inv / St_max
        p_t1 = 0.5*(1 - (f_t1t2**4 - 1)/(f_t1t2**4 + 1))

        # Transition between drift- and turbulence-dominated fragmentation
        dv_dr = np.maximum(np.hypot(dv_rad, dv_az), 1e-300)
        St_mn = 0.5*St_max
        vgas = np.sqrt(self._vg2_factor*alpha)*self.cs
        v_small = vgas*np.sqrt((St_max - St_mn)/(St_max + St_mn) *
                               (St_max**2/(St_max + sqrtRe_inv) -
                                St_mn**2/(St_mn + sqrtRe_inv)))
        v_inter = vgas*np.sqrt(2.292*St_max)
        # No turbulence for Re <= 1 (as vrel_sqd_OC07)
        v_turb = np.where(sqrtRe_inv < 1, p_t1*v_inter + (1 - p_t1)*v_small, 0.)
        f_dt = self._drift_switch*v_turb / dv_dr
        f_dt6 = np.minimum(f_dt, 1e50)**6
        p_dr = 0.5 + 0.5*(1 - f_dt6)/(f_dt6 + 1)

        q_turb = p_t1*self._q_turb1 + (1 - p_t1)*self._q_turb2
        q_frag = p_dr*self._q_drift_frag + (1 - p_dr)*q_turb

        # Transition between growth (sweep-up) and fragmentation
        p_frag = np.exp(-(self._frag_width*(np.minimum(dv_max/self._uf, 1) - 1))**2)

        return p_frag*q_frag + (1 - p_frag)*self._q_sweep

    @staticmethod
    def grow_amax(a0, t_grow, z0, n, dt):
        """Analytic solution for amax growth at fixed t_grow.

        TriPoD's growth rate (Pfeil et al. 2024, eq. 29) can be written
            d ln a / dt = tanh(z/2) / t_grow,  z = s ln(v_frag / dv_max).
        Assuming dv_max ∝ a^(n/s), z = z0 - n ln(a/a0), so that
            dz/dt = -(n/t_grow) tanh(z/2),
        which has the solution
            sinh(z/2) = sinh(z0/2) exp(-n dt / (2 t_grow)),
        and ln(a/a0) = (z0 - z)/n.

        args:
            a0     : initial size
            t_grow : free growth time-scale, a / (da/dt) for dv_max << v_frag
            z0     : initial s ln(v_frag / dv_max) (> 0 for growth, < 0 for
                     fragmentation)
            n      : power-law index of (dv_max/v_frag)^s with a
            dt     : time-step

        returns:
            a : the size after dt
        """
        x = dt / t_grow
        small = np.abs(n*x) < 1e-8
        # n -> 0: z is constant during the step
        a_small = a0*np.exp(np.tanh(0.5*z0)*np.where(small, x, 0))

        n_safe = np.where(small, 1., n)
        w = np.sinh(0.5*z0)*np.exp(np.minimum(-0.5*n_safe*x, 700.))
        z = 2*np.arcsinh(w)
        return np.where(small, a_small, a0*np.exp((z0 - z)/n_safe))

    @staticmethod
    def exchange_fraction(f0, A, B, dt):
        """Analytic solution of df/dt = A f - B f^2 at fixed A, B >= 0"""
        x = A*dt
        return f0 / (np.exp(-x) + B*f0*dt*exprel(-x))

    def _growth_rates(self):
        """Coefficients of the local growth/fragmentation step, evaluated for
        the current state.

        returns a dict with:
            t_grow : free growth time-scale of amax
            z0     : s ln(v_frag / dv_max)
            C1, C2 : sweep-up and fragmentation rate coefficients
        """
        eps_tot, s, L = self.size_distribution()
        amin, amax = self._amin, self._a[1]

        gas_vel = self._gas_velocities()

        # Representative sizes
        f_dv = self._f_dv
        a0 = mean_size(s, 0, 0.5*L, amin)
        a1 = mean_size(s, 0.5*L, L, amin)

        g = self._collision_gas(gas_vel)
        ac = self._f_coll*amax
        # dv_max, its drift components and its index in one evaluation
        dv_max, beta, dv_rad, dv_az = self.relative_velocity_index(ac, f_dv*ac, gas=g,
                                                                   components=True)
        dv01 = self.relative_velocity(a0, a1, gas=g)
        dv11 = self.relative_velocity(a1, f_dv*a1, gas=g)

        St_max = self.Stokes(self.Sigma_G, amax)
        q = self._q_equilibrium(dv_max, dv_rad, dv_az, St_max)

        H1 = self.dust_scale_height(self.Stokes(self.Sigma_G, a1))
        t_grow = self._free_growth_time(amax, H1, dv_max)
        z0 = self._s_frag*np.log(self._uf/dv_max)

        # Power-law index of (dv_max / v_frag)^s with amax: the dv^2-weighted
        # mean of the indices of the contributions (relative_velocity_index),
        # clipped to their physical range as a safeguard
        n = self._s_frag*np.clip(beta, -1.5, 1.0)

        # Near equilibrium (the equilibrium size, where dv_max = v_frag, lies
        # within the window of _eq_window): use the secant index from amax to it,
        # n = z0 / ln(a_eq/amax), so that the analytic step relaxes onto the
        # equilibrium without overshooting it, however steeply dv_max varies
        # (e.g. at the OC07 small-eddy transition). See _equilibrium_size.
        if self._eq_factor > 1:
            l_eq, near = self._equilibrium_size(amax, z0, gas_vel, gas=g)
            dl = l_eq - np.log(amax)
            # |z0/dl|: at a discontinuity in dv_max (amax on the jump) the
            # root, found to tol, can lie just on the far side of amax from
            # the direction z0 points; a negative index would then run away
            sec = near & (np.abs(z0) > 1e-8) & (dl != 0)
            n = np.where(sec, np.abs(z0/np.where(sec, dl, 1.)), n)
        else:
            near = np.zeros_like(z0, dtype=bool)

        # Mass exchange between the populations, eqs. 32-35
        H0 = self.dust_scale_height(self.Stokes(self.Sigma_G, a0))
        sig01 = np.pi*(a0 + a1)**2
        sig11 = np.pi*(a1 + f_dv*a1)**2
        m1 = (4*np.pi/3)*self._rho_s*a1**3
        F = H1*np.sqrt(2/(H0**2 + H1**2)) * (sig01/sig11) * (dv01/dv11) * \
            np.exp(-0.5*(q + 4)*L)
        C1 = sig01*dv01 / (m1*np.sqrt(2*np.pi*(H0**2 + H1**2)))
        C2 = sig11*dv11*F / (2*m1*np.sqrt(np.pi)*H1)

        return dict(t_grow=t_grow, z0=z0, n=n, near=near, C1=C1, C2=C2)

    def _free_growth_time(self, amax, H1, dv_max):
        """Free growth time-scale of amax (eq. 29, integrated at fixed t_grow),
        with H1 the scale height of the large population"""
        Sigma1 = self.Sigma*self._eps[1]
        adot_free = Sigma1*dv_max / (np.sqrt(2*np.pi)*H1*self._rho_s)
        return self._fgrow*amax / np.maximum(adot_free, 1e-300)

    def _eq_window(self, amax, z0):
        """ln of the window in which the equilibrium size is searched for,
        in the direction amax moves: eq_factor for growing cells, down to
        1.5 amin for shrinking ones (which take a single step, _growth_dt)"""
        return np.where(z0 >= 0, np.log(self._eq_factor),
                        np.log(np.maximum(amax/(1.5*self._amin), 1.)))

    def _near_cached(self, amax, z0, tol=1e-3):
        """Whether the cached equilibrium root (_equilibrium_size) lies within
        the window (_eq_window) in the direction amax moves: the band of _growth_dt,
        without a new root search (the root depends only on the gas and
        v_frag)"""
        lc = self._l_eq_cache
        near = z0 == 0
        if lc is None or lc.shape != amax.shape:
            return near
        d = (lc - np.log(amax))*np.where(z0 >= 0, 1., -1.)
        with np.errstate(invalid='ignore'):
            return near | (np.isfinite(lc) & (d >= -tol) & (d <= self._eq_window(amax, z0)))

    def _equilibrium_size(self, amax, z0, gas_vel, n_search=3, tol=1e-3,
                          max_iter=50, gas=None):
        """ln of the equilibrium size, where z = s ln(v_frag/dv_max) = 0, if
        it lies within the window of _eq_window in the direction amax moves
        (up to eq_factor for z0 > 0, down to 1.5 amin for z0 < 0): the root
        the growth law relaxes to.

        The equilibrium depends only on the gas and v_frag, so the last roots
        are cached and checked first (one evaluation of z), with one Newton
        step where they have moved. Cells without a valid cached root are
        bracketed by n_search uniform steps in ln a, refined by Brent's method
        (brent.brentq) to tol in ln a.

        returns:
            l_eq  : ln of the equilibrium size (ln amax where none is found)
            found : whether it lies within the window
        """
        l0 = np.log(amax)
        sgn = np.where(z0 >= 0, 1., -1.)
        W = self._eq_window(amax, z0)
        s = self._s_frag
        f_dv = self._f_dv
        z_tol = 0.5*s*tol           # |z| ~ s beta |dl| with beta >~ 1/2

        def z(l):
            a = self._f_coll*np.exp(l)
            return s*np.log(self._uf/self.relative_velocity(a, f_dv*a, gas_vel, gas=gas))

        found = z0 == 0
        l_eq = l0.copy()
        cache = np.full_like(l0, np.nan)

        # Cached roots, checked and if need be moved by one Newton step
        lc = self._l_eq_cache
        if lc is not None and lc.shape == l0.shape and np.isfinite(lc).any():
            have = np.isfinite(lc)
            lc = np.where(have, lc, l0)
            zc = z(lc)
            ok = have & (np.abs(zc) <= z_tol)
            redo = have & ~ok
            if redo.any():
                h = 1e-3
                slope = (z(lc + h) - zc)/h          # < 0 at a stable root
                step = redo & (slope < 0)
                l1 = np.where(step, lc - zc/np.where(step, slope, -1.), lc)
                ok1 = step & (np.abs(z(l1)) <= z_tol)
                lc = np.where(ok1, l1, lc)
                ok |= ok1
            cache = np.where(ok, lc, np.nan)
            d = (lc - l0)*sgn
            use = ok & (d >= -tol) & (d <= W)
            l_eq = np.where(use, lc, l_eq)
            found |= use

        if not found.all():
            # Bracket [la, lb] with z(la)*sgn > 0 >= z(lb)*sgn
            done = found.copy()
            new_found = np.zeros_like(found)
            la, lb = l0.copy(), l0.copy()
            for k in range(1, n_search + 1):
                l = l0 + sgn*W*k/n_search
                zl = z(l)
                new = ~done & (zl*sgn <= 0)
                lb = np.where(new, l, lb)
                done |= new
                new_found |= new
                la = np.where(~done, l, la)
            if new_found.any():
                # Brent's method (brent.brentq, vectorised) on the bracketed
                # cells; the others get a linear stand-in with its root at l0.
                # (Re-evaluated ends: a cell at equilibrium, z0 ~ 0, can
                # change sign in the last bit; it takes the end with the
                # smaller |z|.)
                fa, fb = z(la)*sgn, z(lb)*sgn
                ok = new_found & (fa*fb <= 0)
                l_end = np.where(np.abs(fa) <= np.abs(fb), la, lb)
                a_end = np.where(ok, la, l0 - 1.)
                b_end = np.where(ok, lb, l0 + 1.)

                def f(l):
                    return np.where(ok, z(l)*sgn, l - l0)
                root = brentq(f, a_end, b_end, xtol=tol, rtol=0., max_iter=max_iter)
                root = np.where(ok, root, l_end)
                l_eq = np.where(new_found, root, l_eq)
                cache = np.where(new_found, root, cache)
                found |= new_found

        self._l_eq_cache = cache
        return l_eq, found

    def _growth_dt(self, rates):
        """Maximum growth step in each cell.

        A growing amax may change by at most growth_cfl e-folds per step,
        unless it is within a factor eq_factor of equilibrium. There, and for
        any shrinking amax, there is no limit: the step uses the secant index
        to the equilibrium size (_growth_rates), so it relaxes onto it without
        overshooting. (A shrinking amax relaxes onto a fixed equilibrium set
        by the gas and v_frag, quickly; the growth phase is where the growth
        time matters.)
        """
        z0 = rates['z0']
        g = np.abs(np.tanh(0.5*z0))
        dt = self._growth_cfl*rates['t_grow'] / np.maximum(g, 1e-300)
        return np.where(rates['near'] | (z0 < 0), np.inf, dt)

    def growth_timestep(self):
        """Time-step limit for resolving the growth of amax.

        Cells within a factor eq_factor of their equilibrium size do not
        limit the time-step (see _growth_dt). Only dv_max, t_grow and z0 are
        evaluated; the band is judged from the cached equilibrium roots of
        the last growth step (_near_cached) rather than a new search.
        """
        eps = self.dust_frac
        ids = (eps[0] > 0) & (eps[1] > 0)
        if not ids.any():
            return np.inf
        _, s, L = self.size_distribution()
        amax = self._a[1]
        a1 = mean_size(s, 0.5*L, L, self._amin)
        ac = self._f_coll*amax
        dv_max = self.relative_velocity(ac, self._f_dv*ac)
        z0 = self._s_frag*np.log(self._uf/dv_max)
        near = self._near_cached(amax, z0) if self._eq_factor > 1 else np.zeros_like(ids)
        H1 = self.dust_scale_height(self.Stokes(self.Sigma_G, a1))
        rates = dict(z0=z0, t_grow=self._free_growth_time(amax, H1, dv_max), near=near)
        return self._growth_dt(rates)[ids].min()

    def do_grain_growth(self, dt):
        """Apply the grain growth.

        amax is advanced with the analytic solution at fixed t_grow (assuming
        a local power law of dv_max in amax) and the mass exchange between populations
        with the analytic solution at fixed coefficients. Cells are
        sub-cycled so that a growing amax changes by at most growth_cfl
        e-folds per sub-step, unless it is within eq_factor of equilibrium;
        a shrinking amax takes a single step (_growth_dt).

        The exchange between the populations also mixes the ices (see
        chemistry.area_ice): a tracer's split between the populations relaxes
        to the mass split at (C1 + C2) Sigma_1.
        """
        if dt <= 0:
            return
        _, s0, L0 = self.size_distribution()
        ids = (self._eps[0] > 0) & (self._eps[1] > 0)
        if not ids.any():
            return

        # The cached parameters are the working variables of the step; the
        # state is updated from their change at the end
        self._snap = None
        remaining = np.where(ids, dt, 0.)
        mix = np.zeros(self.Ncells)
        for _ in range(self._max_substeps):
            active = remaining > 0
            if not active.any():
                break
            rates = self._growth_rates()
            h = np.minimum(remaining, self._growth_dt(rates))
            h = np.where(active, h, 0)
            eps_tot, s, L = self.size_distribution()

            amax_new = self.grow_amax(self._a[1], rates['t_grow'], rates['z0'],
                                      rates['n'], h)
            amax_new = np.maximum(amax_new, 1.5*self._amin)

            S = self.Sigma*eps_tot
            f1 = large_fraction(s, L)
            C1, C2 = rates['C1'], rates['C2']
            f1_new = self.exchange_fraction(f1, S*C1, S*(C1 + C2), h)
            mix += S*(C1 + C2)*0.5*(f1 + f1_new)*h

            self._eps[0][active] = (eps_tot*(1 - f1_new))[active]
            self._eps[1][active] = (eps_tot*f1_new)[active]
            self._a[1][active] = amax_new[active]
            remaining = remaining - h
        else:
            # Finish with a single step for any cells left
            active = remaining > 0
            if active.any():
                rates = self._growth_rates()
                amax_new = self.grow_amax(self._a[1], rates['t_grow'],
                                          rates['z0'], rates['n'], remaining)
                self._a[1][active] = np.maximum(amax_new, 1.5*self._amin)[active]

        # Change the shape of the state by the change in the scaled moments,
        # keeping M_0
        _, s1, L1 = self._cached_params()
        self._y = self._y + (self._shape_moments(s1, L1) -
                             self._shape_moments(s0, L0))
        self._update_a0()
        self._take_snapshot()

        if self._ices is not None:
            self._ices.mix_area_to_mass(-np.expm1(-mix))

    def update_ices(self, grains):
        """Update the dust fraction (at fixed size distribution) and the
        fragmentation velocity due to a change in bulk ice abundance. The
        ices are kept, for growth to mix (ChemicalAbund.mix_area_to_mass)"""
        self._ices = grains
        self._set_dust_fraction(grains.total_abund)
        eps_new = self._eps_tot

        # Update the ice fraction
        f_ice = 0
        for spec in grains:
            if 'grain' not in spec:
                f_ice += grains[spec]
        f_ice /= (eps_new + 1e-300)

        self._uf = self._frag_velocity(f_ice)

    def remove_dust_with_gas(self, dSigma_G, a_cut=None, efficiency=None):
        """Remove gas from each cell, together with the dust entrained by it.

        In a cell losing a fraction f_g = dSigma_G / Sigma_G of its gas, grains
        of size a lose a fraction f_g * eta(a) of their mass, where
        eta(a) = H(a_cut - a) * efficiency(a). The moments M_-1, M_0 and M_1
        removed are computed from the truncated power law (analytically for
        the step alone, otherwise by quadrature below a_cut), and the power
        law is then refitted to the remaining moments with the same projection
        used after transport.

        args:
            dSigma_G   : gas surface density to remove, per cell
            a_cut      : largest grain size entrained (cm), scalar or per cell.
                         None: no dust is removed.
            efficiency : optional function eta_del(a) in [0, 1]. It receives
                         sizes of shape [Nq, Ncells] and must return an array
                         of the same shape. Default: 1.

        returns:
            dSigma_D : dust surface density removed, per cell
        """
        ks = (-1, 0, 1)
        Sigma = self.Sigma.copy()
        dSigma_G = np.broadcast_to(dSigma_G, Sigma.shape)
        Sigma_G = self.Sigma_G
        has_gas = Sigma_G > 0
        f_g = np.zeros_like(Sigma)
        f_g[has_gas] = np.minimum(dSigma_G[has_gas]/Sigma_G[has_gas], 1)

        eps_tot, s, L = self.size_distribution()
        amin = self._amin
        M = self.moments(ks)

        # Moments removed (per unit Sigma) and cells where the shape is kept
        dM = np.zeros_like(M)
        same_shape = np.ones(Sigma.shape, dtype=bool)
        if a_cut is not None:
            a_cut = np.broadcast_to(a_cut, Sigma.shape)
            Lc = np.minimum(np.log(np.maximum(a_cut, amin)/amin), L)
            part = (Lc > 0) & (f_g > 0) & (eps_tot > 0)
            Lc = np.where(part, Lc, L)
            if efficiency is None:
                frac = np.array([np.exp(_lnE(s+k, Lc) - _lnE(s+k, L))
                                 for k in ks])
                dM = np.where(part, f_g*M*frac, 0)
                same_shape = ~part | (Lc >= L)
            else:
                a, w = self._quad(s, Lc, amin)
                eta = np.clip(efficiency(a), 0, 1)
                F_c = np.exp(_lnE(s, Lc) - _lnE(s, L))
                dM = np.array([f_g*eps_tot*F_c*(w*eta*a**k).sum(0)
                               for k in ks])
                dM = np.where(part, dM, 0)
                same_shape = ~part

        dSigma_D = Sigma*dM[1]
        self._Sigma -= dSigma_G
        self._Sigma -= dSigma_D

        # Remaining moments per unit (new) Sigma
        not_empty = self.Sigma > 0
        Q = (M - dM)*Sigma/np.where(not_empty, self.Sigma, 1)

        # Where the shape is unchanged, just rescale the dust fraction
        scale = np.where(not_empty & (eps_tot > 0),
                         Q[1]/np.where(eps_tot > 0, eps_tot, 1), 1)
        fit = not_empty & ~same_shape
        if fit.any():
            # Elsewhere set the state from the remaining moments
            self.n_removal_clamped += self.set_moments(
                np.where(fit, Q, M*scale))
        else:
            self._set_dust_fraction(eps_tot*scale)

        return dSigma_D

    def initialize_dust_density(self, dust_frac):
        """Set the initial dust density"""
        self._set_dust_fraction(dust_frac)

    def _set_dust_fraction(self, eps_new):
        """Set the total dust fraction at fixed shape"""
        _, s, L = self.size_distribution()
        f1 = large_fraction(s, L)
        self._eps_tot = np.array(np.broadcast_to(eps_new, (self.Ncells,)),
                                 dtype='f8')
        self._eps[0] = self._eps_tot*(1 - f1)
        self._eps[1] = self._eps_tot*f1
        self._take_snapshot()

    def update(self, dt):
        """Do the standard disc update, and apply grain growth"""
        super(DustGrowthTriMom, self).update(dt)
        self.do_grain_growth(dt)

################################################################################
# Radial drift
################################################################################
class MomentDrift(object):
    """Moment-consistent radial drift for the DustGrowthTriMom model.

    Transports three linear functionals of the size distribution, each with
    its own weighted velocity, in the single-fluid approximation with the short
    friction time limit (as in dust.SingleFluidDrift). The dust-gas relative
    velocity for each size follows Tanaka+ (2005), with feedback, evaluated on
    the quadrature nodes of the reconstructed power law.

    Schemes:
        'moments' : transport the moments int a^k eps(a) da for k = -1, 0, 1
                    (area, mass and size), which gives exact area and mass
                    fluxes in the small Stokes number limit.
        'tripod'  : TriPoD's scheme: eps0, eps1 and amax*eps1 transported with
                    the velocities of f_drift*a0, f_drift*a1, and optional
                    shrinkage of amax in gaps (Pfeil et al. 2024, Sec. 4.3).
        'trimean' : TriPoD's variables with the mean size in place of amax:
                    eps0, eps1 and M_1 = eps_tot*<a> transported with the
                    velocities of f_drift times the mass-averaged size of
                    each population, and of M_2/M_1 for M_1 (each flux is
                    exact for St << 1). The power law is then refitted to
                    eps0, eps1 and <a> (in closed form, fit_mean_size); no
                    mass moves between the populations and there is no
                    shrinkage. The refit needs no bounds on q: use a disc
                    with q_bounds=(-np.inf, np.inf), or the disc clips s.
        'trimax'  : TriPoD's variables, eps0 and eps1 transported with the
                    velocities of f_drift times the mass-averaged size of
                    each population, and amax advected (not conserved) with
                    the velocity of f_drift*amax: d amax/dt + v d amax/dR = 0,
                    upwinded, so that amax stays within its upwind values.
                    No shrinkage; the closure is TriPoD's (s from eps1/eps0).
                    With the gas and in diffusion amax is a tracer of eps1
                    (eps1*amax, as 'tripod'). Use a disc with no q bounds, as
                    for 'trimean'.

    Advection with the gas is done by the driver's gas step: pre_gas_transport
    hands it the three functionals (as passive tracers) and
    post_gas_transport sets the size distribution from them afterwards. If
    the disc is evolved without a drift module, amax is not advected.
    Diffusion is a separate step too, done by the driver through diffuse().

    args:
        settling  : Use midplane densities for feedback, default=False
        van_leer  : Use 2nd-order Van-Leer reconstruction, default=False
        scheme    : 'moments', 'tripod', 'trimean' or 'trimax',
                    default='moments'
        f_drift   : Factor multiplying grain size in drift velocities,
                    default=1 (0.8 is the TriPoD calibration)
        shrink    : TriPoD shrinkage (only for scheme='tripod'), default=True
        q_min     : 'trimean' only: floor on q in the refit. Steeper cells
                    take q = q_min with amax set to keep Sigma_d and <a>,
                    moving mass between the populations (fit_mean_size).
                    Default None (no floor)
        f_crit    : TriPoD critical large-grain fraction, default=0.425
        a_lim     : TriPoD minimum size for shrinkage, default=1e-4 cm
        area_tracers: 'moments' only: indices of the dust-tracer rows carried
                    by the grain surface area (e.g. the area-distributed ice
                    of chemistry.area_ice), which drift with the area (M_-1)
                    velocity and diffuse with its Stokes number. The other
                    rows follow the dust mass. Default None (all by mass)
    """
    # Orders of the moments of the state, which the 'moments' scheme transports
    _ks = (-1, 0, 1)

    def __init__(self, settling=False, van_leer=False,
                 scheme='moments', f_drift=1.0, shrink=True, f_crit=0.425,
                 a_lim=1e-4, q_min=None, area_tracers=None):
        if scheme not in ('moments', 'tripod', 'trimean', 'trimax'):
            raise ValueError("Unknown scheme: {}".format(scheme))
        if area_tracers is not None and scheme != 'moments':
            raise ValueError("area_tracers needs scheme='moments'")

        self._settling = settling
        self._van_leer = van_leer
        self._scheme = scheme
        self._f_drift = f_drift
        self._shrink = shrink
        self._f_crit = f_crit
        self._a_lim = a_lim
        self._s_min = -np.inf if q_min is None else q_min + 4
        self._vl = None
        self._area_tracers = (None if area_tracers is None
                              else np.atleast_1d(area_tracers))

        # Diagnostics
        self.n_clamped = 0

        # Transport velocities from max_timestep, reused by __call__ while
        # the state they depend on is unchanged
        self._vcache = None

    def ASCII_header(self):
        """Radial drift header"""
        return ('# {} settling: {} scheme: {} moments: {} f_drift: {}'
                ''.format(self.__class__.__name__, self._settling, self._scheme,
                          self._ks, self._f_drift))

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        head = { "settling"  : "{}".format(self._settling),
                 "van-leer"  : "{}".format(self._van_leer),
                 "scheme"    : "{}".format(self._scheme),
                 "moments"   : "{}".format(self._ks),
                 "f_drift"   : "{}".format(self._f_drift),
                 }
        if self._scheme == 'tripod':
            head["shrink"] = "{}".format(self._shrink)
            head["f_crit"] = "{}".format(self._f_crit)
            head["a_lim"] = "{}".format(self._a_lim)
        return self.__class__.__name__ , head

    ############################################################################
    # Transported quantities
    ############################################################################
    def _functionals(self, disc):
        """Transported quantities (per unit total surface density) and the
        node weights defining the corresponding velocity averages.

        returns:
            Q      : transported quantities, shape [3, Ncells]
            phi    : node weights for velocity averages, [3, Nq, Ncells]
            a, w   : quadrature nodes and mass weights
        """
        Q = self._transported(disc)
        a, w = disc.quadrature()

        if self._scheme == 'moments':
            phi = np.array([w*a**k for k in self._ks])
        else:
            phi = None

        return Q, phi, a, w

    def _transported(self, disc):
        """Transported quantities per unit total surface density, [3, Ncells]"""
        if self._scheme == 'moments':
            return disc.moments(self._ks)
        elif self._scheme == 'trimean':
            eps_tot, s, L = disc.size_distribution()
            return np.array([disc.dust_frac[0], disc.dust_frac[1],
                             eps_tot*mean_size(s, 0, L, disc.amin)])
        else:
            return np.array([disc.dust_frac[0], disc.dust_frac[1],
                             disc.dust_frac[1]*disc.amax])

    def _reconstruct(self, disc, Q):
        """Set the disc's size distribution from the transported quantities"""
        amin = disc.amin

        if self._scheme in ('tripod', 'trimax'):
            eps0 = np.maximum(Q[0], 0)
            eps1 = np.maximum(Q[1], 0)
            amax = np.where(eps1 > 0, Q[2]/np.where(eps1 > 0, eps1, 1),
                            disc.amax)
            amax = np.maximum(amax, amin*np.exp(disc._L_min))
            disc.set_populations(eps0, eps1, amax)
            if self._scheme == 'tripod' and self._shrink:
                self._tripod_shrink(disc)
            return

        if self._scheme == 'trimean':
            # Defined for eps0, eps1 > 0 and <a> > amin; other cells (empty
            # or degenerate) keep their amax
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
            return

        # The moments are the disc's state
        self.n_clamped += disc.set_moments(Q)

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
    def _edge_velocities(self, disc, a, w, v_visc=None):
        """Dust-gas relative velocity of each quadrature node at the cell
        edges, evaluated using the nodes of the cells to the left and right.

        returns:
            dV_L, dV_R : node velocities at the edges, from the left / right
                         cell's nodes, [Nq, Ncells-1]
            epsDeltaV  : v_COM - v_gas at the edges
            DeltaVphi  : azimuthal velocity of the nodes (left cell)
        """
        Sigma  = disc.Sigma
        SigmaG = disc.Sigma_G
        eps_tot = disc.integ_dust_frac if disc.feedback else 0*Sigma

        Sig_av  = 0.5*(Sigma[1:] + Sigma[:-1]) + 1e-300
        SigG_av = 0.5*(SigmaG[1:] + SigmaG[:-1])

        aL, aR = a[:, :-1], a[:, 1:]
        St_L = disc.Stokes(SigG_av, aL + 1e-300)
        St_R = disc.Stokes(SigG_av, aR + 1e-300)

        # Feedback: dust fractions of each node at the edges
        if disc.feedback:
            if self._settling:
                St_c = disc.Stokes(SigmaG, a)
                rhoD = (eps_tot*Sigma)*w / \
                    (np.sqrt(2*np.pi)*disc.dust_scale_height(St_c)*AU)
                rhoG = disc.midplane_gas_density
                rho_av = 0.5*(rhoG[1:] + rhoG[:-1]) + \
                    0.25*(rhoD[:, 1:] + rhoD[:, :-1]).sum(0) + 1e-300
                eL = rhoD[:, :-1] / rho_av
                eR = rhoD[:, 1:] / rho_av
                eps_g = np.maximum(0.5*(rhoG[1:] + rhoG[:-1])/rho_av, 1e-300)
            else:
                SigD = eps_tot*Sigma*w
                eL = SigD[:, :-1] / Sig_av
                eR = SigD[:, 1:] / Sig_av
                eps_g = np.maximum(SigG_av / Sig_av, 1e-300)

            f1L, f2L = _drift_factors(St_L)
            f1R, f2R = _drift_factors(St_R)
            la0 = 0.5*((eL*(1 - f2L)).sum(0) + (eR*(1 - f2R)).sum(0))/eps_g
            la1 = 0.5*((eL*f1L).sum(0) + (eR*f1R).sum(0))/eps_g
        else:
            eps_g = 1.
            la0, la1 = 0., 0.

        u_gas, v_gas = self._edge_gas_velocities(disc, la0, la1, eps_g, v_visc)

        def deltaV(St):
            f1, f2 = _drift_factors(St)
            return 2*v_gas*f1 - u_gas*f2

        dV_L = deltaV(St_L)
        dV_R = deltaV(St_R)

        if disc.feedback:
            epsDeltaV = 0.5*((eL*dV_L).sum(0) + (eR*dV_R).sum(0))
        else:
            epsDeltaV = 0*SigG_av

        f1L, f2L = _drift_factors(St_L)
        DeltaVphi = -0.5*u_gas*f1L + v_gas*(1 - f2L)

        return dV_L, dV_R, epsDeltaV, DeltaVphi

    def _edge_gas_velocities(self, disc, la0, la1, eps_g, v_visc=None):
        """Radial and azimuthal gas velocities at the edges, with feedback
        (Tanaka et al. 2005) and the viscous velocity (Dipierro et al. 2018)"""
        Om_k = disc.Omega_k
        rho = disc.midplane_gas_density
        dPdr = np.diff(disc.P) / disc.grid.dRc
        eta = - dPdr / (0.5*(rho[1:] + rho[:-1] + 1e-300) *
                        0.5*(Om_k[1:] + Om_k[:-1]))

        D_1 = 1 / ((1 + la0)**2 + la1**2)
        u_gas =            la1  * eta * D_1
        v_gas = - 0.5*(1 + la0) * eta * D_1

        if v_visc is not None:
            u_gas = u_gas + (1 + la0) * v_visc * D_1 / eps_g
            v_gas = v_gas + 0.5 * la1 * v_visc * D_1 / eps_g
        return u_gas, v_gas

    def _moment_velocities(self, disc, a, w, v_visc=None):
        """Velocities of the transported moments at the edges, relative to the
        centre of mass, for the 'moments' scheme without settling.

        A grain moves relative to the gas at
            dv = 2 v_gas St/(1+St^2) - u_gas St^2/(1+St^2),
        so the velocity of M_k needs <St^p/(1+St^2)>_k, p = 1, 2. Since St is
        linear in a, these are the exact <St^p>_k times G_{k+p}, the mean of
        1/(1+St^2) with weight a^(k+p) eps(a), which alone is evaluated by
        quadrature. The feedback coefficients and v_COM - v_gas use the same
        averages as the mass moment's velocity, so that the centre of mass
        moves with the viscous velocity (Booth et al. 2017).

        returns:
            V_L, V_R  : velocities from the left / right cells, [3, Ncells-1]
            epsDeltaV : v_COM - v_gas at the edges
        """
        Sigma, SigmaG = disc.Sigma, disc.Sigma_G
        Sig_av  = 0.5*(Sigma[1:] + Sigma[:-1]) + 1e-300
        SigG_av = 0.5*(SigmaG[1:] + SigmaG[:-1])
        K = disc.Stokes(SigG_av, self._f_drift)        # St = K a at the edges
        AL, AR = self._edge_blocks(disc, a, w, K)

        if disc.feedback:
            SigD = disc.integ_dust_frac*Sigma
            eL, eR = SigD[:-1]/Sig_av, SigD[1:]/Sig_av
            eps_g = np.maximum(SigG_av/Sig_av, 1e-300)
            # <1/(1+St^2)>_0 as 1 - <St^2/(1+St^2)>_0, from the same averages
            # as the mass velocity
            la0 = 0.5*(eL*(1 - AL[0, 2]) + eR*(1 - AR[0, 2]))/eps_g
            la1 = 0.5*(eL*AL[0, 1] + eR*AR[0, 1])/eps_g
        else:
            eps_g, la0, la1 = 1., 0., 0.

        u_gas, v_gas = self._edge_gas_velocities(disc, la0, la1, eps_g, v_visc)

        def dV(A, k):
            return 2*v_gas*A[k, 1] - u_gas*A[k, 2]
        if disc.feedback:
            epsDeltaV = 0.5*(eL*dV(AL, 0) + eR*dV(AR, 0))
        else:
            epsDeltaV = 0*SigG_av
        V_L = np.array([dV(AL, k) for k in self._ks])
        V_R = np.array([dV(AR, k) for k in self._ks])
        return V_L - epsDeltaV, V_R - epsDeltaV, epsDeltaV

    def _edge_blocks(self, disc, a, w, K):
        """<St^p/(1+St^2)>_k for the left and right cell of each edge"""
        _, s, L = disc.size_distribution()
        lnE = {j: _lnE(s + j, L) for j in range(-1, 4)}
        def side(sl):
            return self._blocks(disc.amin, {j: e[sl] for j, e in lnE.items()},
                                a[:, sl], w[:, sl], K)
        return side(slice(None, -1)), side(slice(1, None))

    def _blocks(self, amin, lnE, a, w, K):
        """{(k, p): <St^p/(1+St^2)>_k} for p = 1, 2 and k = -1, 0, 1, with
        St = K a.

        lnE  : {j: ln E(q+4+j, L)}, which give the exact <St^p>_k
        a, w : quadrature nodes and mass weights, used only for
               G_j = <1/(1+St^2)>_j, normalised by the same nodes so that
               errors in integrating the weight cancel
        """
        g = 1/(1 + (K*a)**2)
        G = {}
        for j in range(0, 4):
            wj = w*a**j
            G[j] = (wj*g).sum(0)/(wj.sum(0) + 1e-300)
        return {(k, p): (K*amin)**p*np.exp(lnE[k+p] - lnE[k])*G[k+p]
                for k in self._ks for p in (1, 2)}

    def _transport_velocities(self, disc, phi, a, w, v_visc=None):
        """Weighted velocities of the transported quantities at the edges.

        returns:
            V_L, V_R : velocities from the left / right cells, [3, Ncells-1]
            epsDeltaV
        """
        # The feedback (lambda), node velocities and epsDeltaV are all computed
        # from the transported species, so that the centre of mass moves with
        # the viscous velocity (Booth et al. 2017, eqs. 5-9).
        if self._scheme in ('tripod', 'trimean', 'trimax'):
            # Species: the two populations at their representative sizes
            # (and, for trimean, M_1 at M_2/M_1, for trimax amax at amax,
            # with no weight in the feedback)
            a_rep = self._representative_sizes(disc)
            w_rep = disc.dust_frac / np.maximum(disc.dust_frac.sum(0), 1e-300)
            if self._scheme in ('trimean', 'trimax'):
                w_rep = np.concatenate([w_rep, 0*w_rep[:1]])
            dV_L, dV_R, epsDeltaV, _ = self._edge_velocities(
                disc, a_rep[:len(w_rep)], w_rep, v_visc)
            dV_L, dV_R = dV_L[self._species], dV_R[self._species]
            return dV_L - epsDeltaV, dV_R - epsDeltaV, epsDeltaV

        if not self._settling:
            return self._moment_velocities(disc, a, w, v_visc)

        # With settling the feedback depends on each size's scale height:
        # average the node velocities
        dV_L, dV_R, epsDeltaV, _ = self._edge_velocities(disc, self._f_drift*a,
                                                         w, v_visc)

        pL, pR = phi[..., :-1], phi[..., 1:]
        V_L = (pL*dV_L).sum(1) / ((pL).sum(1) + 1e-300)
        V_R = (pR*dV_R).sum(1) / ((pR).sum(1) + 1e-300)
        return V_L - epsDeltaV, V_R - epsDeltaV, epsDeltaV

    def _state_key(self, disc, v_visc):
        """Copies of the arrays the transport velocities depend on"""
        disc._sync()
        key = [disc._eps_tot, disc._y, disc.Sigma, disc.cs]
        if self._settling:
            key.append(np.atleast_1d(disc.alpha))
        if v_visc is not None:
            key.append(v_visc)
        return [np.array(k, copy=True) for k in key]

    def _velocities(self, disc, v_visc=None):
        """Transported quantities and their velocities,
        (Q, phi, a, w, V_L, V_R, epsDeltaV), reusing those from the last call
        if the disc state is unchanged (max_timestep followed by __call__)"""
        key = self._state_key(disc, v_visc)
        c = self._vcache
        if c is not None and c[0] is disc and len(c[1]) == len(key) and \
                all(np.array_equal(x, y) for x, y in zip(c[1], key)):
            return c[2]
        Q, phi, a, w = self._functionals(disc)
        V_L, V_R, epsDeltaV = self._transport_velocities(disc, phi, a, w,
                                                         v_visc)
        out = (Q, phi, a, w, V_L, V_R, epsDeltaV)
        self._vcache = (disc, key, out)
        return out

    @property
    def _species(self):
        """Representative size used for each transported quantity"""
        return [0, 1, 2] if self._scheme in ('trimean', 'trimax') else [0, 1, 1]

    def _representative_sizes(self, disc):
        """f_drift times the mass-averaged sizes of the two populations
        (and, for trimean, the M_1-weighted size M_2/M_1, for trimax amax)"""
        _, s, L = disc.size_distribution()
        amin = disc.amin
        a = [mean_size(s, 0, 0.5*L, amin), mean_size(s, 0.5*L, L, amin)]
        if self._scheme == 'trimean':
            a.append(mean_size(s+1, 0, L, amin))
        elif self._scheme == 'trimax':
            a.append(disc.amax)
        return self._f_drift*np.array(a)

    def _mean_stokes(self, disc, phi, a):
        """Weighted mean Stokes number of each transported quantity (for
        trimax, eps1*amax diffuses as eps1)"""
        eps_tot, s, L = disc.size_distribution()
        amin = disc.amin
        if phi is None:
            species = [0, 1, 1] if self._scheme == 'trimax' else self._species
            a_rep = self._representative_sizes(disc)[species]
            return disc.Stokes(disc.Sigma_G, a_rep)
        # St is linear in a, so the mean of M_k is St(M_{k+1}/M_k)
        a_k = np.array([moment_ratio(k+1, s, L, amin)/moment_ratio(k, s, L, amin)
                        for k in self._ks])
        return disc.Stokes(disc.Sigma_G, a_k)

    def diffusion_stokes(self, disc):
        """Stokes numbers with which diffuse() diffuses the transported
        quantities (pre_gas_transport) and the dust tracers.

        returns:
            St_Q [3, Ncells], St of the dust mass [Ncells] (for the tripod
            schemes, which split the tracers by population, the mass-weighted
            mean of the populations')
        """
        Q, phi, a, w = self._functionals(disc)
        St_Q = self._mean_stokes(disc, phi, a)
        if self._scheme != 'moments':
            return St_Q, (self._population_fractions(disc)*St_Q[:2]).sum(0)
        if not self._settling:
            return St_Q, St_Q[self._ks.index(0)]
        return St_Q, (w*disc.Stokes(disc.Sigma_G, a)).sum(0)

    @staticmethod
    def _population_fractions(disc):
        eps = disc.dust_frac
        return eps/(eps.sum(0) + np.finfo(eps.dtype).tiny)

    def diffuse(self, dt, disc, diffusion, Q, gas_tracers=None, dust_tracers=None):
        """Diffuse the transported quantities Q (from pre_gas_transport), the
        dust tracers and the gas tracers over dt, in one step of the diffusion
        object (TracerDiffusion, or rkl.RKLDiffusion), with the Schmidt
        numbers of dust_schmidt: each quantity at its mean Stokes number, the
        tracers as the dust mass (moments) or split by population (tripod
        schemes), the gas tracers with the gas's"""
        St_Q, St_m = self.diffusion_stokes(disc)
        Sc_g = diffusion.gas_schmidt(disc)
        items = [(Q, dust_schmidt(Sc_g, St_Q))]
        split = dust_tracers is not None and self._scheme != 'moments'
        if split:
            f = self._population_fractions(disc)
            t_p = dust_tracers[None]*f[:, None]
            items.append((t_p, dust_schmidt(Sc_g, St_Q[:2])[:, None]))
        elif dust_tracers is not None:
            Sc = dust_schmidt(Sc_g, St_m)
            if self._area_tracers is not None:
                Sc_A = dust_schmidt(Sc_g, St_Q[self._ks.index(-1)])
                Sc = np.where(self._area_rows(dust_tracers)[:, None], Sc_A, Sc)
            items.append((dust_tracers, Sc))
        if gas_tracers is not None:
            items.append((gas_tracers, Sc_g))
        diffusion.diffuse(dt, disc, items)
        if split:
            dust_tracers[:] = t_p.sum(0)

    def _area_rows(self, dust_tracers):
        """Mask of the dust-tracer rows carried by the grain area"""
        mask = np.zeros(len(dust_tracers), dtype=bool)
        mask[self._area_tracers] = True
        return mask

    def _upwind_flux(self, disc, Q, V_L, V_R, dt=0.):
        """Upwind fluxes at all cell edges (including the boundaries).

        V_L, V_R are the velocities at the interior edges from the cells to the
        left and right. dt is used for the half-step predictor in the van Leer
        reconstruction.
        """
        return self._flux(*self._face_states(disc, Q, V_L, V_R, dt), V_L, V_R)

    def _face_states(self, disc, Q, V_L, V_R, dt=0.):
        """Values of Q at all cell edges from the cells to the left and right
        (as _upwind_flux)"""
        if self._van_leer:
            if self._vl is None or self._vl_grid is not disc.grid:
                self._vl = VanLeer(disc.grid.Ree, 1)
                self._vl_grid = disc.grid
            pad = lambda X, n: np.concatenate([X[..., :1]]*n + [X] +
                                              [X[..., -1:]]*n, axis=-1)
            # Velocity at the upper / lower face of each cell, seen from it
            v_up  = pad(np.concatenate([V_L, V_R[..., -1:]], axis=-1), 1)
            v_low = pad(np.concatenate([V_L[..., :1], V_R], axis=-1), 1)
            Q_from_L, Q_from_R = self._vl.face_values(pad(Q, 2), v_up, v_low,
                                                       dt)
        else:
            Q_from_L = np.concatenate([Q[..., :1], Q], axis=-1)
            Q_from_R = np.concatenate([Q, Q[..., -1:]], axis=-1)
        return Q_from_L, Q_from_R

    @staticmethod
    def _flux(Q_from_L, Q_from_R, V_L, V_R):
        """Upwind fluxes at all cell edges from the edge values"""
        flux = np.empty(Q_from_L.shape, dtype='f8')
        flux[..., 1:-1] = Q_from_L[..., 1:-1]*np.maximum(V_L, 0) + \
            Q_from_R[..., 1:-1]*np.minimum(V_R, 0)
        flux[...,  0] = Q_from_R[...,  0]*np.minimum(V_L[...,  0], 0)
        flux[..., -1] = Q_from_L[..., -1]*np.maximum(V_R[..., -1], 0)
        return flux

    def _advect_trimax(self, disc, Q, V_L, V_R, dt):
        """The trimax update: eps0 and eps1 conservatively, and amax along
        its characteristics, d amax/dt + v(amax) d amax/dR = 0. amax and a
        pseudo-density rho, the dust mass per unit ln(a) at amax
        (density_at_amax, from the power law of the cell or edge values of
        eps0, eps1 and amax), are carried together with v(amax), and amax is
        the ratio: the upwind fluxes make it a rho-weighted average of the
        cell's and upwind values, so it stays within them and is set by the
        grains at amax. (Carrying eps1*amax with v(amax) instead would add the
        spurious source -amax div(Sigma eps1 (v(amax) - v1)) and could
        overshoot.)

        returns:
            Q_new : (eps0, eps1, eps1*amax) after the step
        """
        Sigma, amax = disc.Sigma, disc.amax
        L_min = disc._L_min

        def rho(SigD, A):
            # Floors guard exactly empty populations (outside the domain)
            floor = np.maximum(1e-150*(SigD[0] + SigD[1]), 1e-300)
            L = np.log(np.maximum(A/disc.amin, np.exp(L_min)))
            return density_at_amax(np.maximum(SigD[0], floor),
                                   np.maximum(SigD[1], floor), L)

        EL, ER = self._face_states(disc, Sigma*Q[:2], V_L[:2], V_R[:2], dt)
        eps = Q[:2] + dt*self._divergence(disc, self._flux(EL, ER, V_L[:2], V_R[:2]))

        VL, VR = V_L[2], V_R[2]
        AL, AR = self._face_states(disc, amax, VL, VR, dt)
        rL, rR = rho(EL, AL), rho(ER, AR)
        r0 = rho(Sigma*Q[:2], amax)
        r1 = r0 + Sigma*dt*self._divergence(disc, self._flux(rL, rR, VL, VR))
        rA = r0*amax + Sigma*dt*self._divergence(disc, self._flux(rL*AL, rR*AR, VL, VR))
        ok = r1 > 0
        amax = np.where(ok, rA/np.where(ok, r1, 1), amax)
        return np.array([eps[0], eps[1], np.maximum(eps[1], 0)*amax])

    def _divergence(self, disc, flux):
        grid = disc.grid
        return - np.diff(flux*grid.Re) / ((disc.Sigma + 1e-300)*0.5*grid.dRe2)

    def max_timestep(self, disc, v_visc=None):
        """Courant limited time-step"""
        Cou = 0.5
        Q, phi, a, w, V_L, V_R, _ = self._velocities(disc, v_visc)

        dVtot = np.zeros_like(Q)
        dVtot[..., :-1] += np.maximum(V_L, 0)
        dVtot[..., 1:] += np.maximum(-V_R, 0)
        dVtot[...,  0] += np.maximum(-V_L[...,  0], 0)
        dVtot[..., -1] += np.maximum(V_R[..., -1], 0)

        # Prevent empty cells limiting the time-step. Note integ_dust_frac is
        # zero without feedback, so use dust_frac directly.
        dVtot[:, disc.dust_frac.sum(0) < 1e-20] *= 1e-3
        dt = Cou * (disc.grid.dRe / (dVtot + 1e-300)).min()

        # Resolve the growth of amax relative to the drift
        return min(dt, disc.growth_timestep())

    ############################################################################
    # Update
    ############################################################################
    def pre_gas_transport(self, disc):
        """Quantities advected with the gas by the driver's gas step: the
        transported functionals Q (per unit Sigma). All sizes move with the
        gas in that step, so each functional is a passive tracer."""
        return self._transported(disc)

    def post_gas_transport(self, disc, Q):
        """Set the size distribution from the functionals after the gas (and
        diffusion) steps, with the same projection as after drift"""
        self._reconstruct(disc, Q)

    def __call__(self, dt, disc, gas_tracers=None, dust_tracers=None, v_visc=None):
        """Apply the update for radial drift over time-step dt"""
        Q, phi, a, w, V_L, V_R, epsDeltaV = self._velocities(disc, v_visc)
        self._vcache = None      # the state changes below

        Sigma = disc.Sigma
        zero = np.zeros(disc.Ncells - 1)

        # Gas tracers move with the gas (relative to the centre of mass)
        if gas_tracers is not None:
            flux = self._upwind_flux(disc, Sigma*gas_tracers, zero - epsDeltaV,
                                     zero - epsDeltaV, dt)
            gas_tracers[:] += dt*self._divergence(disc, flux)

        # Dust tracers are assumed to have the same mass distribution as the
        # dust. They move with the mass-weighted velocity (area_tracers: the
        # area-weighted one).
        if dust_tracers is not None:
            if self._scheme == 'moments':
                if not self._settling:
                    # Exactly the velocity of the dust mass (area)
                    k0, k1 = self._ks.index(0), self._ks.index(-1)
                    U_L, U_R = V_L[k0], V_R[k0]
                    UA_L, UA_R = V_L[k1], V_R[k1]
                else:
                    dV_L, dV_R, _, _ = self._edge_velocities(
                        disc, self._f_drift*a, w, v_visc)
                    U_L = (w[:, :-1]*dV_L).sum(0) - epsDeltaV
                    U_R = (w[:, 1:]*dV_R).sum(0) - epsDeltaV
                    wA = w/a
                    wA = wA/(wA.sum(0) + 1e-300)
                    UA_L = (wA[:, :-1]*dV_L).sum(0) - epsDeltaV
                    UA_R = (wA[:, 1:]*dV_R).sum(0) - epsDeltaV
                if self._area_tracers is not None:
                    area = self._area_rows(dust_tracers)[:, None]
                    U_L = np.where(area, UA_L, U_L)
                    U_R = np.where(area, UA_R, U_R)
                flux = self._upwind_flux(disc, Sigma*dust_tracers, U_L, U_R, dt)
                d_tr = self._divergence(disc, flux)
            else:
                f = self._population_fractions(disc)
                d_tr = 0
                for p in range(2):
                    flux = self._upwind_flux(disc, Sigma*dust_tracers*f[p], V_L[p], V_R[p], dt)
                    d_tr = d_tr + self._divergence(disc, flux)
            dust_tracers[:] += dt*d_tr

        # Transport the functionals
        if self._scheme == 'trimax':
            Q_new = self._advect_trimax(disc, Q, V_L, V_R, dt)
        else:
            flux = self._upwind_flux(disc, Sigma*Q, V_L, V_R, dt)
            Q_new = Q + dt*self._divergence(disc, flux)

        self._reconstruct(disc, Q_new)

        # Advection with the gas is done by the driver's gas step, which moves
        # the functionals returned by pre_gas_transport.

    def radial_drift_velocity(self, disc, v_visc=None, ret_vphi=False):
        """Mass-weighted radial drift velocity of each population, and
        optionally the azimuthal velocity"""
        a, w = disc.quadrature()
        pop = disc._quad.population[:,None]
        dV_L, _, epsDeltaV, DeltaVphi = self._edge_velocities(disc, a, w,
                                                              v_visc)
        wL = w[:, :-1]
        V = np.array([(wL*(pop == p)*dV_L).sum(0) /
                      ((wL*(pop == p)).sum(0) + 1e-300) for p in range(2)])
        if ret_vphi:
            Vphi = np.array([(wL*(pop == p)*DeltaVphi).sum(0) /
                             ((wL*(pop == p)).sum(0) + 1e-300)
                             for p in range(2)])
            return V - epsDeltaV, Vphi
        return V - epsDeltaV


# Aliases for drop-in replacement of dust.py
DustGrowthTwoPop = DustGrowthTriMom
SingleFluidDrift = MomentDrift
