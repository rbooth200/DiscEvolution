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
# SingleFluidDrift are provided as aliases with compatible interfaces. TriPoD's
# own transport scheme, for comparison, is in dust_tripod.py.
################################################################################
from __future__ import print_function
import numpy as np
from scipy.special import exprel, expit
from .constants import *
from .dust import (DustyDisc, FixedSizeDust, _cell_faces, _divergence,
                   _gas_tracer_flux, _gas_timestep)
from .reconstruction import VanLeer

__all__ = ["DustyDisc", "FixedSizeDust", "DustGrowthTriMom", "MomentDrift",
           "DustGrowthTwoPop", "SingleFluidDrift"]

################################################################################
# Truncated power-law size distribution
################################################################################
# The mass distribution is eps(a) ∝ a^(q+3) on [amin, amax]. We use
#   s = q + 4,  L = ln(amax/amin),  l = ln(a/amin),
# so that eps(a) da ∝ exp(s l) dl. Integrals of the form
#   E(sig, L) = int_0^L exp(sig l) dl = (x^sig - 1)/sig,   x = amax/amin,
# are evaluated via exprel(y) = (e^y - 1)/y, which removes the special cases
# (q = -3, -4, -5) of Pfeil et al. (2024) eqs. 24-26.

def _ln_exprel(x):
    """ln((e^x - 1)/x), without overflow for large x. Uses numpy's expm1,
    which is several times faster than scipy's exprel"""
    x = np.asarray(x, dtype='f8')
    big = x > 700
    zero = x == 0
    special = big | zero
    xs = np.where(special, 1., x)
    r = np.log(np.expm1(xs)/xs)
    if special.any():
        r = np.where(big, x - np.log(np.where(big, x, 1.)), np.where(zero, 0., r))
    return r

def _lnE(sig, L):
    """ln E(sig, L), finite for all sig (including sig = 0)"""
    return np.log(L) + _ln_exprel(sig*L)

def _lnE_newton(s, L):
    """For x = (s+k) L, k = -1, 0, 1: ln exprel(x) = ln E(s+k, L) - ln L (the
    shape of the power law needs only differences of these, in which ln L
    cancels), and the derivatives of ln E(s+k, L) with respect to s and ln L,
    as rows k = -1, 0, 1:
        d ln E / d ln L = x e^x/(e^x - 1),  d ln E / d s = L (that - 1)/x,
    the latter L (1/2 + x/12) for small x"""
    k = np.array([-1., 0., 1.]).reshape((3,) + (1,)*np.ndim(s))
    x = (s + k)*L
    g = _ln_exprel(x)
    xs = x + 1e-300                             # x is 0 or far larger
    em = np.expm1(np.minimum(xs, 700.))         # the derivatives' limits hold above
    d = xs*(em + 1)/em
    m = np.where(np.abs(x) < 1e-3, L*(0.5 + x/12.), L*(d - 1)/xs)
    return g, m, d

def _shape_moments(s, L):
    """y = [ln(M_-1/M_0) + ln(amin), ln(M_1/M_0) - ln(amin)] of the power law"""
    k = np.array([-1., 0., 1.]).reshape((3,) + (1,)*np.ndim(s))
    g = _ln_exprel((s + k)*L)
    return np.array([g[0] - g[1], g[2] - g[1]])

def _mean_l(sig, L):
    """Mean of l on [0, L] with weight exp(sig l), i.e. d lnE / d sig"""
    x = sig*L
    small = np.abs(x) < 1e-3
    xs = np.where(small, 1., x)
    g = np.where(small, 0.5 + x/12. - x*x*x/720.,
                 1/(-np.expm1(-xs)) - 1/xs)
    return L*g

def moment_ratio(k, s, L, amin):
    """M_k / M_0 of the (mass-weighted) truncated power law"""
    g = _ln_exprel(np.array([s + k, s + 0*k])*L)
    return amin**k * np.exp(g[0] - g[1])

def large_fraction(s, L):
    """eps1 / eps_tot, with the populations split at a_int = sqrt(amin amax)"""
    return expit(0.5*s*L)

def s_from_split(eps0, eps1, L):
    """s = q + 4 from the population masses (Pfeil et al. 2024, eq. 23)"""
    return 2*np.log(eps1/eps0) / L

def mean_size(s, l0, l1, amin):
    """Mass-averaged size on [amin e^l0, amin e^l1] (Pfeil et al. 2024, eq. 26)"""
    dl = l1 - l0
    g = _ln_exprel(np.array([s + 1, s + 0*dl])*dl)
    return amin*np.exp(l0 + g[0] - g[1])

def population_sizes(s, L, amin):
    """Mass-averaged sizes (a0, a1) of the two populations, split at
    sqrt(amin amax): the large one's is the small one's times e^(L/2)"""
    a0 = mean_size(s, 0, 0.5*L, amin)
    return a0, a0*np.exp(0.5*L)


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
        # Nodes within a population, as fractions of it
        t = ((np.arange(n_panels)[:,None] + 0.5*(x[None,:] + 1)) / n_panels).ravel()
        # Node positions in l and their weights, as fractions of [0, L]: the
        # small population is [0, L/2], the large one [L/2, L]
        self._frac = np.concatenate([0.5*t, 0.5 + 0.5*t])[:,None]
        self._frac_w = np.tile(0.25*w / n_panels, 2*n_panels)[:,None]
        n = len(t)
        self._pop = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])

    @property
    def population(self):
        """Population (0 or 1) that each node belongs to"""
        return self._pop

    def __call__(self, s, L, amin, lnE0=None):
        """Nodes and mass weights of the distribution.

        args:
            lnE0 : ln E(s, L), the weights' normalisation, if already known

        returns:
            a : node sizes, shape [Nq, Ncells]
            w : fraction of the dust mass in each node, shape [Nq, Ncells]
        """
        if lnE0 is None:
            lnE0 = _lnE(s, L)
        l = self._frac*L
        w = (self._frac_w*L) * np.exp(s*l - lnE0)
        return amin*np.exp(l), w


def mean_drift_factors(s, L, St_min, quad, ks=(-1, 0, 1)):
    """Means of the drift factors St/(1+St^2) and St^2/(1+St^2)
    (_drift_factors) over the power law (s, L), weighted by a^k eps(a) for
    each k in ks, with St = St_min a/amin.

    Since St is linear in a, <St^p/(1+St^2)>_k = <St^p>_k G_(k+p): the exact
    mean <St^p>_k times G_j, the a^j eps(a)-weighted mean of 1/(1+St^2).
    Only G_j is evaluated by quadrature (quad, a SizeQuadrature), normalised
    by the same nodes so that errors in integrating the weight cancel.

    returns:
        F1, F2 : <St/(1+St^2)>_k and <St^2/(1+St^2)>_k, [len(ks), ...]
    """
    k0, k1 = min(ks), max(ks)
    js = np.arange(k0, k1 + 3)
    # ln E(s+j, L), one row for each j in js
    lnE = _lnE(np.add.outer(js, s), L)

    # G_j, j = k0+1 .. k1+2, from the nodes in units of amin (St = St_min b)
    b, w = quad(s, L, 1., lnE[-k0] if k0 <= 0 <= k1 + 2 else None)
    St = St_min*b
    g = 1/(1 + St*St)
    wj = w if k0 == -1 else w*b**(k0 + 1)
    G = []
    for j in range(k0 + 1, k1 + 3):
        if j > k0 + 1:
            wj = wj*b
        G.append((wj*g).sum(0)/(wj.sum(0) + 1e-300))
    G = np.array(G)

    # <St>_k = St_min R_k and <St^2>_k = St_min^2 R_k R_(k+1), with
    # R_j = E(s+j+1, L)/E(s+j, L) the mean size of M_j in units of amin
    R = np.exp(np.diff(lnE, axis=0))
    i = np.array(ks) - k0       # rows of k in R, and of k+1 in G
    St_1 = St_min*R[i]
    St_2 = St_1*St_min*R[i + 1]
    return St_1*G[i], St_2*G[i + 1]

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
        resid : resid(s, l, idx) -> residuals (r0, r1) and Jacobian
                ((J00, J01), (J10, J11)) for the cells idx (all cells if idx is
                None)

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
        (r0, r1), ((J00, J01), (J10, J11)) = resid(sa, la, act)
        det = J00*J11 - J01*J10 - 1e-300     # det < 0: the map is single-signed
        ds = np.clip(-(J11*r0 - J01*r1) / det, -0.5, 0.5)
        dl = np.clip(-(J00*r1 - J10*r0) / det, -0.5, 0.5)
        s_new = np.clip(sa + ds, s_lo, s_hi)
        l_new = np.clip(la + dl, l_lo, l_hi)
        s[act], l[act] = s_new, l_new
        act = act[(np.abs(s_new - sa) >= tol) | (np.abs(l_new - la) >= tol)]

    # Realizability: if s is at its bounds, keep the mass and the
    # mass-flux moment (k=1) and solve for L.
    (r0, r1), _ = resid(s, l, None)
    bad = ok & (np.maximum(np.abs(r0), np.abs(r1)) > 1e-6)
    ib = np.flatnonzero(bad)
    if ib.size:
        sb, lb, yb = s[ib], l[ib], y1[ib]
        for _ in range(max_iter):
            g, _, d = _lnE_newton(sb, np.exp(lb))
            f = g[2] - g[1] - yb
            dl = np.clip(-f/(d[2] - d[1] + 1e-300), -0.5, 0.5)
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
        g, m, d = _lnE_newton(s, np.exp(l))
        ya_i, yb_i = (ya, yb) if idx is None else (ya[idx], yb[idx])
        r = (g[0] - g[1] - ya_i, g[2] - g[1] - yb_i)
        J = ((m[0] - m[1], d[0] - d[1]), (m[2] - m[1], d[2] - d[1]))
        return r, J

    # The fallback keeps the mass-flux moment, y = yb
    s, L, bad = _fit_s_L(resid, ok, s0, L0, s_bounds, L_min, yb)
    return eps_new, s, L, bad


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
    y = _shape_moments(s, L)
    return y[1], y[0] + y[1]

def _D_inf(X):
    return X - np.log(2 - np.exp(-X))

def _logit_T(X, D):
    T = np.clip(D/_D_inf(X), 1e-15, 1 - 1e-15)
    return np.log(T/(1 - T))


class MomentTable(object):
    """Inverse of the map (s, L) -> (X, D), tabulated on the parameter box
    s in s_bounds, L in [L_min, L_max]: bilinear interpolation of s and ln L,
    then one Newton step on the moments. The drift velocities of the result
    (mean_drift_factors) agree with those of the exact fit to <~ 1e-7, median
    ~3e-10; the interpolation alone, at the default resolution (512 x 256),
    gives 1.2e-3, median ~4e-5.

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
        """Project (X, D) onto the box and invert: the table's bilinear
        interpolation, then one Newton step (_newton_step).

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
        if moved.any():
            Dc = np.where(moved, expit(ltc)*_D_inf(Xc), D)
            Xc = np.where(moved, Xc, X)
        else:
            Xc, Dc = X, D

        # Bilinear interpolation in (ln X, v), with the corner weights shared
        # by the two tables
        xi = np.log(np.clip(Xc, self.X_min, self.X_max))
        # (clipped: at the corners lo and hi meet, and can cross by rounding)
        v = np.clip((ltc - lo)/np.maximum(hi - lo, 1e-300), 0., 1.)
        i = np.clip(np.searchsorted(self._xi, xi) - 1, 0, len(self._xi) - 2)
        fx = (xi - self._xi[i])/(self._xi[i+1] - self._xi[i])
        nv = self._nv - 1
        j = np.minimum((v*nv).astype(int), nv - 1)
        fv = v*nv - j
        k00 = i*self._nv + j                     # flat index of the (i, j) corner
        k10 = k00 + self._nv
        w01, w11 = (1 - fx)*fv, fx*fv
        w00, w10 = (1 - fx) - w01, fx - w11
        def interp(t):
            t = t.ravel()
            return (w00*t.take(k00) + w01*t.take(k00 + 1) +
                    w10*t.take(k10) + w11*t.take(k10 + 1))
        s = np.clip(interp(self._s), s_lo, s_hi)
        s, L = self._newton_step(s, np.exp(interp(self._lnL)), Xc, Dc)
        return Xc, Dc, s, L, moved

    def _newton_step(self, s, L, X, D):
        """One Newton step in (s, ln L) towards the power law with the scaled
        moments y = (D - X, X), kept in the box. Only for q >= -5 (s >= -1):
        below, broad distributions are ill-conditioned (M_-1, M_0 and M_1
        barely depend on amax), and the table's values are kept."""
        s_lo, s_hi, L_min, L_max = self._box
        g, m, d = _lnE_newton(s, L)
        r0, r1 = g[0] - g[1] - (D - X), g[2] - g[1] - X
        J00, J10 = m[0] - m[1], m[2] - m[1]
        J01, J11 = d[0] - d[1], d[2] - d[1]
        det = J00*J11 - J01*J10 - 1e-300     # det < 0: the map is single-signed
        ds = -(J11*r0 - J01*r1)/det
        dl = -(J00*r1 - J10*r0)/det
        step = s >= _S_INNER
        return (np.where(step, np.clip(s + ds, s_lo, s_hi), s),
                np.where(step, np.clip(L*np.exp(dl), L_min, L_max), L))


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
    and grain_size = [amin, amax], as in DustGrowthTwoPop (the populations'
    mass-averaged sizes, used by growth and drift, follow from the power law).

    The state is the total dust fraction, M_0, and the scaled moments
    M_-1/M_0 and M_1/M_0. The power-law parameters, and dust_frac and
    grain_size, are set with it on every change (read-only arrays: set the
    dust with set_dust_frac). Growth changes only the scaled moments, so it
    conserves the dust mass exactly.

    Growth of amax is integrated analytically at fixed growth time-scale,
    assuming dv_max is a power law in a during each step, with the local index
    estimated from the contributions to dv_max, or, once amax has crossed its
    fragmentation limit, the secant index since the last step (_limit_slope).
    Mass exchange between the populations (sweep-up and fragmentation) is
    integrated analytically at fixed coefficients.

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
        growth_cfl: Maximum change in ln(amax) per (sub-)step of a growing
                    amax, default=0.2. A shrinking amax takes a single step.
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
        frag_slope, frag_size:
                    The fragmentation velocity varies with amax as
                    v_frag = uf (amax/frag_size)^frag_slope, where uf (from
                    uf_0, uf_ice and the ice fraction) is its value at
                    frag_size [cm]. Defaults 0 (independent of size) and 0.1.
                    Only frag_slope <= 0 is physically meaningful.
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
                 growth_cfl=0.2, max_substeps=1000,
                 vg2_factor=1.5,
                 q_bounds=(-5., -1.), quadrature=None, moment_table=True,
                 drift_switch=0.3, turb_switch=5.0, frag_width=1.5, f_coll=0.9,
                 frag_slope=0., frag_size=0.1):
        super(DustGrowthTriMom, self).__init__(grid, star, eos, Sigma, rho_s,
                                              Sc, feedback)
        # (ln amax, ln dv_max) at the last growth step that moved amax, for
        # the secant index (_limit_slope, _remember_step); NaN until then
        self._last_dv = (np.full(self.Ncells, np.nan), np.full(self.Ncells, np.nan))

        if a0 <= 1.5*amin:
            raise ValueError("Initial size a0 must exceed 1.5*amin")

        self._uf_0   = uf_0 / (AU * Omega0)
        self._uf_ice = uf_ice / (AU * Omega0)
        self._ice_threshold = thresh
        self._uf = self._frag_velocity(np.broadcast_to(f_ice, (self.Ncells,)))

        self._fgrow = f_grow
        self._f_dv = f_dv
        self._f_coll = f_coll
        self._s_frag = s_frag
        self._q_sweep = q_sweep
        self._q_turb1 = q_turb1
        self._q_turb2 = q_turb2
        self._q_drift_frag = q_drift_frag
        self._growth_cfl = growth_cfl
        self._max_substeps = max_substeps
        self._vg2_factor = vg2_factor
        self._drift_switch = drift_switch
        self._turb_switch = turb_switch
        self._frag_width = frag_width
        self._frag_slope = frag_slope
        self._frag_size = frag_size

        if quadrature is None:
            quadrature = SizeQuadrature()
        self._quad = quadrature

        # Initial number distribution slope (DustyDisc._p; remove_dust_with_gas
        # uses the size distribution itself)
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

        self.set_size_distribution(eps*np.ones(Ncells), s, L)

        # Diagnostics: realizability fallbacks in remove_dust_with_gas
        self.n_removal_clamped = 0

        # The ices last passed to update_ices, which growth mixes
        self._ices = None

        self._head = (', uf_0: {}cm s^-1, uf_ice: {}cm s^-1, thresh: {}'
                      ', f_grow: {}, a0: {}cm, amin: {}cm, f_dv: {}'
                      ', s_frag: {}, frag_slope: {}, frag_size: {}cm'.format(
                          uf_0, uf_ice, thresh, f_grow, a0, amin,
                          f_dv, s_frag, frag_slope, frag_size))

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
        """Maximum grain size (read-only)"""
        return self._a[1]

    @property
    def dust_frac(self):
        """Dust fractions of the two populations, [eps0, eps1] (read-only:
        use set_dust_frac)"""
        return self._eps

    @property
    def grain_size(self):
        """Grain sizes, [amin, amax] (read-only: use set_dust_frac)"""
        return self._a

    @property
    def integ_dust_frac(self):
        """Total dust fraction, or zero if not including dust feedback"""
        if self._feedback:
            return self._eps_tot.copy()
        return 0

    def size_distribution(self):
        """Parameters of the truncated power law.

        returns:
            eps_tot : total dust fraction
            s       : q + 4
            L       : ln(amax/amin)
        """
        return self._eps_tot.copy(), self._s, self._L

    def set_size_distribution(self, eps_tot, s, L):
        """Set the state from the power-law parameters"""
        eps_tot, s, L = (np.broadcast_to(x, (self.Ncells,)) for x in (eps_tot, s, L))
        self._set_state(eps_tot, self._shape_moments(s, L), s, L)

    def set_populations(self, eps0, eps1, amax):
        """Set the state from the TriPoD parameters, which are kept exactly
        (negative dust fractions are set to zero), as set_dust_frac"""
        self.set_dust_frac([eps0, eps1], (None, amax))

    def set_dust_frac(self, dust_frac, grain_size=None):
        """Set the dust fractions of the populations, [eps0, eps1], and
        optionally the grain sizes, [amin, amax] (only amax is used; e.g. to
        restart from a snapshot). The populations, kept exactly, define the
        state: s follows from their split, and L from amax.

        M_0 is the populations' total, which rescales the moments; their
        shape (the stored ln M_k/M_0) is kept where the populations reproduce
        it to round-trip accuracy, the fit's (~1e-9 for q >= -5; more below,
        where the table is not polished). So cells the caller left unchanged,
        or only rescaled, keep their shape exactly.
        """
        eps = np.maximum(np.array(dust_frac, dtype='f8'), 0)
        amax = self._a[1] if grain_size is None else \
            np.array(np.broadcast_to(grain_size[1], (self.Ncells,)), dtype='f8')
        if (eps == self._eps).all() and (amax == self._a[1]).all():
            return
        L = self._ln_amax(amax)
        _, s = self._population_params(eps[0], eps[1], L)
        y = self._shape_moments(s, L)
        y = np.where(np.isclose(y, self._y, rtol=0, atol=1e-8), self._y, y)
        self._set_state(eps.sum(0), y, s, L, eps, amax)

    def set_moments(self, Q, L_max=None):
        """Set the state from the moments (M_-1, M_0, M_1) per unit Sigma.

        Moments outside the range of the power law are projected onto it,
        keeping M_0 and M_1. Where M_0, M_-1 or M_1 is not positive the shape
        is kept.

        L_max, optional, is a per-cell upper bound on L = ln(amax/amin) where
        the moments need q > q_max (e.g. the largest amax that transport can
        have brought into the cell). There the projection keeps M_0 and M_1
        by raising amax; where that would exceed the bound, q = q_max and
        amax = amin e^L_max are kept instead of M_1.

        returns:
            the number of cells projected
        """
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

        if L_max is not None:
            # Projected onto q = q_max (the table's projection leaves s
            # within ~1e-6 of it) with amax above the bound
            s_hi = self._s_bounds[1]
            top = moved & (s > s_hi - 1e-4) & (L > L_max*(1 + 1e-9))
            if top.any():
                s, L = np.where(top, s_hi, s), np.where(top, L_max, L)
                y = np.where(top, self._shape_moments(s, L), y)

        self._set_state(eps, y, np.where(ok, s, self._s), np.where(ok, L, self._L))
        return moved.sum()

    def moments(self, ks=(-1, 0, 1)):
        """Mass-weighted moments, int a^k eps(a) da, for each k in ks"""
        # From the state where possible, otherwise from the power law
        y = {-1: self._y[0], 0: 0., 1: self._y[1]}
        if not set(ks) <= set(y):
            _, s, L = self.size_distribution()
        return np.array([self._eps_tot*(np.exp(y[k])*self._amin**k if k in y
                                        else moment_ratio(k, s, L, self._amin))
                         for k in ks])

    @property
    def area_size(self):
        """Area-equivalent grain size, a_A = M_0 / M_-1"""
        return self._amin*np.exp(-self._y[0])

    @property
    def surface_area(self):
        """Total grain surface area per unit gas mass [cm^2 g^-1]"""
        M_m1 = self._eps_tot*np.exp(self._y[0])/self._amin
        return 3*M_m1 / self._rho_s / np.maximum(1 - self._eps_tot, 1e-300)

    @property
    def grain_cross_section(self):
        """Geometric cross-section of the grains per unit dust mass,
        3 / (4 rho_s a_A) [cm^2 g^-1], a quarter of their surface area"""
        return 3 / (4*self._rho_s*self.area_size)

    ############################################################################
    # State
    ############################################################################
    # The state is eps_tot = M_0 and y = [ln(amin M_-1/M_0), ln(M_1/(amin M_0))].
    # Every change goes through _set_state, which also sets the power-law
    # parameters (s, L) and the quantities derived from them (_eps = [eps0,
    # eps1], _a = [amin, amax]). All are read-only arrays, replaced (never
    # modified) on each change: set the dust through set_dust_frac,
    # set_populations, set_size_distribution or set_moments.
    def _shape_moments(self, s, L):
        """y = [ln(M_-1/M_0) + ln(amin), ln(M_1/M_0) - ln(amin)]"""
        return _shape_moments(s, L)

    def _set_state(self, eps_tot, y, s, L, eps=None, amax=None):
        """Set the state (eps_tot, y [2, Ncells]), its power-law parameters
        (s, L) and the populations, as read-only copies (all per cell). eps =
        [eps0, eps1] and amax, if given, are kept exactly (set_dust_frac,
        growth); otherwise they follow from (s, L)."""
        def frozen(x):
            x = np.array(x, dtype='f8')
            x.flags.writeable = False
            return x
        if eps is None:
            f1 = large_fraction(s, L)
            eps = [eps_tot*(1 - f1), eps_tot*f1]
        if amax is None:
            amax = self._amin*np.exp(L)
        self._eps_tot, self._y, self._s, self._L = map(frozen, (eps_tot, y, s, L))
        # dust_frac = [eps0, eps1] and grain_size = [amin, amax]
        self._eps = frozen(eps)
        self._a = frozen([np.full_like(self._s, self._amin), amax])

    def _ln_amax(self, amax):
        """L = ln(amax/amin), at least L_min"""
        return np.log(np.maximum(amax/self._amin, np.exp(self._L_min)))

    def _population_params(self, eps0, eps1, L):
        """[eps0, eps1] (negative values set to zero) and s of the TriPoD
        parameters with L = ln(amax/amin). s follows from the split, or is
        kept where a population is empty."""
        eps0 = np.maximum(eps0, 0)
        eps1 = np.maximum(eps1, 0)
        ok = (eps0 > 0) & (eps1 > 0)
        s = np.where(ok, s_from_split(np.where(ok, eps0, 1),
                                      np.where(ok, eps1, 1), L),
                     self._s)
        return np.array([eps0, eps1]), np.clip(s, *self._s_bounds)

    def _fit_newton(self, y):
        """(s, L) of the scaled moments y by Newton iteration from the current
        parameters, and the cells that needed the realizability fallback"""
        s0, L0 = self._s, self._L
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

    def _pressure_eta(self):
        """eta = -(dP/dR) / (rho_g Omega_k) at cell centres"""
        rho = self.midplane_gas_density
        return - np.gradient(self.P, self.R) / ((rho + 1e-300)*self.Omega_k)

    def _gas_velocities(self, eta=None):
        """Gas velocities relative to Keplerian at cell centres, including
        feedback from the dust (Dipierro+18, without the viscous term).

        args:
            eta : _pressure_eta, if already known

        returns:
            u_gas, v_gas : radial and azimuthal gas velocities
        """
        if eta is None:
            eta = self._pressure_eta()

        if not self.feedback:
            return 0*eta, -0.5*eta

        a, w = self.quadrature()
        eps_tot = self.integ_dust_frac
        eps_g = np.maximum(1 - eps_tot, 1e-300)
        St = self.Stokes(self.Sigma_G, a)
        f1, f2 = _drift_factors(St)
        la0 = (eps_tot/eps_g) * (w*(1 - f2)).sum(0)
        la1 = (eps_tot/eps_g) * (w*f1).sum(0)

        D_1 = 1 / ((1 + la0)**2 + la1**2)
        return la1*eta*D_1, -0.5*(1 + la0)*eta*D_1

    def _collision_gas(self):
        """The gas quantities of relative_velocity(_index): gas velocities
        (with the feedback of the current dust), Sigma_G, cs, Re^-1/2,
        Omega_k, H and alpha/Sc, and the gas factors of the collision
        velocities: K = St/a, alpha cs^2 (turbulence), the Brownian-motion
        factor and Omega_k H (settling)"""
        SigG, cs, alpha, Om, H = self.Sigma_G, self.cs, self.alpha, self.Omega_k, self.H
        eta = self._pressure_eta()
        return dict(eta=eta, vel=self._gas_velocities(eta), SigG=SigG, cs=cs, alpha=alpha,
                    sqrtRe_inv=self._sqrtRe_inv(), Om=Om, H=H, aSc=alpha/self._Sc,
                    K=self._Kdrag/(SigG + 1e-300),
                    turb=self._vg2_factor*alpha*cs*cs,
                    # (8 m_p mu cs^2/pi) (m1 + m2)/(m1 m2), m = (4 pi/3) rho_s a^3
                    brown=(6*m_p/np.pi**2)*self.mu*cs*cs/self._rho_s,
                    OmH=Om*H)

    def relative_velocity(self, a1, a2, gas=None, components=False):
        """Relative velocity of grains of size a1 and a2.

        Includes Brownian motion, turbulence (Ormel & Cuzzi 2007), radial and
        azimuthal drift, and differential settling, as in cuDisc.

        args:
            a1, a2     : grain sizes, broadcastable to [..., Ncells]
            gas        : The gas quantities from _collision_gas (computed if
                         None)
            components : Also return the drift contributions

        returns:
            dv : Total relative velocity (code units)
            dv_rad, dv_az : Radial and azimuthal drift contributions (only if
                            components=True)
        """
        dv, dv_rad, dv_az, _ = self._relative_velocity(
            a1, a2, self._collision_gas() if gas is None else gas)
        if components:
            return dv, np.abs(dv_rad), np.abs(dv_az)
        return dv

    def relative_velocity_index(self, a1, a2, gas=None, components=False):
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
            a1, a2, self._collision_gas() if gas is None else gas, index=True)
        if components:
            return dv, beta, np.abs(dv_rad), np.abs(dv_az)
        return dv, beta

    def _relative_velocity(self, a1, a2, g, index=False):
        """dv, its signed radial and azimuthal drift contributions, and (if
        index) beta, for gas quantities g (_collision_gas)"""
        u_gas, v_gas = g['vel']
        St1, St2 = g['K']*a1, g['K']*a2

        # Turbulence
        turb2 = g['turb']*vrel_sqd_OC07(St1, St2, g['sqrtRe_inv'])

        # Brownian motion
        bm2 = g['brown']*(1/(a1*a1*a1) + 1/(a2*a2*a2))

        # Radial and azimuthal drift: X(St) = A St/(1+St^2) + B St^2/(1+St^2)
        # with (A, B) = (2 v_gas, -u_gas) and (-u_gas/2, -v_gas)
        f11, f21 = _drift_factors(St1)
        f12, f22 = _drift_factors(St2)
        dv_rad = (2*v_gas*f11 - u_gas*f21) - (2*v_gas*f12 - u_gas*f22)
        dv_az = (-0.5*u_gas*f11 - v_gas*f21) - (-0.5*u_gas*f12 - v_gas*f22)

        # Differential settling: Y(St) = Omega H_d(St) min(St, 1/2), with
        # dust_scale_height
        Y1 = g['OmH']*np.minimum(St1, 0.5)/np.sqrt(1 + St1/g['aSc'])
        Y2 = g['OmH']*np.minimum(St2, 0.5)/np.sqrt(1 + St2/g['aSc'])
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

    def _v_frag(self, amax):
        """The fragmentation velocity at amax (frag_slope, frag_size)"""
        return self._uf*(amax/self._frag_size)**self._frag_slope

    def _q_equilibrium(self, dv_max, dv_rad, dv_az, St_max, gas, v_frag):
        """Target size distribution exponent (Pfeil et al. 2024, eqs. 36-38
        and App. C, following the TriPoDPy implementation), with the gas
        quantities gas (_collision_gas) and the fragmentation velocity v_frag"""
        alpha = gas['alpha']
        sqrtRe_inv = gas['sqrtRe_inv']

        # Transition between the two turbulent regimes: p_t1 -> 1 when the
        # largest grains are in the intermediate regime
        # (St_max > turb_switch Re^-1/2; TriPoD: 5)
        f_t1t2 = self._turb_switch*sqrtRe_inv / St_max
        p_t1 = 0.5*(1 - (f_t1t2**4 - 1)/(f_t1t2**4 + 1))

        # Transition between drift- and turbulence-dominated fragmentation
        dv_dr = np.maximum(np.hypot(dv_rad, dv_az), 1e-300)
        St_mn = 0.5*St_max
        vgas = np.sqrt(self._vg2_factor*alpha)*gas['cs']
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
        p_frag = np.exp(-(self._frag_width*(np.minimum(dv_max/v_frag, 1) - 1))**2)

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

    def _growth_rates(self, gas=None):
        """Coefficients of the local growth/fragmentation step, evaluated for
        the current state, with the gas quantities gas (_collision_gas,
        computed if None).

        returns a dict with:
            t_grow : free growth time-scale of amax
            z0     : s ln(v_frag / dv_max), v_frag at amax (_v_frag)
            ln_dv  : ln dv_max
            beta   : local power-law index of dv_max with amax
            C1, C2 : sweep-up and fragmentation rate coefficients
        """
        eps_tot, s, L = self.size_distribution()
        amin, amax = self._amin, self._a[1]

        # Representative sizes: the mass-averaged sizes of the populations
        f_dv = self._f_dv
        a0, a1 = population_sizes(s, L, amin)

        g = self._collision_gas() if gas is None else gas
        ac = self._f_coll*amax
        # dv_max, its drift components and its index beta (the dv^2-weighted
        # mean of the indices of the contributions) in one evaluation
        dv_max, beta, dv_rad, dv_az = self.relative_velocity_index(ac, f_dv*ac, gas=g,
                                                                   components=True)
        dv01 = self.relative_velocity(a0, a1, gas=g)
        dv11 = self.relative_velocity(a1, f_dv*a1, gas=g)

        St_max = g['K']*amax
        v_frag = self._v_frag(amax)
        q = self._q_equilibrium(dv_max, dv_rad, dv_az, St_max, g, v_frag)

        H1 = self.dust_scale_height(g['K']*a1)
        t_grow = self._free_growth_time(amax, H1, dv_max)
        z0 = self._s_frag*np.log(v_frag/dv_max)

        # Mass exchange between the populations, eqs. 32-35
        H0 = self.dust_scale_height(g['K']*a0)
        sig01 = np.pi*(a0 + a1)**2
        sig11 = np.pi*(a1 + f_dv*a1)**2
        m1 = (4*np.pi/3)*self._rho_s*a1**3
        F = H1*np.sqrt(2/(H0**2 + H1**2)) * (sig01/sig11) * (dv01/dv11) * \
            np.exp(-0.5*(q + 4)*L)
        C1 = sig01*dv01 / (m1*np.sqrt(2*np.pi*(H0**2 + H1**2)))
        C2 = sig11*dv11*F / (2*m1*np.sqrt(np.pi)*H1)

        return dict(t_grow=t_grow, z0=z0, ln_dv=np.log(dv_max), beta=beta, C1=C1, C2=C2)

    def _free_growth_time(self, amax, H1, dv_max):
        """Free growth time-scale of amax (eq. 29, integrated at fixed t_grow),
        with H1 the scale height of the large population"""
        Sigma1 = self.Sigma*self._eps[1]
        adot_free = Sigma1*dv_max / (np.sqrt(2*np.pi)*H1*self._rho_s)
        return self._fgrow*amax / np.maximum(adot_free, 1e-300)

    # Smallest change in ln amax for the secant index (_limit_slope)
    _secant_dl = 1e-3

    def _limit_slope(self, ln_dv):
        """A secant approximation to the index, beta, of dv_max ~ a^beta. Only used
        near the fragmentation limit, where the local index can be steep.

        returns:
            bounded, beta_secant
        """
        l1, ln_dv1 = self._last_dv
        l = np.log(self._a[1])
        dl = l - l1
        # ln v_frag at both points, with today's v_frag
        p = self._frag_slope
        ln_uf = np.log(self._uf) - p*np.log(self._frag_size)
        ln_vf, ln_vf1 = ln_uf + p*l, ln_uf + p*l1
        # Check whether we have a bounded interval around the limit that is wide
        # enough for the secant to be meaningful.
        bounded = ((ln_dv > ln_vf) != (ln_dv1 > ln_vf1)) & (np.abs(dl) > self._secant_dl)
        beta_secant = (ln_dv - ln_dv1)/np.where(bounded, dl, 1.)
        # A falling dv_max/v_frag (past the peak of dv_max) crosses an
        # unstable limit
        return bounded & (beta_secant > p), beta_secant

    def _remember_step(self, ln_dv, amax_new):
        """Store (ln amax, ln dv_max) for estimaing the slope of dv(a) near the
        fragmenation limit."""
        l = np.log(self._a[1])
        moved = np.abs(np.log(amax_new) - l) > self._secant_dl
        self._last_dv = (np.where(moved, l, self._last_dv[0]),
                         np.where(moved, ln_dv, self._last_dv[1]))

    def _growth_dt(self, t_grow, z0):
        """Maximum growth step in each cell: growth_cfl e-folds of a growing
        amax; a shrinking amax takes a single step."""
        dt = self._growth_cfl*t_grow/np.maximum(np.abs(np.tanh(0.5*z0)), 1e-300)
        return np.where(z0 < 0, np.inf, dt)

    def growth_timestep(self):
        """Time-step limit for resolving the growth of amax (_growth_dt)"""
        eps = self.dust_frac
        ids = (eps[0] > 0) & (eps[1] > 0)
        if not ids.any():
            return np.inf

        g = self._collision_gas()

        # the large population's mass-averaged size, as _growth_rates
        _, a1 = population_sizes(self._s, self._L, self._amin)
        amax = self._a[1]
        ac = self._f_coll*amax

        dv_max = self.relative_velocity(ac, self._f_dv*ac, gas=g)
        H1 = self.dust_scale_height(g['K']*a1)
        t_grow = self._free_growth_time(amax, H1, dv_max)

        z0 = self._s_frag*np.log(self._v_frag(amax)/dv_max)

        return self._growth_dt(t_grow, z0)[ids].min()

    def do_grain_growth(self, dt):
        """Apply the grain growth.

        amax is advanced with the analytic solution at fixed t_grow (assuming
        a local power law of dv_max in amax, _limit_slope) and the mass
        exchange between populations with the analytic solution at fixed
        coefficients. Cells are sub-cycled so that a growing amax changes by
        at most growth_cfl e-folds per sub-step; a shrinking amax takes a
        single step (_growth_dt). The last allowed sub-step takes the rest.

        The exchange between the populations also mixes the ices (see
        chemistry.area_ice): a tracer's split between the populations relaxes
        to the mass split at (C1 + C2) Sigma_1.
        """
        if dt <= 0:
            return
        ids = (self._eps[0] > 0) & (self._eps[1] > 0)
        if not ids.any():
            return

        remaining = np.where(ids, dt, 0.)
        mix = np.zeros(self.Ncells)
        # The gas quantities, fixed for the call (operator splitting: the
        # feedback is that of the dust at its start), and the power law's
        # shape moments
        gas = self._collision_gas()
        shape = self._shape_moments(self._s, self._L)
        for i in range(self._max_substeps):
            active = remaining > 0
            if not active.any():
                break

            eps_tot, s, L = self.size_distribution()
            rates = self._growth_rates(gas)
            # Estimate dv_max(a) ~ a^beta. Use a local estimate of beta unless
            # we have a bounded interval around the fragmentation limit.
            crossed, beta_secant = self._limit_slope(rates['ln_dv'])
            beta = np.where(crossed, beta_secant, rates['beta'])

            if i < self._max_substeps - 1:
                h = np.minimum(remaining, self._growth_dt(rates['t_grow'], rates['z0']))
            else:
                h = remaining

            amax_new = self.grow_amax(self._a[1], rates['t_grow'], rates['z0'],
                                      self._s_frag*(beta - self._frag_slope), h)
            amax_new = np.maximum(amax_new, 1.5*self._amin)      # the size floor
            self._remember_step(rates['ln_dv'], amax_new)

            S = self.Sigma*eps_tot
            f1 = large_fraction(s, L)
            C1, C2 = rates['C1'], rates['C2']
            f1_new = self.exchange_fraction(f1, S*C1, S*(C1 + C2), h)
            mix += S*(C1 + C2)*0.5*(f1 + f1_new)*h

            # (inactive cells keep their state exactly)
            shape = self._grow_populations(
                np.where(active, eps_tot*(1 - f1_new), self._eps[0]),
                np.where(active, eps_tot*f1_new, self._eps[1]),
                np.where(active, amax_new, self._a[1]), shape)
            remaining = remaining - h

        if self._ices is not None:
            self._ices.mix_area_to_mass(-np.expm1(-mix))

    def _grow_populations(self, eps0, eps1, amax, shape):
        """Set the populations after a growth (sub-)step, keeping M_0: the
        shape of the state changes by the change in the scaled moments of
        the power law (shape: those of the current one, _shape_moments), so
        that cells that do not change keep their state exactly.

        returns:
            the scaled moments of the new power law
        """
        L = self._ln_amax(amax)
        eps, s = self._population_params(eps0, eps1, L)
        new = self._shape_moments(s, L)
        self._set_state(self._eps_tot, self._y + (new - shape), s, L, eps, amax)
        return new

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
        self._set_state(np.broadcast_to(eps_new, (self.Ncells,)), self._y, self._s,
                        self._L, amax=self._a[1])

    def update(self, dt):
        """Do the standard disc update, and apply grain growth"""
        super(DustGrowthTriMom, self).update(dt)
        self.do_grain_growth(dt)

################################################################################
# Radial drift
################################################################################
class MomentDrift(object):
    """Moment-consistent radial drift for the DustGrowthTriMom model.

    Transports the moments int a^k eps(a) da for k = -1, 0, 1 (area, mass and
    size), each with its own weighted velocity, in the single-fluid
    approximation with the short friction time limit (as in
    dust.SingleFluidDrift). The dust-gas relative velocity for each size
    follows Tanaka+ (2005), with feedback. In the small Stokes number limit the
    area and mass fluxes are exact.

    The drift step is Stone & Gardiner's (2009) van Leer integrator (VL2),
    second order in space and time: a donor-cell half step, the size
    distribution refitted to the half-step moments, then fluxes from van Leer
    face states of the half-step cells, with velocities from the face states.
    With van_leer=False it is donor cell (first order).

    Advection with the gas is done by the driver's gas step: pre_gas_transport
    hands it the moments (as passive tracers) and post_gas_transport sets the
    size distribution from them afterwards. If the disc is evolved without a
    drift module, amax is not advected. Diffusion is a separate step too, done
    by the driver through diffuse().

    TriPoD's scheme and its TriMean and TriMax variants, which use the same
    integrator, are in dust_tripod.py.

    args:
        settling  : Use midplane densities for feedback, default=False
        van_leer  : Use the van Leer integrator (VL2), default=True; False
                    for donor cell
        f_drift   : Factor multiplying grain size in drift velocities,
                    default=1
        area_tracers: Indices of the dust-tracer rows carried by the grain
                    surface area (e.g. the area-distributed ice of
                    chemistry.area_ice), which drift with the area (M_-1)
                    velocity and diffuse with its Stokes number. The other
                    rows follow the dust mass. Default None (all by mass)
        limit_amax: Where the moments after transport need q > q_max, keep
                    amax no larger than the largest of the cell and the
                    cells upwind of it (instead of keeping M_1), default=True
    """
    _scheme = 'moments'

    # Orders of the transported moments
    _ks = (-1, 0, 1)

    # VL2: cells below this fraction of the largest dust surface density, and
    # their neighbours, keep their cell values at their faces (their shape is
    # undefined)
    _empty_frac = 1e-6

    # Reach of a drift step (VL2's stencil): the number of cells upwind of a
    # face whose grains can enter through it
    _reach = 2

    def __init__(self, settling=False, van_leer=True, f_drift=1.0,
                 area_tracers=None, limit_amax=True):
        self._settling = settling
        self._van_leer = van_leer
        self._f_drift = f_drift
        self._limit_amax = limit_amax
        self._L_upwind = None       # the bound on L in a drift step
        self._area_tracers = (None if area_tracers is None
                              else np.atleast_1d(area_tracers))
        self._vl = None

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
                 "limit_amax": "{}".format(self._limit_amax),
                 }
        return self.__class__.__name__ , head

    ############################################################################
    # Transported quantities
    ############################################################################
    def _transported(self, disc):
        """Transported quantities per unit total surface density, [3, Ncells]"""
        return disc.moments(self._ks)

    def _reconstruct(self, disc, Q, shrink=True):
        """Set the disc's size distribution from the transported quantities
        (shrink: TriPoD's shrinkage, for the schemes that have it)"""
        # The moments are the disc's state. With limit_amax, where they need
        # q > q_max, amax is kept no larger than upwind (in a drift step,
        # _L_upwind; after the gas and diffusion steps, its value before
        # them). Without it, where all the grains have St > 1 the drift
        # narrows the distribution (the smaller grains, nearer St = 1, drift
        # faster), and keeping M_1 at q_max raises amax. The larger, slower
        # grains then hold back M_1, so amax runs away, and seeds runaways in
        # broad fits where these grains mix with others. A bound from
        # downwind would let the clipped cells raise each other's.
        L_max = None
        if self._limit_amax:
            L_max = self._L_upwind
            if L_max is None:
                L_max = disc.size_distribution()[2]
        self.n_clamped += disc.set_moments(Q, L_max)

    def _upwind_amax(self, disc, V_L, V_R):
        """The largest L = ln(amax/amin) of each cell and the _reach cells
        upwind of its faces, from the velocities of the transported
        quantities at the interior edges [*, Ncells-1] (a face is upwind
        from a side if any of them flows in from it)"""
        _, _, L = disc.size_distribution()
        n, N = self._reach, L.size
        V_L, V_R = np.reshape(V_L, (-1, N - 1)), np.reshape(V_R, (-1, N - 1))
        from_R = np.concatenate([((V_L < 0) | (V_R < 0)).any(0), [False]])
        from_L = np.concatenate([[False], ((V_L > 0) | (V_R > 0)).any(0)])
        pad = np.concatenate([np.full(n, -np.inf), L, np.full(n, -np.inf)])
        right = np.max([pad[n + k:n + k + N] for k in range(1, n + 1)], axis=0)
        left = np.max([pad[n - k:n - k + N] for k in range(1, n + 1)], axis=0)
        return np.maximum(L, np.maximum(np.where(from_R, right, -np.inf),
                                        np.where(from_L, left, -np.inf)))

    ############################################################################
    # Velocities
    ############################################################################
    # The velocities at an interior edge depend on the size distributions on
    # either side of it, each given as a 'side' (Sd, s, L): the dust surface
    # density (the total, [Ne]; for the TriPoD schemes of the two populations,
    # [2, Ne]) and the shape of the power law. The donor cell step and the VL2
    # predictor take the cells on either side (_cell_sides), the VL2 corrector
    # the face states (_vl2_faces).
    def _cell_sides(self, disc):
        """The cells to the left and right of each interior edge, as sides"""
        _, s, L = disc.size_distribution()
        Sd = (disc.Sigma*disc.dust_frac).sum(0)
        return (Sd[..., :-1], s[:-1], L[:-1]), (Sd[..., 1:], s[1:], L[1:])

    def _side_scale_height(self, disc, a, side):
        """Dust scale height of sizes a [Nq, Ne] on the left ('L') or right
        ('R') side of the interior edges, with the gas of that side's cell"""
        sl = slice(None, -1) if side == 'L' else slice(1, None)
        St = disc.Stokes(disc.Sigma_G[sl], a)
        # dust_scale_height takes cell arrays: pad to the cells and back
        if side == 'L':
            return disc.dust_scale_height(np.concatenate([St, St[..., -1:]], axis=-1))[..., :-1]
        return disc.dust_scale_height(np.concatenate([St[..., :1], St], axis=-1))[..., 1:]

    def _node_velocities(self, disc, aL, SdL, aR, SdR, v_visc=None):
        """Velocities relative to the centre of mass of nodes of sizes a, with
        dust surface densities Sd ([Nq, Ne] each), on the left and right of
        each interior edge. The feedback comes from all of them.

        returns:
            V_L, V_R  : node velocities at the edges, [Nq, Ne]
            DeltaVphi : azimuthal velocity of the left nodes relative to the gas
        """
        SigmaG = disc.Sigma_G
        SigG_av = 0.5*(SigmaG[1:] + SigmaG[:-1])

        St_L = disc.Stokes(SigG_av, aL + 1e-300)
        St_R = disc.Stokes(SigG_av, aR + 1e-300)

        # Feedback: the fractions of each node at the edges, of the edge's
        # total density (the gas plus both sides' dust, so that the fractions
        # sum to one and the centre of mass moves with v_visc); with settling,
        # of the midplane densities
        if disc.feedback:
            if self._settling:
                dL = SdL / (np.sqrt(2*np.pi)*self._side_scale_height(disc, aL, 'L')*AU)
                dR = SdR / (np.sqrt(2*np.pi)*self._side_scale_height(disc, aR, 'R')*AU)
                rhoG = disc.midplane_gas_density
                g = 0.5*(rhoG[1:] + rhoG[:-1])
            else:
                dL, dR, g = SdL, SdR, SigG_av
            tot = g + 0.5*(dL + dR).sum(0) + 1e-300
            eL, eR = dL / tot, dR / tot
            eps_g = np.maximum(g / tot, 1e-300)
        else:
            eL = eR = 0.
            eps_g = 1.

        f1L, f2L = _drift_factors(St_L)
        f1R, f2R = _drift_factors(St_R)
        la0 = 0.5*((eL*(1 - f2L)).sum(0) + (eR*(1 - f2R)).sum(0))/eps_g
        la1 = 0.5*((eL*f1L).sum(0) + (eR*f1R).sum(0))/eps_g
        u_gas, v_gas = self._edge_gas_velocities(disc, la0, la1, eps_g, v_visc)

        # Relative to the gas, and to the centre of mass: epsDeltaV =
        # v_COM - v_gas
        dV_L = 2*v_gas*f1L - u_gas*f2L
        dV_R = 2*v_gas*f1R - u_gas*f2R
        epsDeltaV = 0.5*((eL*dV_L).sum(0) + (eR*dV_R).sum(0))
        V_L, V_R = dV_L - epsDeltaV, dV_R - epsDeltaV
        DeltaVphi = -0.5*u_gas*f1L + v_gas*(1 - f2L)

        return V_L, V_R, DeltaVphi

    def _edge_velocities(self, disc, a, w, v_visc=None):
        """Velocities of the quadrature nodes a with mass weights w ([Nq,
        Ncells]) of the cells at the edges, from the cells to the left and
        right (as _node_velocities)"""
        Sd = disc.Sigma*disc.dust_frac.sum(0)*w
        return self._node_velocities(disc, a[:, :-1], Sd[:, :-1], a[:, 1:], Sd[:, 1:], v_visc)

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

    def _moment_velocities(self, disc, left, right, v_visc=None):
        """Velocities of the moments at the edges, relative to the centre of
        mass, without settling, from the sides (Sd, s, L) to the left and
        right.

        A grain moves relative to the gas at
            dv = 2 v_gas St/(1+St^2) - u_gas St^2/(1+St^2),
        so M_k moves at the same expression with the two drift factors
        replaced by their a^k eps(a)-weighted means (mean_drift_factors). The
        feedback coefficients and v_COM - v_gas use the mass moment's (k = 0)
        means, so that the centre of mass moves with the viscous velocity
        (Booth et al. 2017).

        returns:
            V_L, V_R : velocities from the left / right sides, [3, Ncells-1]
        """
        SigmaG = disc.Sigma_G
        SigG_av = 0.5*(SigmaG[1:] + SigmaG[:-1])
        # Stokes number of the smallest grains at the edges
        St_min = disc.Stokes(SigG_av, self._f_drift*disc.amin)
        (F1L, F2L), (F1R, F2R) = [mean_drift_factors(s, L, St_min, disc._quad, self._ks)
                                  for _, s, L in (left, right)]
        m = self._ks.index(0)     # the mass moment

        if disc.feedback:
            # Fractions of the edge's total, the gas plus both sides' dust (as
            # _node_velocities)
            Sig_e = SigG_av + 0.5*(left[0] + right[0]) + 1e-300
            eL, eR = left[0]/Sig_e, right[0]/Sig_e
            eps_g = np.maximum(SigG_av/Sig_e, 1e-300)
        else:
            eL = eR = 0.
            eps_g = 1.
        # <1/(1+St^2)>_0 = 1 - <St^2/(1+St^2)>_0
        la0 = 0.5*(eL*(1 - F2L[m]) + eR*(1 - F2R[m]))/eps_g
        la1 = 0.5*(eL*F1L[m] + eR*F1R[m])/eps_g
        u_gas, v_gas = self._edge_gas_velocities(disc, la0, la1, eps_g, v_visc)

        V_L = 2*v_gas*F1L - u_gas*F2L
        V_R = 2*v_gas*F1R - u_gas*F2R
        epsDeltaV = 0.5*(eL*V_L[m] + eR*V_R[m])     # v_COM - v_gas
        return V_L - epsDeltaV, V_R - epsDeltaV

    def _side_velocities(self, disc, left, right, v_visc=None):
        """Velocities of the transported quantities at the interior edges,
        relative to the centre of mass, from the sides (Sd, s, L) to their
        left and right. The feedback (lambda), node velocities and v_COM -
        v_gas are all computed from the transported species, so that the
        centre of mass moves with the viscous velocity (Booth et al. 2017,
        eqs. 5-9).

        returns:
            V_L, V_R : velocities from the left / right sides, [3, Ncells-1]
        """
        if not self._settling:
            return self._moment_velocities(disc, left, right, v_visc)

        # With settling the feedback depends on each size's scale height:
        # average the node velocities
        amin = disc.amin
        nodes = []
        for Sd, s, L in (left, right):
            a, w = disc._quad(s, L, amin)
            nodes.append((a, w, Sd*w))
        (aL, wL, SdL), (aR, wR, SdR) = nodes
        V_nL, V_nR, _ = self._node_velocities(
            disc, self._f_drift*aL, SdL, self._f_drift*aR, SdR, v_visc)
        pL = np.array([wL*aL**k for k in self._ks])
        pR = np.array([wR*aR**k for k in self._ks])
        V_L = (pL*V_nL).sum(1) / (pL.sum(1) + 1e-300)
        V_R = (pR*V_nR).sum(1) / (pR.sum(1) + 1e-300)
        return V_L, V_R

    def _transport_velocities(self, disc, v_visc=None):
        """Velocities of the transported quantities at the edges from the
        cells to the left and right (as _side_velocities)"""
        left, right = self._cell_sides(disc)
        return self._side_velocities(disc, left, right, v_visc)

    def _state_key(self, disc, v_visc):
        """Copies of the arrays the transport velocities depend on"""
        key = [disc._eps_tot, disc._y, disc.Sigma, disc.cs]
        if self._settling:
            key.append(np.atleast_1d(disc.alpha))
        if v_visc is not None:
            key.append(v_visc)
        return [np.array(k, copy=True) for k in key]

    def _velocities(self, disc, v_visc=None):
        """Transported quantities and their velocities, (Q, V_L, V_R),
        reusing those from the last call if the disc state is
        unchanged (max_timestep followed by __call__)"""
        key = self._state_key(disc, v_visc)
        c = self._vcache
        if c is not None and c[0] is disc and len(c[1]) == len(key) and \
                all(np.array_equal(x, y) for x, y in zip(c[1], key)):
            return c[2]
        Q = self._transported(disc)
        out = (Q,) + tuple(self._transport_velocities(disc, v_visc))
        self._vcache = (disc, key, out)
        return out

    ############################################################################
    # Diffusion
    ############################################################################
    def diffusion_stokes(self, disc):
        """Stokes numbers with which diffuse() diffuses the transported
        quantities (pre_gas_transport) and the dust tracers.

        returns:
            St_Q [3, Ncells], St of the dust mass [Ncells]
        """
        # St is linear in a, so the mean of M_k is St(M_{k+1}/M_k), with
        # M_(j+1)/M_j = amin exprel((s+j+1) L)/exprel((s+j) L)
        _, s, L = disc.size_distribution()
        j = np.arange(min(self._ks), max(self._ks) + 2)[:, None]
        g = _ln_exprel((s + j)*L)
        a_k = disc.amin*np.exp(np.diff(g, axis=0))[np.array(self._ks) - j[0, 0]]
        St_Q = disc.Stokes(disc.Sigma_G, a_k)
        if not self._settling:
            return St_Q, St_Q[self._ks.index(0)]
        a, w = disc.quadrature()
        return St_Q, (w*disc.Stokes(disc.Sigma_G, a)).sum(0)

    def diffuse(self, dt, disc, diffusion, Q, gas_tracers=None, dust_tracers=None):
        """Diffuse the transported quantities Q (from pre_gas_transport), the
        dust tracers and the gas tracers over dt, in one step of the diffusion
        object (TracerDiffusion, or rkl.RKLDiffusion), with the Schmidt
        numbers of its dust_schmidt: each quantity at its mean Stokes number,
        the tracers as the dust mass (or area), the gas tracers as the gas"""
        St_Q, St_m = self.diffusion_stokes(disc)
        items = [(Q, diffusion.dust_schmidt(disc, St_Q))]
        if dust_tracers is not None:
            Sc = diffusion.dust_schmidt(disc, St_m)
            if self._area_tracers is not None:
                Sc_A = diffusion.dust_schmidt(disc, St_Q[self._ks.index(-1)])
                Sc = np.where(self._area_rows(dust_tracers)[:, None], Sc_A, Sc)
            items.append((dust_tracers, Sc))
        if gas_tracers is not None:
            items.append((gas_tracers, diffusion.gas_schmidt(disc)))
        diffusion.diffuse(dt, disc, items)

    def _area_rows(self, dust_tracers):
        """Mask of the dust-tracer rows carried by the grain area"""
        mask = np.zeros(len(dust_tracers), dtype=bool)
        mask[self._area_tracers] = True
        return mask

    ############################################################################
    # Fluxes
    ############################################################################
    # As SingleFluidDrift's (dust.py)
    _cell_faces = staticmethod(_cell_faces)
    _divergence = staticmethod(_divergence)

    @staticmethod
    def _flux_parts(Q_from_L, Q_from_R, V_L, V_R):
        """Upwind fluxes at all cell edges from the edge values, split into
        the parts carried from the left and from the right. V_L, V_R are the
        velocities at the interior edges; the boundary edges are outflow only,
        at the velocities of the nearest interior edges"""
        pos = np.zeros(Q_from_L.shape, dtype='f8')
        neg = np.zeros(Q_from_L.shape, dtype='f8')
        pos[..., 1:-1] = Q_from_L[..., 1:-1]*np.maximum(V_L, 0)
        neg[..., 1:-1] = Q_from_R[..., 1:-1]*np.minimum(V_R, 0)
        neg[...,  0] = Q_from_R[...,  0]*np.minimum(V_L[...,  0], 0)
        pos[..., -1] = Q_from_L[..., -1]*np.maximum(V_R[..., -1], 0)
        return pos, neg

    @classmethod
    def _flux(cls, Q_from_L, Q_from_R, V_L, V_R):
        """Upwind fluxes at all cell edges from the edge values"""
        pos, neg = cls._flux_parts(Q_from_L, Q_from_R, V_L, V_R)
        return pos + neg

    def _cell_state(self, disc, Q):
        """The cell values of the transported quantities that _advance needs
        at the start of the step, and their edge values from the cells
        (donor cell)"""
        return {'Q': Q}, {'Q': self._cell_faces(disc.Sigma*Q)}

    def _advance(self, dt, disc, state, faces, V_L, V_R):
        """The transported quantities after dt, and the parts of their upwind
        fluxes from the left and right"""
        pos, neg = self._flux_parts(*faces['Q'], V_L, V_R)
        return state['Q'] + dt*self._divergence(disc, pos + neg), pos, neg

    def _update(self, dt, disc, state, faces, V_L, V_R):
        """The state after a step of dt from state with the edge values faces
        and the velocities at the interior edges (V_L, V_R, relative to the
        centre of mass).

        state : dict of the cell values at the start of the step: those of
                _cell_state, 'G' (gas tracers, or None) and 'D' (dust tracers,
                or None)
        faces : dict of (from the left, from the right) edge values at all
                edges: those of _cell_state (or _face_quantities), 'G' (the
                gas tracers per unit gas mass, _gas_ratios) and 'X' (the dust
                tracers per unit of their carrier, _tracer_ratios)

        returns:
            Q, G, D : the transported quantities and tracers after the step
        """
        Q_new, pos, neg = self._advance(dt, disc, state, faces, V_L, V_R)
        G, D = state['G'], state['D']

        G_new = D_new = None
        # Gas tracers have the same distribution as the gas, which carries the
        # opposite of the dust's mass flux (the drift leaves Sigma unchanged;
        # without feedback the dust mass is neglected and the gas is at rest):
        # they are carried by it, at their ratio to the gas on its upwind side
        if G is not None:
            GL, GR = faces['G']
            F_dust = sum(self._dust_mass_fluxes(pos, neg))
            G_new = G + dt*self._divergence(disc, _gas_tracer_flux(disc, GL, GR, F_dust))

        # Dust tracers have the same distribution as their carrier (the dust
        # mass, or area): they are carried by its fluxes, at their ratio to it
        # on the upwind side
        if D is not None:
            XL, XR = faces['X']
            cpos, cneg = self._carrier_fluxes(pos, neg, D)
            D_new = D + dt*self._divergence(disc, cpos*XL + cneg*XR)
        return Q_new, G_new, D_new

    def _dust_mass(self, Q):
        """The dust mass per unit Sigma, from the transported quantities Q"""
        return Q[self._ks.index(0)]

    def _dust_mass_fluxes(self, pos, neg):
        """The parts of the dust mass flux from the left and right, from those
        of the transported quantities' fluxes"""
        m = self._ks.index(0)
        return pos[m], neg[m]

    def _gas_ratios(self, Q, G):
        """Gas tracers per unit gas mass (G per unit Sigma, Q the transported
        quantities); zero where there is no gas"""
        g = 1 - self._dust_mass(Q)
        ok = g > 0
        return np.where(ok, G/np.where(ok, g, 1), 0.)

    def _carrier(self, Q, D):
        """The carrier of each dust-tracer row, per unit Sigma (broadcast to
        D): the dust mass, or for area tracers M_-1"""
        C = np.broadcast_to(self._dust_mass(Q), D.shape)
        if self._area_tracers is not None:
            C = np.where(self._area_rows(D)[:, None], Q[self._ks.index(-1)], C)
        return C

    def _carrier_fluxes(self, pos, neg, D):
        """The parts of the fluxes of the dust tracers' carriers (_carrier)"""
        cpos, cneg = self._dust_mass_fluxes(pos, neg)
        if self._area_tracers is not None:
            k1 = self._ks.index(-1)
            area = self._area_rows(D)[:, None]
            cpos = np.where(area, pos[k1], cpos)
            cneg = np.where(area, neg[k1], cneg)
        return cpos, cneg

    def _tracer_ratios(self, Q, D):
        """Dust tracers per unit of their carrier, with Q the transported
        quantities (per unit Sigma); zero where there is none"""
        C = self._carrier(Q, D)
        ok = C > 0
        return np.where(ok, D/np.where(ok, C, 1), 0.)

    ############################################################################
    # The VL2 corrector's face states
    ############################################################################
    def _primitives(self, disc):
        """The reconstructed variables of the cells, (Sigma_d, s, L)"""
        eps_tot, s, L = disc.size_distribution()
        return np.array([disc.Sigma*eps_tot, s, L])

    @staticmethod
    def _dust_density(W):
        """Dust surface density of the primitives W"""
        return W[0]

    def _face_sides(self, disc, W):
        """Sides (Sd, s, L) of the face states W (_primitives) [3, ...]"""
        s_lo, s_hi = disc._s_bounds
        return (W[0], np.clip(W[1], s_lo, s_hi), np.maximum(W[2], disc._L_min))

    def _face_quantities(self, disc, W, W_cell, SQ_cell):
        """Sigma times the transported quantities of the face states W
        [3, Ncells+1], as a dict like the faces of _cell_state. W_cell and
        SQ_cell are the owning cells' primitives and Sigma Q: the face moments
        are those of the face state scaled by the cell's ratio of its stored
        moments to its power law's, so that a face state equal to the cell
        state carries its stored moments"""
        amin = disc.amin
        Sd, s, L = self._face_sides(disc, W)
        Sc, sc, Lc = self._face_sides(disc, W_cell)
        rc = Sc*self._moment_ratios(sc, Lc, amin)
        c = np.where(rc > 0, SQ_cell/np.where(rc > 0, rc, 1), 1.)
        return {'Q': Sd*self._moment_ratios(s, L, amin)*c}

    def _moment_ratios(self, s, L, amin):
        """M_k/M_0 of the power laws (s, L) for each k in _ks, [len(ks), ...]"""
        ks = np.array(self._ks)[:, None]
        return amin**ks*np.exp(_lnE(s + ks, L) - _lnE(s, L))

    def _vl_faces(self, disc, X):
        """van Leer face values of X [..., Ncells] at all edges, from the
        left and right (the end cells' slopes are zero)"""
        if self._vl is None or self._vl_grid is not disc.grid:
            self._vl = VanLeer(disc.grid.Ree, 1)
            self._vl_grid = disc.grid
        pad = np.concatenate([X[..., :1]]*2 + [X] + [X[..., -1:]]*2, axis=-1)
        return self._vl.face_values(pad)

    def _vl2_faces(self, disc, G=None, D=None, v_visc=None):
        """The corrector's edge values (as _update's faces) and velocities
        from the van Leer face states of the disc's (half-step) cells.
        Near-empty cells (below _empty_frac of the largest dust surface
        density), and their neighbours, take their cell values, since their
        shape is undefined.

        returns:
            faces, V_L, V_R
        """
        Sigma = disc.Sigma
        W = self._primitives(disc)
        Q = self._transported(disc)
        WL, WR = self._vl_faces(disc, W)
        # Near-empty cells: keep the cell values at their faces
        Sd = self._dust_density(W)
        empty = np.concatenate([[False], Sd < self._empty_frac*Sd.max(), [False]])
        donor = empty[1:-1] | empty[:-2] | empty[2:]
        CL, CR = self._cell_faces(W)
        dL, dR = self._cell_faces(donor)
        WL, WR = np.where(dL, CL, WL), np.where(dR, CR, WR)

        SQL, SQR = self._cell_faces(Sigma*Q)
        fL = self._face_quantities(disc, WL, CL, SQL)
        fR = self._face_quantities(disc, WR, CR, SQR)
        faces = {k: (fL[k], fR[k]) for k in fL}

        left = self._face_sides(disc, WL[:, 1:-1])
        right = self._face_sides(disc, WR[:, 1:-1])
        V_L, V_R = self._side_velocities(disc, left, right, v_visc)

        if G is not None:
            faces['G'] = self._vl_faces(disc, self._gas_ratios(Q, G))
        if D is not None:
            faces['X'] = self._vl_faces(disc, self._tracer_ratios(Q, D))
        return faces, V_L, V_R

    ############################################################################
    # Update
    ############################################################################
    def max_timestep(self, disc, v_visc=None):
        """Courant limited time-step"""
        Cou = 0.5
        Q, V_L, V_R = self._velocities(disc, v_visc)

        dVtot = np.zeros_like(Q)
        dVtot[..., :-1] += np.maximum(V_L, 0)
        dVtot[..., 1:] += np.maximum(-V_R, 0)
        dVtot[...,  0] += np.maximum(-V_L[...,  0], 0)
        dVtot[..., -1] += np.maximum(V_R[..., -1], 0)

        # Prevent empty cells limiting the time-step. Note integ_dust_frac is
        # zero without feedback, so use dust_frac directly.
        dVtot[:, disc.dust_frac.sum(0) < 1e-20] *= 1e-3
        dt = Cou * (disc.grid.dRe / (dVtot + 1e-300)).min()

        # The gas (tracers), which carries the opposite of the dust's mass flux
        if disc.feedback:
            pos, neg = self._flux_parts(*self._cell_faces(disc.Sigma*Q), V_L, V_R)
            dt = min(dt, _gas_timestep(disc, sum(self._dust_mass_fluxes(pos, neg)), Cou))

        # Resolve the growth of amax relative to the drift
        return min(dt, disc.growth_timestep())

    def pre_gas_transport(self, disc):
        """Quantities advected with the gas by the driver's gas step: the
        transported quantities Q (per unit Sigma). All sizes move with the
        gas in that step, so each is a passive tracer."""
        return self._transported(disc)

    def post_gas_transport(self, disc, Q):
        """Set the size distribution from the transported quantities after
        the gas (and diffusion) steps, with the same projection as after
        drift"""
        self._reconstruct(disc, Q)

    def __call__(self, dt, disc, gas_tracers=None, dust_tracers=None, v_visc=None):
        """Apply the update for radial drift over time-step dt: by Stone &
        Gardiner's (2009) van Leer integrator (van_leer=True), or donor cell.

        VL2: a donor-cell half step (with the velocities of max_timestep, from
        each edge's two cells), the size distribution refitted to the
        half-step quantities (without TriPoD's shrinkage), then the full step
        from the start with the fluxes of the van Leer face states of the
        half-step cells and their velocities. The half-step state sets the
        gas properties of the corrector too (Sigma_G and the pressure, with
        feedback). Tracers take the same steps."""
        Q, V_L, V_R = self._velocities(disc, v_visc)
        self._vcache = None      # the state changes below
        self._L_upwind = self._upwind_amax(disc, V_L, V_R)

        state, faces = self._cell_state(disc, Q)
        state['G'], state['D'] = gas_tracers, dust_tracers
        if gas_tracers is not None:
            faces['G'] = self._cell_faces(self._gas_ratios(Q, gas_tracers))
        if dust_tracers is not None:
            faces['X'] = self._cell_faces(self._tracer_ratios(Q, dust_tracers))

        if not self._van_leer:
            Q_new, G_new, D_new = self._update(dt, disc, state, faces, V_L, V_R)
        else:
            # Predictor: the half-step state
            Q_h, G_h, D_h = self._update(0.5*dt, disc, state, faces, V_L, V_R)
            n_clamped = self.n_clamped
            self._reconstruct(disc, Q_h, shrink=False)
            self.n_clamped = n_clamped
            # Corrector
            faces, V_L, V_R = self._vl2_faces(disc, G_h, D_h, v_visc)
            Q_new, G_new, D_new = self._update(dt, disc, state, faces, V_L, V_R)

        if gas_tracers is not None:
            gas_tracers[:] = G_new
        if dust_tracers is not None:
            dust_tracers[:] = D_new
        self._reconstruct(disc, Q_new)
        self._L_upwind = None

    def radial_drift_velocity(self, disc, v_visc=None, ret_vphi=False):
        """Mass-weighted radial drift velocity of each population, and
        optionally the azimuthal velocity"""
        a, w = disc.quadrature()
        pop = disc._quad.population[:,None]
        V_n, _, DeltaVphi = self._edge_velocities(disc, a, w, v_visc)
        wL = w[:, :-1]
        V = np.array([(wL*(pop == p)*V_n).sum(0) /
                      ((wL*(pop == p)).sum(0) + 1e-300) for p in range(2)])
        if ret_vphi:
            Vphi = np.array([(wL*(pop == p)*DeltaVphi).sum(0) /
                             ((wL*(pop == p)).sum(0) + 1e-300)
                             for p in range(2)])
            return V, Vphi
        return V


# Aliases for drop-in replacement of dust.py
DustGrowthTwoPop = DustGrowthTriMom
SingleFluidDrift = MomentDrift
