# test_dust_trimom.py
#
# Author: R. Booth
# Date: 5 - Oct - 2026
#
# Tests of the moment-consistent TriPoD dust model (dust_trimom.py)
###############################################################################
import numpy as np
import pytest
from scipy.integrate import solve_ivp, quad

from DiscEvolution.grid      import Grid
from DiscEvolution.constants import AU, Msun, sig_H2
from DiscEvolution.eos       import LocallyIsothermalEOS
from DiscEvolution.star      import SimpleStar
from DiscEvolution.diffusion import TracerDiffusion
from DiscEvolution.reconstruction import VanLeer
from DiscEvolution.FV_interpolation import compute_centroids

from DiscEvolution.dust_trimom import (
    DustGrowthTriMom, MomentDrift, SizeQuadrature, moment_ratio, mean_drift_factors,
    large_fraction, s_from_split, mean_size, vrel_sqd_OC07, _lnE,
    fit_moments)
from DiscEvolution.dust_tripod import make_drift, fit_mean_size


def _setup_disc(**kwargs):
    grid = Grid(1, 300, 120, spacing='log')
    star = SimpleStar()
    eos = LocallyIsothermalEOS(star, 1/30., -0.25, 1e-3)
    eos.set_grid(grid)
    Mdot = 1e-8*Msun/(2*np.pi)/AU**2
    Sigma = (Mdot/(3*np.pi*eos.nu))*np.exp(-grid.Rc/100.)
    eos.update(0, Sigma)
    return DustGrowthTriMom(grid, star, eos, 0.01, Sigma=Sigma, **kwargs)


def _zero_dust(disc, *cells):
    """Remove the dust from the cells (slices or index arrays)"""
    eps = disc.dust_frac.copy()
    for c in cells:
        eps[:, c] = 0
    disc.set_dust_frac(eps)

###############################################################################
# Power-law family
###############################################################################
@pytest.mark.parametrize("q", [-5., -4.5, -4., -3.5, -3., -2.5])
@pytest.mark.parametrize("k", [-1, 1, 2])
def test_moment_ratio(q, k):
    amin, amax = 1e-5, 3.
    f = lambda a: a**(q + 3)
    M0 = quad(f, amin, amax, points=np.logspace(-5, 0, 6), limit=200)[0]
    Mk = quad(lambda a: a**k*f(a), amin, amax, points=np.logspace(-5, 0, 6),
              limit=200)[0]
    s, L = q + 4, np.log(amax/amin)
    assert np.isclose(moment_ratio(k, s, L, amin), Mk/M0, rtol=1e-7)


def test_special_cases_continuous():
    """exprel removes the singularities at s+k = 0 and s = 0"""
    L = np.log(1e5)
    for s_c, k in [(1., -1), (0., 1), (-1., 1)]:
        s = s_c + np.array([-1e-9, 0., 1e-9])
        r = moment_ratio(k, s, L, 1e-5)
        assert np.all(np.isfinite(r))
        assert np.allclose(r, r[1], rtol=1e-7)


def test_population_split():
    s = np.array([-0.5, 0.5, 1.0])
    L = np.log(1e4)
    f1 = large_fraction(s, L)
    assert np.allclose(s_from_split(1 - f1, f1, L), s)
    # Mean sizes bracket the split point
    a0 = mean_size(s, 0, 0.5*L, 1e-5)
    a1 = mean_size(s, 0.5*L, L, 1e-5)
    aint = 1e-5*np.exp(0.5*L)
    assert np.all(a0 < aint) and np.all(a1 > aint)


@pytest.mark.parametrize("ks", [(-1, 1), (1, 2)])
def test_inversion_jacobian_single_signed(ks):
    """The map (q, amax) -> moment ratios is invertible"""
    s = np.linspace(-1.5, 2., 71)[:,None]
    L = np.log(np.logspace(0.05, 9, 80))[None,:]
    h = 1e-6
    def f(s, L):
        return [_lnE(s+k, L) - _lnE(s, L) for k in ks]
    fs = [(a - b)/(2*h) for a, b in zip(f(s+h, L), f(s-h, L))]
    fL = [(a - b)/(2*h) for a, b in zip(f(s, L+h), f(s, L-h))]
    det = fs[0]*fL[1] - fs[1]*fL[0]
    assert np.all(det > 0) or np.all(det < 0)


def test_quadrature():
    quadr = SizeQuadrature(n_panels=8)
    amin = 1e-5
    s = np.array([-1., 0., 0.5, 1., 2.])
    L = np.log(np.array([10., 1e3, 1e5, 1e7, 1e9]))
    a, w = quadr(s, L, amin)
    assert np.allclose(w.sum(0), 1, rtol=1e-9)
    for k in (-1, 1, 2, 3):
        assert np.allclose((w*a**k).sum(0), moment_ratio(k, s, L, amin),
                           rtol=1e-6)
    # The two populations have the masses given by large_fraction
    pop = quadr.population[:,None]
    assert np.allclose((w*(pop == 1)).sum(0), large_fraction(s, L),
                       rtol=1e-9)
    # The default (24 nodes) up to amax/amin = 1e5
    a, w = SizeQuadrature()(s[:3], L[:3], amin)
    for k in (-1, 0, 1, 2):
        assert np.allclose((w*a**k).sum(0), moment_ratio(k, s[:3], L[:3], amin),
                           rtol=1e-5)

###############################################################################
# Relative velocities
###############################################################################
def _vrel_OC07_dustpy(St1, St2, Re):
    """Squared turbulent relative velocity / (alpha cs^2), ported from
    DustPy's vrel_ormel_cuzzi_2007 (piecewise form), for comparison."""
    c0, c1, c2, c3 = 1.6015125, -0.63119577, 0.32938936, -0.29847604
    ya = 1.6
    yap1inv = 1/(1 + ya)
    ReInvSqrt = np.sqrt(1/Re)
    ts = ReInvSqrt          # in units of 1/Omega
    vs2 = Re**-0.5          # (Re^-1/4 vn)^2 / vn^2
    vg2 = 1.5
    StL, StS = max(St1, St2), min(St1, St2)
    eps = StS/StL
    ys = c0 + c1*StL + c2*StL**2 + c3*StL**3
    h1 = (StL - StS)/(StL + StS)*(StL*yap1inv - StS**2/(StS + ya*StL))
    h2 = (2*(ya*StL - ReInvSqrt) + StL*yap1inv - StL**2/(StL + ReInvSqrt)
          + StS**2/(ya*StL + StS) - StS**2/(StS + ReInvSqrt))
    if StL < 0.2*ts:
        return 1.5*(vs2/ts**2)*(StL - StS)**2
    elif StL*ya < ts:
        return vg2*(StL - StS)/(StL + StS)*(StL**2/(StL + ReInvSqrt)
                                            - StS**2/(StS + ReInvSqrt))
    elif StL < 5*ts:
        return vg2*(h1 + h2)
    elif StL < 0.2:
        return vg2*StL*(2*ya - 1 - eps + 2/(1 + eps)*(yap1inv + eps**3/(ya + eps)))
    elif StL < 1:
        return vg2*StL*(2*ys - 1 - eps + 2/(1 + eps)*(1/(1 + ys) + eps**3/(ys + eps)))
    return vg2*(2 + StL + StS)/(1 + StL + StS + StL*StS)


@pytest.mark.parametrize("St1", [1e-5, 3e-4, 1e-3, 3e-2, 0.1, 0.5, 0.9, 3.])
@pytest.mark.parametrize("ratio", [0.4, 0.1])
def test_OC07_vs_dustpy(St1, ratio):
    """cuDisc's closed-form OC07 agrees with DustPy's piecewise version.

    cuDisc normalizes to V_g^2 = alpha cs^2, DustPy to V_g^2 = 1.5 alpha cs^2.
    After accounting for this, the two approximations to OC07 agree to ~10%
    in dv (they are identical where both are exact).
    """
    Re = 1e8
    St2 = ratio*St1
    v_cu = 1.5*vrel_sqd_OC07(np.array(St1), np.array(St2), Re**-0.5)
    v_dp = _vrel_OC07_dustpy(St1, St2, Re)
    assert np.isclose(np.sqrt(v_cu), np.sqrt(v_dp), rtol=0.12)
    if St1 < 5e-4 and St1 > 1e-4 or St1 > 1:
        assert np.isclose(v_cu, v_dp, rtol=1e-4)

###############################################################################
# Growth
###############################################################################
@pytest.mark.parametrize("z0", [30., 3., 0.3, -0.3, -3., -30.])
@pytest.mark.parametrize("x", [0.1, 1., 10., 100.])
def test_grow_amax_vs_ode(z0, x):
    """Analytic growth step against direct integration of TriPoD's eq. 29
    with dv ∝ a^(1/2) and fixed t_grow"""
    n, t_grow, a0 = 1.5, 2.0, 1e-3
    def rhs(t, y):
        z = z0 - n*y[0]
        return [np.tanh(0.5*z)/t_grow]
    sol = solve_ivp(rhs, (0, x*t_grow), [0.], rtol=1e-11, atol=1e-13)
    a_ode = a0*np.exp(sol.y[0, -1])
    a = DustGrowthTriMom.grow_amax(a0, t_grow, z0, n, x*t_grow)
    assert np.isclose(a, a_ode, rtol=1e-7)


def test_grow_amax_limits():
    a0, t_grow, n = 1e-3, 1., 1.5
    # Small-grain limit: exponential growth
    a = DustGrowthTriMom.grow_amax(a0, t_grow, 200., n, 3.)
    assert np.isclose(a, a0*np.exp(3.), rtol=1e-10)
    # Long times: relaxes to the fragmentation limit z = 0 without overshoot
    for z0 in (5., -5.):
        a = DustGrowthTriMom.grow_amax(a0, t_grow, z0, n, 1e4)
        assert np.isclose(np.log(a/a0), z0/n, rtol=1e-10)
    # n -> 0: constant rate
    a = DustGrowthTriMom.grow_amax(a0, t_grow, 1., 0., 2.)
    assert np.isclose(a, a0*np.exp(np.tanh(0.5)*2.), rtol=1e-8)


def test_exchange_fraction_vs_ode():
    f0, A, B, dt = 0.2, 3., 5., 0.7
    sol = solve_ivp(lambda t, f: A*f - B*f*f, (0, dt), [f0], rtol=1e-12,
                    atol=1e-14)
    f = DustGrowthTriMom.exchange_fraction(f0, A, B, dt)
    assert np.isclose(f, sol.y[0, -1], rtol=1e-8)
    # A = 0
    f = DustGrowthTriMom.exchange_fraction(f0, 0., B, dt)
    assert np.isclose(f, f0/(1 + B*f0*dt))


def test_growth_conserves_mass():
    disc = _setup_disc()
    M0 = disc.Mdust()
    disc.update(1e4*2*np.pi)
    assert np.isclose(disc.Mdust(), M0, rtol=1e-12)
    assert np.all(disc.amax >= 1.5*disc.amin)

def test_grain_cross_section():
    disc = _setup_disc()
    disc.update(1e4*2*np.pi)
    eps = disc.dust_frac.sum(0)
    assert np.allclose(4*disc.grain_cross_section*eps,
                       disc.surface_area*(1 - eps), rtol=1e-12)
    # Grown dust has less area than grains of amin
    assert np.all(disc.grain_cross_section < 3/(4*disc.amin))

###############################################################################
# Transport
###############################################################################
def _step(drift, disc, diffusion):
    """A drift step, then the diffusion step (as DiscEvolutionDriver)"""
    dt = min(drift.max_timestep(disc), diffusion.max_timestep(disc))
    drift(dt, disc)
    Q = drift.pre_gas_transport(disc)
    drift.diffuse(dt, disc, diffusion, Q)
    drift.post_gas_transport(disc, Q)


@pytest.mark.parametrize("scheme,ks", [('moments', (-1, 0, 1)),
                                       ('tripod', (0,)),
                                       ('trimean', (0, 1)),
                                       ('trimax', (0,))])
@pytest.mark.parametrize("van_leer", [False, True])
def test_transport_conservation(scheme, ks, van_leer):
    """Mass (and, for the moment schemes, the transported moments) are
    conserved up to the boundary fluxes"""
    disc = _setup_disc(a0=0.1)
    # Remove the dust near the boundaries so there is no boundary flux
    _zero_dust(disc, slice(None, 15), slice(-15, None))
    drift = make_drift(scheme=scheme, van_leer=van_leer)
    D = TracerDiffusion()

    Re = disc.grid.Re*AU
    dA = np.pi*np.diff(Re**2)
    def totals():
        m = disc.moments((-1, 0, 1, 2))
        return (m*disc.Sigma*dA).sum(1)

    T0 = totals()
    for _ in range(20):
        _step(drift, disc, D)
    T1 = totals()

    assert np.isclose(T1[1], T0[1], rtol=1e-10)
    if scheme != 'tripod':
        for k in ks:
            i = (-1, 0, 1, 2).index(k)
            assert np.isclose(T1[i], T0[i], rtol=1e-8)
    if scheme == 'trimean':
        assert drift.n_clamped == 0


@pytest.mark.parametrize("scheme,ks", [('moments', (-1, 0, 1)),
                                       ('tripod', (0,)),
                                       ('trimean', (0,)),
                                       ('trimax', (0,))])
def test_gas_transport_conservation(scheme, ks):
    """Advection with the gas via pre/post_gas_transport conserves mass and,
    for the moment scheme, the transported moments up to the boundary fluxes"""
    from DiscEvolution.viscous_evolution import ViscousEvolutionFV
    disc = _setup_disc(a0=0.1)
    # A non-uniform amax, so that cells with different distributions mix
    eps_tot, s, L = disc.size_distribution()
    L = L*(1 + 0.5*np.sin(np.log(disc.R)*3))
    disc.set_size_distribution(eps_tot, s, L)
    _zero_dust(disc, slice(None, 15), slice(-15, None))
    drift = make_drift(scheme=scheme, **({'shrink': False} if scheme == 'tripod' else {}))
    gas = ViscousEvolutionFV()

    Re = disc.grid.Re*AU
    dA = np.pi*np.diff(Re**2)
    def totals():
        m = disc.moments((-1, 0, 1))
        return (m*disc.Sigma*dA).sum(1)

    T0 = totals()
    amax0 = disc.amax.copy()
    for _ in range(20):
        dt = gas.max_timestep(disc)
        Q = drift.pre_gas_transport(disc)
        gas(dt, disc, [], [Q])
        drift.post_gas_transport(disc, Q)
    T1 = totals()

    assert np.isclose(T1[1], T0[1], rtol=1e-10)
    if scheme == 'moments':
        assert drift.n_clamped == 0
        np.testing.assert_allclose(T1, T0, rtol=1e-8)
    if scheme == 'trimean':
        assert drift.n_clamped == 0
        assert np.isclose(T1[2], T0[2], rtol=1e-8)
    # amax has moved with the gas
    assert not np.allclose(disc.amax, amax0, rtol=1e-6)


@pytest.mark.parametrize("van_leer", [False, True])
def test_trimax_amax_bounded(van_leer):
    """trimax advects amax along its characteristics with v(amax): drift
    alone keeps it within the range of its initial values, and moves it"""
    disc = _setup_disc(a0=0.1)
    eps_tot, s, L = disc.size_distribution()
    L = L*(1 + 0.5*np.sin(np.log(disc.R)*3))
    disc.set_size_distribution(eps_tot, s, L)
    drift = make_drift(scheme='trimax', van_leer=van_leer)
    amax0 = disc.amax.copy()
    for _ in range(20):
        drift(drift.max_timestep(disc), disc)
    assert disc.amax.min() >= amax0.min()*(1 - 1e-12)
    assert disc.amax.max() <= amax0.max()*(1 + 1e-12)
    assert not np.allclose(disc.amax, amax0, rtol=1e-6)


def test_trimax_amax_velocity():
    """trimax's third velocity is that of amax itself"""
    disc = _setup_disc(a0=0.1)
    drift = make_drift(scheme='trimax')
    _, s, L = disc.size_distribution()
    a_rep = drift._sizes(s, L, disc.amin)
    np.testing.assert_allclose(a_rep[2], disc.amax, rtol=1e-14)
    assert np.all(a_rep[1] < disc.amax)


def test_gas_transport_uniform_distribution():
    """A spatially uniform size distribution is unchanged by the gas step"""
    from DiscEvolution.viscous_evolution import ViscousEvolutionFV
    disc = _setup_disc(a0=0.1)
    drift = MomentDrift()
    gas = ViscousEvolutionFV()
    _, s0, L0 = disc.size_distribution()
    M0 = disc.moments()

    for _ in range(5):
        dt = gas.max_timestep(disc)
        Q = drift.pre_gas_transport(disc)
        gas(dt, disc, [], [Q])
        drift.post_gas_transport(disc, Q)

    # The scaled moments (the state) are unchanged; the power law is
    # recovered to the accuracy of the moment table
    M = disc.moments()
    np.testing.assert_allclose(M/M[1], M0/M0[1], rtol=1e-10)
    _, s, L = disc.size_distribution()
    np.testing.assert_allclose(s, s0, atol=1e-4)
    np.testing.assert_allclose(L, L0, rtol=1e-4)


def test_moment_velocity_small_St():
    """For St << 1 the flux of M_k is carried by M_{k+1}: <v>_k ∝ M_{k+1}/M_k"""
    disc = _setup_disc(a0=1e-3, feedback=False)
    drift = make_drift(scheme='moments')
    V_L, _ = drift._transport_velocities(disc)
    m = disc.moments((-1, 0, 1, 2))
    ratio_num = V_L[1]/V_L[0]
    ratio_ana = (m[2]/m[1] / (m[1]/m[0]))[:-1]
    assert np.allclose(ratio_num[10:-10], ratio_ana[10:-10], rtol=1e-3)


def test_trimean_velocity_small_St():
    """For St << 1 the trimean fluxes of the mass and of M_1 are exact:
    proportional to M_1 and M_2"""
    disc = _setup_disc(a0=1e-3, feedback=False)
    drift = make_drift(scheme='trimean')
    Q = drift._transported(disc)
    V_L, _ = drift._transport_velocities(disc)
    m = disc.moments((0, 1, 2))
    F_mass = Q[0][:-1]*V_L[0] + Q[1][:-1]*V_L[1]
    np.testing.assert_allclose(Q[2], m[1], rtol=1e-12)
    ratio_num = Q[2][:-1]*V_L[2]/F_mass
    ratio_ana = (m[2]/m[1])[:-1]
    assert np.allclose(ratio_num[10:-10], ratio_ana[10:-10], rtol=1e-3)


def test_fit_mean_size():
    """fit_mean_size recovers (s, L) from the population masses and the mean
    size, including s ~ -1 (the Lambert W branch point)"""
    rng = np.random.default_rng(3)
    n = 4000
    s = np.concatenate([rng.uniform(-8, 4, n), -1 + rng.uniform(-1, 1, n)*1e-3])
    L = rng.uniform(1e-2, 25, 2*n)
    lnX = _lnE(s+1, L) - _lnE(s, L)
    s_fit, L_fit = fit_mean_size(large_fraction(-s, L), large_fraction(s, L),
                                 lnX)
    np.testing.assert_allclose(_lnE(s_fit+1, L_fit) - _lnE(s_fit, L_fit),
                               lnX, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(L_fit, L, rtol=1e-10)
    np.testing.assert_allclose(s_fit, s, rtol=1e-10, atol=1e-12)


def test_trimean_reconstruct_roundtrip():
    """Refitting the trimean variables of a disc leaves it unchanged"""
    disc = _setup_disc(a0=0.1)
    eps_tot, s, L = disc.size_distribution()
    disc.set_size_distribution(eps_tot, s + 0.3*np.sin(np.log(disc.R)*3),
                               L*(1 + 0.5*np.sin(np.log(disc.R)*2)))
    drift = make_drift(scheme='trimean')
    Q = drift._transported(disc)
    eps0, amax0 = disc.dust_frac.copy(), disc.amax.copy()
    drift._reconstruct(disc, Q.copy())
    np.testing.assert_allclose(disc.dust_frac, eps0, rtol=1e-12)
    np.testing.assert_allclose(disc.amax, amax0, rtol=1e-9)
    np.testing.assert_allclose(drift._transported(disc), Q, rtol=1e-10)


def test_van_leer_linear_exact():
    """The corrected van Leer reconstruction is exact for linear data on a
    non-uniform grid"""
    xe = np.logspace(0, 2, 41)
    for m in (0, 1):
        xc = compute_centroids(xe, m)
        Q = 2 + 3*xc
        Qf = VanLeer(xe, m)(np.ones(len(xe) - 2), Q)
        assert np.allclose(Qf, 2 + 3*xe[2:-2], rtol=1e-13)


@pytest.mark.parametrize("feedback", [True, False])
@pytest.mark.parametrize("scheme,ks", [('moments', (-1, 0, 1)),
                                       ('tripod', (0,)),
                                       ('trimean', (0,)),
                                       ('trimax', (0,))])
def test_courant_limit(feedback, scheme, ks):
    """max_timestep keeps the outflow Courant number of every transported
    quantity <= 0.5 (regression: integ_dust_frac is zero without feedback)"""
    disc = _setup_disc(a0=1.0, feedback=feedback)
    drift = make_drift(scheme=scheme)
    Q = drift._transported(disc)
    V_L, V_R = drift._transport_velocities(disc)
    dt = drift.max_timestep(disc)
    out = np.zeros_like(Q)
    out[..., :-1] += np.maximum(V_L, 0)
    out[..., 1:] += np.maximum(-V_R, 0)
    assert (dt*out/disc.grid.dRe).max() <= 0.5 + 1e-12


@pytest.mark.parametrize("beta_true", [0.5, 0.75, 1.0])
@pytest.mark.parametrize("side", [1, -1])
def test_equilibrium_band_converges_monotonically(beta_true, side):
    """Within the equilibrium band, repeated unlimited steps with beta = 1
    converge monotonically to equilibrium for any beta_true in [1/2, 1]
    (with beta = 1/2, beta_true = 1 would oscillate without damping).

    The band, |z0| <= (s/2) ln 2, is a factor 2^(1/(2 beta_true)) in size.
    """
    s, a_eq, t_grow = 3., 1.0, 1.0
    a = a_eq*2**(side*0.99/(2*beta_true))
    err = [np.log(a/a_eq)]
    for _ in range(30):
        z0 = s*beta_true*np.log(a_eq/a)
        assert abs(z0) <= 0.5*s*np.log(2) + 1e-12   # stays in the band
        a = DustGrowthTriMom.grow_amax(a, t_grow, z0, s*1.0, 1e6)
        err.append(np.log(a/a_eq))
    err = np.array(err)
    assert np.all(err*np.sign(err[0]) >= 0)   # never overshoots
    assert np.all(np.abs(np.diff(np.abs(err))) >= 0)
    assert abs(err[-1]) < 1e-8


@pytest.mark.parametrize("fac", [1., 1 + 1e-12])
@pytest.mark.parametrize("x", [0.55, 0.7, 0.8, 1.0, 1.3])
def test_limit_on_steep_transition(x, fac):
    """With the fragmentation limit on or just above the steep rise of dv_max
    at OC07's small-eddy transition (St = x Re^-1/2), stepping at
    growth_timestep settles onto it in a few hundred steps. The index of
    relative_velocity_index follows the rise; a regime-wise index (1, 1/2, -1/2)
    made amax chatter across the limit (5000 steps reached only 4e4 yr at
    x = 0.7). fac > 1 leaves a tiny second sub-step in each call, which must
    not change this."""
    disc = _setup_disc()
    g = disc._collision_gas()
    a_t = x*g['sqrtRe_inv']/g['K']
    disc._uf = disc.relative_velocity(a_t, disc._f_dv*a_t, gas=g)
    t, n, T = 0., 0, 1e5*2*np.pi
    while t < T and n < 1000:
        h = min(disc.growth_timestep()*fac, T - t)
        disc.do_grain_growth(h)
        t, n = t + h, n + 1
    assert t >= T and n < 400


def test_size_dependent_frag_velocity():
    """With v_frag = uf (amax/a0)^p (frag_slope p = -0.5, frag_size a0 = 1 mm),
    cells settle where dv_max(amax) = v_frag(amax): above the size-independent
    limit where that lies below a0 (v_frag larger there), below it above a0"""
    def settle(**kwargs):
        disc = _setup_disc(**kwargs)
        t, n, T = 0., 0, 1e5*2*np.pi
        while t < T and n < 2000:
            h = min(disc.growth_timestep(), T - t)
            disc.do_grain_growth(h)
            t, n = t + h, n + 1
        assert t >= T
        return disc, disc._growth_rates()['z0']
    disc, z = settle(frag_slope=-0.5, frag_size=0.1)
    assert np.allclose(disc._v_frag(disc.amax), disc._uf*(disc.amax/0.1)**-0.5, rtol=1e-12)
    disc0, z0 = settle()
    at_limit = (np.abs(z) < 1e-6) & (np.abs(z0) < 1e-6)
    assert at_limit.sum() > 10
    a, a0 = disc.amax[at_limit], disc0.amax[at_limit]
    assert ((a > a0) == (a0 < 0.1)).all()


def test_local_beta_is_bounded():
    """The weighted index lies within the range of its contributions'
    indices, [-3/2, 1] (unclipped)"""
    disc = _setup_disc()
    r = disc._growth_rates()
    assert np.all(r['beta'] >= -1.5 - 1e-12) and np.all(r['beta'] <= 1 + 1e-12)


@pytest.mark.parametrize("feedback", [True, False])
def test_relative_velocity_index(feedback):
    """The weighted index matches d ln dv / d ln a, including above the
    small-eddy transition of OC07 (dv steepens sharply just above St* =
    Re^-1/2, St ~ 0.62 Re^-1/2), except in the kink's immediate vicinity."""
    disc = _setup_disc(feedback=feedback)
    sri = disc._sqrtRe_inv()
    h = 1e-4
    for la in np.linspace(-4.5, 1, 23):
        a = 10**la*np.ones(disc.Ncells)
        dv, beta = disc.relative_velocity_index(a, 0.4*a)
        assert np.allclose(dv, disc.relative_velocity(a, 0.4*a), rtol=1e-12)
        bn = (np.log(disc.relative_velocity(a*np.exp(h), 0.4*a*np.exp(h))) -
              np.log(disc.relative_velocity(a*np.exp(-h), 0.4*a*np.exp(-h))))/(2*h)
        St = disc.Stokes(disc.Sigma_G, a)
        ok = (St < 0.5*sri) | (St > 0.9*sri)
        assert np.all(np.abs(beta - bn)[ok] < 0.25)
        # The drift components, as relative_velocity's
        _, _, dr, da = disc.relative_velocity_index(a, 0.4*a, components=True)
        _, dr0, da0 = disc.relative_velocity(a, 0.4*a, components=True)
        assert np.allclose(dr, dr0, rtol=1e-12, atol=0) and np.allclose(da, da0, rtol=1e-12, atol=0)


def test_vrel_sqd_OC07_factorised():
    """The factorised V1, V2 of vrel_sqd_OC07 against OC07's form"""
    from DiscEvolution.dust_trimom import vrel_sqd_OC07
    a_, b_ = 0.51505382481121010, 0.12876345620280252
    rng = np.random.default_rng(1)
    St1 = 10**rng.uniform(-8, 3, 20000)
    St2 = St1*rng.uniform(0, 1, St1.size)
    sri = 10**rng.uniform(-5, -1, St1.size)
    St_l = np.maximum(St1, St2)
    St_c = np.minimum(St_l, 1)
    Sts = np.maximum(np.where(St_l > 1, 1., St_c*(1 - b_ + np.sqrt(b_*b_ + a_*(1-St_c)/(1+St_c)))), sri)
    V1 = lambda St: St * (St/(Sts + St) - St/(1 + St))
    V2 = lambda St: (Sts - sri) + St*St/(St + Sts) - St*St/(St + sri)
    ref = np.maximum(((St1 - St2)/(St1 + St2))*(V1(St1) - V1(St2)) + V2(St1) + V2(St2), 0)
    assert np.allclose(vrel_sqd_OC07(St1, St2, sri), ref, rtol=1e-9, atol=0)


###############################################################################
# Dust removal with the gas (photoevaporative entrainment)
###############################################################################
def _trimom(a0=0.1, q=-3.5):
    return _setup_disc(a0=a0, distribution_slope=-q)


def test_trimom_removal_without_entrainment_keeps_dust():
    disc = _trimom()
    Sigma_G0, Sigma_D0 = disc.Sigma_G.copy(), disc.Sigma_D.copy()
    dSigma_D = disc.remove_dust_with_gas(0.5*Sigma_G0, None)
    assert (dSigma_D == 0).all()
    np.testing.assert_allclose(disc.Sigma_G, 0.5*Sigma_G0, rtol=1e-12)
    np.testing.assert_allclose(disc.Sigma_D, Sigma_D0, rtol=1e-12)


@pytest.mark.parametrize("f_g", [0.1, 1.0])
def test_trimom_removal_moments(f_g):
    """The remaining moments are those of the power law with the part below
    a_cut reduced by f_g, and area is lost faster than mass"""
    disc = _trimom()
    amin = disc.amin
    eps_tot, s, L = disc.size_distribution()
    Sigma0 = disc.Sigma.copy()
    Sigma_G0 = disc.Sigma_G.copy()
    ks = (-1, 0, 1)
    M0 = np.array([eps_tot*moment_ratio(k, s, L, amin) for k in ks])*Sigma0

    a_cut = 1e-4
    Lc = np.log(a_cut/amin)
    frac = np.array([np.exp(_lnE(s+k, Lc) - _lnE(s+k, L)) for k in ks])
    expected = M0*(1 - f_g*frac)

    dSigma_D = disc.remove_dust_with_gas(f_g*Sigma_G0, a_cut)

    M1 = disc.moments(ks)*disc.Sigma
    assert disc.n_removal_clamped == 0
    np.testing.assert_allclose(M1, expected, rtol=1e-8)
    np.testing.assert_allclose(dSigma_D, M0[1] - M1[1], rtol=1e-8)
    np.testing.assert_allclose(disc.Sigma_G, (1 - f_g)*Sigma_G0, rtol=1e-8,
                               atol=1e-12*Sigma_G0.max())
    # Fraction of area removed exceeds the fraction of mass removed
    assert ((1 - M1[0]/M0[0]) > 10*(1 - M1[1]/M0[1])).all()


def test_trimom_removal_large_cut_keeps_shape():
    disc = _trimom()
    _, s0, L0 = disc.size_distribution()
    f1 = disc.dust_frac[1]/disc.dust_frac.sum(0)
    Sigma_D0 = disc.Sigma_D.sum(0)
    dSigma_D = disc.remove_dust_with_gas(0.4*disc.Sigma_G, 10.)
    np.testing.assert_allclose(dSigma_D, 0.4*Sigma_D0, rtol=1e-12)
    np.testing.assert_allclose(disc.dust_frac[1]/disc.dust_frac.sum(0), f1,
                               rtol=1e-12)
    np.testing.assert_allclose(disc.amax, np.exp(L0)*disc.amin, rtol=1e-12)


def test_trimom_removal_efficiency_unity_matches_step():
    """A size-dependent efficiency of 1 reproduces the step-only result to
    quadrature accuracy"""
    d1, d2 = _trimom(), _trimom()
    dS = 0.5*d1.Sigma_G
    r1 = d1.remove_dust_with_gas(dS, 1e-4)
    r2 = d2.remove_dust_with_gas(dS, 1e-4, efficiency=lambda a: np.ones_like(a))
    np.testing.assert_allclose(r2, r1, rtol=1e-8)
    np.testing.assert_allclose(d2.moments(), d1.moments(), rtol=1e-6)


def test_trimom_removal_efficiency_reduces_loss():
    d1, d2 = _trimom(), _trimom()
    dS = 0.5*d1.Sigma_G
    r1 = d1.remove_dust_with_gas(dS, 1e-4)
    r2 = d2.remove_dust_with_gas(dS, 1e-4,
                                 efficiency=lambda a: np.where(a < 1e-5*3, 1., 0.))
    assert (r2 < r1).all() and (r2 > 0).all()


###############################################################################
# Moment fit, realizability, bounds and restart
###############################################################################
KS = (-1, 0, 1)
S_BOUNDS = (-1., 3.)
L_MIN = np.log(1.5)


def _moments(eps, s, L, amin=1e-5):
    return np.array([eps*moment_ratio(k, s, L, amin) for k in KS])


def test_fit_moments_roundtrip():
    """The fit recovers parameters inside the realizable range"""
    s_true = np.array([-0.5, 0., 0.5, 1., 2., 2.5])
    L_true = np.array([8., 5., 12., 3., 10., 1.5])
    eps = np.full(len(s_true), 0.01)
    Q = _moments(eps, s_true, L_true)
    # Start from a different guess
    e, s, L, bad = fit_moments(Q, 1e-5, np.full_like(s_true, 0.5),
                               np.full_like(s_true, 4.), S_BOUNDS, L_MIN)
    assert not bad.any()
    np.testing.assert_allclose(e, eps, rtol=1e-12)
    np.testing.assert_allclose(s, s_true, atol=1e-7)
    np.testing.assert_allclose(L, L_true, rtol=1e-7)


def test_fit_moments_unrealizable_fallback():
    """Moments outside the realizable range (s beyond its bounds) are
    flagged, give finite parameters inside the bounds, and keep the mass and
    mass-flux (k=1) moments."""
    eps = np.full(2, 0.01)
    # Generated with s outside the bounds, on either side
    s_out, L_out = np.array([-2.5, 4.5]), np.array([8., 8.])
    Q = _moments(eps, s_out, L_out)
    e, s, L, bad = fit_moments(Q, 1e-5, np.full(2, 0.5), np.full(2, 5.),
                               S_BOUNDS, L_MIN)
    assert bad.all()
    assert np.all(np.isfinite(s)) and np.all(np.isfinite(L))
    assert np.all((s >= S_BOUNDS[0]) & (s <= S_BOUNDS[1]))
    assert np.all(L >= L_MIN)
    Qf = _moments(e, s, L)
    np.testing.assert_allclose(Qf[1], Q[1], rtol=1e-12)
    np.testing.assert_allclose(Qf[2], Q[2], rtol=1e-6)


def test_fit_moments_empty_cells_keep_state():
    """Cells without dust return the starting parameters"""
    Q = np.zeros((3, 4))
    s0, L0 = np.full(4, 0.7), np.full(4, 6.)
    e, s, L, bad = fit_moments(Q, 1e-5, s0, L0, S_BOUNDS, L_MIN)
    assert not bad.any()
    np.testing.assert_array_equal(e, 0)
    np.testing.assert_array_equal(s, s0)
    np.testing.assert_array_equal(L, L0)


###############################################################################
# Moment state
###############################################################################
def test_moment_table_inverts_and_projects():
    """The table recovers the power law to ~1e-4 in the moments, and moves
    unrealizable moments onto the edge of the family, keeping M_1/M_0"""
    from DiscEvolution.dust_trimom import MomentTable, moment_coords
    tab = MomentTable.get(S_BOUNDS, L_MIN)
    rng = np.random.default_rng(1)
    s = rng.uniform(*S_BOUNDS, 2000)
    L = np.exp(rng.uniform(np.log(L_MIN), np.log(12.), 2000))
    X, D = moment_coords(s, L)
    Xp, Dp, s2, L2, moved = tab(X, D)
    assert not moved.any()
    np.testing.assert_array_equal([Xp, Dp], [X, D])
    X2, D2 = moment_coords(s2, L2)
    assert np.abs(X2 - X).max() < 5e-4
    assert np.abs((D2 - X2) - (D - X)).max() < 5e-4

    # Too wide (s below its bound) and too narrow (above it)
    Xo, Do = moment_coords(np.array([-2.5, 4.5]), np.array([8., 8.]))
    Xp, Dp, s2, L2, moved = tab(Xo, Do)
    assert moved.all()
    np.testing.assert_allclose(Xp, Xo, rtol=1e-12)
    np.testing.assert_allclose(s2, S_BOUNDS, atol=1e-3)
    X2, D2 = moment_coords(s2, L2)
    np.testing.assert_allclose([X2, D2], [Xp, Dp], atol=5e-4)


def test_nested_moment_table():
    """With q_min below -5 the q >= -5 states keep the default table's
    inversion exactly, steeper states are inverted by the wide table, and a
    disc with such bounds uses the nested table"""
    from DiscEvolution.dust_trimom import (MomentTable, NestedMomentTable,
                                           moment_coords)
    s_bounds = (-6., S_BOUNDS[1])                       # q >= -10
    nested = NestedMomentTable(s_bounds, L_MIN)
    inner = MomentTable.get(S_BOUNDS, L_MIN)
    rng = np.random.default_rng(2)
    s = rng.uniform(*S_BOUNDS, 2000)
    L = np.exp(rng.uniform(np.log(L_MIN), np.log(7.), 2000))
    X, D = moment_coords(s, L)
    for a, b in zip(nested(X, D), inner(X, D)):
        np.testing.assert_array_equal(a, b)

    # Steep shapes, outside the default box
    s = rng.uniform(-6., -1.05, 2000)
    L = np.exp(rng.uniform(np.log(1.), np.log(7.), 2000))
    X, D = moment_coords(s, L)
    Xp, Dp, s2, L2, moved = nested(X, D)
    assert moved.sum() < 0.01*len(s)
    X2, D2 = moment_coords(s2, L2)
    assert np.abs(X2 - X)[~moved].max() < 5e-3
    assert np.abs((D2 - X2) - (D - X))[~moved].max() < 5e-3

    disc = _setup_disc(a0=0.1, q_bounds=(-10., -1.))
    assert isinstance(disc._table, NestedMomentTable)


def test_moment_state_has_no_round_trip_error():
    """Setting the state from its own moments, with the power law evaluated
    in between, leaves the moments unchanged"""
    disc = _setup_disc(a0=0.1)
    M0 = disc.moments()
    for _ in range(100):
        disc.set_moments(disc.moments())
        disc.size_distribution()
    np.testing.assert_allclose(disc.moments(), M0, rtol=1e-13)


def test_growth_keeps_dust_fraction_exactly():
    """Growth changes only the scaled moments"""
    disc = _setup_disc()
    M = disc.moments()
    disc.update(1e3*2*np.pi)
    M1 = disc.moments()
    np.testing.assert_array_equal(M1[1], M[1])
    assert not np.allclose(M1[2], M[2])


def test_set_dust_frac_updates_state():
    """set_dust_frac defines the state from the populations: an equal
    rescaling of them keeps the shape exactly (the stored moments agree to
    round-trip accuracy); a new amax keeps the populations. dust_frac and
    grain_size are read-only."""
    disc = _setup_disc(a0=0.1)
    with pytest.raises(ValueError):
        disc.dust_frac[:] *= 3
    with pytest.raises(ValueError):
        disc.grain_size[1] = 1.
    M = disc.moments()
    disc.set_dust_frac(3*disc.dust_frac)
    M3 = disc.moments()
    np.testing.assert_allclose(M3[1], 3*M[1], rtol=1e-14)
    np.testing.assert_array_equal(M3/M3[1], M/M[1])

    eps, s, L = disc.size_distribution()
    a = disc.grain_size.copy()
    a[1] *= 2
    disc.set_dust_frac(disc.dust_frac, a)
    eps2, s2, L2 = disc.size_distribution()
    np.testing.assert_allclose(L2, L + np.log(2), rtol=1e-12)
    np.testing.assert_allclose(eps2, eps, rtol=1e-14)
    np.testing.assert_allclose(disc.moments(),
                               _moments(eps2, s2, L2, disc.amin), rtol=1e-12)


@pytest.mark.parametrize("van_leer", [False, True])
def test_transport_conservation_newton(van_leer):
    """Without a lower bound on q the moments are inverted by Newton
    iteration; the moments are still conserved"""
    disc = _setup_disc(a0=0.1, q_bounds=(-np.inf, -1.))
    assert disc._table is None
    _zero_dust(disc, slice(None, 15), slice(-15, None))
    drift = MomentDrift(van_leer=van_leer)
    D = TracerDiffusion()
    dA = np.pi*np.diff((disc.grid.Re*AU)**2)
    T0 = (disc.moments()*disc.Sigma*dA).sum(1)
    for _ in range(20):
        _step(drift, disc, D)
    T1 = (disc.moments()*disc.Sigma*dA).sum(1)
    np.testing.assert_allclose(T1, T0, rtol=1e-8)


def test_gap_transport_stays_realizable():
    """Drifting dust into a cleared gap, where the distribution is steepened
    and truncated, keeps the parameters bounded and finite, and conserves
    mass"""
    disc = _setup_disc(a0=0.3)
    _zero_dust(disc, slice(None, 15), slice(-15, None))
    # Gap: no gas and no dust in a ring
    ring = slice(55, 62)
    disc._Sigma[ring] *= 1e-3
    _zero_dust(disc, ring)
    drift = MomentDrift(van_leer=True)
    D = TracerDiffusion()

    dA = np.pi*np.diff((disc.grid.Re*AU)**2)
    M0 = (disc.Sigma*disc.integ_dust_frac*dA).sum()
    for _ in range(40):
        _step(drift, disc, D)
        disc.do_grain_growth(0.)

    eps_tot, s, L = disc.size_distribution()
    assert np.all(np.isfinite(disc.dust_frac)) and np.all(disc.dust_frac >= 0)
    assert np.all((s >= S_BOUNDS[0]) & (s <= S_BOUNDS[1]))
    assert np.all(L >= L_MIN - 1e-12)
    assert np.all(disc.amax >= disc.amin*np.exp(L_MIN) - 1e-20)
    assert np.isclose((disc.Sigma*disc.integ_dust_frac*dA).sum(), M0,
                      rtol=1e-10)


def test_state_roundtrip():
    """Setting the state from size_distribution() reproduces it exactly, so a
    restart from the stored (eps, a) is lossless"""
    disc = _setup_disc(a0=0.2)
    eps_tot, s, L = disc.size_distribution()
    L = L*(1 + 0.3*np.sin(np.log(disc.R)*3))
    s = s + 0.5*np.cos(np.log(disc.R)*2)
    disc.set_size_distribution(eps_tot, s, L)

    eps, a = disc.dust_frac.copy(), disc.grain_size.copy()
    M = disc.moments(KS)

    new = _setup_disc(a0=0.2)
    new.set_dust_frac(eps, a)
    np.testing.assert_allclose(new.moments(KS), M, rtol=1e-12)
    np.testing.assert_allclose(new.grain_size, a, rtol=1e-12)

    e2, s2, L2 = new.size_distribution()
    np.testing.assert_allclose(s2, s, atol=1e-10)
    np.testing.assert_allclose(L2, L, rtol=1e-12)


def test_empty_cells_are_finite():
    """Cells with no dust give finite state, velocities and time-step"""
    disc = _setup_disc(a0=0.1)
    _zero_dust(disc, slice(40, 50))
    drift = MomentDrift()
    dt = drift.max_timestep(disc)
    assert np.isfinite(dt) and dt > 0
    drift(dt, disc)
    disc.update(dt)
    assert np.all(np.isfinite(disc.dust_frac))
    assert np.all(np.isfinite(disc.grain_size))
    assert np.all(disc.dust_frac[:, 41:49] < 1e-10)


def test_hdf5_attributes():
    """Headers can be generated and parsed back"""
    disc = _setup_disc()
    name, head = disc.HDF5_attributes()
    assert name == 'DustGrowthTriMom'
    assert 'f_dv' in head
    drift = MomentDrift()
    name, head = drift.HDF5_attributes()
    assert head['scheme'] == 'moments'
    assert isinstance(drift.ASCII_header(), str)


###############################################################################
# Feedback consistency (Booth et al. 2017, eqs. 5-9)
###############################################################################
def _tanaka_u_gas(disc, a_L, a_R, e_L, e_R, la=None):
    """Inviscid Tanaka+05 gas radial velocity at the cell edges for species of
    sizes a_L/a_R (left / right cells) and dust fractions e_L/e_R (relative
    to the edge-averaged total Sigma), or for given (lambda_0, lambda_1)"""
    Sigma, SigG = disc.Sigma, disc.Sigma_G
    Sig_av = 0.5*(Sigma[1:] + Sigma[:-1]) + 1e-300
    SigG_av = 0.5*(SigG[1:] + SigG[:-1])
    eps_g = SigG_av/Sig_av
    if la is None:
        St_L, St_R = disc.Stokes(SigG_av, a_L), disc.Stokes(SigG_av, a_R)
        la0 = 0.5*((e_L/(1 + St_L**2)).sum(0) +
                   (e_R/(1 + St_R**2)).sum(0))/eps_g
        la1 = 0.5*((e_L*St_L/(1 + St_L**2)).sum(0) +
                   (e_R*St_R/(1 + St_R**2)).sum(0))/eps_g
    else:
        la0, la1 = la
    rho, Om = disc.midplane_gas_density, disc.Omega_k
    eta = -(np.diff(disc.P)/disc.grid.dRc) / \
        (0.5*(rho[1:] + rho[:-1])*0.5*(Om[1:] + Om[:-1]))
    return la1*eta/((1 + la0)**2 + la1**2)


def _com_minus_gas(drift, disc, faces=False, v_visc=None):
    """v_gas and v_COM - v_gas at the edges of the cell states (or the VL2
    corrector's face states), recovered from the drift's velocities: the
    velocity relative to the gas of a node (or, for the moment scheme
    without settling, of the mass moment) minus its velocity relative to the
    centre of mass"""
    got = {}
    edge_gas, nodes, moments = (drift._edge_gas_velocities, drift._node_velocities,
                                drift._moment_velocities)
    def capture_gas(*args):
        got['u'], got['v'] = edge_gas(*args)
        return got['u'], got['v']
    def capture_nodes(*args):
        out = nodes(*args)
        got['nodes'] = args[1], out[0]          # sizes and velocities, left
        return out
    def capture_moments(*args):
        out = moments(*args)
        got['moments'] = args[1], out[0]        # side and velocities, left
        return out
    drift._edge_gas_velocities, drift._node_velocities = capture_gas, capture_nodes
    drift._moment_velocities = capture_moments
    if faces:
        drift._vl2_faces(disc, v_visc=v_visc)
    else:
        drift._transport_velocities(disc, v_visc)
    u, v = got['u'], got['v']
    SigG = disc.Sigma_G
    SigG_av = 0.5*(SigG[1:] + SigG[:-1])
    if 'nodes' in got:
        a, V = got['nodes']
        St = disc.Stokes(SigG_av, a[0] + 1e-300)
        dV = (2*v*St - u*St**2)/(1 + St**2)
        return u, dV - V[0]
    (_, s, L), V = got['moments']
    m = drift._ks.index(0)
    F1, F2 = mean_drift_factors(s, L, disc.Stokes(SigG_av, drift._f_drift*disc.amin),
                                disc._quad, drift._ks)
    return u, 2*v*F1[m] - u*F2[m] - V[m]


@pytest.mark.parametrize("scheme,f_drift", [('moments', 1.0),
                                            ('moments', 0.8),
                                            ('tripod', 1.0),
                                            ('tripod', 0.8),
                                            ('trimean', 1.0),
                                            ('trimax', 1.0)])
def test_feedback_centre_of_mass(scheme, f_drift):
    """With the feedback computed from the transported species, the inviscid
    centre of mass is at rest: v_COM - v_gas = -v_gas, with v_gas checked
    against Tanaka et al. (2005)"""
    disc = _setup_disc(a0=0.3)
    disc.set_dust_frac(10*disc.dust_frac)     # eps = 0.1, so that feedback matters
    drift = make_drift(scheme=scheme, f_drift=f_drift)
    _, epsDeltaV = _com_minus_gas(drift, disc)

    Sigma = disc.Sigma
    Sig_av = 0.5*(Sigma[1:] + Sigma[:-1]) + 1e-300
    if scheme in ('tripod', 'trimean', 'trimax'):
        eps_tot, s, L = disc.size_distribution()
        a = f_drift*np.array([mean_size(s, 0, 0.5*L, disc.amin),
                              mean_size(s, 0.5*L, L, disc.amin)])
        SigD = disc.dust_frac*Sigma
        u_gas = _tanaka_u_gas(disc, a[:, :-1], a[:, 1:],
                              SigD[:, :-1]/Sig_av, SigD[:, 1:]/Sig_av)
    else:
        # The moment scheme's lambdas, from its averages of St/(1+St^2) and
        # St^2/(1+St^2) for the mass
        SigG = disc.Sigma_G
        SigG_av = 0.5*(SigG[1:] + SigG[:-1])
        left, right = drift._cell_sides(disc)
        St_min = disc.Stokes(SigG_av, f_drift*disc.amin)
        (F1L, F2L), (F1R, F2R) = [mean_drift_factors(sd[1], sd[2], St_min, disc._quad)
                                  for sd in (left, right)]
        SigD = disc.integ_dust_frac*Sigma
        eL, eR = SigD[:-1]/Sig_av, SigD[1:]/Sig_av
        eps_g = SigG_av/Sig_av
        la0 = 0.5*(eL*(1 - F2L[1]) + eR*(1 - F2R[1]))/eps_g
        la1 = 0.5*(eL*F1L[1] + eR*F1R[1])/eps_g
        u_gas = _tanaka_u_gas(disc, None, None, None, None, la=(la0, la1))
        # ... which agree with the node values
        a, w = disc.quadrature()
        u_nodes = _tanaka_u_gas(disc, f_drift*a[:, :-1], f_drift*a[:, 1:],
                                (SigD*w)[:, :-1]/Sig_av, (SigD*w)[:, 1:]/Sig_av)
        np.testing.assert_allclose(u_gas, u_nodes, rtol=1e-6)

    assert np.abs(u_gas).max() > 0
    np.testing.assert_allclose(epsDeltaV, -u_gas, rtol=1e-10,
                               atol=1e-12*np.abs(u_gas).max())


@pytest.mark.parametrize("scheme", ['moments', 'tripod', 'trimean', 'trimax'])
@pytest.mark.parametrize("settling", [False, True])
@pytest.mark.parametrize("faces", [False, True])
@pytest.mark.parametrize("v_visc", [None, 1e-3])
def test_centre_of_mass_velocity(scheme, settling, faces, v_visc):
    """The centre of mass moves with the viscous velocity, and is at rest
    without it: v_gas + (v_COM - v_gas) = v_visc, with and without settling
    (midplane densities in the feedback), for the cell states and for the
    VL2 corrector's face states, which differ from the cells' (structure in
    the dust fraction and the shape)"""
    disc = _setup_disc(a0=0.3)
    lR = np.log(disc.R)
    disc.set_size_distribution(0.1*(1 + 0.5*np.sin(2*lR)), 0.5 + 0.8*np.cos(1.5*lR),
                               np.log(0.3/disc.amin)*(1 + 0.3*np.sin(lR)))
    drift = make_drift(scheme=scheme, settling=settling)
    vv = None if v_visc is None else v_visc*np.ones(disc.Ncells - 1)
    u, w = _com_minus_gas(drift, disc, faces, vv)
    np.testing.assert_allclose(u + w, 0 if v_visc is None else v_visc,
                               rtol=0, atol=1e-12*np.abs(u).max())


@pytest.mark.parametrize("scheme", ['moments', 'tripod', 'trimean', 'trimax'])
@pytest.mark.parametrize("settling", [False, True])
@pytest.mark.parametrize("van_leer", [False, True])
def test_gas_tracer_follows_gas(scheme, settling, van_leer):
    """A gas tracer proportional to the gas stays so, with and without
    settling: the gas tracers are carried by the gas, which carries the
    opposite of the dust's mass flux (the drift leaves Sigma unchanged)"""
    disc = _setup_disc(a0=0.3)
    disc.set_dust_frac(10*disc.dust_frac)     # eps = 0.1, so that feedback matters
    _zero_dust(disc, slice(None, 15), slice(-15, None))
    drift = make_drift(scheme=scheme, settling=settling, van_leer=van_leer)
    G = 0.7*(1 - disc.integ_dust_frac)[None]
    for _ in range(20):
        drift(drift.max_timestep(disc), disc, gas_tracers=G)
    np.testing.assert_allclose(G[0], 0.7*(1 - disc.integ_dust_frac), rtol=1e-12)


@pytest.mark.parametrize("scheme", ['moments', 'tripod'])
def test_timestep_limits_gas_outflow(scheme):
    """Where the dust outweighs the gas, the gas (which carries the opposite
    of the dust's mass flux) can move faster than the dust: in a step of
    max_timestep no cell loses more than half its gas (dust-to-gas 9, a
    steep distribution, q = -1, so the dust moves nearly as one size)"""
    disc = _setup_disc(a0=0.3)
    n = disc.Ncells
    disc.set_size_distribution(0.9*np.ones(n), 3.*np.ones(n),
                               np.log(1./disc.amin)*np.ones(n))
    disc.growth_timestep = lambda: np.inf     # the drift's own limit only
    drift = make_drift(scheme=scheme)
    dt = drift.max_timestep(disc)
    Q, V_L, V_R = drift._velocities(disc)
    pos, neg = drift._flux_parts(*drift._cell_faces(disc.Sigma*Q), V_L, V_R)
    F_gas = -sum(drift._dust_mass_fluxes(pos, neg))
    gas_out = np.maximum(F_gas[1:], 0) + np.maximum(-F_gas[:-1], 0)
    assert (dt*gas_out/(disc.Sigma_G*disc.grid.dRe)).max() <= 0.5*(1 + 1e-12)


@pytest.mark.parametrize("q", [-4.5, -3.5, -2., -8.])
@pytest.mark.parametrize("amax", [1e-3, 0.1, 10.])
def test_moment_velocities_vs_fine_quadrature(q, amax):
    """The factorised means <St^p/(1+St^2)>_k (mean_drift_factors) agree with
    a fine quadrature of the node velocities, including steep power laws"""
    disc = _setup_disc(a0=0.1, q_bounds=(-np.inf, -1.))
    n = disc.Ncells
    disc.set_size_distribution(0.01*np.ones(n), (q + 4)*np.ones(n),
                               np.log(amax/disc.amin)*np.ones(n))
    drift = MomentDrift()
    SigG = disc.Sigma_G
    K = disc.Stokes(0.5*(SigG[1:] + SigG[:-1]), 1.)
    _, sL, LL = drift._cell_sides(disc)[0]
    F = mean_drift_factors(sL, LL, K*disc.amin, disc._quad)
    _, s, L = disc.size_distribution()
    af, wf = SizeQuadrature(32, 12)(s, L, disc.amin)
    af, wf = af[:, :-1], wf[:, :-1]
    St = K*af
    for k in (-1, 0, 1):
        wk = wf*af**k
        for p in (1, 2):
            ref = (wk*St**p/(1 + St**2)).sum(0)/wk.sum(0)
            np.testing.assert_allclose(F[p - 1][k + 1], ref, rtol=2e-3)


###############################################################################
# Transport by the van Leer integrator (VL2)
###############################################################################
@pytest.mark.parametrize("scheme", ['moments', 'tripod', 'trimean', 'trimax'])
@pytest.mark.parametrize("feedback,settling", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("van_leer", [False, True])
def test_transport_tracers(scheme, feedback, settling, van_leer):
    """Dust tracers proportional to the dust mass stay so (and a thin shell,
    area_tracers, proportional to the area, M_-1, to the quadrature's
    accuracy), and the mass is conserved, with feedback and settling; a
    uniform gas tracer without feedback stays uniform. (Cells filled from
    empty, below 1e-6 of the peak, have their moments projected onto the
    power law, which changes M_-1: they are left out of the area
    comparison.)"""
    disc = _setup_disc(a0=0.1, feedback=feedback)
    if scheme in ('trimean', 'trimax'):
        disc._s_bounds = (-np.inf, np.inf)
    _zero_dust(disc, slice(None, 15), slice(-15, None))
    area = scheme == 'moments'
    drift = make_drift(scheme=scheme, settling=settling, van_leer=van_leer,
                       **({'area_tracers': [1]} if area else {}))
    n = disc.Ncells
    gas = np.ones((1, n))
    dust = np.array([0.3*disc.moments((0,))[0],
                     1e-8*disc.amin*disc.moments((-1,))[0] if area
                     else 0.2*disc.moments((0,))[0]])
    dA = np.pi*np.diff((disc.grid.Re*AU)**2)
    M0 = (disc.Sigma*disc.dust_frac.sum(0)*dA).sum()
    for _ in range(20):
        drift(drift.max_timestep(disc), disc, gas_tracers=gas, dust_tracers=dust)
    assert np.all(np.isfinite(disc.moments()))
    assert np.isclose((disc.Sigma*disc.dust_frac.sum(0)*dA).sum(), M0, rtol=1e-10)
    m = disc.moments((-1, 0))
    np.testing.assert_allclose(dust[0], 0.3*m[1], rtol=1e-10, atol=1e-14*m[1].max())
    if area:
        full = m[1] > 1e-6*m[1].max()
        np.testing.assert_allclose(dust[1][full], 1e-8*disc.amin*m[0][full], rtol=1e-7)
    else:
        np.testing.assert_allclose(dust[1], 0.2*m[1], rtol=1e-10, atol=1e-14*m[1].max())
    if not feedback:
        np.testing.assert_allclose(gas, 1., rtol=1e-12)


@pytest.mark.parametrize("scheme", ['moments', 'tripod'])
def test_vl2_reuses_courant_velocities(scheme):
    """The VL2 predictor's velocities, from max_timestep or recomputed, give
    the same step"""
    a, b = _setup_disc(a0=0.1), _setup_disc(a0=0.1)
    da, db = make_drift(scheme=scheme), make_drift(scheme=scheme)
    for _ in range(5):
        dt = da.max_timestep(a)
        da(dt, a)
        db._vcache = None
        db(dt, b)
    np.testing.assert_array_equal(a.moments(), b.moments())


def _disc_n(n):
    grid = Grid(1, 300, n, spacing='log')
    star = SimpleStar()
    eos = LocallyIsothermalEOS(star, 1/30., -0.25, 1e-3)
    eos.set_grid(grid)
    Mdot = 1e-8*Msun/(2*np.pi)/AU**2
    Sigma = (Mdot/(3*np.pi*eos.nu))*np.exp(-grid.Rc/100.)
    eos.update(0, Sigma)
    disc = DustGrowthTriMom(grid, star, eos, 0.01, Sigma=Sigma, a0=0.1)
    disc.growth_timestep = lambda: np.inf
    return disc


def _temporal_error(van_leer, n, dt, n_steps, f=0.25, f_ref=1/32., window=(10., 50.)):
    """Error of n_steps/f steps of f dt against f_ref dt, in a smooth window"""
    out = {}
    for g in (f, f_ref):
        disc, drift = _disc_n(n), MomentDrift(van_leer=van_leer)
        for _ in range(int(round(n_steps/g))):
            drift(g*dt, disc)
        out[g] = disc.Sigma*disc.moments()
    R = disc.grid.Rc
    m = (R >= window[0]) & (R < window[1])
    dA = np.diff(disc.grid.Re**2)
    return max((np.abs(out[f][k] - out[f_ref][k])*dA)[m].sum() /
               (out[f_ref][k]*dA)[m].sum() for k in range(3))


def test_vl2_temporal_error_is_a_cross_term():
    """At a fixed time-step, halving the cell size halves VL2's temporal
    error (O(dt dx), second order) but not donor cell's (O(dt))"""
    dt = MomentDrift(van_leer=False).max_timestep(_disc_n(120))
    r = {vl: _temporal_error(vl, 120, dt, 40)/_temporal_error(vl, 240, dt, 40)
         for vl in (False, True)}
    assert r[True] > 1.7
    assert r[False] < 1.1


###############################################################################
# Grain sizes with condensing / sublimating ice (ice_sizes)
###############################################################################
def _ice_disc(ice_sizes, water):
    from DiscEvolution.chemistry import ChemicalAbund
    disc = _setup_disc(a0=0.1, ice_sizes=ice_sizes)
    ice = ChemicalAbund(['grain', 'H2O'], [100., 18.], disc.Ncells)
    ice['grain'] = 0.008
    ice['H2O'] = water
    disc.update_ices(ice)
    return disc, ice


@pytest.mark.parametrize("delta", [0.5, -0.2])
def test_ice_sizes_scale_moments(delta):
    """Mass-distributed ice raising every grain's mass by 1 + delta scales
    its size by f = (1 + delta)^(1/3): M_k -> (1 + delta) f^k M_k"""
    disc, ice = _ice_disc(True, 0.002)
    M0 = disc.moments()
    eps = 0.01
    ice['H2O'] = 0.002 + delta*eps
    disc.update_ices(ice)
    M1 = disc.moments()
    f = (1 + delta)**(1/3.)
    for i, k in enumerate((-1, 0, 1)):
        assert np.allclose(M1[i], (1 + delta)*f**k*M0[i], rtol=1e-6)


def test_ice_sizes_round_trip():
    """Condensing and then sublimating the same ice restores the state"""
    disc, ice = _ice_disc(True, 0.002)
    M0 = disc.moments()
    ice['H2O'] = 0.006
    disc.update_ices(ice)
    ice['H2O'] = 0.002
    disc.update_ices(ice)
    assert np.allclose(disc.moments(), M0, rtol=1e-6)


def test_ice_sizes_off_fixed_shape():
    """Without ice_sizes the ice changes the mass at fixed sizes"""
    disc, ice = _ice_disc(False, 0.002)
    M0 = disc.moments()
    ice['H2O'] = 0.007
    disc.update_ices(ice)
    M1 = disc.moments()
    for i in range(3):
        assert np.allclose(M1[i]/M0[i], 1.5, rtol=1e-12)
