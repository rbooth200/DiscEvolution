# test_area_ice.py
#
# Ice shells and mass-distributed ice with the TriMom dust (chemistry.area_ice)
###############################################################################
import numpy as np
import pytest

from DiscEvolution.grid      import Grid
from DiscEvolution.constants import AU, Msun, m_H
from DiscEvolution.eos       import LocallyIsothermalEOS
from DiscEvolution.star      import SimpleStar
from DiscEvolution.dust_trimom import (DustGrowthTriMom, MomentDrift, SizeQuadrature,
                                       IceProfile, moment_ratio, shell_thickness,
                                       _drift_factors)
from DiscEvolution.chemistry import ChemicalAbund, MolecularIceAbund
from DiscEvolution.chemistry.area_ice import AreaMassIceAbund, TimeDependentChemTriMom

YR = 2*np.pi
SPECIES, MASSES = ['H2O', 'CO', 'Si-grain'], [18., 28., 100.]
VOLATILES = ['H2O', 'CO']


def _setup_disc(**kwargs):
    grid = Grid(1, 300, 120, spacing='log')
    star = SimpleStar()
    eos = LocallyIsothermalEOS(star, 1/30., -0.25, 1e-3)
    eos.set_grid(grid)
    Mdot = 1e-8*Msun/(2*np.pi)/AU**2
    Sigma = (Mdot/(3*np.pi*eos.nu))*np.exp(-grid.Rc/100.)
    eos.update(0, Sigma)
    disc = DustGrowthTriMom(grid, star, eos, 0.01, Sigma=Sigma, amin=1e-5, **kwargs)
    disc.update(1e3*YR)          # a grown, non-trivial distribution
    return disc


def _chem(disc, area_frac, seed=1):
    """Out-of-equilibrium abundances, a fraction area_frac of the ice in the
    shells"""
    rng = np.random.default_rng(seed)
    N = disc.Ncells
    gas = ChemicalAbund(SPECIES, MASSES, N)
    ice = ChemicalAbund(SPECIES, MASSES, N)
    for s, X in (('H2O', 5e-3), ('CO', 1e-3)):
        f = rng.uniform(0, 1, N)
        ice[s], gas[s] = f*X, (1 - f)*X
    ice['Si-grain'] = 0.01
    c = MolecularIceAbund(gas=gas, ice=ice)
    c.ice = AreaMassIceAbund.from_ice(ice, VOLATILES)
    for s in VOLATILES:
        tot = c.ice[s].copy()
        c.ice.set_parts(s, area_frac*tot, (1 - area_frac)*tot)
    return c


def _conditions(disc):
    # Temperatures spanning both regimes for both species
    T = np.logspace(np.log10(12.), np.log10(250.), disc.Ncells)
    return T, disc.midplane_gas_density


###############################################################################
# Abundances and geometry
###############################################################################
def test_clip_negative_area_mass():
    # The shell can be negative (peeled grains); a negative mass-distributed
    # part goes into the shell, keeping the total; a negative total empties
    # both rows
    ice = AreaMassIceAbund(SPECIES, MASSES, VOLATILES, 4)
    ice.set_parts('H2O', [-0.5, 1., -1., 0.5], [1., -1., 0.5, -1.])
    ice['Si-grain'] = [-1., 1., 0., 2.]
    ice.clip_negative()
    assert np.array_equal(ice.area('H2O'), [-0.5, 0., 0., 0.])
    assert np.array_equal(ice.mass_part('H2O'), [1., 0., 0., 0.])
    assert np.array_equal(ice['Si-grain'], [0., 1., 0., 2.])


def test_mixing_thins_the_floor():
    # Growth mixing moves the shell into the mass-distributed part and thins
    # the floor by the same fraction (towards ice distributed by mass)
    ice = AreaMassIceAbund(SPECIES, MASSES, VOLATILES, 2)
    ice.set_parts('CO', [-0.5, 0.5], [1., 1.])
    ice.set_floor('CO', [1e-6, 1e-6])
    ice.mix_area_to_mass(np.array([0.25, 1.]))
    assert np.allclose(ice.area('CO'), [-0.375, 0.])
    assert np.allclose(ice.mass_part('CO'), [0.875, 1.5])
    assert np.allclose(ice.floor('CO'), [0.75e-6, 0.])


def test_profile_offset():
    # The offset holds the ice, with and without a floor, peeled or not
    amin = 1e-5
    s, L = np.full(6, 0.5), np.full(6, np.log(1./amin))
    v = np.array([0., 0.2, 0.2, 0.2, 0.5, 0.5])
    r = np.array([0.3, 0.5, 0.1, 0.1, 0.2, 1e-6])
    t_f = np.array([0., 0., 0., 5e-8, 5e-8, 5e-8])
    prof = IceProfile(s, L, amin, v)
    d, tf, xf = prof.offset(r, t_f)
    I, _ = prof.ice(d, tf)
    assert np.allclose(I, r, rtol=1e-10)
    assert np.all((d < 0) == (r < v))
    # with too little ice for the floor alone, all the grains are at a
    # lower floor
    assert tf[-1] < t_f[-1] and xf[-1] == L[-1]
    # against a fine grid of sizes
    x = np.linspace(0, L[0], 400001)
    a = amin*np.exp(x)
    for i in range(6):
        c = prof.kappa[i]*a
        th = np.maximum(tf[i], prof.beta[i]*a + d[i])
        w = np.exp(s[i]*x)
        assert np.isclose(np.trapezoid(w*((c + th)**3 - c**3)/a**3, x)/np.trapezoid(w, x),
                          r[i], rtol=1e-5)


def test_shell_thickness():
    # The cubic, and the mantle's mass on a fine grid of sizes
    amin = 1e-5
    for q, amax in ((-3.5, 1.), (-2.5, 1e-2), (-4.5, 10.), (-3., 1e-4)):
        s, L = np.array([q + 4.]), np.array([np.log(amax/amin)])
        R = [moment_ratio(-k, s, L, amin) for k in (1, 2, 3)]
        a = amin*np.exp(np.linspace(0, L[0], 200001))
        w = a**(q + 4)
        for r in (1e-12, 1e-3, 0.3, 3., 100.):
            d = shell_thickness(np.array([r]), s, L, amin)
            assert np.isclose(3*d*R[0] + 3*d*d*R[1] + d**3*R[2], r, rtol=1e-12)
            la = np.log(a)              # (w: mass per ln a)
            assert np.isclose(np.trapezoid(w*((1 + d/a)**3 - 1), la)/np.trapezoid(w, la),
                              r, rtol=1e-6)
    assert shell_thickness(np.array([0., -1.]), s[[0, 0]], L[[0, 0]], amin).tolist() == [0., 0.]


###############################################################################
# Chemistry
###############################################################################
@pytest.mark.parametrize("area_frac", [0., 0.3, 1.])
@pytest.mark.parametrize("dt", [1e-6, 1e-2, 1., 1e2, 1e5])
def test_conservation_and_bounds(area_frac, dt):
    disc = _setup_disc()
    T, rho = _conditions(disc)
    c = _chem(disc, area_frac)
    tot0 = {s: c.ice[s] + c.gas[s] for s in VOLATILES}
    M0 = {s: c.ice.mass_part(s).copy() for s in VOLATILES}
    chem = TimeDependentChemTriMom(disc)
    chem.update(dt*YR, T, rho, c.ice.total_abund, c)
    assert chem.n_unconverged == 0
    for s in VOLATILES:
        assert np.allclose(c.ice[s] + c.gas[s], tot0[s], rtol=1e-12)
        assert np.all(c.ice[s] >= 0)
        assert np.array_equal(c.ice.mass_part(s), M0[s])     # the shell takes it all
        assert np.all(c.gas[s] >= 0)
        assert np.all(c.ice.floor(s) >= 0)


@pytest.mark.parametrize("area_frac", [0., 0.5, 1.])
def test_long_step_equilibrium(area_frac):
    # A long step reaches ThermalChem's equilibrium on the grains (the dust
    # less the shells): exactly where there are multilayers (the saturated
    # vapour does not depend on the area). Below a monolayer, without
    # mass-distributed ice, to the mantles' effect on the area (one layer on
    # the smallest grains); with it the stripped grains are smaller (the
    # cores)
    disc = _setup_disc()
    T, rho = _conditions(disc)
    c = _chem(disc, area_frac)
    chem = TimeDependentChemTriMom(disc)
    tot = ChemicalAbund(SPECIES, MASSES, disc.Ncells)
    for s in VOLATILES:
        tot[s] = c.ice[s] + c.gas[s]
    chem.update(1e12*YR, T, rho, c.ice.total_abund, c)
    base = c.ice.total_abund - sum(c.ice.area(s) for s in VOLATILES)
    eps = chem._effective_dust_frac(base, disc.grain_cross_section)
    for s in VOLATILES:
        eq = chem._equilibrium_ice_abund(T, rho, eps, s, tot)
        multi = eq > 1.01*chem._etaNbind*eps*c.ice.mass(s)/chem._mu
        assert multi.any() and (~multi).any()
        assert np.allclose(c.ice[s][multi], eq[multi], rtol=1e-6)
        if area_frac == 1:
            assert np.allclose(c.ice[s], eq, rtol=0.03, atol=1e-12*tot[s].max())


def test_multilayer_desorption_is_exact():
    # Into empty gas, every grain above a monolayer loses the same thickness,
    # Sd dt monolayers, in one step of any length
    disc = _setup_disc()
    T = np.full(disc.Ncells, 40.)
    rho = 1e-30*disc.midplane_gas_density
    c = _chem(disc, 1.)
    for s in VOLATILES:
        c.gas[s] = 0.
    chem = TimeDependentChemTriMom(disc)
    d0 = chem.thickness(c.ice.total_abund, c.ice, 'CO')
    m = 28.
    conv = chem._mu/m
    base = c.ice.total_abund - sum(c.ice.area(s) for s in VOLATILES)
    _, s, L = disc.size_distribution()
    Xm = chem._etaNbind*chem._effective_dust_frac(base, disc.grain_cross_section)
    delta1 = Xm/(conv*3*base*moment_ratio(-1, s, L, disc.amin))
    Sd = chem._f_des*chem._nu_i(chem._Tbind['CO'], m)*np.exp(-chem._Tbind['CO']/T[0])
    dt = 0.5*np.median(d0/delta1)/Sd
    chem.update(dt, T, rho, c.ice.total_abund, c)
    d1 = chem.thickness(c.ice.total_abund, c.ice, 'CO')
    thick = d0 > 2*Sd*dt*delta1
    assert thick.sum() > 10
    assert np.allclose(d1[thick], d0[thick] - Sd*dt*delta1[thick], rtol=1e-8)


# One step against the per-grain backward Euler step on a fine grid of sizes
###############################################################################
P_AMIN, P_AMAX, P_Q, P_RHO_S, P_EPS = 1e-5, 1., -3.5, 1., 0.01


class _Parcel(object):
    """The parts of a DustGrowthTriMom that the chemistry uses: one cell with
    a fixed power law"""
    amin = P_AMIN

    def __init__(self):
        self._s, self._L = np.array([P_Q + 4]), np.array([np.log(P_AMAX/P_AMIN)])
        aA = 1/moment_ratio(-1, self._s, self._L, P_AMIN)
        self.grain_cross_section = 3/(4*P_RHO_S*aA)

    def size_distribution(self):
        return np.array([P_EPS]), self._s, self._L


def _parcel_chem(A, M, X_g):
    """The grains P_EPS (with the mass-distributed ice M) and a shell A of
    CO, X_g of CO vapour (mass fractions)"""
    gas = ChemicalAbund(['CO', 'grain'], [28., 100.], 1)
    ice = ChemicalAbund(['CO', 'grain'], [28., 100.], 1)
    c = MolecularIceAbund(gas=gas, ice=ice)
    c.ice = AreaMassIceAbund.from_ice(ice, ['CO'])
    c.ice['grain'] = P_EPS - M
    c.ice.set_parts('CO', np.array([A]), np.array([M]))
    c.gas['CO'] = X_g
    return c


def _grain_step(chem, par, A, M, X_g, T, rho, dt, t_f=0., nb=2000):
    """(ice, vapour) after one backward-Euler step of every grain of a fine
    grid of sizes: the thickness above the core kappa a, from the profile
    max(t_f, beta a + d), changing by dt (gain - Sd min(t/delta1, 1)), the
    vapour a root by bisection"""
    x = (np.arange(nb) + 0.5)*par._L[0]/nb
    a = P_AMIN*np.exp(x)
    w = np.exp(par._s[0]*x)
    w /= w.sum()
    m, conv = 28., chem._mu/28.
    eps = chem._effective_dust_frac(np.array([P_EPS]), par.grain_cross_section)
    Xm = chem._etaNbind*eps[0]
    Tb = chem._Tbind['CO']
    ra = chem._f_ads*chem._v_therm(T, m)*eps[0]*rho/(chem._mu*m_H)/Xm
    Sd = chem._f_des*chem._nu_i(Tb, m)*np.exp(-Tb/T)
    delta1 = Xm/(conv*3*P_EPS*moment_ratio(-1, par._s, par._L, P_AMIN)[0])
    prof = IceProfile(par._s, par._L, P_AMIN, np.array([M/P_EPS]))
    d, t_f, _ = prof.offset(np.array([(A + M)/P_EPS]), np.array([t_f]))
    c = prof.kappa[0]*a
    t0 = np.maximum(t_f[0], prof.beta[0]*a + d[0])

    def ice(Xg1):
        g, l = dt*ra*Xg1*delta1, dt*Sd
        t_multi = t0 + g - l*delta1
        t = np.where(t_multi >= delta1, t_multi, (t0 + g)/(1 + l))
        return P_EPS*(w*((c + t)**3 - c**3)/a**3).sum()

    X_t = (A + M + X_g)*conv
    lo, hi = 0., X_t
    for _ in range(200):
        mid = 0.5*(lo + hi)
        lo, hi = (mid, hi) if mid + conv*ice(mid) < X_t else (lo, mid)
    return ice(lo), 0.5*(lo + hi)/conv


@pytest.mark.parametrize("A, M, X_g, T, t, t_f", [
    (0., 0., 1e-3, 18., 0.3, 0.),          # adsorption onto bare grains
    (1e-3, 0., 1e-3, 18., 3., 0.),         # onto a shell
    (1e-4, 0., 0., 30., 50., 0.),          # desorption of a shell
    (2.5e-3, 2.5e-3, 0., 30., 100., 0.),   # through the shell, peeling the grains
    (0., 1e-3, 1e-6, 30., 1e3, 0.),        # the mass part, the small grains bare
    (1e-3, 1e-3, 1e-3, 23., 10., 0.),      # near saturation
    (-1e-3, 2e-3, 1e-6, 30., 300., 3e-8),  # peeled, the small grains at a floor
    (-1e-3, 2e-3, 2e-3, 23., 10., 3e-8),   # peeled, near saturation
    (-1e-3, 2e-3, 1e-3, 18., 0.1, 1e-6),   # peeled, re-adsorbing
])
def test_step_against_grains(A, M, X_g, T, t, t_f):
    par = _Parcel()
    chem = TimeDependentChemTriMom(par)
    T, rho = np.array([T]), np.array([1e-13])
    c = _parcel_chem(A, M, X_g)
    c.ice.set_floor('CO', t_f)
    ice, gas = _grain_step(chem, par, A, M, X_g, T[0], rho[0], t*YR, t_f)
    chem.update(t*YR, T, rho, c.ice.total_abund, c)
    tot = A + M + X_g
    assert np.isclose(c.ice['CO'][0], ice, rtol=1e-4, atol=1e-6*tot)
    assert np.isclose(c.gas['CO'][0], gas, rtol=1e-4, atol=1e-6*tot)
    assert c.ice.mass_part('CO')[0] == M


def test_peeling_independent_of_steps():
    # Desorption through the shell into the mass-distributed ice: the peeled
    # state carries over from step to step, so N steps give one step's
    # result (exact: zeroth order into nearly empty gas)
    par = _Parcel()
    chem = TimeDependentChemTriMom(par)
    T, rho = np.array([30.]), np.array([1e-13])
    t = 300.
    ice, _ = _grain_step(chem, par, 2.5e-3, 2.5e-3, 0., T[0], rho[0], t*YR)
    for N in (1, 16, 128):
        c = _parcel_chem(2.5e-3, 2.5e-3, 0.)
        for _ in range(N):
            chem.update(t*YR/N, T, rho, c.ice.total_abund, c)
        assert c.ice.area('CO')[0] < 0          # peeled
        assert np.isclose(c.ice['CO'][0], ice, rtol=3e-4)


def test_steps_converge():
    # Adsorption in N steps: first order towards the fine-step solution
    par = _Parcel()
    chem = TimeDependentChemTriMom(par)
    T, rho = np.array([18.]), np.array([1e-13])

    def run(N, t=1.):
        c = _parcel_chem(0., 0., 1e-3)
        for _ in range(N):
            chem.update(t*YR/N, T, rho, c.ice.total_abund, c)
        return c.gas['CO'][0]
    ref = run(1024)
    err = [abs(run(N) - ref) for N in (4, 16, 64)]
    assert all(e1 < 0.4*e0 for e0, e1 in zip(err, err[1:]))


###############################################################################
# Growth and transport
###############################################################################
@pytest.mark.parametrize("area_frac", [0.5, -0.5])
@pytest.mark.parametrize("ice_sizes", [False, True])
def test_growth_mixing(ice_sizes, area_frac):
    # Growth mixes the shells into the grains, keeping the totals; with
    # ice_sizes the grains grow with the mixed ice (shrink, for a negative
    # shell: the peeled ice is spread over the grains)
    disc = _setup_disc(ice_sizes=ice_sizes)
    c = _chem(disc, area_frac=area_frac)
    disc.update_ices(c.ice)
    A0 = {s: c.ice.area(s).copy() for s in VOLATILES}
    tot0 = {s: c.ice[s].copy() for s in VOLATILES}
    disc.do_grain_growth(10*YR)
    y1 = disc._y.copy()
    disc.update_ices(c.ice)
    for s in VOLATILES:
        assert np.allclose(c.ice[s], tot0[s], rtol=1e-12)
        assert np.all(np.abs(c.ice.area(s)) <= np.abs(A0[s]))
        assert np.any(np.abs(c.ice.area(s)) < 0.99*np.abs(A0[s]))
    mixed = np.abs(c.ice.area('H2O')) < 0.99*np.abs(A0['H2O'])
    grew = disc._y[1] - y1[1]                   # ln(M_1/M_0): the sizes
    if ice_sizes:
        assert np.all(np.sign(grew[mixed]) == np.sign(area_frac))
    else:
        assert np.all(grew == 0)


def _shell_state(disc, r):
    """Dust tracers: the grains and a CO shell r times their mass"""
    eps = disc.dust_frac.sum(0)
    base = eps/(1 + r)
    D = np.array([base, r*base])
    return D


def _cell_velocities(drift, disc, D):
    """The velocities of the volatiles' ice from the cells (donor cell)"""
    Q = disc.moments()
    _, faces = drift._cell_state(disc, Q)
    faces['X'] = drift._cell_faces(drift._tracer_ratios(Q, D))
    return drift._shell_velocities(disc, faces, *drift._cell_sides(disc))


def test_shell_velocity():
    # The shells' velocity: thin, the area's (M_-1); thick, the mass-weighted
    # velocity with St at the coated size, against a fine quadrature
    disc = _setup_disc(feedback=False)
    drift = MomentDrift(area_tracers=[1])
    V_Q = drift._transport_velocities(disc)
    V_thin = _cell_velocities(drift, disc, _shell_state(disc, 1e-12))
    assert np.allclose(V_thin[0][0], V_Q[0][0], rtol=1e-6)
    assert np.allclose(V_thin[1][0], V_Q[1][0], rtol=1e-6)

    r = 0.7
    D = _shell_state(disc, r)
    V_L, _ = _cell_velocities(drift, disc, D)
    eps, s, L = disc.size_distribution()
    d = shell_thickness(r*np.ones_like(s), s, L, disc.amin)[:-1]
    a, w = SizeQuadrature(32, 12)(s[:-1], L[:-1], disc.amin)
    W = w*((1 + d/a)**3 - 1)
    SigG = disc.Sigma_G
    f1, f2 = _drift_factors(disc.Stokes(0.5*(SigG[1:] + SigG[:-1]), a + d))
    F = ((W*f1).sum(0)/W.sum(0), (W*f2).sum(0)/W.sum(0))
    u_gas, v_gas = drift._edge_gas_velocities(disc, 0., 0., 1.)
    ref = 2*v_gas*F[0] - u_gas*F[1]
    assert np.allclose(V_L[0], ref, rtol=1e-3)


def test_shells_conserved_in_transport():
    # The shell rows are conserved by the drift (up to the outflow at the
    # inner boundary), and stay non-negative
    disc = _setup_disc(feedback=True)
    drift = MomentDrift(area_tracers=[1])
    D = _shell_state(disc, 0.5)
    dA = np.pi*np.diff((disc.grid.Re*AU)**2)
    m0 = (disc.Sigma*D[1]*dA).sum()
    for _ in range(5):
        drift(drift.max_timestep(disc), disc, dust_tracers=D)
    m1 = (disc.Sigma*D[1]*dA).sum()
    assert np.all(D >= 0)
    assert m1 <= m0 and m1 > 0.99*m0


def _peeled_state(disc, v, I_of_base):
    """CO ice with the mass-distributed part v of the grains and the total
    I_of_base(profile) times them, [grain, CO, CO:A]"""
    ice = AreaMassIceAbund(['grain', 'CO'], [100., 28.], ['CO'], disc.Ncells)
    eps, s, L = disc.size_distribution()
    prof = IceProfile(s, L, disc.amin, np.full_like(s, v))
    I = I_of_base(prof)
    base = eps/(1 + I - v)                      # eps = base + shell
    ice['grain'] = (1 - v)*base
    ice.set_parts('CO', (I - v)*base, v*base)
    return ice, prof


def test_peeled_ice_moves_with_its_grains():
    # Ice peeled down to the floor (a coverage on every core) moves with the
    # cores' area; fully peeled ice (none left) stays none, however large the
    # mass-distributed part (the shell row cancels it exactly)
    disc = _setup_disc(feedback=False)
    t_f = 3e-8
    v = 0.4
    ice, prof = _peeled_state(disc, v, lambda p: p.segment(p.kappa, t_f, 0., p.R)[0])
    ice.set_floor('CO', t_f)
    drift = MomentDrift(area_tracers=ice)
    V_L, _ = _cell_velocities(drift, disc, ice.data)
    eps, s, L = disc.size_distribution()
    a, w = SizeQuadrature(32, 12)(s[:-1], L[:-1], disc.amin)
    c = prof.kappa[:-1]*a
    W = w*((c + t_f)**3 - c**3)/a**3
    SigG = disc.Sigma_G
    f1, f2 = _drift_factors(disc.Stokes(0.5*(SigG[1:] + SigG[:-1]), c + t_f))
    u_gas, v_gas = drift._edge_gas_velocities(disc, 0., 0., 1.)
    ref = 2*v_gas*(W*f1).sum(0)/W.sum(0) - u_gas*(W*f2).sum(0)/W.sum(0)
    assert np.allclose(V_L[0], ref, rtol=1e-3)

    ice, _ = _peeled_state(disc, v, lambda p: 0*p.v)
    drift = MomentDrift(area_tracers=ice)
    M0 = ice.mass_part('CO').max()
    for _ in range(5):
        drift(drift.max_timestep(disc), disc, dust_tracers=ice.data)
    assert np.all(np.abs(ice['CO']) < 1e-13*M0)
    assert ice.mass_part('CO').max() > 0.5*M0


def test_floor_moves_with_the_area():
    # The floor (not a row of the ice) is moved with the grains' area: a
    # uniform floor stays uniform, and a step moves as a thin shell's ratio
    # to M_-1 does
    disc = _setup_disc(feedback=False)
    ice, _ = _peeled_state(disc, 0.3, lambda p: p.v)          # I = M, no shell
    step = np.where(disc.R > 30., 1e-7, 2e-7)
    ice.set_floor('CO', step)
    drift = MomentDrift(area_tracers=ice)
    thin = disc.moments((-1,))[0]*np.array([step*1e-6])       # thin shell, ratio ~ step
    ref = MomentDrift(area_tracers=[0])
    disc2 = _setup_disc(feedback=False)
    for _ in range(5):
        drift(drift.max_timestep(disc), disc, dust_tracers=ice.data)
        ref(drift.max_timestep(disc2), disc2, dust_tracers=thin)
    ratio = thin[0]/disc2.moments((-1,))[0]/1e-6
    assert np.allclose(ice.floor('CO'), ratio, rtol=1e-3)
    assert ice.floor('CO').min() < 1.5e-7 < ice.floor('CO').max()

    ice.set_floor('CO', 1e-7)
    drift(drift.max_timestep(disc), disc, dust_tracers=ice.data)
    assert np.allclose(ice.floor('CO'), 1e-7, rtol=1e-12)


def test_tripod_has_no_area_tracers():
    from DiscEvolution.dust_tripod import TriPoDDrift
    with pytest.raises(TypeError):
        TriPoDDrift(area_tracers=[0])
