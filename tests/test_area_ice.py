# test_area_ice.py
#
# Area- and mass-distributed ice with the TriMom dust (chemistry.area_ice)
###############################################################################
import numpy as np
import pytest

from DiscEvolution.grid      import Grid
from DiscEvolution.constants import AU, Msun, m_H
from DiscEvolution.eos       import LocallyIsothermalEOS
from DiscEvolution.star      import SimpleStar
from DiscEvolution.dust_trimom import DustGrowthTriMom, MomentDrift
from DiscEvolution.chemistry import ChemicalAbund, MolecularIceAbund
from DiscEvolution.chemistry.base_chem import TimeDependentChem
from DiscEvolution.chemistry.area_ice import (AreaMassIceAbund,
                                              TimeDependentChemTriMom,
                                              _shape, _profile, _moments,
                                              _fracs)

YR = 2*np.pi
SPECIES, MASSES = ['H2O', 'CO', 'Si-grain'], [18., 28., 100.]
VOLATILES = ['H2O', 'CO']


def _setup_disc():
    grid = Grid(1, 300, 120, spacing='log')
    star = SimpleStar()
    eos = LocallyIsothermalEOS(star, 1/30., -0.25, 1e-3)
    eos.set_grid(grid)
    Mdot = 1e-8*Msun/(2*np.pi)/AU**2
    Sigma = (Mdot/(3*np.pi*eos.nu))*np.exp(-grid.Rc/100.)
    eos.update(0, Sigma)
    disc = DustGrowthTriMom(grid, star, eos, 0.01, Sigma=Sigma, amin=1e-5)
    disc.update(1e3*YR)          # a grown, non-trivial distribution
    return disc


def _chem(disc, area_frac, seed=1):
    """Out-of-equilibrium abundances: a fraction area_frac of the ice by area"""
    rng = np.random.default_rng(seed)
    N = disc.Ncells
    gas = ChemicalAbund(SPECIES, MASSES, N)
    ice = ChemicalAbund(SPECIES, MASSES, N)
    for s, X in (('H2O', 5e-3), ('CO', 1e-3)):
        f = rng.uniform(0, 1, N)
        ice[s], gas[s] = f*X, (1 - f)*X
    ice['Si-grain'] = 0.01
    ref = MolecularIceAbund(gas=gas.copy(), ice=ice.copy())
    new = MolecularIceAbund(gas=gas, ice=ice)
    new.ice = AreaMassIceAbund.from_ice(ice, VOLATILES)
    for s in VOLATILES:
        tot = new.ice[s].copy()
        new.ice.set_parts(s, area_frac*tot, (1 - area_frac)*tot)
    return ref, new


def _conditions(disc):
    # Temperatures spanning both regimes for both species
    T = np.logspace(np.log10(12.), np.log10(250.), disc.Ncells)
    return T, disc.midplane_gas_density, disc.dust_frac.sum(0)


@pytest.mark.parametrize("dt", [1e-6, 1e-2, 1., 1e2, 1e5])
def test_area_only_is_time_dependent_chem(dt):
    # With no mass-distributed ice the coverage is uniform: TimeDependentChem
    disc = _setup_disc()
    T, rho, eps = _conditions(disc)
    ref, new = _chem(disc, area_frac=1.)
    TimeDependentChem().update(dt*YR, T, rho, eps, ref,
                               grain_cross_section=disc.grain_cross_section)
    TimeDependentChemTriMom(disc).update(dt*YR, T, rho, eps, new)
    for s in VOLATILES:
        assert np.allclose(new.ice[s], ref.ice[s], rtol=1e-8, atol=1e-14)
        assert np.allclose(new.gas[s], ref.gas[s], rtol=1e-8, atol=1e-14)
        assert np.all(new.ice.mass_part(s) == 0)


def test_mass_ice_below_monolayer():
    # All grains below a monolayer: the total follows TimeDependentChem and
    # the mass-distributed ice decays at the desorption rate
    disc = _setup_disc()
    _, rho, eps = _conditions(disc)
    T = 200*np.ones(disc.Ncells)
    ref, new = _chem(disc, area_frac=0.)
    for c in (ref, new):
        for s in VOLATILES:
            c.ice[s] = 1e-8*c.ice[s]
    M0 = {s: new.ice.mass_part(s).copy() for s in VOLATILES}
    dt = 1e-9*YR
    chem = TimeDependentChemTriMom(disc)
    TimeDependentChem().update(dt, T, rho, eps, ref,
                               grain_cross_section=disc.grain_cross_section)
    chem.update(dt, T, rho, eps, new)
    for s in VOLATILES:
        m = new.ice.mass(s)
        Sd = chem._f_des*chem._nu_i(chem._Tbind[s], m)*np.exp(-chem._Tbind[s]/T)
        assert np.allclose(new.ice.mass_part(s), M0[s]*np.exp(-Sd*dt), rtol=1e-10)
        assert np.allclose(new.ice[s], ref.ice[s], rtol=1e-8)


@pytest.mark.parametrize("area_frac", [0., 0.3, 1.])
@pytest.mark.parametrize("dt", [1e-2, 1., 1e2, 1e5])
def test_conservation_and_bounds(area_frac, dt):
    disc = _setup_disc()
    T, rho, eps = _conditions(disc)
    _, new = _chem(disc, area_frac)
    tot0 = {s: new.ice[s] + new.gas[s] for s in VOLATILES}
    chem = TimeDependentChemTriMom(disc)
    chem.update(dt*YR, T, rho, eps, new)
    for s in VOLATILES:
        assert np.allclose(new.ice[s] + new.gas[s], tot0[s], rtol=1e-12)
        M = new.ice.mass_part(s)
        assert np.all(M >= 0)
        assert np.all(M <= _M_max(disc, new.ice[s])*(1 + 1e-12))
        assert np.all(new.gas[s] >= 0)


def _M_max(disc, X):
    # Realisable: Y/X <= amax/a_A, i.e. M <= X (amax/a_A - 1)/(c - 1)
    _, s, L = disc.size_distribution()
    g = _shape(s, L)
    return X*(g['amax_r'] - 1)/(g['c'] - 1)


def test_clip_negative_default():
    ab = ChemicalAbund(SPECIES, MASSES, 3)
    ab.data[:] = [[-1., 2., 0.], [1., -2., 3.], [0., 0., -1e-30]]
    ab.clip_negative()
    assert np.array_equal(ab.data, [[0., 2., 0.], [1., 0., 3.], [0., 0., 0.]])


def test_clip_negative_area_mass():
    # Negative area ice is left alone; negative mass-distributed ice goes
    # into the area part; a negative total empties both rows
    ice = AreaMassIceAbund(SPECIES, MASSES, VOLATILES, 4)
    ice.set_parts('H2O', [-0.5, 1., -1., 0.5], [1., -1., 0.5, -1.])
    ice.set_parts('CO', [-0.1, 0., 1., 2.], [1., 1., 1., 1.])
    ice['Si-grain'] = [-1., 1., 0., 2.]
    ice.clip_negative()
    assert np.array_equal(ice.area('H2O'), [-0.5, 0., 0., 0.])
    assert np.array_equal(ice.mass_part('H2O'), [1., 0., 0., 0.])
    assert np.array_equal(ice.area('CO'), [-0.1, 0., 1., 2.])
    assert np.array_equal(ice['Si-grain'], [0., 1., 0., 2.])


def test_unrealisable_start_is_projected():
    # Transport can leave Y/X > amax/a_A (more concentrated than all the ice
    # at amax): the step moves mass-distributed ice into the area part,
    # conserving the total
    disc = _setup_disc()
    T, rho, eps = _conditions(disc)
    _, new = _chem(disc, area_frac=0.)
    for s in VOLATILES:
        X = new.ice[s].copy()
        M = 1.2*_M_max(disc, X)
        new.ice.set_parts(s, X - M, M)
    tot0 = {s: new.ice[s] + new.gas[s] for s in VOLATILES}
    chem = TimeDependentChemTriMom(disc)
    chem.update(1e-8*YR, T, rho, eps, new)
    for s in VOLATILES:
        M = new.ice.mass_part(s)
        assert np.allclose(new.ice[s] + new.gas[s], tot0[s], rtol=1e-12)
        assert np.all(M <= _M_max(disc, new.ice[s])*(1 + 1e-12))
        assert np.all(chem.last_clip[s] > 0)


@pytest.mark.parametrize("area_frac", [0., 0.5, 1.])
def test_long_step_equilibrium(area_frac):
    # A long step reaches ThermalChem's equilibrium, whatever the split
    disc = _setup_disc()
    T, rho, eps = _conditions(disc)
    _, new = _chem(disc, area_frac)
    chem = TimeDependentChemTriMom(disc)
    eps_eff = chem._effective_dust_frac(eps, disc.grain_cross_section)
    tot = ChemicalAbund(SPECIES, MASSES, disc.Ncells)
    for s in VOLATILES:
        tot[s] = new.ice[s] + new.gas[s]
    chem.update(1e12*YR, T, rho, eps, new)
    for s in VOLATILES:
        eq = chem._equilibrium_ice_abund(T, rho, eps_eff, s, tot)
        assert np.allclose(new.ice[s], eq, rtol=1e-6, atol=1e-12*tot[s].max())


def test_growth_mixing():
    disc = _setup_disc()
    _, new = _chem(disc, area_frac=0.5)
    disc.update_ices(new.ice)
    A0 = {s: new.ice.area(s).copy() for s in VOLATILES}
    tot0 = {s: new.ice[s].copy() for s in VOLATILES}
    disc.update(10*YR)
    for s in VOLATILES:
        assert np.allclose(new.ice[s], tot0[s], rtol=1e-12)
        assert np.all(new.ice.area(s) <= A0[s])
        assert np.any(new.ice.area(s) < 0.99*A0[s])


def test_area_tracers_drift_with_area():
    # Dust tracers equal to M_-1 and M_0 (per unit mass) move exactly as the
    # transported moments, area_tracers with the M_-1 velocity
    disc = _setup_disc()
    drift = MomentDrift(area_tracers=[0])
    M = disc.moments((-1, 0))
    tracers = M.copy()
    dt = 0.5*drift.max_timestep(disc)
    drift(dt, disc, dust_tracers=tracers)
    assert np.allclose(tracers, disc.moments((-1, 0)), rtol=1e-6)

    from DiscEvolution.dust_tripod import TriPoDDrift
    with pytest.raises(TypeError):
        TriPoDDrift(area_tracers=[0])


def test_stripping_step_is_zeroth_order():
    # A step whose desorption depth exceeds the coverage of the smallest grains
    # (mixed regime, undersaturated gas) must not jump to 'all grains below a
    # monolayer' and desorb everything: above a monolayer at most S_d X_mono dt
    disc = _setup_disc()
    _, rho, eps = _conditions(disc)
    T = 150*np.ones(disc.Ncells)
    _, new = _chem(disc, area_frac=0.)
    chem = TimeDependentChemTriMom(disc)
    m = new.ice.mass('H2O')
    conv = chem._mu/m
    Sd = chem._f_des*chem._nu_i(chem._Tbind['H2O'], m)*np.exp(-chem._Tbind['H2O']/T)
    Xm = chem._etaNbind*chem._effective_dust_frac(eps, disc.grain_cross_section)
    _, s, L = disc.size_distribution()
    theta_min = new.ice['H2O']*conv/Xm*_shape(s, L)['amin_r']   # at amin
    thick = theta_min > 10
    dt = 3*np.median(theta_min[thick])/Sd[0]
    X0 = new.ice['H2O']*conv
    chem.update(dt, T, 1e-6*rho, eps, new)
    loss = X0 - new.ice['H2O']*conv
    assert np.all(loss[thick] <= Sd[thick]*Xm[thick]*dt*(1 + 1e-6))
    deep = thick & (X0 > 2*Sd*Xm*dt)
    assert deep.any()
    assert np.all(new.ice['H2O'][deep]*conv > 0.5*X0[deep])


# The floor + cut-off line profile against the exact per-grain kinetics
###############################################################################
P_AMIN, P_AMAX, P_Q, P_RHO_S, P_EPS = 1e-5, 0.1, -3.5, 1.67, 0.01


class _Parcel(object):
    """The parts of a DustGrowthTriMom that the chemistry uses: one cell with
    a fixed power law"""
    def __init__(self):
        self.amin = P_AMIN
        self._s, self._L = np.array([P_Q + 4]), np.array([np.log(P_AMAX/P_AMIN)])
        aA = P_AMIN/_shape(self._s, self._L)['amin_r']
        self.grain_cross_section = 3/(4*P_RHO_S*aA)

    def size_distribution(self):
        return np.array([P_EPS]), self._s, self._L


def _parcel_setup(theta_eq, T=150.):
    """Parcel, chemistry, X_mono, shape, rates; the density sets
    theta_eq = Sa X_g/(Sd X_mono) for 10 monolayers of vapour"""
    par = _Parcel()
    chem = TimeDependentChemTriMom(par)
    eps = chem._effective_dust_frac(np.array([P_EPS]), par.grain_cross_section)
    Xm = chem._etaNbind*eps
    T = np.array([T])
    Tb = chem._Tbind['H2O']
    Sd = chem._f_des*chem._nu_i(Tb, 18.)*np.exp(-Tb/T)
    Sa1 = chem._f_ads*chem._v_therm(T, 18.)*eps/(chem._mu*m_H)
    Xg = 10*Xm
    rho = theta_eq*Sd*Xm/(Sa1*Xg)
    return par, chem, Xm, _shape(par._s, par._L), T, rho, Sa1*rho, Sd, Xg


def _parcel_chem(chem, X, M, Xg):
    conv = chem._mu/18.
    gas = ChemicalAbund(['H2O', 'Si-grain'], [18., 100.], 1)
    ice = ChemicalAbund(['H2O', 'Si-grain'], [18., 100.], 1)
    ice['Si-grain'] = P_EPS
    c = MolecularIceAbund(gas=gas, ice=ice)
    c.ice = AreaMassIceAbund.from_ice(ice, ['H2O'])
    c.ice.set_parts('H2O', (X - M)/conv, M/conv)
    c.gas['H2O'] = Xg/conv
    return c


def _size_grid(g, nx=2000):
    x = (np.arange(nx) + 0.5)*g['L'][0]/nx
    w = np.exp((g['s'][0] - 1)*x)
    return x, w/w.sum(), g['amin_r'][0]*np.exp(x)


def _exact(theta, w, Xm, Xt, Sa, Sd, t, nsub=5000):
    """dtheta/dt = r - Sd min(theta, 1) for every grain, r = Sa (Xt - X)/Xm,
    the gas held over each of nsub sub-steps (each grain exact for it)"""
    h = t/nsub
    for _ in range(nsub):
        r = Sa*(Xt - Xm*(w*theta).sum())/Xm
        te = r/Sd
        if r >= Sd:
            theta = np.where(theta >= 1, theta + (r - Sd)*h,
                             te + (theta - te)*np.exp(-Sd*h))
        else:
            t1 = (theta - 1)/(Sd - r)
            theta = np.where(theta < 1, te + (theta - te)*np.exp(-Sd*h),
                             np.where(t1 >= h, theta - (Sd - r)*h,
                                      te + (1 - te)*np.exp(-Sd*(h - np.minimum(t1, h)))))
    return theta


def _model_theta(chem, c, T, rho, at):
    p = chem.ice_profile(T, rho, np.array([P_EPS]), c, 'H2O')
    return np.maximum(p['phi'], (p['alpha'] + p['beta']*at)/p['Xm'])


def test_profile_round_trip():
    # (X, Y) of the floor + (cut-off) line profile give back the profile,
    # continuously across a_f = amin
    par = _Parcel()
    g = _shape(par._s, par._L)
    Xm = np.array([1.])
    amin_r = g['amin_r'][0]
    for phi in (0., 0.3, 1.):
        for xf in (0., 1e-6, 0.5, 4., 0.9*par._L[0]):
            beta = np.array([100.])
            alpha = phi*Xm - beta*amin_r*np.exp(xf)
            X, Y = _moments(g, Xm, np.array([phi]), alpha, beta, np.array([xf]))
            a, b, x = _profile(g, Xm, X, Y, np.array([phi]))
            assert np.allclose([a[0], b[0], x[0]], [alpha[0], beta[0], xf],
                               rtol=1e-8, atol=1e-8)
        # a line above the floor everywhere: the line's coefficients
        alpha = phi*Xm + 1.
        X, Y = _moments(g, Xm, np.array([phi]), alpha, beta, np.zeros(1))
        a, b, x = _profile(g, Xm, X, Y, np.array([phi]))
        assert np.allclose([a[0], b[0], x[0]], [alpha[0], beta[0], 0.], atol=1e-10)


@pytest.mark.parametrize("theta_A, theta_eq", [(1e4, 0.01), (100., 0.3), (10., 0.9)])
def test_against_exact_per_grain(theta_A, theta_eq):
    # Mass-distributed ice at a front (with vapour re-adsorbing): the total and
    # the large grains' ice follow the exact per-grain kinetics. The large
    # grains are stripped as a uniform layer, not in proportion to the total.
    par, chem, Xm, g, T, rho, Sa, Sd, Xg = _parcel_setup(theta_eq)
    x, w, at = _size_grid(g)
    large = x > 0.5*g['L'][0]
    M0 = theta_A*Xm
    Xt = M0 + Xg
    t = 0.25*theta_A*g['amax_r'][0]/Sd[0]
    th = _exact(M0[0]*at/Xm[0], w, Xm[0], Xt[0], Sa[0], Sd[0], t)
    c = _parcel_chem(chem, M0, M0, Xg)
    for _ in range(256):
        chem.update(t/256, T, rho, np.array([P_EPS]), c)
    thm = _model_theta(chem, c, T, rho, at)
    assert chem.n_unconverged == 0
    assert np.isclose((w*thm).sum(), (w*th).sum(), rtol=3e-3)
    assert np.isclose((w*thm)[large].sum(), (w*th)[large].sum(), rtol=3e-3)
    # at the front the large grains keep more than the total's fraction
    if theta_eq < 0.1:
        assert (w*th)[large].sum()/(w*at)[large].sum() > (w*th).sum()/(w*at).sum()


def test_implicit_step_converges():
    # One step against many: first order, towards the exact solution
    par, chem, Xm, g, T, rho, Sa, Sd, Xg = _parcel_setup(0.3)
    x, w, at = _size_grid(g)
    M0 = 100*Xm
    t = 0.25*100*g['amax_r'][0]/Sd[0]
    ex = Xm[0]*(w*_exact(M0[0]*at/Xm[0], w, Xm[0], M0[0] + Xg[0], Sa[0], Sd[0], t)).sum()
    err = []
    for N in (1, 4, 16, 64):
        c = _parcel_chem(chem, M0, M0, Xg)
        for _ in range(N):
            chem.update(t/N, T, rho, np.array([P_EPS]), c)
        err.append(abs(c.ice['H2O'][0]*chem._mu/18./ex - 1))
    assert err[0] < 0.1
    assert all(e1 < 0.4*e0 for e0, e1 in zip(err, err[1:]))


def test_pebble_ice_near_the_line():
    # Only the large grains icy (small grains on the floor), the gas half
    # saturated: the floor holds theta_eq with no net exchange, so the step
    # loses ice as the exact kinetics do (no spurious gain onto bare grains)
    par, chem, Xm, g, T, rho, Sa, Sd, Xg = _parcel_setup(0.5)
    x, w, at = _size_grid(g)
    af = 0.3*g['amax_r'][0]
    beta = np.array([1e5*Xm[0]])
    th0 = np.maximum(0.5, beta[0]*(at - af)/Xm[0])
    X0, Y0 = Xm*(w*th0).sum(), Xm*(w*th0*at).sum()
    M0 = (Y0 - X0)/(g['c'] - 1)
    t = 10/Sd[0]                        # ten layers
    th = _exact(th0, w, Xm[0], X0[0] + Xg[0], Sa[0], Sd[0], t)
    c = _parcel_chem(chem, X0, M0, Xg)
    chem.update(t, T, rho, np.array([P_EPS]), c)
    X1 = c.ice['H2O'][0]*chem._mu/18.
    loss, loss_ex = X0[0] - X1, X0[0] - Xm[0]*(w*th).sum()
    assert loss > 0
    assert np.isclose(loss, loss_ex, rtol=0.05)


def test_supersaturated_bare_grains_reice():
    # Bare small grains in super-saturated gas: they re-ice as uniform layers
    # (the floor at one layer is taken out of A, then every grain gains the
    # same), so M is unchanged
    par, chem, Xm, g, T, rho, Sa, Sd, Xg = _parcel_setup(5.)
    x, w, at = _size_grid(g)
    af = 0.1*g['amax_r'][0]
    th0 = np.maximum(0., 1e3*(at - af))
    X0, Y0 = Xm*(w*th0).sum(), Xm*(w*th0*at).sum()
    M0 = (Y0 - X0)/(g['c'] - 1)
    c = _parcel_chem(chem, X0, M0, Xg)
    chem.update(1/Sd[0], T, rho, np.array([P_EPS]), c)
    assert np.isclose(c.ice.mass_part('H2O')[0]*chem._mu/18., M0[0], rtol=1e-10)
    assert c.ice['H2O'][0]*chem._mu/18. > X0[0]
    thm = _model_theta(chem, c, T, rho, at)
    assert np.all(thm >= 1 - 1e-9)
