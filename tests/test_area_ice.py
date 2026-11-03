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
                                              TimeDependentChemTriMom)

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
        assert np.all(new.ice.area(s) >= 0)
        assert np.all(new.ice.mass_part(s) >= 0)
        assert np.all(new.gas[s] >= 0)


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
    drift = MomentDrift(scheme='moments', area_tracers=[0])
    M = disc.moments((-1, 0))
    tracers = M.copy()
    dt = 0.5*drift.max_timestep(disc)
    drift(dt, disc, dust_tracers=tracers)
    assert np.allclose(tracers, disc.moments((-1, 0)), rtol=1e-6)

    with pytest.raises(ValueError):
        MomentDrift(scheme='tripod', area_tracers=[0])


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
    theta_min = new.ice['H2O']*conv/Xm*np.exp(-chem._ln_aA(s, L))   # at amin
    thick = theta_min > 10
    dt = 3*np.median(theta_min[thick])/Sd[0]
    X0 = new.ice['H2O']*conv
    chem.update(dt, T, 1e-6*rho, eps, new)
    loss = X0 - new.ice['H2O']*conv
    assert np.all(loss[thick] <= Sd[thick]*Xm[thick]*dt*(1 + 1e-6))
    deep = thick & (X0 > 2*Sd*Xm*dt)
    assert deep.any()
    assert np.all(new.ice['H2O'][deep]*conv > 0.5*X0[deep])
