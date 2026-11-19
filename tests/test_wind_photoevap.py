# test_wind_photoevap.py
#
# Tests of tracer and dust handling in the MHD wind model and the external
# photoevaporation models.
###############################################################################
import numpy as np
import pytest

from DiscEvolution.disc      import AccretionDisc
from DiscEvolution.dust      import FixedSizeDust
from DiscEvolution.grid      import Grid
from DiscEvolution.constants import AU, Msun
from DiscEvolution.eos       import LocallyIsothermalEOS
from DiscEvolution.star      import SimpleStar
from DiscEvolution.viscous_evolution import HybridWindModel, TaboneSolution

yr = 2*np.pi


def _setup(N=200, alpha=5e-3, Rd=30.):
    grid = Grid(0.5, 1000, N, spacing='natural')
    star = SimpleStar()
    eos = LocallyIsothermalEOS(star, 1/30., -0.25, alpha)
    eos.set_grid(grid)
    nud = np.interp(Rd, grid.Rc, eos.nu)
    return grid, star, eos, nud


def test_wind_retained_tracer_single_step():
    """A uniform tracer in `adv` is advected but not lost with the wind, so
    its density is what advection alone gives: Sigma' t' = t (Sigma + dt f)"""
    grid, star, eos, nud = _setup()
    sol = TaboneSolution(1e-2*Msun/AU**2, 30., 2*nud, psi_DW=1)
    disc = AccretionDisc(grid, star, eos, sol(grid.Rc, 0.))
    visc = HybridWindModel(psi_DW=1)

    t0 = 0.01
    tracer = t0*np.ones(grid.Ncells)
    Sigma0 = disc.Sigma.copy()
    dt = visc.max_timestep(disc)

    visc(dt, disc, [], [tracer])
    f = visc._fluxes()

    np.testing.assert_allclose(disc.Sigma*tracer, t0*(Sigma0 + dt*f),
                               rtol=1e-12, atol=1e-300)
    # Gas is lost, the tracer is not, so its concentration rises
    assert (tracer >= t0*(1 - 1e-12)).all()
    assert (tracer > t0).any()


def test_wind_lost_gas_tracer_dusty():
    """Gas tracers (mass fractions of the total Sigma) are lost with the gas at
    the rate s_wind * t * Sigma / Sigma_G"""
    grid, star, eos, nud = _setup()
    sol = TaboneSolution(1e-2*Msun/AU**2, 30., 2*nud, psi_DW=1)
    eps = 0.1
    disc = FixedSizeDust(grid, star, eos, eps, 1e-4, Sigma=sol(grid.Rc, 0.))
    visc = HybridWindModel(psi_DW=1)

    X0 = 1e-3
    X = X0*np.ones(grid.Ncells)
    Sigma0 = disc.Sigma.copy()
    dt = visc.max_timestep(disc)

    visc(dt, disc, [X], [])
    f, s = visc._fluxes(), visc._s_wind

    expected = X0*(Sigma0 + dt*f) - dt*s*X0/(1 - eps)
    np.testing.assert_allclose(disc.Sigma*X, expected, rtol=1e-12)


def test_wind_tabone_solution():
    """HybridWindModel reproduces Tabone et al. (2022), given the total
    effective viscosity (1 + psi) nu"""
    psi = 1.0
    grid, star, eos, nud = _setup()
    sol = TaboneSolution(1e-2*Msun/AU**2, 30., (1 + psi)*nud, psi_DW=psi)
    disc = AccretionDisc(grid, star, eos, sol(grid.Rc, 0.))
    visc = HybridWindModel(psi_DW=psi)

    t, tmax = 0., 1e5*yr
    while t < tmax:
        dt = min(visc.max_timestep(disc), tmax - t)
        visc(dt, disc)
        t += dt

    m = (grid.Rc < 200) & (disc.Sigma > 1e-6*disc.Sigma.max())
    err = np.abs(np.log10(disc.Sigma[m]/sol(grid.Rc[m], t)))
    assert np.median(err) < 0.01


###############################################################################
# External photoevaporation
###############################################################################
from DiscEvolution.photoevaporation import (FixedExternalEvaporation,
                                            TimeExternalEvaporation)


def _mass_lost(disc, Sigma0):
    """Gas mass removed, summed cell by cell to avoid cancellation"""
    dA = np.pi*np.diff((disc.R_edge*AU)**2)
    return ((Sigma0 - disc.Sigma_G)*dA).sum()


def _lbp_disc(N=100):
    grid, star, eos, nud = _setup(N=N)
    sol = TaboneSolution(1e-2*Msun/AU**2, 30., nud, psi_DW=0)
    return AccretionDisc(grid, star, eos, sol(grid.Rc, 0.))


@pytest.mark.parametrize("dt_yr", [1e3, 3e4])
def test_timescale_remove_mass_bookkeeping(dt_yr):
    """_Mcum_gas records the gas actually removed, including cells that were
    already empty (which contribute nothing)"""
    disc = _lbp_disc()
    pe = FixedExternalEvaporation(disc, Mdot=1e-8)

    Sigma0 = disc.Sigma_G.copy()
    for _ in range(5):
        pe(disc, dt_yr*yr)
    assert pe._Mcum_gas > 0
    np.testing.assert_allclose(_mass_lost(disc, Sigma0), pe._Mcum_gas,
                               rtol=1e-10)


def test_time_external_evaporation_runs():
    disc = _lbp_disc()
    with pytest.warns(DeprecationWarning):
        pe = TimeExternalEvaporation(time=1e5)

    Sigma0 = disc.Sigma_G.copy()
    pe(disc, 3e4*yr)
    dM = _mass_lost(disc, Sigma0)
    assert dM > 0
    assert (disc.Sigma >= 0).all()
    np.testing.assert_allclose(dM, pe._Mcum_gas, rtol=1e-10)


###############################################################################
# Dust removal with the gas (remove_dust_with_gas)
###############################################################################
from DiscEvolution.dust import DustGrowthTwoPop


def _dusty(cls, **kw):
    # (N >= 88: with fewer cells the natural grid's ghost cells reach R <= 0)
    grid, star, eos, nud = _setup(N=100)
    sol = TaboneSolution(1e-2*Msun/AU**2, 30., nud, psi_DW=0)
    return cls(grid, star, eos, 0.01, Sigma=sol(grid.Rc, 0.), **kw)


def test_twopop_removal_formula():
    """Default (power-law) entrainment: dust lost = f_g * Sigma_D * f_ent,
    with f_ent = min(1, (a_cut/amax)^(4-p)), split in proportion"""
    disc = _dusty(DustGrowthTwoPop)
    disc._a[1] = 0.1
    disc._eps[1] = 0.6*disc._eps.sum(0)
    disc._eps[0] = 0.4*disc._eps[1]/0.6

    Sigma_G0, Sigma_D0 = disc.Sigma_G.copy(), disc.Sigma_D.copy()
    f_g = 0.3
    a_cut = np.logspace(-4, 0, disc.Ncells)
    dSigma_D = disc.remove_dust_with_gas(f_g*Sigma_G0, a_cut)

    f_ent = np.minimum(1, (a_cut/0.1)**(4 - disc._p))
    np.testing.assert_allclose(dSigma_D, f_g*f_ent*Sigma_D0.sum(0), rtol=1e-13)
    np.testing.assert_allclose(disc.Sigma_G, (1 - f_g)*Sigma_G0, rtol=1e-12)
    np.testing.assert_allclose(disc.Sigma_D.sum(0), Sigma_D0.sum(0) - dSigma_D,
                               rtol=1e-12)
    np.testing.assert_allclose(disc.dust_frac[1]/disc.dust_frac.sum(0), 0.6,
                               rtol=1e-12)


def test_removal_without_entrainment_keeps_dust():
    disc = _dusty(DustGrowthTwoPop)
    Sigma_G0, Sigma_D0 = disc.Sigma_G.copy(), disc.Sigma_D.copy()
    dSigma_D = disc.remove_dust_with_gas(0.5*Sigma_G0, None)
    assert (dSigma_D == 0).all()
    np.testing.assert_allclose(disc.Sigma_G, 0.5*Sigma_G0, rtol=1e-12)
    np.testing.assert_allclose(disc.Sigma_D, Sigma_D0, rtol=1e-12)
