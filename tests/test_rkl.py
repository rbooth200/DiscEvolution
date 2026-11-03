# Tests of the RKL2 super-time-stepping (DiscEvolution/rkl.py)
import numpy as np
import pytest

from DiscEvolution.grid import Grid
from DiscEvolution.constants import AU, Msun
from DiscEvolution.eos import LocallyIsothermalEOS
from DiscEvolution.star import SimpleStar
from DiscEvolution.disc import AccretionDisc
from DiscEvolution.viscous_evolution import ViscousEvolutionFV
from DiscEvolution.rkl import rkl2, rkl2_flux, rkl2_stages, ViscousEvolutionRKL


def _laplacian(N):
    """1D periodic diffusion operator with unit diffusivity and spacing"""
    return lambda Y: np.roll(Y, 1) - 2*Y + np.roll(Y, -1)


@pytest.mark.parametrize("ratio", [1., 10., 100.])
def test_rkl2_mode_decay(ratio):
    """A Fourier mode of the heat equation decays at the exact rate, to second
    order, with steps far above the explicit limit (dt_e = 1/2)"""
    N = 64
    x = np.arange(N)
    k = 2*np.pi*2/N
    lam = 2*(np.cos(k) - 1)                     # eigenvalue of the operator
    L = _laplacian(N)
    t_end = 50.
    errs = []
    for n in (8, 16):
        dt = t_end/n
        s = rkl2_stages(dt, 0.5)
        Y = np.sin(k*x)
        for _ in range(n):
            Y = rkl2(L, Y, dt, s)
        errs.append(np.abs(Y - np.exp(lam*t_end)*np.sin(k*x)).max())
    assert errs[1] < 0.3*errs[0] + 1e-12           # converging (order ~2)
    # and stable for the stiffest (zigzag) mode
    dt = ratio*0.5
    s = rkl2_stages(dt, 0.5)
    Y = (-1.)**x
    for _ in range(20):
        Y = rkl2(L, Y, dt, s)
    assert np.abs(Y).max() <= 1 + 1e-12


def test_rkl2_flux_consistent():
    """rkl2_flux returns the state of rkl2 and Y0 + div(G)"""
    N = 50
    rng = np.random.default_rng(2)
    Y0 = 1 + rng.uniform(size=N)
    flux = lambda Y: np.concatenate([[0.], np.diff(Y), [0.]])     # closed boundaries
    div = lambda F: np.diff(F)
    for dt in (0.3, 5., 40.):
        s = rkl2_stages(dt, 0.5)
        Y1 = rkl2(lambda Y: div(flux(Y)), Y0, dt, s)
        Y2, G = rkl2_flux(flux, div, Y0, dt, s)
        assert np.allclose(Y1, Y2, rtol=1e-12, atol=1e-14)
        assert np.allclose(Y2, Y0 + div(G), rtol=1e-14)
        assert np.isclose(Y2.sum(), Y0.sum(), rtol=1e-13)            # conserved


def _lbp_disc():
    grid = Grid(0.5, 500, 200, spacing='natural')
    star = SimpleStar()
    eos = LocallyIsothermalEOS(star, 1/30., -0.25, 1e-3)
    eos.set_grid(grid)
    Mdot = 1e-8*Msun/(2*np.pi)/AU**2
    Sigma = (Mdot/(3*np.pi*eos.nu))*np.exp(-grid.Rc/50.)
    eos.update(0, Sigma)
    return AccretionDisc(grid, star, eos, Sigma)


def test_viscous_rkl_matches_explicit():
    """The RKL2 viscous step with a large step matches the explicit solver,
    and moves tracers consistently with the gas"""
    t_end = 2e4*2*np.pi
    disc_e, disc_r = _lbp_disc(), _lbp_disc()
    ve, vr = ViscousEvolutionFV(), ViscousEvolutionRKL()
    t = 0.
    while t < t_end:
        dt = min(ve.max_timestep(disc_e), t_end - t)
        ve(dt, disc_e)
        t += dt
    n_e = 0
    uniform = 0.3*np.ones(disc_r.Ncells)
    step = np.where(disc_r.R < 20, 1., 0.)
    M0 = (disc_r.Sigma*step*disc_r.grid.dRe2).sum()
    t = 0.
    while t < t_end:
        dt = min(vr.max_timestep(disc_r), 200*ve.max_timestep(disc_r), t_end - t)
        vr(dt, disc_r, [uniform, step])
        t += dt
        n_e += 1
    assert vr.stages > 2
    ok = disc_e.Sigma > 1e-8*disc_e.Sigma.max()
    assert np.allclose(disc_r.Sigma[ok], disc_e.Sigma[ok], rtol=1e-2)
    assert np.allclose(uniform, 0.3, rtol=1e-12)
    # tracer mass changes only through the inner boundary
    assert np.all(step >= -1e-14) and np.all(step <= 1 + 1e-12)
    assert (disc_r.Sigma*step*disc_r.grid.dRe2).sum() <= M0*(1 + 1e-12)


###############################################################################
# Diffusion steppers and the driver
###############################################################################
from DiscEvolution.diffusion import TracerDiffusion
from DiscEvolution.rkl import RKLDiffusion


def test_rkl_diffusion_matches_explicit():
    """RKLDiffusion over a long time agrees with explicit diffusion, and takes
    no part in the time-step"""
    disc = _lbp_disc()
    x = np.exp(-0.5*((np.log(disc.R) - np.log(20.))/0.3)**2)
    D, R = TracerDiffusion(), RKLDiffusion()
    assert R.max_timestep(disc) == np.inf
    dt_e = D.max_timestep(disc)
    ye, yr = x.copy(), x.copy()
    for _ in range(400):
        D.step(dt_e, disc, ye)
    for _ in range(10):
        R.step(40*dt_e, disc, yr)
    assert R.stages > 2
    assert np.abs(yr - ye).max() < 2e-3*x.max()


def _dusty_disc():
    from DiscEvolution.dust_trimom import DustGrowthTriMom
    grid = Grid(0.5, 300, 150, spacing='natural')
    star = SimpleStar()
    eos = LocallyIsothermalEOS(star, 1/30., -0.25, 1e-3)
    eos.set_grid(grid)
    Mdot = 1e-8*Msun/(2*np.pi)/AU**2
    Sigma = (Mdot/(3*np.pi*eos.nu))*np.exp(-grid.Rc/50.)
    eos.update(0, Sigma)
    disc = DustGrowthTriMom(grid, star, eos, 0.01, Sigma=Sigma, feedback=False)
    from DiscEvolution.chemistry import ChemicalAbund, MolecularIceAbund
    gas = ChemicalAbund(['A'], [1.], disc.Ncells)
    ice = ChemicalAbund(['B'], [1.], disc.Ncells)
    gas['A'] = 1e-3
    ice['B'] = 0.5*disc.dust_frac.sum(0)
    disc.chem = MolecularIceAbund(gas=gas, ice=ice)
    return disc


@pytest.mark.parametrize("rkl", [False, True])
def test_driver_conserves_with_tracers(rkl):
    """DiscEvolutionDriver with MomentDrift and a diffusion step (explicit or
    RKL): the dust and tracer masses change only through the boundaries, a
    uniform gas tracer stays uniform, and the ice stays proportional to the
    dust"""
    from DiscEvolution.driver import DiscEvolutionDriver
    from DiscEvolution.dust_trimom import MomentDrift
    disc = _dusty_disc()
    gas = ViscousEvolutionRKL() if rkl else ViscousEvolutionFV()
    diffusion = RKLDiffusion() if rkl else TracerDiffusion()
    drv = DiscEvolutionDriver(disc, gas=gas, dust=MomentDrift(), diffusion=diffusion)
    for _ in range(30):
        drv(1e5*2*np.pi)
    assert np.allclose(disc.chem.gas['A'], 1e-3, rtol=1e-10)
    eps = disc.dust_frac.sum(0)
    ok = eps > 1e-6*eps.max()
    assert np.allclose(disc.chem.ice['B'][ok], 0.5*eps[ok], rtol=1e-6)


def test_driver_explicit_rkl_agree():
    """The same disc evolved by the explicit and the RKL classes agrees to
    the splitting error"""
    from DiscEvolution.driver import DiscEvolutionDriver
    from DiscEvolution.dust_trimom import MomentDrift
    out = []
    for rkl in (False, True):
        disc = _dusty_disc()
        drv = DiscEvolutionDriver(disc, gas=ViscousEvolutionRKL() if rkl else ViscousEvolutionFV(),
                                  dust=MomentDrift(),
                                  diffusion=RKLDiffusion() if rkl else TracerDiffusion())
        while drv.t < 3e3*2*np.pi:
            drv(3e3*2*np.pi)
        out.append((disc.Sigma.copy(), disc.Sigma*disc.dust_frac.sum(0), drv.num_steps))
    (S0, D0, n0), (S1, D1, n1) = out
    assert n1 <= n0
    ok = D0 > 1e-6*D0.max()
    assert np.allclose(S1, S0, rtol=1e-3)
    assert np.abs(D1[ok]/D0[ok] - 1).max() < 0.05


def test_single_fluid_drift_diffuse():
    """SingleFluidDrift.diffuse conserves the dust and the dust tracers (no
    boundary flux) and keeps the tracers proportional to the dust"""
    from DiscEvolution.dust import FixedSizeDust, SingleFluidDrift
    grid = Grid(1, 300, 120, spacing='log')
    star = SimpleStar()
    eos = LocallyIsothermalEOS(star, 1/30., -0.25, 1e-3)
    eos.set_grid(grid)
    Sigma = 100*grid.Rc**-1*np.exp(-grid.Rc/50.)
    eos.update(0, Sigma)
    disc = FixedSizeDust(grid, star, eos, 0.01, [1e-4, 0.1], Sigma=Sigma, feedback=False)
    disc.dust_frac[:, :10] = 0
    disc.dust_frac[:, -10:] = 0
    tr = 0.3*disc.dust_frac.sum(0)
    drift, D = SingleFluidDrift(), TracerDiffusion()
    dA = np.diff(grid.Re**2)
    M0 = (disc.Sigma*disc.dust_frac*dA).sum(1)
    for _ in range(50):
        drift.diffuse(D.max_timestep(disc), disc, D, drift.pre_gas_transport(disc), dust_tracers=tr)
    assert np.allclose((disc.Sigma*disc.dust_frac*dA).sum(1), M0, rtol=1e-10)
    assert np.allclose(tr, 0.3*disc.dust_frac.sum(0), rtol=1e-8, atol=1e-30)
