import numpy as np
import pytest

from DiscEvolution.grid      import Grid
from DiscEvolution.constants import AU, Msun
from DiscEvolution.eos       import LocallyIsothermalEOS
from DiscEvolution.star      import SimpleStar
from DiscEvolution.dust      import FixedSizeDust, DustGrowthTwoPop, SingleFluidDrift


def _setup_disc(model='fixed', n=120, feedback=False, sizes=(1e-3, 0.1)):
    grid = Grid(1, 300, n, spacing='log')
    star = SimpleStar()
    eos = LocallyIsothermalEOS(star, 1/30., -0.25, 1e-3)
    eos.set_grid(grid)
    Mdot = 1e-8*Msun/(2*np.pi)/AU**2
    Sigma = (Mdot/(3*np.pi*eos.nu))*np.exp(-grid.Rc/100.)
    eos.update(0, Sigma)
    if model == 'fixed':
        return FixedSizeDust(grid, star, eos, 0.01, list(sizes), Sigma=Sigma,
                             feedback=feedback)
    disc = DustGrowthTwoPop(grid, star, eos, 0.01, Sigma=Sigma, feedback=feedback)
    disc.grain_size[1] = 0.1
    return disc


@pytest.mark.parametrize("model", ['fixed', 'twopop'])
@pytest.mark.parametrize("feedback,settling", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("van_leer", [False, True])
def test_drift_conserves_and_carries_tracers(model, feedback, settling, van_leer):
    """The dust mass is conserved (no dust near the boundaries), it stays
    non-negative, and a dust tracer proportional to the dust mass stays so"""
    disc = _setup_disc(model, feedback=feedback)
    disc.dust_frac[:, :15] = 0
    disc.dust_frac[:, -15:] = 0
    drift = SingleFluidDrift(settling=settling, van_leer=van_leer)
    dust = 0.3*disc.dust_frac.sum(0)[None]
    gas = np.ones((1, disc.Ncells))
    dA = np.pi*np.diff((disc.grid.Re*AU)**2)
    M0 = (disc.Sigma*disc.dust_frac.sum(0)*dA).sum()
    for _ in range(20):
        drift(drift.max_timestep(disc), disc, gas_tracers=gas, dust_tracers=dust)
    eps = disc.dust_frac
    assert np.all(np.isfinite(eps)) and eps.min() >= -1e-16*eps.max()
    assert np.isclose((disc.Sigma*eps.sum(0)*dA).sum(), M0, rtol=1e-10)
    np.testing.assert_allclose(dust[0], 0.3*eps.sum(0), rtol=1e-10, atol=1e-14*eps.max())
    if not feedback:
        np.testing.assert_allclose(gas, 1., rtol=1e-12)


@pytest.mark.parametrize("model", ['fixed', 'twopop'])
@pytest.mark.parametrize("settling", [False, True])
@pytest.mark.parametrize("van_leer", [False, True])
def test_gas_tracer_follows_gas(model, settling, van_leer):
    """A gas tracer proportional to the gas stays so, with and without
    settling: the gas tracers are carried by the gas, which carries the
    opposite of the dust's mass flux (the drift leaves Sigma unchanged)"""
    disc = _setup_disc(model, feedback=True)
    disc.dust_frac[:] *= 10                     # eps = 0.1, so that feedback matters
    drift = SingleFluidDrift(settling=settling, van_leer=van_leer)
    G = 0.7*(1 - disc.dust_frac.sum(0))[None]
    for _ in range(20):
        drift(drift.max_timestep(disc), disc, gas_tracers=G)
    np.testing.assert_allclose(G[0], 0.7*(1 - disc.dust_frac.sum(0)), rtol=1e-12)


@pytest.mark.parametrize("van_leer", [False, True])
def test_gas_tracers_bounded_where_dust_dominates(van_leer):
    """Where the dust outweighs the gas, the gas (which carries the opposite
    of the dust's mass flux) can move faster than the dust: max_timestep
    limits the step so that the gas tracers' ratio to the gas stays within
    its initial bounds (single-size dust, dust-to-gas 9)"""
    disc = _setup_disc(feedback=True, sizes=[0.1])
    disc.dust_frac[:] = 0.9
    drift = SingleFluidDrift(van_leer=van_leer)
    X0 = 1 + 0.5*np.sin(4*np.log(disc.R))
    G = (X0*(1 - disc.dust_frac.sum(0)))[None]
    for _ in range(200):
        drift(drift.max_timestep(disc), disc, gas_tracers=G)
    X = G[0]/(1 - disc.dust_frac.sum(0))
    assert X.min() >= X0.min() - 1e-12 and X.max() <= X0.max() + 1e-12


def _temporal_error(van_leer, n, dt, n_steps, f=0.25, f_ref=1/32., window=(10., 50.)):
    """Error of n_steps/f steps of f dt against f_ref dt, in a smooth window"""
    out = {}
    for g in (f, f_ref):
        disc, drift = _setup_disc(n=n), SingleFluidDrift(van_leer=van_leer)
        for _ in range(int(round(n_steps/g))):
            drift(g*dt, disc)
        out[g] = disc.Sigma*disc.dust_frac
    R = disc.grid.Rc
    m = (R >= window[0]) & (R < window[1])
    dA = np.diff(disc.grid.Re**2)
    return max((np.abs(out[f][k] - out[f_ref][k])*dA)[m].sum() /
               (out[f_ref][k]*dA)[m].sum() for k in range(len(out[f])))


def test_vl2_temporal_error_is_a_cross_term():
    """At a fixed time-step, halving the cell size halves the van Leer
    integrator's temporal error (O(dt dx), second order) but not donor
    cell's (O(dt))"""
    dt = SingleFluidDrift().max_timestep(_setup_disc(n=120))
    r = {vl: _temporal_error(vl, 120, dt, 40)/_temporal_error(vl, 240, dt, 40)
         for vl in (False, True)}
    assert r[True] > 1.7
    assert r[False] < 1.1


def test_vl2_spatial_convergence():
    """At a fixed Courant number, the van Leer integrator converges faster
    than donor cell towards a fine-grid solution (in a smooth window)"""
    def run(van_leer, n, T):
        disc, drift = _setup_disc(n=n), SingleFluidDrift(van_leer=van_leer)
        t = 0.
        while t < T:
            dt = min(drift.max_timestep(disc), T - t)
            drift(dt, disc)
            t += dt
        return disc
    T = 40*SingleFluidDrift().max_timestep(_setup_disc(n=120))
    ref = run(True, 960, T)
    def err(van_leer, n):
        d = run(van_leer, n, T)
        k = 960//n
        # Coarse cells are unions of k fine cells (same log grid)
        m_ref = (ref.Sigma*ref.dust_frac*np.diff(ref.grid.Re**2)).reshape(2, n, k).sum(-1)
        m = d.Sigma*d.dust_frac*np.diff(d.grid.Re**2)
        w = (d.grid.Rc >= 10) & (d.grid.Rc < 50)
        return np.abs(m - m_ref)[:, w].sum()/m_ref[:, w].sum()
    p_vl = np.log2(err(True, 120)/err(True, 240))
    p_dc = np.log2(err(False, 120)/err(False, 240))
    assert p_vl > 1.5
    assert p_dc < 1.3
