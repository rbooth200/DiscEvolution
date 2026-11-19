# test_viscous_evo.py
#
# Author: R. Booth
# Date: 22 - May - 2018
#
# ChecksLynden-Bell & Pringle solutions are reproduced
###############################################################################
import numpy as np
import pytest

from DiscEvolution.disc      import AccretionDisc
from DiscEvolution.grid      import Grid
from DiscEvolution.constants import AU, Msun
from DiscEvolution.eos       import LocallyIsothermalEOS
from DiscEvolution.star      import SimpleStar

from DiscEvolution.viscous_evolution import (LBP_Solution, ViscousEvolution,
                                             ViscousEvolutionFV)
from DiscEvolution.rkl import ViscousEvolutionRKL


@pytest.mark.filterwarnings("ignore:ViscousEvolution is deprecated")
@pytest.mark.parametrize("Visc", [ViscousEvolutionFV, ViscousEvolutionRKL, ViscousEvolution])
def test_LP_model(Visc):
    alpha = 5e-3

    M = 1e-2 * Msun
    Rd = 30.
    T0 = (2 * np.pi)
    N = 100

    grid = Grid(1.0, 1000, N, spacing='natural')
    star = SimpleStar()
    eos = LocallyIsothermalEOS(star, 1 / 30., -0.25, alpha)
    eos.set_grid(grid)

    nud = np.interp(Rd, grid.Rc, eos.nu)
    sol = LBP_Solution(M, Rd, nud, 1)

    Sigma = sol(grid.Rc, 0.)

    disc = AccretionDisc(grid, star, eos, Sigma)

    visc = Visc()

    yr = 2*np.pi
    
    # Integrate to specfi

    times = np.logspace(0, 6, 7)*yr

    # ~1% above each scheme's errors (FV's, at this resolution, are up to 6%
    # larger, mostly from the flux through the boundaries)
    if Visc is ViscousEvolution:
        max_err = [3.6e-4, 3.6e-3, 3.6e-2, 3.6e-1, 2.2, 7.9, 17.5]
    else:
        max_err = [3.7e-4, 3.7e-3, 3.7e-2, 3.6e-1, 2.3, 8.5, 18.8]

    t = 0
    for i, ti in enumerate(times):
        while t < ti:
            dt = visc.max_timestep(disc)
            dti = min(dt, ti - t)

            visc(dti, disc)

            t = min(t + dt, ti)

        # Check L1, L2 and max norm
        exact = sol(grid.Rc, t)
        error = (exact - disc.Sigma) / exact.mean()

        L1   = np.linalg.norm(error, ord=1) 

        assert(L1   < max_err[i])



def test_ViscousEvolution_deprecated():
    with pytest.warns(DeprecationWarning):
        ViscousEvolution()


@pytest.mark.parametrize("spacing,R0", [('natural', 0.1), ('log', 0.1), ('linear', 10.), (0.25, 0.1)])
@pytest.mark.parametrize("bound,power", [('Mdot', 0.5), ('Mdot_out', -0.5),
                                         ('Mdot_inn', 0.5), ('power_law', -1.3)])
def test_ghost_cells_power_law(spacing, R0, bound, power):
    """The boundary extrapolations are exact for the power laws they assume,
    on any grid (not only log-spaced ones)"""
    from DiscEvolution.viscous_evolution import _set_ghost_cells
    grid = Grid(R0, 1000., 200, spacing=spacing)
    S_exact = grid.Rce**power
    S = S_exact.copy()
    S[0] = S[-1] = np.nan
    in_bound = 'Mdot' if bound == 'Mdot' else 'Zero'
    _set_ghost_cells(S, grid, in_bound, 'Zero' if bound == 'Mdot' else bound)
    if bound == 'Mdot':
        assert np.isclose(S[0], S_exact[0], rtol=1e-12)
    else:
        assert np.isclose(S[-1], S_exact[-1], rtol=1e-12)


@pytest.mark.parametrize("spacing", ['natural', 'log'])
@pytest.mark.parametrize("Visc", [ViscousEvolutionFV, ViscousEvolutionRKL])
def test_steady_accretion_inner_boundary(spacing, Visc):
    """With the constant-Mdot inner boundary, steady accretion (nu Sigma =
    const) passes through the inner edge unchanged"""
    from types import SimpleNamespace
    grid = Grid(0.1, 1000., 500, spacing=spacing)
    nu = grid.Rc.copy()
    disc = SimpleNamespace(grid=grid, nu=nu, Sigma_G=1/nu, Sigma=1/nu)
    visc = Visc()
    visc._setup_grid(grid)
    visc._init_fluxes(disc)
    rate = visc._fluxes() * grid.Rc**2 / nu / disc.Sigma    # dSigma/dt in Sigma / t_nu
    flux = visc._dA * visc._dS
    assert abs(rate[0]) < 1e-2
    assert np.isclose(flux[0], flux[len(flux)//2], rtol=1e-2)
