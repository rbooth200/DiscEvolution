# test_viscous_evo.py
#
# Author: R. Booth
# Date: 22 - May - 2018
#
# ChecksLynden-Bell & Pringle solutions are reproduced
###############################################################################
import numpy as np

from DiscEvolution.disc      import AccretionDisc
from DiscEvolution.grid      import Grid
from DiscEvolution.constants import AU, Msun
from DiscEvolution.eos       import LocallyIsothermalEOS
from DiscEvolution.star      import SimpleStar

from DiscEvolution.viscous_evolution import LBP_Solution, ViscousEvolution



def test_LP_model():
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

    visc = ViscousEvolution()

    yr = 2*np.pi
    
    # Integrate to specfi

    times = np.logspace(0, 6, 7)*yr

    max_err   = [3.6e-4, 3.6e-3, 3.6e-2, 3.6e-1, 2.2, 7.9, 17.5]

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

