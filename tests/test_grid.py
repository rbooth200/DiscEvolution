# test_grid.py
#
# Grid spacings and their ghost cells
###############################################################################
import numpy as np
import pytest

from DiscEvolution.grid import Grid, MultiResolutionGrid


@pytest.mark.parametrize("name,alpha", [('natural', 0.5), ('linear', 1.0)])
def test_numeric_spacing_matches_named(name, alpha):
    """A numeric power-law index gives the same grid as its named spacing"""
    g0 = Grid(10., 1000., 500, spacing=name)
    g1 = Grid(10., 1000., 500, spacing=alpha)
    assert np.allclose(g1.Ree, g0.Ree, rtol=1e-12)
    assert np.allclose(g1.Rce, g0.Rce, rtol=1e-12)


@pytest.mark.parametrize("alpha", [0.25, -0.5, 0.75])
def test_numeric_spacing(alpha):
    """Power-law grids: increasing edges, centres between them, uniform in R^alpha"""
    g = Grid(10., 1000., 200, spacing=alpha)
    assert (np.diff(g.Ree) > 0).all() and (g.Ree > 0).all()
    assert ((g.Rce > g.Ree[1:-2]) & (g.Rce < g.Ree[2:-1])).all()
    assert np.allclose(np.diff(g.Re**alpha), (g.Re[-1]**alpha - g.Re[0]**alpha)/200)


@pytest.mark.parametrize("R0,N,spacing", [(0.1, 500, 'linear'), (0.1, 50, 'natural'),
                                          (0.1, 200, -1.0)])
def test_ghost_cells_at_negative_radius(R0, N, spacing):
    """Grids whose ghost cells would reach R <= 0 are rejected"""
    with pytest.raises(ValueError):
        Grid(R0, 1000., N, spacing=spacing)


@pytest.mark.parametrize("R0,R1,spacing", [(0., 10., 'log'), (-1., 10., 'linear'),
                                           (10., 1., 'natural')])
def test_invalid_range(R0, R1, spacing):
    with pytest.raises(ValueError):
        Grid(R0, R1, 100, spacing=spacing)


def test_invalid_spacing():
    with pytest.raises(AttributeError):
        Grid(1., 10., 100, spacing='cubic')
    with pytest.raises(ValueError):
        Grid(1., 10., 100, spacing=0)


def test_multiresolution_natural():
    """MultiResolutionGrid still builds its centres from the spacing name"""
    g = MultiResolutionGrid([(1., 10.), (10., 100.)], [50, 20], spacing='natural')
    assert (np.diff(g.Rce) > 0).all()
