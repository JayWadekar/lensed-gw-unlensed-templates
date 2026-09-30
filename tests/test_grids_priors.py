"""Lens grid geometry and prior normalization."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lensing import amplification as amp
from lensing import grids
from lensing import priors


def test_grid_mass_consistency():
    """M_Lz implied by (t_d, y) must reproduce t_d exactly."""
    g = grids.build_grid(n_td=13, n_y=11)
    td_back = amp.time_delay(g.M_Lz, g.y[None, :])
    assert np.allclose(td_back, g.t_d[:, None], rtol=1e-12)


def test_grid_mask_respects_mass_range():
    g = grids.build_grid(n_td=40, n_y=20, M_Lz_min=1e2, M_Lz_max=1e5)
    assert g.mask.any()
    assert np.all(g.M_Lz[g.mask] >= 1e2 - 1e-9)
    assert np.all(g.M_Lz[g.mask] <= 1e5 + 1e-9)
    assert np.all(g.a > 0) and np.all(g.a < 1)


def test_snapping_makes_shifts_exact():
    """After snapping, every t_d is an integer number of samples."""
    g = grids.build_grid(n_td=45, n_y=10)
    dt = 1.0 / 4096.0
    gs = grids.snap_grid_to_samples(g, dt)
    n = grids.sample_shifts(gs, dt)
    assert np.allclose(gs.t_d, n * dt, rtol=0, atol=1e-15)
    assert gs.t_d.size == np.unique(gs.t_d).size
    assert gs.t_d.size <= g.t_d.size
    assert np.all(np.diff(gs.t_d) > 0)


def test_snapping_keeps_mass_consistency():
    g = grids.build_grid(n_td=45, n_y=10)
    gs = grids.snap_grid_to_samples(g, 1.0 / 4096.0)
    td_back = amp.time_delay(gs.M_Lz, gs.y[None, :])
    assert np.allclose(td_back, gs.t_d[:, None], rtol=1e-12)


@pytest.mark.parametrize("name", ["P0", "P1", "P2", "P3"])
def test_prior_weights_normalized(name):
    """Normalize quadrature weights so sum_k w_k = 1."""
    g = priors.grid_for_prior(name, n_td=30, n_y=20)
    priors.assign_weights(g, name)
    assert g.weights.sum() == pytest.approx(1.0, rel=1e-12)
    assert np.all(g.weights >= 0)
    # masked-out points carry exactly zero weight
    assert np.all(g.weights[~g.mask] == 0.0)
    assert priors.effective_n_points(g) > 1.0


def test_priors_are_actually_different():
    """P0..P3 must not silently collapse to the same weights."""
    ws = {}
    for name in ("P0", "P1", "P3"):
        g = priors.grid_for_prior(name, n_td=30, n_y=20)
        priors.assign_weights(g, name)
        ws[name] = g.weights.copy()
    assert not np.allclose(ws["P0"], ws["P1"])
    assert not np.allclose(ws["P0"], ws["P3"])
    # P3 pushes weight to small y; P0's area prior pushes it to large y
    g = priors.grid_for_prior("P0", n_td=30, n_y=20)
    y = g.y
    mean_y_P0 = (ws["P0"].sum(axis=0) * y).sum() / ws["P0"].sum()
    mean_y_P3 = (ws["P3"].sum(axis=0) * y).sum() / ws["P3"].sum()
    assert mean_y_P3 < mean_y_P0


def test_P2_domain_is_broader():
    g0 = priors.grid_for_prior("P0", n_td=20, n_y=15)
    g2 = priors.grid_for_prior("P2", n_td=20, n_y=15)
    assert g2.y[0] < g0.y[0] and g2.y[-1] > g0.y[-1]
    assert g2.meta["M_Lz_min"] < g0.meta["M_Lz_min"]
    assert g2.meta["M_Lz_max"] > g0.meta["M_Lz_max"]


def test_log_uniform_mass_prior_has_no_extra_jacobian():
    """Guard against multiplying by 1/M twice.

    A prior flat in u = log M_Lz must give, at fixed y, weights proportional to
    d(log t_d) alone -- because du = dv exactly in these coordinates.
    """
    for spacing in ("log", "linear"):
        g = priors.grid_for_prior(
            "P1", n_td=25, n_y=1, y_min=0.5, y_max=0.5, td_spacing=spacing
        )
        priors.assign_weights(g, "P1")
        w = g.weights[g.mask]
        if spacing == "log":
            dv = priors.trapezoid_widths(g.t_d, log_spaced=True)
        else:
            dv = priors.trapezoid_widths(g.t_d, log_spaced=False) / g.t_d
        dv = dv[g.mask[:, 0]]
        assert np.allclose(w / w.sum(), dv / dv.sum(), rtol=1e-12), spacing


def test_prior_is_independent_of_td_spacing():
    """The physical prior must not depend on how the delay axis is gridded.

    A dense grid in either spacing must give the same prior-weighted mean of
    log M_Lz, to within quadrature error.
    """
    means = {}
    for spacing in ("log", "linear"):
        g = priors.grid_for_prior(
            "P0", n_td=4000, n_y=40, td_spacing=spacing
        )
        priors.assign_weights(g, "P0")
        means[spacing] = float(
            (g.weights * np.log(g.M_Lz)).sum() / g.weights.sum()
        )
    assert means["log"] == pytest.approx(means["linear"], rel=2e-3)


def test_unknown_prior_raises():
    g = grids.build_grid(n_td=5, n_y=5)
    with pytest.raises(ValueError, match="unknown prior"):
        priors.assign_weights(g, "nope")
