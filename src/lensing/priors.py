"""Lens-parameter priors and their quadrature weights.

All priors are defined on the physical parameters ``(u = log M_Lz, y)`` and
evaluated on the ``(t_d, y)`` product grid of :mod:`lensing.grids`.  At fixed
``y``, ``u = log t_d - log(4 T_sun DT(y))``, so ``du = d(log t_d) = dt_d / t_d``.
:func:`assign_weights` therefore supplies a ``1/t_d`` Jacobian for a linearly
spaced delay axis and omits it when the axis already is ``log t_d`` -- the
physical prior is the same either way ("if gridding uniformly in
``u = log M``, a log-uniform mass prior has constant density in ``u``; do not
multiply by an extra ``1/M`` again").

Required priors:

===== ===================================================================
P0    log-uniform ``M_Lz``, area prior ``p(y) ~ y``          (primary)
P1    log-uniform ``M_Lz``, uniform ``y``
P2    deliberately broader lens-mass / domain prior
P3    population-mismatched prior, fixed before looking at efficiencies
===== ===================================================================

Every prior's normalization is unit-tested (``tests/test_priors.py``).
"""

from __future__ import annotations

import numpy as np

from . import grids as _grids


# --------------------------------------------------------------------------
# Density definitions on (log M_Lz, y)
# --------------------------------------------------------------------------


def density_P0(u, y):
    """Log-uniform in ``M_Lz``, area prior ``p(y) ~ y``. The primary prior."""
    return np.ones_like(u) * y


def density_P1(u, y):
    """Log-uniform in ``M_Lz``, uniform in ``y``."""
    return np.ones_like(u) * np.ones_like(y)


def density_P2(u, y):
    """Broader prior: still log-uniform in mass, ``p(y) ~ y``, but the domain
    is widened by :func:`grid_for_prior` rather than the density reshaped.

    Keeping the shape identical to P0 isolates the effect of the *domain*,
    i.e. a deliberately broader lens-mass/domain prior.
    """
    return np.ones_like(u) * y


def density_P3(u, y):
    """Population-mismatched prior, declared before any efficiency was seen.

    Mass-weighted toward *light* lenses (``p(M_Lz) ~ 1/M_Lz^2``, i.e.
    ``p(u) ~ exp(-u)``) and toward *small* impact parameters
    (``p(y) ~ 1/y``).  Both choices push prior mass into the corner where the
    two images overlap most strongly and where geometric optics is worst, so
    it is a genuinely adversarial mismatch rather than a mild reweighting.
    """
    return np.exp(-u) / y


DENSITIES = {
    "P0": density_P0,
    "P1": density_P1,
    "P2": density_P2,
    "P3": density_P3,
}

DESCRIPTIONS = {
    "P0": "log-uniform M_Lz, p(y) ~ y (area), M_Lz in [1e2, 1e5], y in [0.01, 2]",
    "P1": "log-uniform M_Lz, uniform y, same domain as P0",
    "P2": "as P0 but broader domain: M_Lz in [10, 1e6], y in [0.005, 3]",
    "P3": "mismatched: p(M_Lz) ~ 1/M_Lz^2, p(y) ~ 1/y, same domain as P0",
}


# --------------------------------------------------------------------------
# Grids that go with each prior
# --------------------------------------------------------------------------


def grid_for_prior(name, n_td=45, n_y=25, t_d_min=1e-3, t_d_max=0.5, **kw):
    """Build the grid appropriate to a named prior.

    Only P2 changes the domain; the others share P0's.
    """
    if name == "P2":
        defaults = dict(
            y_min=0.005, y_max=3.0, M_Lz_min=10.0, M_Lz_max=1e6
        )
    else:
        defaults = dict(y_min=0.01, y_max=2.0, M_Lz_min=1e2, M_Lz_max=1e5)
    defaults.update(kw)
    return _grids.build_grid(
        n_td=n_td, n_y=n_y, t_d_min=t_d_min, t_d_max=t_d_max, **defaults
    )


# --------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------


def trapezoid_widths(x, log_spaced=True):
    """Quadrature widths for a 1-D grid, in the grid's own coordinate.

    For a log-spaced axis the natural coordinate is ``log x``, so the widths
    returned are ``d(log x)``; the caller must then use a density expressed
    per unit ``log x``.
    """
    coord = np.log(x) if log_spaced else np.asarray(x, dtype=float)
    w = np.empty_like(coord)
    if coord.size == 1:
        return np.ones_like(coord)
    w[1:-1] = 0.5 * (coord[2:] - coord[:-2])
    w[0] = coord[1] - coord[0]
    w[-1] = coord[-1] - coord[-2]
    return w


def assign_weights(grid, name="P0", y_log_spaced=True):
    """Attach normalized quadrature weights for prior ``name`` to ``grid``.

    The weight of an active grid point is

        w_ij  ~  p(u_ij, y_j) * dv_i * dY_j

    with ``dv = du = d(log t_d)`` and ``dY`` the width in the
    ``y`` grid's own coordinate.  Because the ``y`` axis is log-spaced by
    default, ``dY = d(log y)`` and the density must be multiplied by ``y`` to
    convert ``p(y) dy -> p(y) y d(log y)``.  Weights are normalized so that
    ``sum w = 1`` over the mask.

    The masked-out points get weight exactly zero.
    """
    if name not in DENSITIES:
        raise ValueError(f"unknown prior {name!r}; choose from {sorted(DENSITIES)}")

    u = np.log(grid.M_Lz)  # (n_td, n_y)
    Y = grid.y[None, :]
    dens = DENSITIES[name](u, Y)

    # The physical prior is a density in u = log M_Lz.  At fixed y,
    # u = log t_d + const, so du = d(log t_d) = dt_d / t_d.  For a log-spaced
    # delay axis the grid coordinate already *is* log t_d and no Jacobian is
    # needed; for a linear axis the 1/t_d factor must be supplied.  Getting
    # this wrong would tilt every prior along the mass direction.
    td_log_spaced = grid.meta.get("td_spacing", "log") == "log"
    if td_log_spaced:
        dv = trapezoid_widths(grid.t_d, log_spaced=True)[:, None]
    else:
        dv = (
            trapezoid_widths(grid.t_d, log_spaced=False) / grid.t_d
        )[:, None]
    if y_log_spaced:
        dY = trapezoid_widths(grid.y, log_spaced=True)[None, :]
        measure = dens * grid.y[None, :]  # p(y) dy = p(y) y dlog y
    else:
        dY = trapezoid_widths(grid.y, log_spaced=False)[None, :]
        measure = dens

    w = measure * dv * dY
    w = np.where(grid.mask, w, 0.0)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"prior {name!r} produced non-positive total weight")
    grid.weights = w / total
    grid.meta = dict(grid.meta)
    grid.meta.update(
        {
            "prior": name,
            "prior_description": DESCRIPTIONS[name],
            "y_coordinate": "log y" if y_log_spaced else "y",
            "weight_sum": 1.0,
        }
    )
    return grid


def effective_n_points(grid) -> float:
    """Kish effective sample size ``1 / sum w^2``: how many points the prior
    really uses.  A prior concentrated on a handful of grid points has a small
    value here and its soft statistic will behave like a maximum."""
    w = grid.weights
    return float(1.0 / np.sum(w * w))
