"""Lens search grids and their quadrature measure.

Coordinate choice
-----------------

The search statistic needs ``z_1(t + tau)`` for every lens point, and a shift
by an integer number of (oversampled) samples is exact and free, while a
fractional one is not.  The grid is therefore laid out as a **product grid in
``(t_d, y)``**:

- every ``y`` shares the same set of ``t_d`` values, so the SNR series is
  rolled ``n_td`` times rather than ``n_td * n_y`` times;
- ``a = a(y)`` depends on ``y`` alone, so the amplitude axis is shared too;
- the delay-domain mask ``t_d in [t_d_min, t_d_max]`` is exact by construction
  rather than an afterthought.

The physical prior lives in ``(u = log M_Lz, y)``.  At fixed ``y``,

    u = log t_d - log(4 T_sun DT(y)),   so   du = d(log t_d) = dt_d / t_d,

and the mass-range restriction becomes a mask.  :mod:`lensing.priors` supplies
the ``1/t_d`` Jacobian when the delay axis is linear and omits it when the axis
already *is* ``log t_d``, so the physical prior does not depend on the gridding
(pinned by ``test_prior_is_independent_of_td_spacing``).

On a log-spaced axis the grid coordinate *is* the log,
so a log-uniform mass prior has constant density there and is **not** multiplied
by a further ``1/M``.

**Delay spacing.** The default is uniform in ``t_d``, not in ``log t_d``,
because the delay metric element is roughly constant beyond ~20 ms, so a
minimal-match criterion needs uniform spacing.  See :func:`build_grid`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import amplification as amp


@dataclass
class LensGrid:
    """A product grid in ``(t_d, y)`` with its quadrature weights.

    Attributes
    ----------
    t_d : (n_td,) array
        Image delays in seconds; spacing per ``meta["td_spacing"]``.
    y : (n_y,) array
        Dimensionless impact parameters, log-spaced.
    a : (n_y,) array
        ``a(y) = sqrt(|mu_-|/mu_+)``, one per ``y``.
    M_Lz : (n_td, n_y) array
        Implied redshifted lens mass at each grid point.
    mask : (n_td, n_y) bool array
        True where the point lies inside the searched domain.
    weights : (n_td, n_y) array
        Prior-weighted quadrature weights, normalized to sum to 1 over the
        mask.  Set by :mod:`lensing.priors`.
    """

    t_d: np.ndarray
    y: np.ndarray
    a: np.ndarray
    M_Lz: np.ndarray
    mask: np.ndarray
    weights: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    # -- shape helpers ----------------------------------------------------

    @property
    def n_td(self) -> int:
        return self.t_d.size

    @property
    def n_y(self) -> int:
        return self.y.size

    @property
    def n_points(self) -> int:
        """Number of *active* grid points."""
        return int(self.mask.sum())

    @property
    def shape(self):
        return (self.n_td, self.n_y)

    def mu_r(self) -> np.ndarray:
        """G26's ``mu_r = 1/a``, for comparison-boundary plotting only."""
        return amp.a_to_mu_r(self.a)

    def as_dict(self) -> dict:
        return {
            "n_td": self.n_td,
            "n_y": self.n_y,
            "n_active": self.n_points,
            "t_d_range_s": [float(self.t_d[0]), float(self.t_d[-1])],
            "y_range": [float(self.y[0]), float(self.y[-1])],
            "a_range": [float(self.a.min()), float(self.a.max())],
            "M_Lz_range_active": [
                float(self.M_Lz[self.mask].min()),
                float(self.M_Lz[self.mask].max()),
            ],
            **self.meta,
        }


def build_grid(
    n_td=45,
    n_y=25,
    t_d_min=1e-3,
    t_d_max=0.5,
    y_min=0.01,
    y_max=2.0,
    M_Lz_min=1e2,
    M_Lz_max=1e5,
    td_spacing="linear",
):
    """Build the default search grid.

    Defaults follow the region covered by the G26 template bank:
    ``t_d in [1, 500] ms``, ``y in [0.01, 2]``, ``M_Lz in [1e2, 1e5] Msun``.
    The mass range enters only as a mask.

    ``td_spacing``
        ``"linear"`` (default) spaces the delays uniformly in ``t_d``;
        ``"log"`` spaces them uniformly in ``log t_d``.

        **Linear is the metric-appropriate choice and the default.** G26's
        Fig. 6 shows the delay metric element ``g_tt`` is roughly constant for
        ``t_d >= 20 ms``, so a minimal-match criterion demands roughly uniform
        spacing in ``t_d``.  A log-spaced axis over ``[1, 500] ms`` places its
        points where they are least needed: with ``n_td = 90`` the spacing near
        500 ms is ~35 ms, hundreds of times coarser than the metric requires,
        which showed up in Phase 3 as scattered recovery losses at large
        ``M_Lz`` on an otherwise 0.99 median.

        The prior measure is handled correspondingly in
        :func:`lensing.priors.assign_weights`, so the physical prior is
        unchanged by this choice.
    """
    if td_spacing == "log":
        t_d = np.geomspace(t_d_min, t_d_max, n_td)
    elif td_spacing == "linear":
        t_d = np.linspace(t_d_min, t_d_max, n_td)
    else:
        raise ValueError(f"td_spacing must be 'linear' or 'log', got {td_spacing!r}")
    y = np.geomspace(y_min, y_max, n_y)
    a = amp.a_of_y(y)

    dt_dimless = amp.delay_dimensionless(y)  # (n_y,)
    # t_d = 4 T_sun M_Lz DT(y)  =>  M_Lz = t_d / (4 T_sun DT(y))
    M_Lz = t_d[:, None] / (4.0 * amp.T_SUN * dt_dimless[None, :])
    mask = (M_Lz >= M_Lz_min) & (M_Lz <= M_Lz_max)

    return LensGrid(
        t_d=t_d,
        y=y,
        a=a,
        M_Lz=M_Lz,
        mask=mask,
        meta={
            "coordinates": f"product grid in (t_d [{td_spacing}], y)",
            "td_spacing": td_spacing,
            "t_d_min_s": t_d_min,
            "t_d_max_s": t_d_max,
            "y_min": y_min,
            "y_max": y_max,
            "M_Lz_min": M_Lz_min,
            "M_Lz_max": M_Lz_max,
        },
    )


def snap_grid_to_samples(grid: LensGrid, delta_t: float) -> LensGrid:
    """Round every ``t_d`` to the nearest available sample and drop duplicates.

    Returned grid has strictly increasing, sample-exact delays, so that the
    shift ``z_1(t + tau)`` involves no interpolation at all.  The number of
    distinct delays can therefore *fall* below ``n_td`` for a coarse
    ``delta_t``; an oversampling study measures the consequence.
    """
    snapped = np.rint(grid.t_d / delta_t) * delta_t
    uniq, idx = np.unique(snapped, return_index=True)
    keep = np.sort(idx)
    dt_dimless = amp.delay_dimensionless(grid.y)
    t_d = snapped[keep]
    M_Lz = t_d[:, None] / (4.0 * amp.T_SUN * dt_dimless[None, :])
    meta = dict(grid.meta)
    meta.update(
        {
            "snapped_to_delta_t": float(delta_t),
            "n_td_before_snap": int(grid.n_td),
            "n_td_after_snap": int(t_d.size),
        }
    )
    lo = meta.get("M_Lz_min", -np.inf)
    hi = meta.get("M_Lz_max", np.inf)
    return LensGrid(
        t_d=t_d,
        y=grid.y.copy(),
        a=grid.a.copy(),
        M_Lz=M_Lz,
        mask=(M_Lz >= lo) & (M_Lz <= hi),
        meta=meta,
    )


def sample_shifts(grid: LensGrid, delta_t: float) -> np.ndarray:
    """Integer sample offsets realizing each ``t_d`` on a grid of step ``delta_t``."""
    return np.rint(grid.t_d / delta_t).astype(int)
