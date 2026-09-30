"""Mono-mass strong-lensing pair population for the faint-image study.

Scope
-----
This module supplies the *deterministic* part of the faint-image study defined
by ``STRONG_LENSING_FAINT_IMAGE_STUDY.md``: the point-lens geometric-optics map
between impact parameter, time delay and amplitude ratio at a **fixed
redshifted lens mass**, the priors on that one-parameter family, and the
decomposition of the 10-60 minute delayed search window into independent
blocks.

Nothing here knows about noise, statistics or thresholds; those live in
:mod:`lensing.conditional_statistics`.

Why a one-parameter family
--------------------------
The lens mass is fixed, so ``y`` alone determines both the delay ``t_d(y)`` and
the amplitude ratio ``a(y)``.  There is therefore **no independent
two-dimensional scan** over delay and amplitude ratio in this experiment, in
contrast to the microlensing search grid of :mod:`lensing.grids`: every delay
node carries the amplitude ratio that the population forces on it.  That
constraint is the whole point -- it is the information the conditional
statistic exploits.

Geometric optics is exact here.  For ``M_Lz ~ 1e8 Msun`` the dimensionless
frequency ``w = 8 pi G M_Lz f / c^3`` is ``~ 2e6`` at 20 Hz, so wave-optics
corrections are utterly negligible and the two images are cleanly separated
pulses.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import brentq

from . import amplification as amp
from . import conventions

#: ``G M_sun / c^3`` in seconds, the same constant :mod:`lensing.amplification`
#: uses for ``time_delay``.
T_SUN = amp.T_SUN


# --------------------------------------------------------------------------
# The population
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MonoMassPopulation:
    """Fixed-``M_Lz`` point-lens population restricted to a delay window.

    ``M_Lz`` and ``y_min`` are *solved* from the requested delay window rather
    than hard-coded, as the specification requires; :meth:`solve` is the
    intended constructor.
    """

    M_Lz: float
    y_min: float
    y_max: float
    t_d_min: float
    t_d_max: float

    # -- construction ------------------------------------------------------

    @classmethod
    def solve(cls, t_d_min: float = 600.0, t_d_max: float = 3600.0,
              y_max: float = 1.0, M_bracket=(1e4, 1e12)) -> "MonoMassPopulation":
        """Solve ``t_d(M_Lz, y_max) = t_d_max`` then ``t_d(M_Lz, y_min) = t_d_min``."""
        if not 0.0 < t_d_min < t_d_max:
            raise ValueError("need 0 < t_d_min < t_d_max")
        M_Lz = brentq(
            lambda M: amp.time_delay(M, y_max) - t_d_max, *M_bracket, xtol=1e-6,
            rtol=1e-15,
        )
        y_min = brentq(
            lambda y: amp.time_delay(M_Lz, y) - t_d_min, 1e-8, y_max,
            xtol=1e-14, rtol=1e-15,
        )
        return cls(float(M_Lz), float(y_min), float(y_max),
                   float(t_d_min), float(t_d_max))

    # -- the map -----------------------------------------------------------

    def t_d(self, y):
        """Delay of the trailing (saddle) image, seconds."""
        return amp.time_delay(self.M_Lz, np.asarray(y, dtype=float))

    def y_of_t_d(self, t_d):
        """Inverse of :meth:`t_d`.  Vectorized, bracketed by the domain."""
        t_d = np.atleast_1d(np.asarray(t_d, dtype=float))
        out = np.empty_like(t_d)
        lo, hi = 1e-8, 10.0 * self.y_max
        for i, td in enumerate(t_d):
            out[i] = brentq(lambda y: amp.time_delay(self.M_Lz, y) - td,
                            lo, hi, xtol=1e-14, rtol=1e-15)
        return out if out.size > 1 else float(out[0])

    def a(self, y):
        """Amplitude ratio ``rho_- / rho_+ = sqrt(|mu_-| / mu_+)``."""
        return amp.a_of_y(np.asarray(y, dtype=float))

    def mu(self, y):
        """``(mu_+, mu_-)`` of the leading minimum and trailing saddle."""
        return amp.magnifications(np.asarray(y, dtype=float))

    def dt_d_dy(self, y):
        """``d t_d / d y = (4 G M_Lz / c^3) sqrt(y^2 + 4)``, seconds."""
        y = np.asarray(y, dtype=float)
        return 4.0 * T_SUN * self.M_Lz * np.sqrt(y * y + 4.0)

    # -- priors ------------------------------------------------------------

    def p_y(self, y):
        """Area prior ``2 y / (y_max^2 - y_min^2)`` on ``[y_min, y_max]``."""
        y = np.asarray(y, dtype=float)
        norm = self.y_max ** 2 - self.y_min ** 2
        inside = (y >= self.y_min) & (y <= self.y_max)
        return np.where(inside, 2.0 * y / norm, 0.0)

    def p_t_d(self, t_d):
        """Induced delay prior ``p[y(t_d)] |dy/dt_d|``, per second."""
        y = np.atleast_1d(self.y_of_t_d(t_d))
        return np.squeeze(self.p_y(y) / self.dt_d_dy(y))

    def relative_morse_quarters(self) -> int:
        """``q`` for a leading minimum and a trailing saddle: ``1``."""
        from . import lens_models
        return lens_models.required_rel_morse(0, 1)

    def image_coeff(self, y, conv=None):
        """``c = s_M a(y) i^q`` from :mod:`lensing.conventions`, never inline."""
        conv = conv or conventions.DEFAULT
        q = self.relative_morse_quarters()
        if conv.rel_morse_quarters != q:
            conv = conv.with_rel_morse(q)
        return conv.image_coeff(self.a(y))

    def filter_coeff(self, y, conv=None):
        """Coefficient multiplying the delayed SNR sample, ``conj(c)``."""
        conv = conv or conventions.DEFAULT
        q = self.relative_morse_quarters()
        if conv.rel_morse_quarters != q:
            conv = conv.with_rel_morse(q)
        return conv.filter_coeff(self.a(y))

    # -- bookkeeping -------------------------------------------------------

    def as_dict(self) -> dict:
        d = asdict(self)
        d["a_at_y_min"] = float(self.a(self.y_min))
        d["a_at_y_max"] = float(self.a(self.y_max))
        d["rel_morse_quarters"] = self.relative_morse_quarters()
        return d


# --------------------------------------------------------------------------
# Quadrature over the family
# --------------------------------------------------------------------------


def prior_nodes(pop: MonoMassPopulation, n: int, kind: str = "area"):
    """Gauss-Legendre nodes in ``y`` with normalized prior weights.

    ``kind`` selects the prior: ``"area"`` is ``p(y) ~ y`` (the physical area
    prior), ``"uniform"`` is ``p(y) = const`` (the SL5 ablation).
    """
    x, w = np.polynomial.legendre.leggauss(int(n))
    half = 0.5 * (pop.y_max - pop.y_min)
    mid = 0.5 * (pop.y_max + pop.y_min)
    y = mid + half * x
    jac = half * w
    if kind == "area":
        dens = 2.0 * y / (pop.y_max ** 2 - pop.y_min ** 2)
    elif kind == "uniform":
        dens = np.full_like(y, 1.0 / (pop.y_max - pop.y_min))
    else:
        raise ValueError(f"unknown prior kind {kind!r}")
    wt = dens * jac
    return y, wt / wt.sum()


# --------------------------------------------------------------------------
# Block decomposition of the delayed search window
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowBlocks:
    """The delayed search window as ``n_block`` independent blocks.

    The window ``[t_d_min, t_d_max]`` is cut into blocks of duration
    ``block_dur``.  Within a block the amplitude ratio ``a(y)`` is constant to
    better than one part in ``1e3`` (the delay changes by ``block_dur`` out of
    ``>= 600`` s), so a single ``a`` per block is used; :attr:`a_spread` records
    the worst within-block variation actually incurred so the approximation is
    reported rather than assumed.
    """

    t_d_min: float
    t_d_max: float
    block_dur: float
    centers: np.ndarray      # block-center delays, seconds
    a: np.ndarray            # amplitude ratio at the block center
    y: np.ndarray            # impact parameter at the block center
    log_p_t_d: np.ndarray    # log of the induced delay prior at the center
    a_spread: float          # max |a(edge) - a(center)| / a(center)

    @property
    def n_block(self) -> int:
        return int(self.centers.size)

    def sigma(self, rho_plus):
        """``sigma_k = a_k rho_+``, the per-block predicted faint-image SNR."""
        return self.a * float(rho_plus)


def make_blocks(pop: MonoMassPopulation, block_dur: float = 4.0) -> WindowBlocks:
    """Cut the delayed window into blocks of ``block_dur`` seconds."""
    span = pop.t_d_max - pop.t_d_min
    n = int(np.floor(span / block_dur + 1e-9))
    if n < 2:
        raise ValueError("block_dur too long for the requested delay window")
    edges = pop.t_d_min + block_dur * np.arange(n + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    y_c = np.atleast_1d(pop.y_of_t_d(centers))
    a_c = pop.a(y_c)
    y_e = np.atleast_1d(pop.y_of_t_d(edges))
    a_e = pop.a(y_e)
    spread = float(np.max(np.abs(
        np.stack([a_e[:-1], a_e[1:]]) - a_c[None, :]) / a_c[None, :]))
    p = pop.p_y(y_c) / pop.dt_d_dy(y_c)
    return WindowBlocks(
        t_d_min=float(pop.t_d_min), t_d_max=float(pop.t_d_min + n * block_dur),
        block_dur=float(block_dur), centers=centers, a=a_c, y=y_c,
        log_p_t_d=np.log(p), a_spread=spread,
    )


# --------------------------------------------------------------------------
# The long-lag autocorrelation test (gate SL1)
# --------------------------------------------------------------------------


def autocorr_long_lag(template, psd, cfg, lags, n_f=1 << 20):
    """``C(tau)`` at lags of minutes to hours, exactly, by oscillatory quadrature.

    A segment FFT cannot be used here.
    :func:`lensing.waveforms.autocorr_series` resolves lags only up to
    ``cfg.seg_dur`` and aliases anything longer, and
    :func:`lensing.waveforms.autocorr_at_lags` sums over the segment's coarse
    frequency grid, whose spacing ``1 / seg_dur`` cannot represent
    ``exp(2 pi i f tau)`` for ``tau ~ 1e3`` s.  A dense trapezoid rule is no
    better: it would need ``df << 1 / tau ~ 3e-4`` Hz *and* many points per
    oscillation, i.e. ``> 1e8`` samples.

    Instead the normalized power ``p(f) = 4 df |h|^2 / S_n`` is interpolated
    linearly onto ``n_f`` nodes and the oscillatory integral of that
    **piecewise-linear interpolant is evaluated in closed form**.  The result is
    exact for the interpolant at any lag, and the interpolation error
    contributes at most ``O(df_fine^2 max|p''|)`` to ``|C|`` independently of
    ``tau`` -- so a modest ``n_f`` suffices even for hour-scale lags.

    Returns the complex ``C(tau)`` normalized so that ``C(0) = 1``.
    """
    power = 4.0 * cfg.delta_f * np.abs(template) ** 2 / psd
    power = np.where(np.isfinite(power), power, 0.0)
    f = cfg.frequencies()
    band = power > 0.0
    if not band.any():
        raise ValueError("template has no in-band power")
    f_lo, f_hi = float(f[band][0]), float(f[band][-1])

    f_fine = np.linspace(f_lo, f_hi, int(n_f))
    p_fine = np.interp(f_fine, f, power)
    p_fine = p_fine / np.trapezoid(p_fine, f_fine)

    h = f_fine[1] - f_fine[0]
    p_a = p_fine[:-1]
    slope = (p_fine[1:] - p_a) / h
    f_a = f_fine[:-1]

    sign = -conventions.DEFAULT.fd_delay_sign
    lags = np.atleast_1d(np.asarray(lags, dtype=float))
    out = np.empty(lags.size, dtype=complex)
    for i, tau in enumerate(lags):
        k = sign * 2.0 * np.pi * float(tau)
        if abs(k * h) < 1e-10:                     # lag ~ 0: no oscillation
            out[i] = np.trapezoid(p_fine * np.exp(1j * k * f_fine), f_fine)
            continue
        e_h = np.exp(1j * k * h)
        i0 = (e_h - 1.0) / (1j * k)                # int_0^h e^{iku} du
        i1 = h * e_h / (1j * k) - i0 / (1j * k)    # int_0^h u e^{iku} du
        out[i] = np.sum(np.exp(1j * k * f_a) * (p_a * i0 + slope * i1))
    return out if out.size > 1 else out[0]


def f_moments(template, psd, cfg):
    """``(f_bar, f_rms, sigma_f)`` of the normalized template power.

    ``f_rms`` is the zero-up-crossing rate of the real process
    ``g(t) = Re[e^{-i theta} z(t)]`` in units of s^-1, i.e. the quantity that
    sets the look-elsewhere penalty of the conditional statistic.  ``sigma_f``
    is the effective bandwidth, which sets the envelope up-crossing rate of
    ``|z|``.
    """
    power = 4.0 * cfg.delta_f * np.abs(template) ** 2 / psd
    power = np.where(np.isfinite(power), power, 0.0)
    p = power / power.sum()
    f = cfg.frequencies()
    f_bar = float(np.sum(p * f))
    f_rms = float(np.sqrt(np.sum(p * f * f)))
    return f_bar, f_rms, float(np.sqrt(max(f_rms ** 2 - f_bar ** 2, 0.0)))


# --------------------------------------------------------------------------
# Two-dimensional lens prior: M_Lz unknown as well as y
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LensPopulation2D:
    """Point-lens population with **both** ``M_Lz`` and ``y`` unknown.

    The mono-mass population of :class:`MonoMassPopulation` fixes ``M_Lz``, so
    ``t_d(y)`` and ``a(y)`` are locked and the ``(t_d, a)`` plane is traversed
    along a single curve.  A real search does not know the lens mass, and at
    fixed delay the amplitude ratio is then essentially unconstrained -- at
    ``t_d = 1800`` s, ``M_Lz`` from ``3e7`` to ``3e8 Msun`` gives ``a`` from
    ``0.27`` to ``0.86``.  The search therefore has to cover a two-dimensional
    region, exactly as the overlapping-image search of :mod:`lensing.grids`
    already does.

    The prior transforms unusually cleanly.  With ``p(M_Lz) ~ 1/M_Lz``
    (log-uniform) and ``p(y) ~ y`` (area), and

        t_d = 4 T_sun M_Lz tau(y),      a = a(y),

    the Jacobian is ``|dt_d/dM_Lz| |da/dy| = 4 T_sun tau(y) |a'(y)|``, so

        p(t_d, a) ~ y(a) / ( t_d |a'(y(a))| )

    which is **separable**: log-uniform in ``t_d``, times a fixed density in
    ``a``.  That is the same log-uniform-in-delay structure the overlapping
    search has, and it means the delay and amplitude dimensions can be gridded
    independently.
    """

    M_min: float
    M_max: float
    y_min: float
    y_max: float
    t_d_min: float
    t_d_max: float

    @classmethod
    def default(cls, t_d_min=600.0, t_d_max=3600.0,
                M_min=3e7, M_max=3e8, y_min=0.05, y_max=2.0):
        return cls(float(M_min), float(M_max), float(y_min), float(y_max),
                   float(t_d_min), float(t_d_max))

    # -- the a axis --------------------------------------------------------

    def a_range(self):
        """``a`` is monotone decreasing in ``y``, so the range inverts."""
        return float(amp.a_of_y(self.y_max)), float(amp.a_of_y(self.y_min))

    def y_of_a(self, a):
        return amp.y_of_a(np.asarray(a, dtype=float))

    def density_a(self, a):
        """``f(a) ~ y(a) / |a'(y(a))|``, unnormalized."""
        a = np.asarray(a, dtype=float)
        y = amp.y_of_a(a)
        h = 1e-6 * np.maximum(y, 1e-3)
        dady = (amp.a_of_y(y + h) - amp.a_of_y(y - h)) / (2.0 * h)
        return y / np.abs(dady)

    def M_of(self, t_d, a):
        """The lens mass implied by a ``(t_d, a)`` cell."""
        y = amp.y_of_a(np.asarray(a, dtype=float))
        return np.asarray(t_d, dtype=float) / (
            4.0 * T_SUN * amp.delay_dimensionless(y))

    def in_support(self, t_d, a):
        """Is ``(t_d, a)`` reachable with both ``M_Lz`` and ``y`` in range?"""
        t_d = np.asarray(t_d, dtype=float)
        a = np.asarray(a, dtype=float)
        lo, hi = self.a_range()
        M = self.M_of(t_d, a)
        return ((a >= lo) & (a <= hi) & (M >= self.M_min) & (M <= self.M_max)
                & (t_d >= self.t_d_min) & (t_d <= self.t_d_max))

    def as_dict(self) -> dict:
        d = asdict(self)
        lo, hi = self.a_range()
        d["a_min"], d["a_max"] = lo, hi
        return d


def injection_nodes_2d(pop: "LensPopulation2D", n_a: int = 8, n_t: int = 12):
    """Quadrature nodes over the 2-D prior, for **injections**.

    Returns ``(t_d, a, y, mu_plus, weights)``.  The weights are the normalized
    prior mass of each cell, so a volume integral over this node set is an
    integral over the same population the 2-D search prior describes.

    This exists because the injected population and the marginalization prior
    must be the *same* distribution for the headline comparison.  Injecting on
    the mono-mass curve while searching a 2-D region measures the cost of a
    mismatched prior instead -- a legitimate robustness test, but a different
    question, and one that makes the marginalized statistic pay for amplitudes
    the population never produces.
    """
    lo, hi = pop.a_range()
    a_edges = np.linspace(lo, hi, n_a + 1)
    a_nodes = 0.5 * (a_edges[:-1] + a_edges[1:])
    da = a_edges[1] - a_edges[0]

    # log-uniform in t_d (the separable structure of the prior)
    lt = np.linspace(np.log(pop.t_d_min), np.log(pop.t_d_max), n_t + 1)
    t_nodes = np.exp(0.5 * (lt[:-1] + lt[1:]))
    dlt = lt[1] - lt[0]

    T, A = np.meshgrid(t_nodes, a_nodes, indexing="ij")
    w = np.broadcast_to(pop.density_a(a_nodes) * da, T.shape) * dlt
    w = np.where(pop.in_support(T, A), w, 0.0)
    if w.sum() <= 0:
        raise ValueError("empty 2-D injection support")
    w = w / w.sum()

    y = amp.y_of_a(A)
    mu_plus = amp.magnifications(y)[0]
    keep = w > 0
    return (T[keep], A[keep], y[keep], mu_plus[keep], w[keep])


@dataclass(frozen=True)
class SearchGrid2D:
    """Delay blocks x amplitude rows, with normalized prior weights.

    ``a_rows`` is shared by every block -- the prior is separable, so the
    amplitude axis does not depend on the delay.  ``log_w`` has shape
    ``(n_block, n_a)`` and sums to one over the supported cells; unsupported
    cells carry ``-inf``.
    """

    centers: np.ndarray       # (n_block,) block-center delays, s
    block_dur: float
    a_rows: np.ndarray        # (n_a,)
    log_w: np.ndarray         # (n_block, n_a), normalized over the grid
    n_a_effective: float      # participation ratio of the amplitude axis

    @property
    def n_block(self) -> int:
        return int(self.centers.size)

    def sigma(self, rho_plus):
        """``sigma_kj = a_j rho_+``, materialized at shape ``(n_block, n_a)``.

        The amplitude rows are shared by every block, but the array is
        broadcast out in full so callers can index ``[k, j]`` directly.
        """
        row = self.a_rows * float(rho_plus)
        return np.broadcast_to(row, (self.n_block, row.size)).copy()


def make_search_grid_2d(pop: LensPopulation2D, block_dur: float = 4.0,
                        n_a: int = 8) -> SearchGrid2D:
    """Build the two-dimensional search grid for a 2-D lens prior.

    ``n_a = 8`` matches the amplitude-row count of the frozen overlapping-image
    grid at ``0.97`` minimal match, so the two searches are gridded to
    comparable density.
    """
    span = pop.t_d_max - pop.t_d_min
    n_b = int(np.floor(span / block_dur + 1e-9))
    if n_b < 2:
        raise ValueError("block_dur too long for the delay window")
    centers = pop.t_d_min + block_dur * (np.arange(n_b) + 0.5)

    lo, hi = pop.a_range()
    edges = np.linspace(lo, hi, n_a + 1)
    a_rows = 0.5 * (edges[:-1] + edges[1:])
    da = edges[1] - edges[0]

    # p(t_d, a) ~ f(a) / t_d on the supported region
    f_a = pop.density_a(a_rows)
    w = (f_a[None, :] * da) * (block_dur / centers[:, None])
    w = np.where(pop.in_support(centers[:, None], a_rows[None, :]), w, 0.0)
    tot = w.sum()
    if tot <= 0:
        raise ValueError("the 2-D prior has empty support on this grid")
    w = w / tot

    col = w.sum(axis=0)
    n_a_eff = float(1.0 / np.sum((col / col.sum()) ** 2))

    with np.errstate(divide="ignore"):
        log_w = np.log(w)
    return SearchGrid2D(centers=centers, block_dur=float(block_dur),
                        a_rows=a_rows, log_w=log_w, n_a_effective=n_a_eff)
