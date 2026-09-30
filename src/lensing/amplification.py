"""Isolated point-mass lens: magnifications, time delay, and the amplification
factor F(w, y) in both wave optics and geometric optics.

Conventions.  With ``M_Lz`` the redshifted lens mass and ``y`` the
dimensionless impact parameter in units of the Einstein radius:

    x_m(y)   = (y + sqrt(y^2 + 4)) / 2
    mu_pm(y) = 1/2 +/- (y^2 + 2) / (2 y sqrt(y^2 + 4))
    DT(y)    = y sqrt(y^2+4)/2 + log[(sqrt(y^2+4)+y)/(sqrt(y^2+4)-y)]
    t_d      = (4 G M_Lz / c^3) DT(y)

    a(y)     = sqrt(|mu_-| / mu_+),   0 < a < 1        [this project]
    mu_r     = sqrt(mu_+ / |mu_-|) = 1/a               [G26]

The dimensionless frequency is

    w = 8 pi G M_Lz f / c^3,

so that  2 pi f t_d = w * DT(y)  exactly, and the geometric-optics limit is

    F_GO(w, y) = sqrt(mu_+) - i sqrt(|mu_-|) exp(i w DT(y))

(the relative -i is the Morse phase of the saddle image at positive
frequencies; its sign is *not* re-derived here but taken from
``conventions.DEFAULT`` and verified by the Phase 0 gate).

The exact wave-optics factor for a point lens is

    F(w, y) = exp[ pi w / 4 + i (w/2) ( log(w/2) - 2 phi_m(y) ) ]
              * Gamma(1 - i w / 2)
              * 1F1(i w / 2, 1; i w y^2 / 2),

    phi_m(y) = (x_m - y)^2 / 2 - log(x_m).

The exponential prefactor and the gamma function are combined in log space
 because individually they overflow/underflow like exp(+/- pi w/4).

``scipy.special.hyp1f1`` in the pinned environment (scipy 1.16.2) does **not**
accept complex arguments, so 1F1 is provided here by two backends:

``"mpmath"`` (default, production)
    Arbitrary-precision ``mpmath.hyp1f1``.  Verified self-consistent between
    ``dps=30`` and ``dps=60`` to machine precision in ``log F`` over
    ``w <= 3000``, ``y in [0.01, 2]``, and fast in the domain that matters
    (0.3-0.5 ms per point at small ``y``, where ``|z| = w y^2 / 2`` is small).

``"series"`` (diagnostic only)
    An overflow-scaled Kummer series in ``longdouble``.  It is **unusable** for
    ``w y >~ 30``: the terms peak near ``n ~ w y / 2`` with modulus up to
    ~1e19 while the sum is O(1), so double/longdouble cancellation destroys
    every digit.  Kept only to document the failure and to cross-check the
    small-argument corner.

We avoid a slow arbitrary-precision path in the main Monte
Carlo; the mitigation here is the frequency-grid cache in
:func:`F_ML_on_grid` plus process-level parallelism, not a faster kernel.
"""

from __future__ import annotations

import numpy as np
from scipy.special import loggamma

# G M_sun / c^3 in seconds.  Taken from lal to match the waveform generator.
try:  # pragma: no cover - environment dependent
    import lal

    T_SUN = lal.MTSUN_SI
except Exception:  # pragma: no cover
    T_SUN = 4.925490947641267e-06


# --------------------------------------------------------------------------
# Geometry: magnifications, time delay, coordinate conversions
# --------------------------------------------------------------------------


def x_m(y):
    """Image position of the minimum (in Einstein radii)."""
    y = np.asarray(y, dtype=float)
    return 0.5 * (y + np.sqrt(y * y + 4.0))


def magnifications(y):
    """Signed magnifications ``(mu_plus, mu_minus)`` of the two images.

    ``mu_minus`` is negative (saddle point); use ``abs`` for its modulus.
    """
    y = np.asarray(y, dtype=float)
    s = np.sqrt(y * y + 4.0)
    common = (y * y + 2.0) / (2.0 * y * s)
    return 0.5 + common, 0.5 - common


def a_of_y(y):
    """Amplitude ratio ``a = sqrt(|mu_-| / mu_+)``, strictly in (0, 1)."""
    mu_p, mu_m = magnifications(y)
    return np.sqrt(np.abs(mu_m) / mu_p)


def mu_r_of_y(y):
    """G26's relative magnification ``mu_r = sqrt(mu_+/|mu_-|) = 1/a``."""
    return 1.0 / a_of_y(y)


def a_to_mu_r(a):
    """Convert this project's ``a`` to G26's ``mu_r``."""
    return 1.0 / np.asarray(a, dtype=float)


def mu_r_to_a(mu_r):
    """Convert G26's ``mu_r`` to this project's ``a``."""
    return 1.0 / np.asarray(mu_r, dtype=float)


def y_of_a(a):
    """Invert ``a(y)`` analytically.

    With  A = a^2 = |mu_-|/mu_+  and  q = (y^2+2)/(y sqrt(y^2+4)),
    mu_pm = (1 +/- q)/2  so  A = (q-1)/(q+1)  and  q = (1+A)/(1-A).
    Then  y^2 (y^2+4) q^2 = (y^2+2)^2  gives, with  v = y^2 + 2,

        (v^2 - 4) q^2 = v^2   =>   v^2 = 4 q^2 / (q^2 - 1).
    """
    a = np.asarray(a, dtype=float)
    A = a * a
    q = (1.0 + A) / (1.0 - A)
    v = 2.0 * q / np.sqrt(q * q - 1.0)
    return np.sqrt(v - 2.0)


def delay_dimensionless(y):
    """``DT(y)``: time delay in units of ``4 G M_Lz / c^3``."""
    y = np.asarray(y, dtype=float)
    s = np.sqrt(y * y + 4.0)
    return 0.5 * y * s + np.log((s + y) / (s - y))


def time_delay(M_Lz, y):
    """Image time delay ``t_d`` in seconds. ``M_Lz`` in solar masses."""
    M_Lz = np.asarray(M_Lz, dtype=float)
    return 4.0 * T_SUN * M_Lz * delay_dimensionless(y)


def y_of_time_delay(M_Lz, t_d, y_bracket=(1e-8, 1e4)):
    """Invert ``time_delay`` for ``y`` at fixed ``M_Lz`` (scalar arguments)."""
    from scipy.optimize import brentq

    target = float(t_d) / (4.0 * T_SUN * float(M_Lz))

    def f(yy):
        return float(delay_dimensionless(yy)) - target

    return brentq(f, *y_bracket, xtol=1e-14, rtol=1e-15)


def w_of_f(f, M_Lz):
    """Dimensionless frequency ``w = 8 pi G M_Lz f / c^3``."""
    return 8.0 * np.pi * T_SUN * np.asarray(M_Lz, dtype=float) * np.asarray(
        f, dtype=float
    )


def f_of_w(w, M_Lz):
    """Inverse of :func:`w_of_f`."""
    return np.asarray(w, dtype=float) / (8.0 * np.pi * T_SUN * float(M_Lz))


def f_w1(M_Lz):
    """Frequency at which ``w = 1``: ``c^3 / (8 pi G M_Lz)``.

    This is *not* the quantity the standard benchmarks call ``f_ML``; see
    :func:`f_ML`.
    """
    return 1.0 / (8.0 * np.pi * T_SUN * np.asarray(M_Lz, dtype=float))


def M_Lz_of_f_w1(f):
    """Inverse of :func:`f_w1`."""
    return 1.0 / (8.0 * np.pi * T_SUN * np.asarray(f, dtype=float))


def f_ML(M_Lz, y):
    """Characteristic lens frequency ``f_ML = 1 / t_d(M_Lz, y)``.

    Fixed empirically against the standard point-lens benchmarks: at ``y = 0.127``,
    ``M_Lz = 1e4`` and ``250 Msun`` give ``f_ML = 19.97`` and ``798.8 Hz``,
    matching the stated "approximately 20 Hz and 800 Hz".  The ``w = 1``
    frequency ``c^3/(8 pi G M_Lz)`` gives 0.81 and 32.3 Hz instead, i.e. a
    constant factor 24.7 low, so it is *not* the intended definition.
    """
    return 1.0 / time_delay(M_Lz, y)


def M_Lz_of_f_ML(f, y):
    """Inverse of :func:`f_ML` at fixed ``y``."""
    return 1.0 / (4.0 * T_SUN * delay_dimensionless(y) * np.asarray(f, dtype=float))


def phi_m(y):
    """``phi_m(y) = (x_m - y)^2 / 2 - log(x_m)``, the Fermat-potential offset."""
    xm = x_m(y)
    return 0.5 * (xm - np.asarray(y, dtype=float)) ** 2 - np.log(xm)


# --------------------------------------------------------------------------
# Geometric optics
# --------------------------------------------------------------------------


def to_project_convention(F, conv=None):
    r"""Map an amplification factor from the literature to the project convention.

    The microlensing literature (and the 1F1 formula above) works in the
    ``htilde(f) = \int h(t) e^{+2 pi i f t} dt`` convention, in which the
    trailing image carries ``exp(+i w DT)``.  PyCBC -- and therefore this
    project -- uses ``exp(-2 pi i f t)``.  For a real-valued strain,
    ``htilde_lit(f) = conj(htilde_proj(f))``, so

        htilde_L,lit = F_lit htilde_lit  =>  F_proj(f) = conj(F_lit(f)).

    Conjugation flips *both* the sign of the delay phase and the Morse factor,
    so in the project convention the trailing image carries ``exp(-i w DT)``
    (arriving later, as it must) with a ``+i`` Morse factor.

    This is not a matter of taste: :file:`scripts/checks/phase0_conventions.py`
    verifies empirically that ``exp(+i w DT)`` places the second image at
    ``t_0 - t_d``, i.e. *before* the first, which is unphysical.
    """
    from . import conventions

    if conv is None:
        conv = conventions.DEFAULT
    return np.conjugate(F) if conv.fd_delay_sign < 0 else F


def F_GO_lit(f, M_Lz, y):
    """Geometric-optics factor in the *literature* convention.

    ``F = sqrt(mu_+) - i sqrt(|mu_-|) exp(+i w DT(y))``.
    """
    mu_p, mu_m = magnifications(y)
    w = w_of_f(f, M_Lz)
    phase = np.exp(1j * w * delay_dimensionless(y))
    return np.sqrt(mu_p) - 1j * np.sqrt(np.abs(mu_m)) * phase


def F_GO(f, M_Lz, y, conv=None):
    """Geometric-optics amplification factor in the project convention."""
    return to_project_convention(F_GO_lit(f, M_Lz, y), conv=conv)


def F_GO_relative(f, M_Lz, y, conv=None):
    """``F_GO / sqrt(mu_+)``: the search template's ``1 + s_M i a e^{-i w DT}``."""
    mu_p, _ = magnifications(y)
    return F_GO(f, M_Lz, y, conv=conv) / np.sqrt(mu_p)


# --------------------------------------------------------------------------
# Confluent hypergeometric 1F1(a, 1; z) for complex a, z
# --------------------------------------------------------------------------

_LOG_SCALE = 1e250


def log_hyp1f1_b1(a, z, max_terms=400000, tol=1e-17):
    """``log 1F1(a, 1; z)`` for complex scalars/arrays via a scaled series.

    Returns the complex logarithm, computed with an overflow-scaled Kummer
    series in ``longdouble`` precision:

        1F1(a, 1; z) = sum_n (a)_n z^n / (n!)^2,
        term_{n+1} = term_n * (a + n) * z / (n + 1)^2.

    The series is truncated when the term modulus falls below ``tol`` times
    the running-sum modulus *and* ``n`` is safely past the term-modulus peak.
    Terms grow before they decay, so a bare smallness test is not a
    convergence test.  The ratio ``|term_{n+1}/term_n| = |a+n||z|/(n+1)^2``
    equals one near ``n ~ sqrt(|a z|)`` (for ``n << |a|``), so the guard is
    ``n > n_safe = 4 sqrt(|a z|) + 4 |z| + 16``.  With ``a = i w / 2`` and
    ``z = i w y^2 / 2`` the peak sits at ``n ~ w y / 2``, which *exceeds*
    ``|z| = w y^2 / 2`` for every ``y < 1`` -- an earlier version of this
    routine guarded on ``|z|`` alone and truncated early, giving errors of
    order ``1e2`` in the logarithm across most of the wave-optics domain.

    Accuracy degrades by cancellation for large ``|z|``; :func:`hyp1f1_b1_mpmath`
    is the arbitrary-precision reference used to map the reliable domain.
    """
    a_arr = np.atleast_1d(np.asarray(a, dtype=np.clongdouble))
    z_arr = np.atleast_1d(np.asarray(z, dtype=np.clongdouble))
    a_arr, z_arr = np.broadcast_arrays(a_arr, z_arr)
    shape = a_arr.shape
    a_flat = np.ravel(a_arr).copy()
    z_flat = np.ravel(z_arr).copy()

    total = np.ones_like(a_flat)
    term = np.ones_like(a_flat)
    log_scale = np.zeros(a_flat.shape, dtype=np.longdouble)
    active = np.ones(a_flat.shape, dtype=bool)
    abs_z = np.abs(z_flat).astype(float)
    n_safe = 4.0 * np.sqrt(np.abs(a_flat).astype(float) * abs_z) + 4.0 * abs_z + 16.0

    n = 0
    while n < max_terms and active.any():
        idx = np.nonzero(active)[0]
        term[idx] *= (a_flat[idx] + n) * z_flat[idx] / np.longdouble((n + 1) ** 2)
        total[idx] += term[idx]

        big = np.abs(term[idx]) > _LOG_SCALE
        if big.any():
            jdx = idx[big]
            term[jdx] /= _LOG_SCALE
            total[jdx] /= _LOG_SCALE
            log_scale[jdx] += np.log(np.longdouble(_LOG_SCALE))

        converged = (np.abs(term[idx]) <= tol * np.abs(total[idx])) & (
            n > n_safe[idx]
        )
        if converged.any():
            active[idx[converged]] = False
        n += 1

    if active.any():
        raise RuntimeError(
            f"log_hyp1f1_b1 did not converge in {max_terms} terms for "
            f"{int(active.sum())} of {a_flat.size} points "
            f"(max |z| = {float(abs_z[active].max()):.3e}, "
            f"max n_safe = {float(n_safe[active].max()):.3e})"
        )

    out = (np.log(total.astype(np.clongdouble)) + log_scale).astype(complex)
    return out.reshape(shape) if np.ndim(a) or np.ndim(z) else out[0]


def hyp1f1_b1_mpmath(a, z, dps=30, maxterms=10**7):
    """Arbitrary-precision reference for ``1F1(a, 1; z)`` (slow; reference only)."""
    import mpmath

    with mpmath.workdps(dps):
        a_arr = np.atleast_1d(np.asarray(a, dtype=complex))
        z_arr = np.atleast_1d(np.asarray(z, dtype=complex))
        a_arr, z_arr = np.broadcast_arrays(a_arr, z_arr)
        out = np.empty(a_arr.shape, dtype=complex)
        it = np.nditer(a_arr, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            val = mpmath.hyp1f1(
                mpmath.mpc(a_arr[i]),
                mpmath.mpf(1),
                mpmath.mpc(z_arr[i]),
                maxterms=maxterms,
            )
            out[i] = complex(val)
        return out if (np.ndim(a) or np.ndim(z)) else out.reshape(())[()]


def log_hyp1f1_b1_mpmath(a, z, dps=30, maxterms=10**7):
    """``log 1F1(a, 1; z)`` at arbitrary precision (handles huge magnitudes)."""
    import mpmath

    with mpmath.workdps(dps):
        a_arr = np.atleast_1d(np.asarray(a, dtype=complex))
        z_arr = np.atleast_1d(np.asarray(z, dtype=complex))
        a_arr, z_arr = np.broadcast_arrays(a_arr, z_arr)
        out = np.empty(a_arr.shape, dtype=complex)
        it = np.nditer(a_arr, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            val = mpmath.hyp1f1(
                mpmath.mpc(a_arr[i]),
                mpmath.mpf(1),
                mpmath.mpc(z_arr[i]),
                maxterms=maxterms,
            )
            out[i] = complex(mpmath.log(val))
        return out if (np.ndim(a) or np.ndim(z)) else out.reshape(())[()]


# --------------------------------------------------------------------------
# Wave optics
# --------------------------------------------------------------------------


def log_F_ML_from_w(w, y, backend="mpmath", **kwargs):
    """``log F(w, y)`` for the exact point-lens wave-optics factor.

    ``backend`` is ``"mpmath"`` (the validated default) or ``"series"``.  The
    plain series is **not** trustworthy over most of the physical domain: see
    :func:`log_hyp1f1_b1`.
    """
    w = np.asarray(w, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    a_par = 0.5j * w
    z_arg = 0.5j * w * y_arr * y_arr

    log_prefactor = (
        0.25 * np.pi * w
        + 0.5j * w * (np.log(0.5 * w) - 2.0 * phi_m(y_arr))
        + loggamma(1.0 - 0.5j * w)
    )
    if backend == "series":
        kwargs.pop("dps", None)
        log_1f1 = log_hyp1f1_b1(a_par, z_arg, **kwargs)
    elif backend == "mpmath":
        log_1f1 = log_hyp1f1_b1_mpmath(a_par, z_arg, **kwargs)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return log_prefactor + log_1f1


def F_ML_from_w(w, y, backend="mpmath", **kwargs):
    """Exact point-lens wave-optics amplification factor ``F(w, y)``."""
    return np.exp(log_F_ML_from_w(w, y, backend=backend, **kwargs))


def F_ML(f, M_Lz, y, backend="mpmath", conv=None, **kwargs):
    """Exact ``F`` as a function of physical frequency and lens parameters.

    Returned in the **project convention** (see :func:`to_project_convention`);
    pass ``conv`` to override.  ``f = 0`` maps to ``w = 0`` where ``F = 1``; it
    is handled explicitly because ``log(w/2)`` diverges there.
    """
    f_arr = np.asarray(f, dtype=float)
    w = w_of_f(f_arr, M_Lz)
    out = np.ones(np.broadcast(w, np.asarray(y, dtype=float)).shape, dtype=complex)
    good = w > 0
    if np.ndim(out) == 0:
        if not good:
            return complex(1.0)
        return complex(
            to_project_convention(
                F_ML_from_w(w, y, backend=backend, **kwargs), conv=conv
            )
        )
    yb = np.broadcast_to(np.asarray(y, dtype=float), out.shape)
    wb = np.broadcast_to(w, out.shape)
    if good.any():
        out[good] = F_ML_from_w(wb[good], yb[good], backend=backend, **kwargs)
    return to_project_convention(out, conv=conv)


def F_ML_relative(f, M_Lz, y, **kwargs):
    """``F / sqrt(mu_+)``, matching :func:`F_GO_relative`'s normalization."""
    mu_p, _ = magnifications(y)
    return F_ML(f, M_Lz, y, **kwargs) / np.sqrt(mu_p)


# --------------------------------------------------------------------------
# Cached / parallel evaluation on a frequency grid
# --------------------------------------------------------------------------


def _cache_key(f, M_Lz, y, backend, dps):
    """Content hash covering every physical and numerical setting."""
    import hashlib

    f = np.asarray(f, dtype=float)
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(f).tobytes())
    h.update(repr((float(M_Lz), float(y), str(backend), int(dps), 3)).encode())
    return h.hexdigest()[:32]


def _chunk_worker(args):
    w_chunk, y, backend, dps = args
    return log_F_ML_from_w(w_chunk, y, backend=backend, dps=dps)


def F_ML_on_grid(
    f,
    M_Lz,
    y,
    backend="mpmath",
    conv=None,
    dps=30,
    cache_dir="cache/F_ML",
    n_proc=1,
    chunk=2048,
):
    """``F_ML`` on a frequency grid, with an on-disk cache and optional pool.

    The cache key hashes the full frequency array together with ``M_Lz``,
    ``y``, the backend and ``dps``.  ``f = 0`` samples are returned as ``F = 1`` without evaluation.

    ``n_proc > 1`` spreads the frequency grid over a process pool; keep the
    total at or below 20 workers per the project compute budget.
    """
    import os

    f = np.asarray(f, dtype=float)
    key = _cache_key(f, M_Lz, y, backend, dps)
    path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"F_{key}.npy")
        if os.path.exists(path):
            return np.load(path)

    out = np.ones(f.shape, dtype=complex)
    if backend == "auto":
        # Split the band: geometric optics above the calibrated w_switch(y),
        # exact mpmath below it (parallelized like the pure-mpmath path).
        w_all = w_of_f(f, M_Lz)
        w_sw = w_switch_of_y(float(y))
        use_go = (w_all >= w_sw) & (f > 0)
        good = np.nonzero((w_all > 0) & ~use_go)[0]
        if np.any(use_go):
            out[use_go] = F_GO(f[use_go], M_Lz, y, conv=conv)
        backend = "mpmath"
    else:
        good = np.nonzero(f > 0)[0]

    if good.size:
        w = w_of_f(f[good], M_Lz)
        if n_proc > 1 and w.size > chunk:
            from multiprocessing import Pool

            pieces = [w[i : i + chunk] for i in range(0, w.size, chunk)]
            with Pool(processes=n_proc) as pool:
                res = pool.map(
                    _chunk_worker, [(p, float(y), backend, dps) for p in pieces]
                )
            log_vals = np.concatenate(res)
        else:
            log_vals = log_F_ML_from_w(w, y, backend=backend, dps=dps)
        out[good] = to_project_convention(np.exp(log_vals), conv=conv)

    if path:
        tmp = path + f".tmp{os.getpid()}.npy"
        np.save(tmp, out)
        os.replace(tmp, path)
    return out


# --------------------------------------------------------------------------
# Measured geometric-optics handoff (the "auto" backend)
# --------------------------------------------------------------------------

_HANDOFF_CACHE = {}


def go_handoff_table(path=None):
    """Load the measured ``w_switch(y)`` table .

    Produced by ``scripts/simulate/calibrate_go_handoff.py``.  Entries are ``inf`` where
    geometric optics never reached the tolerance within the scanned range.
    """
    import json
    import os

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "go_handoff.json")
    key = os.path.abspath(path)
    if key not in _HANDOFF_CACHE:
        with open(path) as fh:
            data = json.load(fh)
        _HANDOFF_CACHE[key] = {
            "y": np.asarray(data["y"], dtype=float),
            "w_switch": np.asarray(data["w_switch"], dtype=float),
            "tolerance": float(data["tolerance"]),
        }
    return _HANDOFF_CACHE[key]


def w_switch_of_y(y, safety=1.5, path=None):
    """Conservative ``w`` above which geometric optics may replace ``F_ML``.

    Because the measured ``w_switch(y)`` is non-monotonic (the error envelope
    oscillates) and contains ``inf`` entries, the value returned for a query
    ``y`` is the **maximum** over the two bracketing table entries, times
    ``safety``.  Any bracketing ``inf`` propagates, i.e. no switch is allowed.
    """
    tab = go_handoff_table(path)
    ys, ws = tab["y"], tab["w_switch"]
    y_arr = np.atleast_1d(np.asarray(y, dtype=float))
    idx = np.clip(np.searchsorted(ys, y_arr), 1, ys.size - 1)
    lo = ws[idx - 1]
    hi = ws[idx]
    out = safety * np.maximum(lo, hi)
    # queries outside the tabulated range get no switch
    out = np.where((y_arr < ys[0]) | (y_arr > ys[-1]), np.inf, out)
    return out if np.ndim(y) else float(out[0])


def F_ML_auto(f, M_Lz, y, dps=30, safety=1.5, handoff_path=None, conv=None):
    """``F_ML`` with a measured geometric-optics handoff at large ``w``.

    Exact (mpmath) below ``w_switch(y)``, geometric optics above it.  The
    handoff exists purely for cost: mpmath's expense is driven by
    ``|z| = w y^2 / 2``, and the expensive corner (large ``w`` *and* large
    ``y``) is exactly where geometric optics is accurate to the calibrated
    tolerance.  Measured: ``M_Lz = 1e5, y = 2`` over a 16385-bin grid takes
    **1656 s** with pure mpmath.

    Returns the factor in the project convention.  Use ``backend="mpmath"``
    (the default elsewhere) for anything where the exactness of the wave-optics
    model is itself the claim.
    """
    kw = {} if handoff_path is None else {"path": handoff_path}
    f_arr = np.asarray(f, dtype=float)
    w = w_of_f(f_arr, M_Lz)
    w_sw = w_switch_of_y(float(y), safety=safety, **kw)

    out = np.ones(f_arr.shape, dtype=complex)
    use_go = (w >= w_sw) & (f_arr > 0)
    use_ex = (w > 0) & ~use_go

    if np.any(use_ex):
        out[use_ex] = to_project_convention(
            F_ML_from_w(w[use_ex], y, backend="mpmath", dps=dps), conv=conv
        )
    if np.any(use_go):
        out[use_go] = F_GO(f_arr[use_go], M_Lz, y, conv=conv)
    return out, {
        "w_switch": float(w_sw),
        "n_exact": int(use_ex.sum()),
        "n_geometric_optics": int(use_go.sum()),
        "handoff_tolerance": float(go_handoff_table(**kw)["tolerance"]),
    }
