"""Wave-optics ``F(w)`` for a lens whose Fermat contours close, via GLoW's
time-domain integral and a regularized Fourier transform written here.

Why this module exists
----------------------
GLoW's ``It_MultiContour_C`` computes ``I(tau)`` for a general 2-D lens, and it
does so correctly for the Chang-Refsdal lens wherever the contours close --
``kappa + gamma < 1`` (:mod:`lensing.lens_models`).  What GLoW cannot do for this lens is the *transform*
``I(tau) -> F(w)``, because every one of its regularization stages assumes the
lens is isolated, i.e. that

    I(tau -> infinity) -> 2 pi .

An external convergence and shear break that assumption: the asymptotic
"unlensed" reference is the **macro-image itself**, so

    I(tau -> infinity) -> 2 pi sqrt(mu_macro),
    mu_macro = 1 / |(1 - kappa)^2 - gamma^2| .

GLoW's ``reg_stage=2`` fits a decaying tail ``A_2 / tau^sigma_2`` to a quantity
that in fact tends to a nonzero constant, and needs ``asymp_amplitude`` from the
lens, which ``CombinedLens`` reports as ``None``; the result is ``NaN``.
Supplying the point-mass asymptotics by hand does not fix it -- the tail model
is simply the wrong shape -- and the transform then fails to converge to
geometric optics at large ``w`` (measured: relative error 0.2 to 2 at
``w = 10^4``, growing with ``w``).

What is computed here
---------------------
The transform, and only the transform.  ``I(tau)`` remains GLoW's.  In the
literature convention GLoW returns,

    F(w) = (-i w / 2 pi) int I(tau) e^{i w tau} d tau .

Split ``I`` into the analytic contribution of the critical points plus a
constant plus a remainder:

    I(tau) = I_sing(tau) + c_inf theta(tau) + Ihat(tau),      Ihat -> 0,

where ``I_sing`` uses GLoW's own singular forms -- a ``2 pi sqrt(mu)`` step at
each minimum, the mirror step at each maximum, and a windowed logarithm
``-2 sqrt(mu) ln|tau - tau_c| exp(-|tau - tau_c| / T)`` at each saddle -- whose
transforms are known in closed form and sum to ``F_GO`` as ``w -> infinity``.
The constant ``c_inf`` is what the external field adds; it transforms to
``c_inf / 2 pi`` exactly.  Only ``Ihat`` is integrated numerically, with a
Filon rule that is *exact* in the oscillatory factor, on a grid split at every
discontinuity and graded geometrically into every singular endpoint.

Accuracy, measured not assumed (see ``scripts/appendix_cr_wave_validation.py``)

* against GLoW's analytic point lens, over ``w`` in ``[1, 100]``: ``2e-4``
  relative, rising to ``3e-3`` at ``w = 3000``;
* the bookkeeping identity ``c_inf / 2 pi + sum_minima sqrt(mu) = sqrt(mu_macro)``
  holds to ``4e-4`` in every cell tested;
* Chang-Refsdal ``F -> F_GO`` at large ``w`` to ``~1e-3``.

The remaining floor is GLoW's own contour-ODE tolerance amplified by the factor
``w`` in the transform, which is why :data:`C_PREC` tightens it well below the
GLoW default.
"""

from __future__ import annotations

import warnings

import numpy as np

#: Width of the exponential window on the saddle logarithm.  GLoW's own
#: default (``glow.freq_domain.FwGeneral``), kept so the analytic transform
#: below is theirs verbatim.
T_SADDLE = 3.0

#: GLoW contour-ODE tolerances, tightened from the ``1e-5``/``1e-6`` defaults.
#: The transform multiplies the error on ``I(tau)`` by ``w``, so the default
#: leaves a ``1.7e-2`` relative error at ``w = 3000`` where ``1e-8`` leaves
#: ``4e-3``.  Below ``1e-8`` nothing improves.
C_PREC = {
    "mc_intContour":       {"h": 1e-8, "epsabs": 1e-8, "epsrel": 0.0, "type": "rk8pd"},
    "mc_getContour":       {"h": 1e-8, "epsabs": 1e-8, "epsrel": 0.0, "type": "rk8pd"},
    "mc_intContourSaddle": {"h": 1e-8, "epsabs": 1e-8, "epsrel": 0.0, "type": "rk8pd"},
    "mc_intRtau":          {"h": 1e-8, "epsabs": 1e-8, "epsrel": 0.0, "type": "rk8pd"},
    "mc_findRbracket":     {"max_iter": 200, "epsabs": 1e-8, "epsrel": 1e-8,
                            "type": "brent"},
    "mc_getContour_x1x2":  {"h": 1e-8, "epsabs": 1e-9, "epsrel": 0.0, "type": "rk8pd"},
}

#: Upper limit of the numerical ``tau`` integral.  The neglected tail is
#: ``O(A / 2 pi tau_max)`` with ``A ~ 4``, i.e. ``6e-5``, independent of ``w``.
TAU_MAX = 1.0e4

#: Points per grid segment.  Segments are split at every discontinuity, so this
#: is the resolution *between* singular points, not overall.
N_SEG = 4000

#: Largest ``w`` at which the numerical transform is trusted.  Above this the
#: caller should use geometric optics, which by then agrees to ~1e-3 anyway.
W_MAX_TRUSTED = 3.0e3


# --------------------------------------------------------------------------
# Filon quadrature: piecewise-linear amplitude, exact oscillatory factor
# --------------------------------------------------------------------------


def _phi12(z):
    """``(e^z - 1)/z`` and ``(z e^z - e^z + 1)/z^2``, stable as ``z -> 0``."""
    z = np.asarray(z, dtype=complex)
    small = np.abs(z) < 1e-3
    zs = np.where(small, 1.0, z)
    p1 = np.expm1(zs) / zs
    p2 = (zs * np.exp(zs) - np.expm1(zs)) / (zs * zs)
    zt = np.where(small, z, 0.0)
    return (
        np.where(small, 1.0 + zt/2 + zt**2/6 + zt**3/24 + zt**4/120, p1),
        np.where(small, 0.5 + zt/3 + zt**2/8 + zt**3/30 + zt**4/144, p2),
    )


def filon_ft(tau, f, w, chunk: int = 256):
    """``int f(tau) e^{i w tau} d tau`` with ``f`` piecewise linear on ``tau``.

    Exact in the oscillatory factor, so accuracy is set by how well the nodes
    resolve ``f`` and not by ``w``.  Verified against
    ``int_0^inf e^{-t} e^{i w t} dt = 1 / (1 - i w)`` to ``1e-6`` or better for
    ``w`` up to ``3000``.

    ``chunk`` bounds the ``(n_w, n_panel)`` temporary; the cost is unchanged.
    """
    tau = np.asarray(tau, dtype=float)
    f = np.asarray(f, dtype=float)
    a = tau[:-1]
    h = np.diff(tau)
    fa = f[:-1]
    bh = np.diff(f)                       # beta * h
    w = np.atleast_1d(np.asarray(w, dtype=float))

    out = np.empty(w.size, dtype=complex)
    for lo in range(0, w.size, chunk):
        wc = w[lo:lo + chunk][:, None]
        p1, p2 = _phi12(1j * wc * h[None, :])
        out[lo:lo + chunk] = np.sum(
            np.exp(1j * wc * a[None, :]) * h[None, :]
            * (fa[None, :] * p1 + bh[None, :] * p2), axis=1)
    return out


def _graded_both(a, b, n, r=1e-10):
    """``n`` nodes on ``[a, b]``, geometrically dense at both ends."""
    g = np.geomspace(r, 1.0, max(2, n // 2))
    return np.unique(np.concatenate([[a], a + 0.5*(b-a)*g, b - 0.5*(b-a)*g, [b]]))


def _graded_left(a, b, n, r=1e-10):
    """``n`` nodes on ``[a, b]``, geometrically dense at ``a`` only."""
    return np.unique(np.concatenate([[a], a + (b - a) * np.geomspace(r, 1.0, n)]))


# --------------------------------------------------------------------------
# The singular model and its analytic transform (GLoW's forms)
# --------------------------------------------------------------------------


def critical_points(It):
    """``(kind, tau_c, sqrt_mu)`` per image, ``tau`` from the global minimum.

    ``sing/cusp`` entries are the lens's own singular centre, not images, and
    are dropped -- exactly as ``glow.freq_domain`` and
    :func:`lensing.glow_backend.images_from_glow` drop them.  For a point-mass
    core the contours around it contribute ``O(e^{-2 tau})``.
    """
    return [
        (p["type"], float(p["t"]) - float(It.tmin), float(np.sqrt(p["mag"])))
        for p in It.p_crits
        if not p["type"].startswith("sing/cusp")
    ]


def It_sing(tau, crits, T=T_SADDLE):
    """Analytic singular part of ``I(tau)``."""
    out = np.zeros_like(np.asarray(tau, dtype=float))
    for kind, tc, sm in crits:
        if kind == "min":
            out = out + np.where(tau >= tc, 2.0 * np.pi * sm, 0.0)
        elif kind == "max":
            out = out + np.where(tau <= tc, 2.0 * np.pi * sm, 0.0)
        else:
            d = np.maximum(np.abs(tau - tc), 1e-300)
            out = out - 2.0 * sm * np.log(d) * np.exp(-d / T)
    return out


def Fw_sing(w, crits, T=T_SADDLE):
    """Exact transform of :func:`It_sing`; tends to ``F_GO`` as ``w -> inf``.

    Verified against a direct quadrature of the windowed logarithm to ``1e-9``,
    and against its own large-``w`` limit ``-i sqrt(mu)``.
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    out = np.zeros(w.size, dtype=complex)
    for kind, tc, sm in crits:
        if kind == "min":
            out = out + sm * np.exp(1j * w * tc)
        elif kind == "max":
            out = out - sm * np.exp(1j * w * tc)
        else:
            tmp = 1.0 / T - 1j * w
            I_plus = -(np.euler_gamma + np.log(tmp)) / tmp
            out = out + 1j * w / np.pi * sm * np.exp(1j * w * tc) * 2.0 * np.real(I_plus)
    return out


# --------------------------------------------------------------------------
# The transform
# --------------------------------------------------------------------------


def build_It(model, y: float, c_prec=None):
    """GLoW's ``I(tau)`` for ``model`` at ``y``, in exact-evaluation mode.

    Raises if the lens has open Fermat contours, which is a property of the
    lens and not a numerical accident: see :meth:`ChangRefsdal.contours_close`.
    """
    from glow import time_domain_c

    closes = getattr(model, "contours_close", None)
    if closes is not None and not closes:
        raise ValueError(
            "Fermat contours do not close for this lens (kappa + gamma > 1); "
            "the contour method has nothing to integrate. Use F_go."
        )
    p_prec = {"eval_mode": "exact",
              "C_prec": dict(C_PREC if c_prec is None else c_prec)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return time_domain_c.It_MultiContour_C(model.glow_lens(), float(y),
                                               p_prec=p_prec)


def F_of_w_contour(It, w, tau_max=TAU_MAX, n_seg=N_SEG, T=T_SADDLE,
                   return_diagnostics=False):
    """``F(w)`` in GLoW's (literature) convention from a GLoW ``I(tau)``.

    Parameters
    ----------
    It
        A GLoW time-domain object, from :func:`build_It`.
    w
        Dimensionless frequencies, ``w = 8 pi G M_Lz f / c^3``.
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    crits = critical_points(It)
    has_saddle = any(k == "saddle" for k, _, _ in crits)

    # A true maximum would break the bookkeeping below: its analytic transform
    # -sqrt(mu) e^{i w tau_c} is the step integrated over (-inf, tau_c], but the
    # numerical grid starts at a finite tau_min, so the uncorrected remainder
    # (-inf, tau_min] would leave a spurious O(sqrt(mu)) term -- a constant does
    # not decay, so no choice of tau_min removes it. No cell in this study has
    # one (Chang-Refsdal in the macro-minimum region gives minima and saddles
    # only; the point-mass core is a `sing/cusp`, not an image), so this refuses
    # rather than returning a plausible wrong number.
    if any(k == "max" for k, _, _ in crits):
        raise NotImplementedError(
            "a true maximum is present; its singular step extends to "
            "tau = -infinity and the truncated grid would leave a spurious "
            "O(sqrt(mu)) term. Handle it explicitly before trusting F."
        )

    # the saddle window has support at tau < 0, where I(tau) = 0; truncate it
    # where exp(-|tau| / T) has fallen to ~1e-14
    tau_min = -T * 32.0 if has_saddle else 0.0

    breaks = np.array(sorted(
        {0.0, tau_min, tau_max}
        | {tc for _, tc, _ in crits if tau_min < tc < tau_max}))

    # the constant the external field leaves behind, from the last half-decade
    probe = np.geomspace(0.3 * tau_max, tau_max, 200)
    c_inf = float(np.mean(np.asarray(It(probe), dtype=float)
                          - It_sing(probe, crits, T)))

    def Ihat(t):
        I = np.where(t > 0.0, np.asarray(It(np.maximum(t, 1e-12)), dtype=float), 0.0)
        return I - It_sing(t, crits, T) - np.where(t > 0.0, c_inf, 0.0)

    acc = np.zeros(w.size, dtype=complex)
    last = breaks.size - 2
    for i in range(breaks.size - 1):
        a, b = float(breaks[i]), float(breaks[i + 1])
        if b <= a:
            continue
        smooth_right = (i == last) and b > 100.0 * max(abs(a), 1e-3)
        t = _graded_left(a, b, n_seg) if smooth_right else _graded_both(a, b, n_seg)
        eps = 1e-12 * max(1.0, abs(a), abs(b))
        t[0], t[-1] = a + eps, b - eps
        acc += filon_ft(t, Ihat(t), w)

    F = Fw_sing(w, crits, T) + c_inf / (2.0 * np.pi) + (-1j * w / (2.0 * np.pi)) * acc
    if not return_diagnostics:
        return F
    return F, {
        "c_inf": c_inf,
        "sqrt_mu_asym": c_inf / (2.0 * np.pi)
                        + sum(sm for k, _, sm in crits if k == "min"),
        "n_crits": len(crits),
        "tau_min": tau_min,
        "tau_max": tau_max,
    }
