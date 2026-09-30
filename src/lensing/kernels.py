"""Fused evaluation of the recombined statistic.

``recombine.recombine_expanded_sq`` is written for clarity: it broadcasts
``(n_time, 1)`` against ``(1, n_lens)`` and lets NumPy allocate the
intermediates.  That is the right reference implementation and the wrong
production one.  Profiling the cost benchmark on G26's bank showed the
broadcast temporaries, not the inputs, to be the bottleneck: the arrays the
expression *must* read are ``z_1`` and its shift, 8.4 MB per delay, while the
five ``(n_time, n_lens)`` temporaries it actually allocates are ten times
that.  The single FFT the whole method exists to save is 0.05% of the runtime.

The kernels here evaluate the same expression in one pass, keeping the
per-sample quantities in registers and writing only the result.  Measured on
one core, per delay:

===================  ==========  =========  ======
block                 broadcast     fused    gain
===================  ==========  =========  ======
full segment          26.8 ms     1.71 ms    15.7x
candidate-restricted  0.259 ms    0.051 ms    5.1x
===================  ==========  =========  ======

The fused path reaches 9.8 GB/s on the minimum necessary traffic, i.e. it is
genuinely memory-bandwidth bound where the broadcast form is temporary bound.

**Numerics.** The kernels compute ``|z|^2`` as ``x*x + y*y``; NumPy's
``abs(z)**2`` takes a square root and squares it back.  Results therefore
differ from the reference path by about **2 ulps** of the peak value
(``2.8e-16`` relative to the peak, against the ``1.4e-14`` at which the
identity itself is verified), and the fused form is the more accurate of the
two.  Pointwise *relative* differences look much larger, up to ``3e-13``, but
only where the numerator nearly cancels and the value is far below threshold;
peak-relative is the normalization Phase 0 uses and the one that matters.

Recovered arguments are unaffected.  Set ``LENSING_NO_NUMBA=1`` to force the
NumPy path, which is what reproduces the pre-2026-09-21 frozen products
bit-for-bit.
"""

from __future__ import annotations

import os

import numpy as np

__all__ = ["HAVE_NUMBA", "half_q_from_series", "half_q_from_indices"]


def _try_numba():
    if os.environ.get("LENSING_NO_NUMBA"):
        return None
    try:
        from numba import njit
    except Exception:  # pragma: no cover - numba is optional
        return None
    return njit


_njit = _try_numba()
HAVE_NUMBA = _njit is not None


# --------------------------------------------------------------------------
# NumPy reference paths.  Kept exercised by the tests, and by
# LENSING_NO_NUMBA=1, so the fused kernels always have something to agree with.
# --------------------------------------------------------------------------


def _np_from_series(z1, z1s, ur, ui, a, norm, out):
    # Deliberately expression-for-expression with
    # ``recombine.recombine_expanded_sq``, including ``abs(z)**2`` rather than
    # the algebraically equivalent ``x*x + y*y``: this path exists so that
    # LENSING_NO_NUMBA=1 reproduces the pre-2026-09-21 frozen products
    # bit-for-bit, which a "better" formulation would silently break.
    u = complex(ur, -ui)
    ps = np.abs(z1) ** 2
    pf = np.abs(z1s) ** 2
    cr = np.real(np.conjugate(u) * z1 * np.conjugate(z1s))
    num = (ps[:, None] + (a * a)[None, :] * pf[:, None]
           + 2.0 * a[None, :] * cr[:, None])
    np.divide(0.5 * num, norm[None, :], out=out)
    return out


def _np_from_indices(z1, idx_self, idx_shift, ur, ui, a, norm, out):
    return _np_from_series(z1[idx_self], z1[idx_shift], ur, ui, a, norm, out)


if HAVE_NUMBA:

    @_njit(cache=True)
    def _nb_from_series(z1, z1s, ur, ui, a, norm, out):  # pragma: no cover
        for i in range(z1.shape[0]):
            x1 = z1[i].real
            y1 = z1[i].imag
            x2 = z1s[i].real
            y2 = z1s[i].imag
            ps = x1 * x1 + y1 * y1
            pf = x2 * x2 + y2 * y2
            cr = ur * (x1 * x2 + y1 * y2) - ui * (y1 * x2 - x1 * y2)
            for j in range(a.shape[0]):
                aj = a[j]
                out[i, j] = 0.5 * (ps + aj * aj * pf + 2.0 * aj * cr) / norm[j]
        return out

    @_njit(cache=True)
    def _nb_from_indices(z1, idx_self, idx_shift, ur, ui,
                         a, norm, out):  # pragma: no cover
        for i in range(idx_self.shape[0]):
            c = idx_self[i]
            s = idx_shift[i]
            x1 = z1[c].real
            y1 = z1[c].imag
            x2 = z1[s].real
            y2 = z1[s].imag
            ps = x1 * x1 + y1 * y1
            pf = x2 * x2 + y2 * y2
            cr = ur * (x1 * x2 + y1 * y2) - ui * (y1 * x2 - x1 * y2)
            for j in range(a.shape[0]):
                aj = a[j]
                out[i, j] = 0.5 * (ps + aj * aj * pf + 2.0 * aj * cr) / norm[j]
        return out


def _coeff(u):
    """``conj(u)`` split into real and imaginary parts.

    The cross term is ``Re[conj(u) z_1 z_{1s}^*]``; writing it out in real
    arithmetic is what lets the kernel avoid a complex temporary. ``u = i^{-q}``,
    so this carries the relative Morse phase and nothing else.
    """
    w = np.conjugate(complex(u))
    return float(w.real), float(w.imag)


def half_q_from_series(z1, z1s, u, a, norm, out=None):
    """``0.5 |z_L|^2`` for contiguous ``z_1`` and its shift, shape ``(n, n_a)``."""
    a = np.ascontiguousarray(a, dtype=float)
    norm = np.ascontiguousarray(norm, dtype=float)
    z1 = np.ascontiguousarray(z1, dtype=complex)
    z1s = np.ascontiguousarray(z1s, dtype=complex)
    if out is None:
        out = np.empty((z1.size, a.size), dtype=float)
    ur, ui = _coeff(u)
    f = _nb_from_series if HAVE_NUMBA else _np_from_series
    return f(z1, z1s, ur, ui, a, norm, out)


def half_q_from_indices(z1, idx_self, idx_shift, u, a, norm, out=None):
    """``0.5 |z_L|^2`` on a candidate set, gathering inside the kernel.

    Avoids materializing ``z_1[cand]`` and ``z_1[cand + n_sh]``, which the
    restricted path would otherwise rebuild once per delay.
    """
    a = np.ascontiguousarray(a, dtype=float)
    norm = np.ascontiguousarray(norm, dtype=float)
    z1 = np.ascontiguousarray(z1, dtype=complex)
    idx_self = np.ascontiguousarray(idx_self, dtype=np.int64)
    idx_shift = np.ascontiguousarray(idx_shift, dtype=np.int64)
    if out is None:
        out = np.empty((idx_self.size, a.size), dtype=float)
    ur, ui = _coeff(u)
    f = _nb_from_indices if HAVE_NUMBA else _np_from_indices
    return f(z1, idx_self, idx_shift, ur, ui, a, norm, out)
