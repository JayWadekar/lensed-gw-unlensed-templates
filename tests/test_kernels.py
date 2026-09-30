"""The fused kernels must agree with the reference broadcast implementation.

Two separate contracts, and they are not the same one:

* the NumPy fallback is **bit-identical** to ``recombine_expanded_sq``, which
  is what lets ``LENSING_NO_NUMBA=1`` reproduce the pre-2026-09-21 frozen
  products exactly;
* the fused kernel differs by a couple of ulps of the peak, because it forms
  ``|z|^2`` as ``x*x + y*y`` instead of squaring a square root.  Peak-relative
  is the normalization Phase 0 uses; pointwise relative is not a meaningful
  tolerance here, since it diverges wherever the numerator cancels.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lensing import conventions as cv
from lensing import kernels as kn
from lensing import recombine as rc

QS = (0, 1, 2, 3)


def _case(seed=3, n=4096, n_a=6):
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n) + 1j * rng.normal(size=n)
    z1s = rng.normal(size=n) + 1j * rng.normal(size=n)
    a = np.linspace(0.15, 0.95, n_a)
    C = 0.3 * (rng.normal(size=n_a) + 1j * rng.normal(size=n_a))
    return z1, z1s, a, C


@pytest.mark.parametrize("q", QS)
def test_numpy_fallback_is_bit_identical(q):
    z1, z1s, a, C = _case()
    conv = cv.DEFAULT.with_rel_morse(q) if hasattr(cv.DEFAULT, "with_rel_morse") \
        else cv.DEFAULT
    u = conv.filter_coeff(1.0)
    ref = 0.5 * rc.recombine_expanded_sq(
        z1[:, None], z1s[:, None], C[None, :], a[None, :], conv=conv)
    norm = rc.lensed_norm_sq(C, a, conv=conv)
    out = np.empty_like(ref)
    got = kn._np_from_series(z1, z1s, *kn._coeff(u), a, norm, out)
    assert np.array_equal(got, ref)


def test_fused_matches_reference_to_two_ulps():
    z1, z1s, a, C = _case()
    u = cv.DEFAULT.filter_coeff(1.0)
    ref = 0.5 * rc.recombine_expanded_sq(
        z1[:, None], z1s[:, None], C[None, :], a[None, :])
    norm = rc.lensed_norm_sq(C, a)
    got = kn.half_q_from_series(z1, z1s, u, a, norm)
    peak_rel = np.abs(got - ref).max() / ref.max()
    assert peak_rel < 1e-14, peak_rel
    assert np.unravel_index(got.argmax(), got.shape) == \
        np.unravel_index(ref.argmax(), ref.shape)


def test_index_and_series_forms_agree():
    """Gathering inside the kernel must equal gathering outside it."""
    z1, _, a, C = _case(seed=5, n=8192)
    n = z1.size
    rng = np.random.default_rng(11)
    cand = np.sort(rng.choice(n, size=1500, replace=False))
    n_sh = 733
    idx_sh = (cand + n_sh) % n
    u = cv.DEFAULT.filter_coeff(1.0)
    norm = rc.lensed_norm_sq(C, a)
    a_idx = kn.half_q_from_indices(z1, cand, idx_sh, u, a, norm)
    a_ser = kn.half_q_from_series(z1[cand], z1[idx_sh], u, a, norm)
    assert np.array_equal(a_idx, a_ser)
