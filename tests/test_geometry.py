"""Point-lens geometry: standard-benchmark unit checks and coordinate round trips."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lensing import amplification as amp


def test_time_delay_benchmark_10ms():
    """Benchmark: M_Lz = 2000 Msun, y = 0.127 gives t_d ~ 10 ms."""
    t_d = float(amp.time_delay(2000.0, 0.127))
    assert t_d == pytest.approx(10.0e-3, rel=2e-3)


def test_f_ML_benchmarks():
    """Benchmark: at y = 0.127, M_Lz = 1e4 / 250 give f_ML ~ 20 / 800 Hz.

    This is the check that established f_ML = 1/t_d.
    """
    assert float(amp.f_ML(1e4, 0.127)) == pytest.approx(20.0, rel=2e-3)
    assert float(amp.f_ML(250.0, 0.127)) == pytest.approx(800.0, rel=2e-3)


def test_f_w1_is_not_f_ML():
    """The w = 1 frequency is a different quantity; guard against confusion."""
    assert float(amp.f_w1(1e4)) == pytest.approx(0.8078, rel=1e-3)
    assert float(amp.f_ML(1e4, 0.127)) / float(amp.f_w1(1e4)) > 20.0


def test_small_y_delay_limit():
    """Benchmark: at small y, t_d -> 8 G M_Lz y / c^3."""
    for y in (1e-5, 1e-4, 1e-3):
        exact = float(amp.time_delay(1000.0, y))
        approx = 8.0 * amp.T_SUN * 1000.0 * y
        assert exact == pytest.approx(approx, rel=1e-5)


def test_magnification_identity():
    """mu_+ - |mu_-| = 1 exactly for a point lens."""
    y = np.geomspace(1e-3, 10.0, 200)
    mu_p, mu_m = amp.magnifications(y)
    assert np.allclose(mu_p - np.abs(mu_m), 1.0, atol=1e-12)


def test_a_in_unit_interval():
    """Benchmark: 0 < a < 1 over the whole physical domain."""
    y = np.geomspace(1e-4, 100.0, 500)
    a = amp.a_of_y(y)
    assert np.all(a > 0) and np.all(a < 1)


def test_a_y_round_trip():
    """a(y) and y(a) invert each other."""
    y = np.geomspace(0.005, 5.0, 300)
    assert np.allclose(amp.y_of_a(amp.a_of_y(y)), y, rtol=1e-9)


def test_a_mu_r_round_trip():
    """The a <-> mu_r round trip."""
    a = np.linspace(0.05, 0.99, 100)
    assert np.allclose(amp.mu_r_to_a(amp.a_to_mu_r(a)), a, rtol=1e-14)
    assert np.allclose(amp.a_to_mu_r(a), 1.0 / a, rtol=1e-14)
    y = np.geomspace(0.01, 2.0, 50)
    assert np.allclose(amp.mu_r_of_y(y), 1.0 / amp.a_of_y(y), rtol=1e-14)


def test_g26_mu_r_range():
    """G26's quoted 1 < mu_r <= 5.5 for y in [0.01, 2] is approximate.

    The audit found mu_r(y=2) = 5.83, not 5.5 (5.5 corresponds to y = 1.89),
    and mu_r(y=0.01) = 1.0100.  Pin the true values so the discrepancy is
    documented rather than rediscovered.
    """
    assert float(amp.mu_r_of_y(2.0)) == pytest.approx(5.8284, rel=1e-4)
    assert float(amp.mu_r_of_y(0.01)) == pytest.approx(1.01005, rel=1e-5)
    assert float(amp.y_of_a(amp.mu_r_to_a(5.5))) == pytest.approx(1.9188, rel=1e-4)


def test_time_delay_inversion():
    for M_Lz, y0 in ((1e3, 0.5), (1e4, 0.05), (1e2, 1.5)):
        t_d = float(amp.time_delay(M_Lz, y0))
        assert amp.y_of_time_delay(M_Lz, t_d) == pytest.approx(y0, rel=1e-9)


def test_w_of_f_consistency():
    """2 pi f t_d = w DT(y) exactly, which fixes w = 8 pi G M_Lz f / c^3."""
    f, M_Lz, y = 137.0, 3000.0, 0.4
    lhs = 2.0 * np.pi * f * float(amp.time_delay(M_Lz, y))
    rhs = float(amp.w_of_f(f, M_Lz)) * float(amp.delay_dimensionless(y))
    assert lhs == pytest.approx(rhs, rel=1e-14)
    assert float(amp.f_of_w(amp.w_of_f(f, M_Lz), M_Lz)) == pytest.approx(f)
