"""The lens population of Sec. V (separated images).

Checks that the fiducial 10--60 minute window, the point-mass population it
implies, and its impact-parameter priors are solved rather than assumed.
"""

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lensing import amplification as amp                    # noqa: E402
from lensing import conventions                             # noqa: E402
from lensing import strong_pair as sp                       # noqa: E402
from lensing import waveforms as wf                         # noqa: E402


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pop():
    return sp.MonoMassPopulation.solve()


@pytest.fixture(scope="module")
def setup():
    cfg = wf.AnalysisConfig(seg_dur=8.0)
    psd = wf.make_psd(cfg)
    h = wf.make_waveform(cfg)
    return cfg, psd, h


# --------------------------------------------------------------------------
# SL1 -- population and priors
# --------------------------------------------------------------------------


def test_delay_endpoints_are_solved_not_assumed(pop):
    assert pop.t_d(pop.y_max) == pytest.approx(3600.0, rel=1e-9)
    assert pop.t_d(pop.y_min) == pytest.approx(600.0, rel=1e-9)
    # the spec's rounded values, reproduced rather than hard-coded
    assert pop.M_Lz == pytest.approx(8.7828e7, rel=1e-4)
    assert pop.y_min == pytest.approx(0.173155, rel=1e-5)


def test_benchmark_amplitude_ratios(pop):
    assert pop.a(pop.y_min) == pytest.approx(0.841, abs=5e-4)
    assert pop.a(0.5) == pytest.approx(0.610, abs=5e-4)
    assert pop.a(1.0) == pytest.approx(0.382, abs=5e-4)
    assert pop.t_d(0.5) / 60.0 == pytest.approx(29.1, abs=0.05)


def test_delay_is_monotone_and_invertible(pop):
    t = np.linspace(600.0, 3600.0, 41)
    y = np.atleast_1d(pop.y_of_t_d(t))
    assert np.all(np.diff(y) > 0)
    assert np.allclose(pop.t_d(y), t, rtol=1e-10)


def test_priors_normalize(pop):
    from scipy.integrate import quad
    iy = quad(lambda y: float(pop.p_y(y)), pop.y_min, pop.y_max)[0]
    it = quad(lambda t: float(pop.p_t_d(t)), 600.0, 3600.0, limit=200)[0]
    assert iy == pytest.approx(1.0, rel=1e-9)
    assert it == pytest.approx(1.0, rel=1e-6)


def test_delay_prior_is_the_jacobian_transform(pop):
    y = np.array([0.25, 0.5, 0.75, 0.95])
    t = pop.t_d(y)
    assert np.allclose(pop.p_t_d(t), pop.p_y(y) / pop.dt_d_dy(y), rtol=1e-8)


@pytest.mark.parametrize("kind", ["area", "uniform"])
def test_prior_nodes_sum_to_one(pop, kind):
    y, w = sp.prior_nodes(pop, 16, kind=kind)
    assert w.sum() == pytest.approx(1.0, rel=1e-12)
    assert np.all((y > pop.y_min) & (y < pop.y_max))
    # first moment against direct quadrature
    from scipy.integrate import quad
    if kind == "area":
        want = quad(lambda t: t * float(pop.p_y(t)), pop.y_min, pop.y_max)[0]
        assert float(np.sum(w * y)) == pytest.approx(want, rel=1e-8)


def test_long_lag_autocorrelation_is_negligible(setup, pop):
    """The two images cannot overlap: |C(t_d)| must be tiny over the domain.

    A segment FFT would alias these lags, so this uses the closed-form
    oscillatory quadrature of :func:`strong_pair.autocorr_long_lag`.
    """
    cfg, psd, h = setup
    lags = np.linspace(600.0, 3600.0, 13)
    C = sp.autocorr_long_lag(h, psd, cfg, lags)
    assert np.max(np.abs(C)) < 1e-5
    # the pair norm is 1 + a^2 to better than a part in 1e4 everywhere
    a = pop.a(np.atleast_1d(pop.y_of_t_d(lags)))
    err = 2.0 * a * np.abs(C) / (1.0 + a ** 2)
    assert np.max(err) < 1e-4
    # and the decay is the 1/tau band-edge tail, not numerical noise
    assert abs(C[0]) > abs(C[-1])


def test_long_lag_quadrature_is_converged(setup):
    cfg, psd, h = setup
    lags = np.array([600.0, 3600.0])
    coarse = sp.autocorr_long_lag(h, psd, cfg, lags, n_f=1 << 16)
    fine = sp.autocorr_long_lag(h, psd, cfg, lags, n_f=1 << 20)
    assert np.allclose(np.abs(coarse), np.abs(fine), rtol=1e-6)


def test_segment_fft_would_have_aliased(setup):
    """Guard the reason the closed-form quadrature exists."""
    cfg, psd, h = setup
    bad = wf.autocorr_at_lags(h, psd, cfg, [1748.25])
    good = sp.autocorr_long_lag(h, psd, cfg, [1748.25])
    assert abs(bad) > 100.0 * abs(good)


def test_blocks_tile_the_window_with_constant_a(pop):
    blk = sp.make_blocks(pop, 4.0)
    assert blk.n_block == 750
    assert blk.a_spread < 1e-3
    assert blk.centers[0] == pytest.approx(602.0)
    assert blk.centers[-1] == pytest.approx(3598.0)
    assert np.all(np.diff(blk.a) < 0)          # a falls as the delay grows


def test_f_moments_match_the_predeclared_values(setup):
    cfg, psd, h = setup
    f_bar, f_rms, sig_f = sp.f_moments(h, psd, cfg)
    assert f_bar == pytest.approx(93.0, abs=0.5)
    assert f_rms == pytest.approx(121.9, abs=0.5)
    assert sig_f == pytest.approx(78.8, abs=0.5)


# --------------------------------------------------------------------------
# SL2 -- the conditional statistic
# --------------------------------------------------------------------------


def test_morse_phase_comes_from_the_population_not_inline(pop):
    assert pop.relative_morse_quarters() == 1
    # the argument is the impact parameter, so c = i a(y)
    assert pop.image_coeff(0.5) == pytest.approx(1j * pop.a(0.5))
    assert pop.filter_coeff(0.5) == pytest.approx(-1j * pop.a(0.5))
    assert pop.a(0.5) == pytest.approx(0.6096, abs=1e-4)


