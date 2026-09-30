"""Noise normalization and segment-length validation.

The noise scale is the single most consequential constant in Phase 4: a factor
sqrt(2) error would rescale every calibrated threshold and efficiency curve.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lensing import simulation as sim
from lensing import waveforms as wf


@pytest.fixture(scope="module")
def setup():
    cfg = wf.AnalysisConfig(seg_dur=8.0, sample_rate=4096.0, mass1=25.0, mass2=25.0)
    psd = wf.make_psd(cfg)
    h = wf.normalize(wf.make_waveform(cfg), psd, cfg.delta_f)
    return cfg, psd, h


def test_snr_variance_is_unity_per_quadrature(setup):
    """E|z|^2 = 2, i.e. unit variance in each quadrature of the complex SNR.

    This is the definition that makes |z|^2 a chi-squared with 2 dof, so that
    a quoted "SNR 8" means the standard thing.
    """
    cfg, psd, h = setup
    rng = np.random.default_rng(12345)
    # one interior sample per realization -> independent draws
    vals = []
    for _ in range(400):
        z, _ = sim.noise_snr_series(cfg, psd, h, rng)
        vals.append(z[len(z) // 3])
    vals = np.array(vals)
    var_re = np.var(vals.real)
    var_im = np.var(vals.imag)
    n = vals.size
    # 400 samples -> ~7% fractional error on a variance; allow 5 sigma
    tol = 5.0 * np.sqrt(2.0 / n)
    assert var_re == pytest.approx(1.0, abs=tol)
    assert var_im == pytest.approx(1.0, abs=tol)
    assert np.mean(np.abs(vals) ** 2) == pytest.approx(2.0, abs=2 * tol)
    # quadratures uncorrelated
    assert abs(np.corrcoef(vals.real, vals.imag)[0, 1]) < 5.0 / np.sqrt(n)


def test_noise_psd_normalization(setup):
    """E|n_k|^2 = S_k / (2 df), directly."""
    cfg, psd, _ = setup
    rng = np.random.default_rng(7)
    k = np.array([200, 800, 2000])  # in-band bins
    acc = np.zeros(k.size)
    n_real = 3000
    for _ in range(n_real):
        n = sim.noise_frequency_series(cfg, psd, rng)
        acc += np.abs(n[k]) ** 2
    acc /= n_real
    expected = psd[k] / (2.0 * cfg.delta_f)
    assert np.allclose(acc / expected, 1.0, atol=6.0 / np.sqrt(n_real))


def test_noise_is_zero_out_of_band(setup):
    cfg, psd, _ = setup
    n = sim.noise_frequency_series(cfg, psd, np.random.default_rng(0))
    assert np.all(n[~cfg.band_mask()] == 0)


def test_chirp_time_11_msun_overflows_16s():
    """The fact that forced the segment-length guard: 11 Msun from 20 Hz lasts
    16.43 s, longer than a 16 s segment."""
    cfg = wf.AnalysisConfig(seg_dur=16.0, mass1=5.5, mass2=5.5, f_lower=20.0)
    assert sim.chirp_time(cfg) == pytest.approx(16.426, rel=1e-3)
    with pytest.raises(ValueError, match="too short"):
        sim.validate_segment(cfg)
    assert sim.suggested_seg_dur(cfg) >= 32.0


def test_validate_segment_accepts_adequate_config():
    cfg = wf.AnalysisConfig(seg_dur=32.0, mass1=25.0, mass2=25.0, f_lower=20.0)
    info = sim.validate_segment(cfg)
    assert info["chirp_time_s"] < info["seg_dur_s"]


def test_injection_optimal_snr_is_exact(setup):
    """The injected optimal SNR must be exactly what was requested."""
    cfg, psd, h = setup
    data, info = sim.lensed_injection(
        cfg, psd, h, 2000.0, 0.3, 4.0, optimal_snr=12.0, wave_optics=False
    )
    assert wf.sigma(data, psd, cfg.delta_f) == pytest.approx(12.0, rel=1e-12)
    assert info["optimal_snr"] == pytest.approx(12.0, rel=1e-12)
