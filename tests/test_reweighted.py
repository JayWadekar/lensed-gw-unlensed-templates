"""Re-weighted SNR and the fixed-threshold sensitive volume.

The load-bearing tests are ``test_matches_pycbc_newsnr`` (the veto re-weighting
must be PyCBC's, not an approximation of it) and
``test_chi2_scaling_matches_a_direct_recomputation`` (the ``rho^2`` scaling of
the chi-squared excess, on which the whole distance extrapolation rests).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lensing import reweighted as rw


def test_matches_pycbc_newsnr():
    from pycbc.events.ranking import newsnr

    rng = np.random.default_rng(3)
    rho = rng.uniform(4.0, 60.0, 400)
    chi = rng.uniform(0.2, 40.0, 400)
    assert np.allclose(rw.new_snr(rho, chi), newsnr(rho, chi), rtol=1e-14)


def test_no_penalty_below_unity():
    """chi2_r <= 1 must leave the SNR untouched, exactly."""
    for c in (0.01, 0.5, 1.0):
        assert rw.new_snr(20.0, c) == pytest.approx(20.0, rel=0, abs=0)


def test_chi2_scaling_matches_a_direct_recomputation():
    """The rho^2 scaling of a noise-free mismatch, verified against pycbc.

    A fixed mismatched signal is rescaled in amplitude and the chi-squared
    recomputed; the excess must scale as rho^2. This is the assumption that
    lets one reference-SNR map be extrapolated to threshold distance.
    """
    from pycbc.types import FrequencySeries
    from pycbc.vetoes import power_chisq

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from lensing import amplification as amp
    from lensing import waveforms as wf

    cfg = wf.AnalysisConfig(seg_dur=8.0, sample_rate=4096.0,
                            mass1=25.0, mass2=25.0)
    psd = wf.make_psd(cfg)
    h = wf.normalize(wf.make_waveform(cfg), psd, cfg.delta_f)
    freqs = cfg.frequencies()
    nz = np.nonzero(np.abs(h) > 0)[0]
    safe = np.where(np.isfinite(psd) & (psd > 0), psd, 1e50)
    psd_fs = FrequencySeries(safe, delta_f=cfg.delta_f)
    nbins, t0 = 16, round(2.0 * cfg.sample_rate) / cfg.sample_rate
    dof = 2 * nbins - 2

    sig = np.zeros(freqs.size, dtype=complex)
    sig[nz] = h[nz] * amp.F_ML(freqs[nz], 1e3, 0.3, backend="mpmath")
    sig = wf.fd_delay(sig, freqs, t0)
    sig = sig / wf.sigma(sig, psd, cfg.delta_f)

    def excess(rho):
        d = sig * rho
        z, _ = wf.snr_series(d, h, psd, cfg, oversample=1)
        k = int(np.argmax(np.abs(z)))
        cs = power_chisq(
            FrequencySeries(h.astype(complex), delta_f=cfg.delta_f),
            FrequencySeries(d.astype(complex), delta_f=cfg.delta_f),
            nbins, psd_fs, low_frequency_cutoff=cfg.f_lower,
            high_frequency_cutoff=cfg.f_final,
        )
        return float(np.array(cs.data)[k] / dof)

    e20 = excess(20.0)
    for rho in (10.0, 40.0):
        assert rw.chi2_at_snr(e20, rho, 20.0) - 1.0 == pytest.approx(
            excess(rho), rel=2e-3
        ), rho


def test_newsnr_saturates_and_the_ceiling_is_right():
    """rho_hat is bounded for a mismatched signal, and the closed-form ceiling
    matches the numerical limit."""
    rho_ref, e = 20.0, 0.5
    ceil = float(rw.saturation_newsnr(e, rho_ref))
    vals = [float(rw.new_snr(r, rw.chi2_at_snr(e, r, rho_ref)))
            for r in (1e3, 1e4, 1e5)]
    assert vals[0] < vals[1] < vals[2] <= ceil
    assert vals[-1] == pytest.approx(ceil, rel=1e-3)
    assert np.isinf(rw.saturation_newsnr(0.0, rho_ref))


def test_newsnr_is_monotonic_in_rho():
    """Required for the bisection in threshold_optimal_snr to be valid."""
    for e in (0.0, 0.05, 1.0, 20.0):
        r = np.geomspace(1.0, 1e5, 400)
        v = rw.new_snr(r, rw.chi2_at_snr(e, r, 20.0))
        assert np.all(np.diff(v) >= -1e-9), e


def test_threshold_recovers_the_trivial_case():
    """With no mismatch and perfect recovery, threshold SNR == threshold."""
    got = rw.threshold_optimal_snr(0.0, 1.0, 20.0, 8.0)
    assert float(got[0]) == pytest.approx(8.0, rel=1e-6)
    # halving the recovered fraction doubles the required optimal SNR
    got = rw.threshold_optimal_snr(0.0, 0.5, 20.0, 8.0)
    assert float(got[0]) == pytest.approx(16.0, rel=1e-6)


def test_undetectable_when_saturation_is_below_threshold():
    """A badly mismatched signal has zero sensitive volume, not merely less."""
    # ceiling = 2^(1/6) * 20 / sqrt(e) < 8  ->  e > (2^(1/6)*20/8)^2 ~ 7.9
    big = 50.0
    assert float(rw.saturation_newsnr(big, 20.0)) < 8.0
    assert np.isinf(float(rw.threshold_optimal_snr(big, 1.0, 20.0, 8.0)[0]))
    v = rw.volume_ratio(big, 1.0, 0.0, 1.0, 20.0, 8.0)
    assert float(v[0]) == 0.0


def test_volume_ratio_is_the_cube_of_the_threshold_ratio():
    e_a, e_b = 2.0, 0.05
    ra = rw.threshold_optimal_snr(e_a, 0.9, 20.0, 8.0)
    rb = rw.threshold_optimal_snr(e_b, 1.0, 20.0, 8.0)
    v = rw.volume_ratio(e_a, 0.9, e_b, 1.0, 20.0, 8.0)
    assert float(v.ravel()[0]) == pytest.approx(
        float((rb / ra).ravel()[0] ** 3), rel=1e-12
    )
    assert float(v[0]) < 1.0            # the mismatched search is worse


def test_volume_ratio_shapes_broadcast():
    e = np.array([[0.0, 1.0], [4.0, 9.0]])
    m = np.full_like(e, 0.95)
    v = rw.volume_ratio(e, m, 0.0, 1.0, 20.0, 8.0)
    assert v.shape == e.shape
    assert np.all(np.diff(v.ravel()) <= 1e-12)   # worse chi2 -> less volume


def test_volume_ratio_accepts_per_arm_thresholds():
    """Different thresholds per search is what makes the comparison
    fixed-false-alarm rather than fixed-threshold.

    With no mismatch anywhere, the ratio must reduce to the cube of the
    threshold ratio -- so a search that pays a higher threshold for examining
    more templates loses exactly that much volume and no more.
    """
    v = rw.volume_ratio(0.0, 1.0, 0.0, 1.0, 20.0, 6.4992,
                        rho_hat_th_b=5.4743)
    assert float(v.ravel()[0]) == pytest.approx((5.4743 / 6.4992) ** 3, rel=1e-6)
    # equal thresholds reproduce the single-threshold behaviour
    v1 = rw.volume_ratio(0.3, 0.9, 0.05, 1.0, 20.0, 8.0)
    v2 = rw.volume_ratio(0.3, 0.9, 0.05, 1.0, 20.0, 8.0, rho_hat_th_b=8.0)
    assert float(v1.ravel()[0]) == pytest.approx(float(v2.ravel()[0]), rel=1e-12)
