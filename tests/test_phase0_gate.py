"""One test per Phase 0 gate item.

These run on a small fast configuration so they are usable in CI; the full
gate lives in ``scripts/checks/phase0_conventions.py``.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "checks"))

from lensing import amplification as amp
from lensing import conventions as cv
from lensing import recombine as rc
from lensing import waveforms as wf

GATE_TOL = 1e-10


@pytest.fixture(scope="module")
def setup():
    cfg = wf.AnalysisConfig(seg_dur=8.0, sample_rate=4096.0, mass1=25.0, mass2=25.0)
    psd = wf.make_psd(cfg)
    h = wf.normalize(wf.make_waveform(cfg), psd, cfg.delta_f)
    return cfg, psd, h


def test_matched_filter_agrees_with_pycbc(setup):
    """The hand-rolled matched filter must reproduce pycbc.filter."""
    cfg, psd, h = setup
    from pycbc.filter import matched_filter
    from pycbc.types import FrequencySeries

    z, _ = wf.snr_series(h, h, psd, cfg)
    safe_psd = np.where(np.isfinite(psd), psd, 1e50)
    zp = np.asarray(
        matched_filter(
            FrequencySeries(h, delta_f=cfg.delta_f),
            FrequencySeries(h, delta_f=cfg.delta_f),
            psd=FrequencySeries(safe_psd, delta_f=cfg.delta_f),
            low_frequency_cutoff=cfg.f_lower,
            high_frequency_cutoff=cfg.f_final,
        ).numpy()
    )
    assert np.max(np.abs(z - zp)) < 1e-13
    assert abs(np.abs(z).max() - 1.0) < 1e-12  # self-match of a unit-norm template


def test_convention_is_self_consistent():
    """s_N must be *derived* from the Morse sign, not set independently."""
    for conv in cv.all_candidates():
        assert conv.norm_cross_sign() == conv.morse_sign  # antilinear template
        u = conv.filter_coeff(1.0)
        C = 0.3 + 0.7j
        a = 0.5
        # the expanded denominator must equal the direct one
        direct = 1.0 + a * a + 2.0 * np.real(conv.filter_coeff(a) * C)
        assert rc.lensed_norm_sq(C, a, conv=conv) == pytest.approx(direct, rel=1e-15)
        assert 2.0 * a * np.real(u * C) == pytest.approx(
            2.0 * np.real(conv.filter_coeff(a) * C), rel=1e-15
        )


def test_item3_sign_scan_selects_default_convention(setup):
    """Gate items 1-3: only one of the four sign combinations recovers the
    injection, and it is the project default."""
    cfg, psd, h = setup
    from phase0_conventions import item_123_sign_scan

    res = item_123_sign_scan(cfg, psd, h, 2000.0, 0.3, 0.5 * cfg.seg_dur)
    by_name = {c["name"]: c for c in res["candidates"]}
    best = max(res["candidates"], key=lambda c: c["match_at_truth"])

    assert best["morse_sign"] == cv.DEFAULT.morse_sign == +1
    assert best["shift_sign"] == cv.DEFAULT.shift_sign == +1
    assert abs(best["match_at_truth"] - 1.0) < GATE_TOL
    # the other three must be clearly worse, not marginally so
    others = [c["match_at_truth"] for c in res["candidates"] if c is not best]
    assert max(others) < 0.9
    # and the recombination reproduces explicit filtering for *every* candidate
    for c in res["candidates"]:
        assert c["err_direct_vs_explicit"] < GATE_TOL
        assert c["err_expanded_vs_explicit"] < GATE_TOL
        assert c["err_exact_direct_vs_explicit"] < GATE_TOL
    assert set(by_name) == {"m+1_t+1", "m+1_t-1", "m-1_t+1", "m-1_t-1"}


def test_item3_delay_direction_is_physical(setup):
    """The trailing (saddle) image must arrive LATER by t_d.

    This is the test that forced the conjugation of F: with
    the literature's e^{+i w DT} the second image lands at t0 - t_d.
    """
    cfg, psd, h = setup
    freqs = cfg.frequencies()
    M_Lz, y, t0 = 2000.0, 0.3, 0.5 * cfg.seg_dur
    t_d = float(amp.time_delay(M_Lz, y))

    for use_project, expected in ((True, t0 + t_d), (False, t0 - t_d)):
        F = amp.F_GO_lit(freqs, M_Lz, y)
        if use_project:
            F = amp.to_project_convention(F)
        data = wf.fd_delay(h * F, freqs, t0)
        z, dt = wf.snr_series(data, h, psd, cfg)
        az = np.abs(z)
        # mask out the primary peak, then find the secondary
        n_prim = int(round(t0 / dt))
        guard = int(round(2e-3 / dt))
        az_masked = az.copy()
        az_masked[max(0, n_prim - guard) : n_prim + guard] = 0.0
        t_sec = int(np.argmax(az_masked)) * dt
        assert abs(t_sec - expected) < 2.0 * dt


def test_item4_norm_from_autocorrelation(setup):
    """Gate item 4: N from C(tau) equals the explicit template norm."""
    cfg, psd, h = setup
    rng = np.random.default_rng(0)
    for t_d, a in zip(rng.uniform(1e-3, 0.5, 20), rng.uniform(0.05, 0.999, 20)):
        C = wf.autocorr_at_lags(h, psd, cfg, [cv.DEFAULT.lag(t_d)])[0]
        n_alg = float(rc.lensed_norm_sq(C, a))
        n_exp = float(rc.explicit_lensed_norm(h, psd, cfg, t_d, a) ** 2)
        assert n_alg == pytest.approx(n_exp, rel=GATE_TOL)


def test_item4_autocorr_properties(setup):
    """C(0) = 1 and C(-tau) = conj(C(tau)) for a normalized template."""
    cfg, psd, h = setup
    lags = np.array([0.0, 1e-3, 5e-3, 2e-2])
    C = wf.autocorr_at_lags(h, psd, cfg, lags)
    Cm = wf.autocorr_at_lags(h, psd, cfg, -lags)
    assert C[0] == pytest.approx(1.0, abs=1e-14)
    assert np.allclose(C, np.conjugate(Cm), atol=1e-14)
    assert np.all(np.abs(C) <= 1.0 + 1e-14)
    # the exact and FFT-grid autocorrelations must agree
    Cs, lg = wf.autocorr_series(h, psd, cfg, oversample=1)
    j = np.argmin(np.abs(lg - 5e-3))
    assert Cs[j] == pytest.approx(
        wf.autocorr_at_lags(h, psd, cfg, [lg[j]])[0], abs=1e-13
    )


def test_item5_pointwise_with_and_without_noise(setup):
    """Gate item 5: direct, expanded and explicit agree at every time sample."""
    cfg, psd, h = setup
    from phase0_conventions import item_5_pointwise

    for with_noise in (False, True):
        r = item_5_pointwise(
            cfg, psd, h, 2000.0, 0.3, 0.5 * cfg.seg_dur, with_noise=with_noise
        )
        assert r["max_rel_err_direct"] < GATE_TOL
        assert r["max_rel_err_expanded"] < GATE_TOL


def test_item6_zero_delay_degeneracy(setup):
    """Gate item 6: at t_d = 0 the lensed template differs from the unlensed one
    only by a constant complex phase, so |z| agrees for every a."""
    cfg, psd, h = setup
    freqs = cfg.frequencies()
    t0 = 0.5 * cfg.seg_dur
    data = wf.fd_delay(h, freqs, t0)
    z_ul, _ = wf.snr_series(data, h, psd, cfg)
    ref = np.abs(z_ul).max()

    for a in (0.05, 0.3, 0.7, 0.95, 0.999):
        z_l, _ = rc.explicit_lensed_snr(data, h, psd, cfg, 0.0, a)
        assert np.abs(z_l).max() == pytest.approx(ref, rel=GATE_TOL)
        # and the template really is h times a constant phase.  Restrict to
        # bins where the waveform is actually nonzero: IMRPhenomD is exactly
        # zero above its own cutoff, which is inside the analysis band.
        hl = rc.explicit_lensed_template(h, freqs, 0.0, a)
        nz = np.abs(h) > 0
        ratio = hl[nz] / h[nz]
        assert np.allclose(ratio, ratio[0], rtol=1e-12)
        assert ratio[0] == pytest.approx(1.0 + 1j * a, rel=1e-14)


def test_item7_norm_positive_on_grid(setup):
    """Gate item 7: N(t_d, a) > 0 across the whole search domain."""
    cfg, psd, h = setup
    t_ds = np.linspace(1e-3, 0.5, 200)
    a_s = amp.a_of_y(np.geomspace(0.01, 2.0, 100))
    C = wf.autocorr_at_lags(h, psd, cfg, cv.DEFAULT.lag(t_ds))
    N = rc.lensed_norm_sq(C[:, None], a_s[None, :])
    assert np.all(N > 0)
    # analytic floor (1 - a)^2 must be respected since |C| <= 1
    assert np.all(N >= (1.0 - a_s[None, :]) ** 2 - 1e-12)
    assert rc.norm_is_positive(C[:, None], a_s[None, :])
