"""The threshold-triggered prefilter must be LOSSLESS.

If these tests fail, every Phase 4 false-alarm probability is wrong.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lensing import background as bg
from lensing import grids as gr
from lensing import priors as pr
from lensing import simulation as sim
from lensing import waveforms as wf

ETAS = (0.01, 0.1, 0.5)


@pytest.fixture(scope="module")
def setup():
    cfg = wf.AnalysisConfig(seg_dur=4.0, sample_rate=4096.0, mass1=25.0, mass2=25.0)
    psd = wf.make_psd(cfg)
    h = wf.normalize(wf.make_waveform(cfg), psd, cfg.delta_f)
    grid = gr.snap_grid_to_samples(
        gr.build_grid(n_td=20, n_y=10), cfg.delta_t / 2
    )
    pr.assign_weights(grid, "P0")
    s = bg.SearchSetup.build(cfg, psd, h, grid, oversample=2)
    return cfg, psd, h, s


def test_bound_constant_is_finite_and_sane(setup):
    cfg, psd, h, s = setup
    assert np.isfinite(s.bound_K)
    # K = max sqrt(2(1+a^2)/N); with N ~ 1+a^2 this is ~sqrt(2)
    assert 1.0 < s.bound_K < 3.0
    assert s.image_floor(6.0) == pytest.approx(6.0 / s.bound_K)


def test_prefilter_is_lossless_on_noise(setup):
    """Prefiltered statistics must EXACTLY equal brute force whenever the
    segment is not censored."""
    cfg, psd, h, s = setup
    rng = np.random.default_rng(11)
    n_checked = 0
    for _ in range(25):
        z1, _ = sim.noise_snr_series(cfg, psd, h, rng, oversample=2)
        ref = bg.evaluate_statistics_bruteforce(z1, s, etas=ETAS)
        # use a floor below the brute-force max so the segment is not censored
        floor = np.sqrt(2.0 * ref["S_max"]) * 0.5
        got = bg.evaluate_statistics(z1, s, floor, etas=ETAS)
        assert not got["censored"]
        for k in ("S_UL", "S_max", "S_soft") + tuple(
            f"S_mix_eta{e:g}" for e in ETAS
        ):
            assert got[k] == pytest.approx(ref[k], rel=1e-12), k
        n_checked += 1
    assert n_checked == 25


def test_prefilter_lossless_with_a_loud_injection(setup):
    """Same, with a real lensed signal present."""
    cfg, psd, h, s = setup
    rng = np.random.default_rng(3)
    for snr in (8.0, 15.0, 30.0):
        data, _ = sim.lensed_injection(
            cfg, psd, h, 3000.0, 0.35, 1.0, optimal_snr=snr, wave_optics=False
        )
        data = data + sim.noise_frequency_series(cfg, psd, rng)
        z1, _ = wf.snr_series(data, h, psd, cfg, oversample=2)
        ref = bg.evaluate_statistics_bruteforce(z1, s, etas=ETAS)
        got = bg.evaluate_statistics(z1, s, 4.0, etas=ETAS)
        assert not got["censored"]
        for k in ("S_max", "S_soft"):
            assert got[k] == pytest.approx(ref[k], rel=1e-12), (k, snr)


def test_censoring_bound_is_never_violated(setup):
    """When a segment IS censored, brute force must confirm it really is below
    the floor.  A violation here would silently bias every threshold."""
    cfg, psd, h, s = setup
    rng = np.random.default_rng(5)
    n_censored = 0
    for _ in range(60):
        z1, _ = sim.noise_snr_series(cfg, psd, h, rng, oversample=2)
        floor = 8.0
        got = bg.evaluate_statistics(z1, s, floor, etas=ETAS)
        if got["censored"]:
            n_censored += 1
            ref = bg.evaluate_statistics_bruteforce(z1, s, etas=ETAS)
            assert ref["S_max"] <= got["floor_stat"] + 1e-9
            assert ref["S_soft"] <= got["floor_stat"] + 1e-9
    assert n_censored > 0, "floor too low to exercise the censoring path"


def test_candidate_set_shrinks_with_floor(setup):
    cfg, psd, h, s = setup
    z1, _ = sim.noise_snr_series(
        cfg, psd, h, np.random.default_rng(9), oversample=2
    )
    sizes = [bg.candidate_times(z1, s, f).size for f in (3.0, 5.0, 7.0, 10.0)]
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[-1] < z1.size


def test_threshold_at_fap_reports_insufficiency():
    v = np.random.default_rng(0).normal(size=500)
    r = bg.threshold_at_fap(v, 1e-4)
    assert not r["sufficient"]
    r2 = bg.threshold_at_fap(np.random.default_rng(0).normal(size=200000), 1e-3)
    assert r2["sufficient"] and r2["n_exceedances"] >= 100
    assert r2["threshold_lo_2sigma"] <= r2["threshold"] <= r2["threshold_hi_2sigma"]


def test_efficiency_interval_brackets_estimate():
    v = np.random.default_rng(1).normal(loc=3.0, size=2000)
    e = bg.efficiency_at_threshold(v, 2.0)
    assert e["eff_lo"] <= e["efficiency"] <= e["eff_hi"]
    assert 0.0 <= e["eff_lo"] and e["eff_hi"] <= 1.0


def test_multi_prior_matches_single_prior(setup):
    """evaluate_statistics_multi must reproduce the single-prior path exactly."""
    from lensing import background as bg2

    cfg, psd, h, s = setup
    rng = np.random.default_rng(21)
    lw = bg2.log_weights_for_priors(s.grid, ["P0", "P1", "P3"])
    for _ in range(6):
        z1, _ = sim.noise_snr_series(cfg, psd, h, rng, oversample=2)
        single = bg2.evaluate_statistics(z1, s, 4.0, etas=ETAS)
        multi = bg2.evaluate_statistics_multi(z1, s, 4.0, lw, etas=ETAS)
        assert multi["S_UL"] == pytest.approx(single["S_UL"], rel=1e-13)
        assert multi["S_max"] == pytest.approx(single["S_max"], rel=1e-13)
        # the grid fixture carries P0 weights, so P0 must agree
        assert multi["S_soft_P0"] == pytest.approx(single["S_soft"], rel=1e-12)
        for e in ETAS:
            assert multi[f"S_mix_P0_eta{e:g}"] == pytest.approx(
                single[f"S_mix_eta{e:g}"], rel=1e-12
            )
        # different priors must actually give different soft statistics
        assert multi["S_soft_P0"] != multi["S_soft_P3"]


def test_fast_path_exact_for_max_and_brackets_soft(setup):
    """evaluate_statistics_fast: S_UL/S_max exact, S_soft bracketed."""
    from lensing import background as bg2

    cfg, psd, h, s = setup
    rng = np.random.default_rng(31)
    widths = []
    for _ in range(20):
        z1, _ = sim.noise_snr_series(cfg, psd, h, rng, oversample=2)
        ref = bg2.evaluate_statistics_bruteforce(z1, s, etas=ETAS)
        got = bg2.evaluate_statistics_fast(z1, s, 4.0, etas=ETAS)
        if got["censored"]:
            continue
        # exact
        assert got["S_UL"] == pytest.approx(ref["S_UL"], rel=1e-12)
        assert got["S_max"] == pytest.approx(ref["S_max"], rel=1e-12)
        # bracketed: the truth must lie inside [lower, upper]
        assert got["S_soft"] <= ref["S_soft"] + 1e-9
        assert ref["S_soft"] <= got["S_soft_upper"] + 1e-9
        # The bracket only needs to be narrow where the statistic is large
        # enough to matter for calibration; well below the floor it may be
        # wide and is harmless, since no quoted threshold sits there.
        if got["S_soft"] > got["floor_stat"] + 4.0:
            widths.append(got["S_soft_bracket"])
        for e in ETAS:
            assert got[f"S_mix_eta{e:g}"] <= ref[f"S_mix_eta{e:g}"] + 1e-6
    # (pure-noise segments rarely clear floor+4; the near-threshold regime is
    # exercised by test_soft_bracket_is_negligible_near_threshold below.)
    assert all(w < 0.5 for w in widths)


def test_soft_bracket_is_negligible_near_threshold(setup):
    """Where it matters -- statistic well above the floor -- the bracket on
    S_soft must be far smaller than the spacing between calibrated FAPs."""
    from lensing import background as bg2

    cfg, psd, h, s = setup
    rng = np.random.default_rng(77)
    checked = 0
    for _ in range(40):
        # a loud injection puts the statistic firmly above the floor
        data, _ = sim.lensed_injection(
            cfg, psd, h, 3000.0, 0.35, 1.0, optimal_snr=14.0,
            wave_optics=False,
        )
        data = data + sim.noise_frequency_series(cfg, psd, rng)
        z1, _ = wf.snr_series(data, h, psd, cfg, oversample=2)
        got = bg2.evaluate_statistics_fast(z1, s, 5.0, etas=(0.1,))
        if got["censored"]:
            continue
        ref = bg2.evaluate_statistics_bruteforce(z1, s, etas=(0.1,))
        assert got["S_max"] == pytest.approx(ref["S_max"], rel=1e-12)
        assert got["S_soft"] <= ref["S_soft"] + 1e-9
        assert ref["S_soft"] <= got["S_soft_upper"] + 1e-9
        assert got["S_soft_bracket"] < 0.05, got["S_soft_bracket"]
        checked += 1
    assert checked >= 30


def test_fast_path_much_cheaper_on_a_dense_grid():
    """The per-delay restriction is what makes a dense grid affordable."""
    import time

    from lensing import background as bg2
    from lensing import grids as gr2
    from lensing import priors as pr2

    cfg = wf.AnalysisConfig(seg_dur=4.0, sample_rate=4096.0,
                            mass1=25.0, mass2=25.0)
    psd = wf.make_psd(cfg)
    h = wf.normalize(wf.make_waveform(cfg), psd, cfg.delta_f)
    grid = gr2.snap_grid_to_samples(
        gr2.build_grid(n_td=400, n_y=8), cfg.delta_t / 2
    )
    pr2.assign_weights(grid, "P0")
    s = bg2.SearchSetup.build(cfg, psd, h, grid, oversample=2)
    z1, _ = sim.noise_snr_series(
        cfg, psd, h, np.random.default_rng(2), oversample=2
    )

    t0 = time.perf_counter()
    fast = bg2.evaluate_statistics_fast(z1, s, 5.0, etas=(0.1,))
    t_fast = time.perf_counter() - t0
    t0 = time.perf_counter()
    slow = bg2.evaluate_statistics(z1, s, 5.0, etas=(0.1,))
    t_slow = time.perf_counter() - t0

    assert fast["S_max"] == pytest.approx(slow["S_max"], rel=1e-10)
    assert t_fast < t_slow, (t_fast, t_slow)
    # the saving should be substantial, not marginal
    assert t_slow / t_fast > 3.0


def test_multi_fast_matches_single_fast(setup):
    """evaluate_statistics_multi_fast must agree with the single-prior fast path."""
    from lensing import background as bg2

    cfg, psd, h, s = setup
    rng = np.random.default_rng(51)
    lw = bg2.log_weights_for_priors(s.grid, ["P0", "P1", "P3"])
    for _ in range(8):
        z1, _ = sim.noise_snr_series(cfg, psd, h, rng, oversample=2)
        one = bg2.evaluate_statistics_fast(z1, s, 4.0, etas=ETAS)
        many = bg2.evaluate_statistics_multi_fast(z1, s, 4.0, lw, etas=ETAS)
        assert many["S_UL"] == pytest.approx(one["S_UL"], rel=1e-13)
        assert many["S_max"] == pytest.approx(one["S_max"], rel=1e-13)
        assert many["S_soft_P0"] == pytest.approx(one["S_soft"], rel=1e-12)
        for e in ETAS:
            assert many[f"S_mix_P0_eta{e:g}"] == pytest.approx(
                one[f"S_mix_eta{e:g}"], rel=1e-12
            )
        if not many["censored"]:
            assert many["S_soft_P0"] != many["S_soft_P3"]
