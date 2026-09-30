#!/usr/bin/env python
"""Convention checks: signs, normalization and degeneracies of the recombination.

Runs seven checks against explicit frequency-domain filtering and writes a
machine-readable result file; ``tests/test_phase0_gate.py`` runs the same checks.

    python scripts/checks/phase0_conventions.py

Tolerance: relative error below 1e-10 in controlled double-precision tests.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from lensing import amplification as amp  # noqa: E402
from lensing import conventions as cv  # noqa: E402
from lensing import recombine as rc  # noqa: E402
from lensing import simulation as sim  # noqa: E402
from lensing import waveforms as wf  # noqa: E402

GATE_TOL = 1e-10


def provenance(cfg, seed):
    """Provenance block stored in every result file."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        commit = "unknown"
    import scipy
    import pycbc

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "seed": seed,
        "config": cfg.as_dict(),
        "convention": cv.DEFAULT.as_dict(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pycbc": pycbc.__version__,
        },
        "gate_tol": GATE_TOL,
        "segment": sim.validate_segment(cfg, t_d_max=0.5),
    }


# --------------------------------------------------------------------------
# Item 1-3: build the injection and scan the four sign combinations
# --------------------------------------------------------------------------


def snap_to_grid(t, cfg, oversample=1):
    """Round a delay to the nearest available SNR sample."""
    dt = cfg.delta_t / oversample
    return float(np.rint(t / dt) * dt)


def item_123_sign_scan(cfg, psd, h, M_Lz, y, t0):
    """Zero-noise F_GO injection; scan {+i,-i} x {t+t_d, t-t_d}.

    The injection is built from ``amp.F_GO``, which applies the project
    convention.  The *correct* template convention is the one that
    recovers the injection with match 1.
    """
    freqs = cfg.frequencies()
    t_d_exact = float(amp.time_delay(M_Lz, y))
    t_d_snapped = snap_to_grid(t_d_exact, cfg)
    a = float(amp.a_of_y(y))

    # Injection: signal = F_GO * h, coalescence of image 1 placed at t0.
    # F_GO comes from the amplification module, an independent code path from
    # the template construction in `recombine`, so this is not a circular test.
    data = wf.fd_delay(h * amp.F_GO(freqs, M_Lz, y), freqs, t0)
    sigma_d = wf.sigma(data, psd, cfg.delta_f)

    z1, dt = wf.snr_series(data, h, psd, cfg)
    out = {
        "t_d_exact_s": t_d_exact,
        "t_d_snapped_s": t_d_snapped,
        "a": a,
        "sigma_data": sigma_d,
        "candidates": [],
    }

    for conv in cv.all_candidates():
        # -- match test: EXACT t_d, evaluated at EXACTLY t0 by direct
        #    summation.  Since F_GO_proj = sqrt(mu_+) [1 + i a e^{-2 pi i f
        #    t_d}], the injection is exactly proportional to h_L for the
        #    correct convention, so |z_L(t0)| must equal sigma_d identically --
        #    no time grid, no oversampling, no interpolation involved.
        z_ex_t0 = rc.explicit_lensed_snr_at_times(
            data, h, psd, cfg, t_d_exact, a, [t0], conv=conv
        )[0]
        z1_t0 = wf.snr_at_times(data, h, psd, cfg, [t0])[0]
        z1s_t0 = wf.snr_at_times(
            data, h, psd, cfg, [t0 + conv.lag(t_d_exact)]
        )[0]
        C_exact = wf.autocorr_at_lags(
            h, psd, cfg, [conv.lag(t_d_exact)]
        )[0]
        z_dir_t0 = rc.recombine_direct(z1_t0, z1s_t0, C_exact, a, conv=conv)

        # -- algebra test: snapped t_d so the roll is exact, full time series
        tau = conv.lag(t_d_snapped)
        z1s = np.roll(z1, -int(np.rint(tau / dt)))
        C_tau = wf.autocorr_at_lags(h, psd, cfg, [tau])[0]
        z_dir = rc.recombine_direct(z1, z1s, C_tau, a, conv=conv)
        sq_exp = rc.recombine_expanded_sq(z1, z1s, C_tau, a, conv=conv)
        z_exp, _ = rc.explicit_lensed_snr(
            data, h, psd, cfg, t_d_snapped, a, conv=conv
        )

        out["candidates"].append(
            {
                "name": conv.name,
                "morse_sign": conv.morse_sign,
                "shift_sign": conv.shift_sign,
                "norm_cross_sign": conv.norm_cross_sign(),
                # recovered match at the TRUE lens parameters, exact arithmetic
                "match_at_truth": float(abs(z_ex_t0) / sigma_d),
                "match_at_truth_recombined": float(abs(z_dir_t0) / sigma_d),
                "peak_time_s": float(int(np.argmax(np.abs(z_exp))) * dt),
                "peak_match_on_grid": float(np.abs(z_exp).max() / sigma_d),
                # algebraic self-consistency of recombination vs explicit
                "err_direct_vs_explicit": float(
                    np.max(np.abs(z_dir - z_exp)) / np.max(np.abs(z_exp))
                ),
                "err_expanded_vs_explicit": float(
                    np.max(np.abs(sq_exp - np.abs(z_exp) ** 2))
                    / np.max(np.abs(z_exp) ** 2)
                ),
                "err_exact_direct_vs_explicit": float(
                    abs(z_dir_t0 - z_ex_t0) / abs(z_ex_t0)
                ),
            }
        )
    return out


# --------------------------------------------------------------------------
# Item 4: autocorrelation normalization vs explicit template norm
# --------------------------------------------------------------------------


def item_4_norm_consistency(cfg, psd, h, n_check=64, seed=0):
    """``N`` from ``C(tau)`` must equal ``(h_L|h_L)`` from the explicit template."""
    rng = np.random.default_rng(seed)
    t_ds = rng.uniform(1e-3, 0.5, size=n_check)
    a_s = rng.uniform(0.05, 0.999, size=n_check)
    errs = []
    for t_d, a in zip(t_ds, a_s):
        tau = cv.DEFAULT.lag(t_d)
        C_tau = wf.autocorr_at_lags(h, psd, cfg, [tau])[0]
        n_alg = float(rc.lensed_norm_sq(C_tau, a))
        n_exp = float(rc.explicit_lensed_norm(h, psd, cfg, t_d, a) ** 2)
        errs.append(abs(n_alg - n_exp) / n_exp)
    return {
        "n_check": int(n_check),
        "max_rel_err": float(np.max(errs)),
        "median_rel_err": float(np.median(errs)),
        "norm_cross_sign_used": cv.DEFAULT.norm_cross_sign(),
    }


# --------------------------------------------------------------------------
# Item 5: pointwise comparison over the whole time series
# --------------------------------------------------------------------------


def item_5_pointwise(cfg, psd, h, M_Lz, y, t0, seed=1, with_noise=True, conv=None):
    """Direct, expanded and explicit forms compared at every time sample.

    ``t_d`` is snapped to the sample grid so that the shift is exact and the
    test isolates the *algebra* from interpolation error; fractional delays are
    the subject of a separate oversampling study.

    ``conv`` defaults to the project convention.  It is exposed so the same
    exactness check can be run for every relative Morse phase: the
    identity is generated from ``conv.filter_coeff(1.0)`` alone and so must
    hold for all four, which is the basis of the generalized search.
    """
    freqs = cfg.frequencies()
    t_d = snap_to_grid(float(amp.time_delay(M_Lz, y)), cfg)
    a = float(amp.a_of_y(y))
    if conv is None:
        conv = cv.DEFAULT

    data = wf.fd_delay(h * amp.F_GO(freqs, M_Lz, y), freqs, t0)
    if with_noise:
        # Correctly normalized coloured Gaussian noise: E|n_k|^2 = S_k/(2 df),
        # so the complex SNR has unit variance per quadrature.  Defined once in
        # lensing.simulation and pinned by tests/test_simulation.py.
        data = data + sim.noise_frequency_series(
            cfg, psd, np.random.default_rng(seed)
        )

    z1, dt = wf.snr_series(data, h, psd, cfg)
    tau = conv.lag(t_d)
    z1s = np.roll(z1, -int(np.rint(tau / dt)))
    C_tau = wf.autocorr_at_lags(h, psd, cfg, [tau])[0]

    z_dir = rc.recombine_direct(z1, z1s, C_tau, a, conv=conv)
    sq_exp = rc.recombine_expanded_sq(z1, z1s, C_tau, a, conv=conv)
    z_ref, _ = rc.explicit_lensed_snr(data, h, psd, cfg, t_d, a, conv=conv)

    scale = float(np.max(np.abs(z_ref)))
    return {
        "with_noise": bool(with_noise),
        "rel_morse_quarters": int(conv.rel_morse_quarters),
        "t_d_s": t_d,
        "a": a,
        "n_samples": int(z_ref.size),
        "peak_snr": scale,
        "max_rel_err_direct": float(np.max(np.abs(z_dir - z_ref)) / scale),
        "max_rel_err_expanded": float(
            np.max(np.abs(sq_exp - np.abs(z_ref) ** 2)) / scale**2
        ),
        "max_abs_err_direct": float(np.max(np.abs(z_dir - z_ref))),
    }


# --------------------------------------------------------------------------
# Item 6: the t_d -> 0 degeneracy
# --------------------------------------------------------------------------


def item_6_zero_delay(cfg, psd, h, t0, a_values=(0.05, 0.3, 0.7, 0.95, 0.999)):
    """At ``t_d = 0`` the normalized lensed template is ``h`` times a constant
    complex phase, so the phase-maximized magnitudes must agree exactly for
    every ``a``.  Also checks the approach to that limit at small ``t_d``.
    """
    freqs = cfg.frequencies()
    data = wf.fd_delay(h, freqs, t0)
    z_ul, dt = wf.snr_series(data, h, psd, cfg)
    ref = float(np.max(np.abs(z_ul)))

    exact, approach = [], []
    for a in a_values:
        z_l, _ = rc.explicit_lensed_snr(data, h, psd, cfg, 0.0, a)
        exact.append(
            {
                "a": float(a),
                "max_abs_z_lensed": float(np.max(np.abs(z_l))),
                "rel_err_vs_unlensed": float(abs(np.max(np.abs(z_l)) - ref) / ref),
            }
        )
        for t_d in (1e-5, 1e-4, 1e-3):
            z_s, _ = rc.explicit_lensed_snr(data, h, psd, cfg, t_d, a)
            approach.append(
                {
                    "a": float(a),
                    "t_d_s": t_d,
                    "rel_diff": float(abs(np.max(np.abs(z_s)) - ref) / ref),
                }
            )
    return {
        "unlensed_peak": ref,
        "exact_zero_delay": exact,
        "max_rel_err_zero_delay": float(
            max(e["rel_err_vs_unlensed"] for e in exact)
        ),
        "small_delay_approach": approach,
    }


# --------------------------------------------------------------------------
# Item 7: positivity of the template norm over the whole grid
# --------------------------------------------------------------------------


def item_7_positivity(cfg, psd, h, n_td=400, n_y=200):
    """``N(t_d, a) > 0`` everywhere on the search domain."""
    t_ds = np.linspace(1e-3, 0.5, n_td)
    ys = np.geomspace(0.01, 2.0, n_y)
    a_s = amp.a_of_y(ys)
    taus = cv.DEFAULT.lag(t_ds)
    C = wf.autocorr_at_lags(h, psd, cfg, taus)

    N = rc.lensed_norm_sq(C[:, None], a_s[None, :])
    # analytic floor: |C| <= 1  =>  N >= (1 - a)^2
    floor = (1.0 - a_s) ** 2
    return {
        "grid": {"n_td": int(n_td), "n_y": int(n_y)},
        "min_norm_sq": float(N.min()),
        "max_norm_sq": float(N.max()),
        "all_positive": bool(np.all(N > 0)),
        "min_margin_vs_analytic_floor": float(np.min(N - floor[None, :])),
        "max_abs_C": float(np.max(np.abs(C))),
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-dur", type=float, default=16.0)  # 25+25 Msun: 1.2 s chirp
    ap.add_argument("--sample-rate", type=float, default=4096.0)
    ap.add_argument("--mass1", type=float, default=25.0)
    ap.add_argument("--mass2", type=float, default=25.0)
    ap.add_argument("--M-Lz", type=float, default=2000.0)
    ap.add_argument("--y", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=20260831)
    ap.add_argument(
        "--out", default="results/development/phase0/phase0_gate.json"
    )
    args = ap.parse_args()

    cfg = wf.AnalysisConfig(
        seg_dur=args.seg_dur,
        sample_rate=args.sample_rate,
        mass1=args.mass1,
        mass2=args.mass2,
    )
    psd = wf.make_psd(cfg)
    h = wf.normalize(wf.make_waveform(cfg), psd, cfg.delta_f)
    t0 = 0.5 * cfg.seg_dur

    res = {"provenance": provenance(cfg, args.seed)}
    res["injection"] = {"M_Lz": args.M_Lz, "y": args.y, "t0_s": t0}

    print("== item 1-3: sign scan ==")
    res["item_123_sign_scan"] = item_123_sign_scan(
        cfg, psd, h, args.M_Lz, args.y, t0
    )
    for c in res["item_123_sign_scan"]["candidates"]:
        print(
            "  %-9s match=%.12f (recomb %.12f) peak_t=%.6f  "
            "err(direct)=%.2e err(exp)=%.2e"
            % (
                c["name"],
                c["match_at_truth"],
                c["match_at_truth_recombined"],
                c["peak_time_s"],
                c["err_direct_vs_explicit"],
                c["err_expanded_vs_explicit"],
            )
        )

    print("== item 4: norm consistency ==")
    res["item_4_norm"] = item_4_norm_consistency(cfg, psd, h, seed=args.seed)
    print("  max rel err = %.3e" % res["item_4_norm"]["max_rel_err"])

    print("== item 5: pointwise ==")
    res["item_5_pointwise_zero_noise"] = item_5_pointwise(
        cfg, psd, h, args.M_Lz, args.y, t0, seed=args.seed, with_noise=False
    )
    res["item_5_pointwise_noise"] = item_5_pointwise(
        cfg, psd, h, args.M_Lz, args.y, t0, seed=args.seed, with_noise=True
    )
    for k in ("item_5_pointwise_zero_noise", "item_5_pointwise_noise"):
        print(
            "  %-28s direct=%.3e expanded=%.3e (peak SNR %.3f)"
            % (
                k,
                res[k]["max_rel_err_direct"],
                res[k]["max_rel_err_expanded"],
                res[k]["peak_snr"],
            )
        )

    print("== item 6: t_d -> 0 degeneracy ==")
    res["item_6_zero_delay"] = item_6_zero_delay(cfg, psd, h, t0)
    print(
        "  max rel err at t_d=0 over a: %.3e"
        % res["item_6_zero_delay"]["max_rel_err_zero_delay"]
    )

    print("== item 7: positivity ==")
    res["item_7_positivity"] = item_7_positivity(cfg, psd, h)
    print(
        "  min N = %.6f (all positive: %s), max|C| = %.6f"
        % (
            res["item_7_positivity"]["min_norm_sq"],
            res["item_7_positivity"]["all_positive"],
            res["item_7_positivity"]["max_abs_C"],
        )
    )

    # ---- gate verdict --------------------------------------------------
    cands = res["item_123_sign_scan"]["candidates"]
    best = max(cands, key=lambda c: c["match_at_truth"])
    checks = {
        "algebra_direct_matches_explicit": max(
            c["err_direct_vs_explicit"] for c in cands
        )
        < GATE_TOL,
        "algebra_expanded_matches_explicit": max(
            c["err_expanded_vs_explicit"] for c in cands
        )
        < GATE_TOL,
        "norm_from_autocorr_consistent": res["item_4_norm"]["max_rel_err"]
        < GATE_TOL,
        "pointwise_zero_noise": res["item_5_pointwise_zero_noise"][
            "max_rel_err_direct"
        ]
        < GATE_TOL,
        "pointwise_with_noise": res["item_5_pointwise_noise"][
            "max_rel_err_direct"
        ]
        < GATE_TOL,
        "zero_delay_degeneracy": res["item_6_zero_delay"][
            "max_rel_err_zero_delay"
        ]
        < GATE_TOL,
        "norm_positive_on_grid": res["item_7_positivity"]["all_positive"],
        "selected_convention_recovers_injection": abs(
            best["match_at_truth"] - 1.0
        )
        < GATE_TOL,
        "recombination_matches_explicit_at_exact_delay": max(
            c["err_exact_direct_vs_explicit"] for c in cands
        )
        < GATE_TOL,
        "selected_convention_is_default": (
            best["morse_sign"] == cv.DEFAULT.morse_sign
            and best["shift_sign"] == cv.DEFAULT.shift_sign
        ),
    }
    res["selected_convention"] = best
    res["checks"] = checks
    res["gate_passed"] = bool(all(checks.values()))

    print("\n== gate ==")
    for k, v in checks.items():
        print("  [%s] %s" % ("PASS" if v else "FAIL", k))
    print(
        "  selected convention: %s (match %.12f)"
        % (best["name"], best["match_at_truth"])
    )
    print("  GATE %s" % ("PASSED" if res["gate_passed"] else "FAILED"))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("  wrote %s" % args.out)
    # Gate evidence must survive in version control: results/development/ is
    # gitignored, so a copy goes to results/frozen/.
    frozen = os.path.join(
        "results", "frozen", "phase0", os.path.basename(args.out)
    )
    os.makedirs(os.path.dirname(frozen), exist_ok=True)
    with open(frozen, "w") as fh:
        json.dump(res, fh, indent=2)
    print("  wrote %s" % frozen)
    return 0 if res["gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
