#!/usr/bin/env python
"""Signal-consistency (power chi-squared) of lensed signals, and its repair.

Two questions, one map each:

1. **Does an overlapping-image lensed signal look like a glitch?** A lensed
   injection is filtered with the ordinary *unlensed* template and PyCBC's
   ``power_chisq`` veto is evaluated at the trigger time. Where the two images
   overlap within the waveform, the composite is not a clean chirp and the
   veto fires -- which is precisely the regime the two-image search targets.

2. **Does modelling the second image repair it?** The two-image statistic is
   run on the *frozen* production grid, giving a best-fit ``(t_d, a)``. Two
   forms of the answer are computed, and the primary one is the conventional:

   * ``chi2_lensed`` -- the **conventional** veto: a pipeline evaluates the
     consistency test for *the template that produced the trigger*, so the
     best-fit two-image template ``h_L = h + c(a) h(t + tau)`` is built and
     ``power_chisq`` is run on the original data against it.  Its equal-power
     bins come from ``|h_L|^2``, which carries the interference pattern
     ``|1 + c e^{-2 pi i f tau}|^2``, so the binning moves with ``(t_d, a)``
     rather than being fixed by ``|h|^2`` across the whole map.
   * ``chi2_residual`` -- the earlier construction, kept as a cross-check: the
     best-fit trailing image is subtracted from the data and the veto is
     re-evaluated on the residual with the *unlensed* template.  Their
     difference isolates how much of the repair is bin re-placement rather
     than genuine residual reduction.

   Neither can remove the geometric-optics/wave-optics mismatch of the
   injection, which is the irreducible floor of both.

Noise-free, at a stated reference optimal SNR. That makes the reported quantity
the deterministic, reproducible *excess*: with Gaussian noise added,
``E[chi2_r] = 1 + excess``, which ``--check-noise`` verifies on a few cells
rather than assuming it. Reporting the excess avoids folding a single noise
realization's scatter into every pixel of a map.

    python scripts/simulate/appendix_chisq_consistency.py \\
        --m-tot 50 --n-proc 16
"""

from __future__ import annotations

import os

# Single-threaded BLAS, set BEFORE numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from lensing import amplification as amp  # noqa: E402
from lensing import conventions as cv  # noqa: E402
from lensing import grids as gr  # noqa: E402
from lensing import priors as pr  # noqa: E402
from lensing import recombine as rc  # noqa: E402
from lensing import simulation as sim  # noqa: E402
from lensing import waveforms as wf  # noqa: E402

_W = {}


def _init(cfg_kw, t0, oversample, grid_kw, nbins, rho_ref, backend="auto"):
    from pycbc.types import FrequencySeries

    cfg = wf.AnalysisConfig(**cfg_kw)
    psd = wf.make_psd(cfg)
    h = wf.normalize(wf.make_waveform(cfg), psd, cfg.delta_f)
    dt = cfg.delta_t / oversample
    grid = gr.snap_grid_to_samples(gr.build_grid(**grid_kw), dt)
    pr.assign_weights(grid, "P0")
    C_vals = wf.autocorr_at_lags(h, psd, cfg, cv.DEFAULT.lag(grid.t_d))
    safe = np.where(np.isfinite(psd) & (psd > 0), psd, 1e50)
    _W.update(
        cfg=cfg, psd=psd, h=h, freqs=cfg.frequencies(),
        nz=np.nonzero(np.abs(h) > 0)[0], t0=t0, oversample=oversample,
        grid=grid, C_vals=C_vals, dt=dt, nbins=nbins, rho_ref=rho_ref,
        dof=2 * nbins - 2, backend=backend,
        psd_fs=FrequencySeries(safe, delta_f=cfg.delta_f),
        shifts=np.rint(cv.DEFAULT.lag(grid.t_d) / dt).astype(int),
        u=cv.DEFAULT.filter_coeff(1.0),
    )


def _F_inj(f_nz, M_Lz, y):
    """Wave-optics ``F`` for the injection, project convention (Phase 3 path)."""
    if _W["backend"] == "auto":
        F, _info = amp.F_ML_auto(f_nz, M_Lz, y)
        return F
    return amp.F_ML(f_nz, M_Lz, y, backend=_W["backend"])


def _trigger_chisq(data, template):
    """Reduced chi-squared at the trigger sample.

    The chi-squared series lives on the *base* sample grid, so the trigger is
    taken from a base-rate SNR series too. Mapping an oversampled index onto
    the base grid by rounding leaves a sub-sample offset which shows up as a
    spurious chi-squared excess (it inflated the null control from 3e-4 to
    3e-2), so it is not done.
    """
    cfg, psd, h = _W["cfg"], _W["psd"], _W["h"]
    z, _dt = wf.snr_series(data, template, psd, cfg, oversample=1)
    k = int(np.argmax(np.abs(z)))
    return float(_chisq_reduced(data, template)[k]), float(np.abs(z[k])), k


def _chisq_reduced(data, template):
    """PyCBC power chi-squared per degree of freedom, as a time series."""
    from pycbc.types import FrequencySeries
    from pycbc.vetoes import power_chisq

    cfg = _W["cfg"]
    cs = power_chisq(
        FrequencySeries(np.asarray(template, dtype=complex), delta_f=cfg.delta_f),
        FrequencySeries(np.asarray(data, dtype=complex), delta_f=cfg.delta_f),
        _W["nbins"], _W["psd_fs"],
        low_frequency_cutoff=cfg.f_lower, high_frequency_cutoff=cfg.f_final,
    )
    return np.array(cs.data) / _W["dof"]


def _lensed_template(a_fit, t_d_fit):
    """The best-fit two-image template, ``t = 0`` referenced and normalized.

    This is the only place in the project that builds a lensed template, and
    it is a *follow-up* of one trigger, not a search: the recombination method
    eliminates the 1944 per-lens-point filters, and evaluating one template's
    veto after the fact does not reintroduce them.
    """
    cfg, psd, h, freqs = _W["cfg"], _W["psd"], _W["h"], _W["freqs"]
    c = cv.DEFAULT.image_coeff(a_fit)
    hl = h + c * wf.fd_delay(h, freqs, cv.DEFAULT.lag(t_d_fit))
    return wf.normalize(hl, psd, cfg.delta_f)


def _best_fit(z1, dt, t_d_span):
    """Best (t_d, a, peak) on the FROZEN grid, plus the ML complex amplitude.

    The maximum-likelihood amplitude of the two-image model is
    ``A = (d | h_L) / (h_L | h_L)``, and ``(d | h_L)`` is exactly the
    recombination numerator, so nothing new is filtered.
    """
    cfg, grid, shifts = _W["cfg"], _W["grid"], _W["shifts"]
    C_vals, u, t0 = _W["C_vals"], _W["u"], _W["t0"]
    n = z1.size

    t_d_max = float(grid.t_d[-1])
    lo = max(0, int(np.floor((t0 - 2.0 * t_d_max) / dt)))
    hi = min(n, int(np.ceil((t0 + t_d_span + 2.0 * t_d_max) / dt)) + 1)
    win = np.arange(lo, hi)
    z_w = z1[win]
    p_self = np.abs(z_w) ** 2

    best, out = -np.inf, None
    for i, n_sh in enumerate(shifts):
        active = np.nonzero(grid.mask[i])[0]
        if active.size == 0:
            continue
        z_sh = z1[(win + n_sh) % n]
        cross = np.real(np.conjugate(u) * z_w * np.conjugate(z_sh))
        a = grid.a[active]
        num = (p_self[:, None] + (a * a)[None, :] * (np.abs(z_sh) ** 2)[:, None]
               + 2.0 * a[None, :] * cross[:, None])
        N = rc.lensed_norm_sq(C_vals[i], a)
        q = num / N[None, :]
        k = int(np.argmax(q))
        if float(q.flat[k]) > best:
            best = float(q.flat[k])
            it, ja = divmod(k, active.size)
            j = int(active[ja])
            out = (i, j, int(win[it]), n_sh, float(grid.a[j]), float(N[ja]))
    if out is None:
        return None
    i, j, k_time, n_sh, a_fit, N = out
    numerator = z1[k_time] + cv.DEFAULT.filter_coeff(a_fit) * z1[(k_time + n_sh) % n]
    return {
        "snr": float(np.sqrt(best)),
        "t_d_fit": float(grid.t_d[i]),
        "a_fit": a_fit,
        "t_peak": float(k_time * dt),
        "amplitude": complex(numerator / N),
        "grid_i": i, "grid_j": j,
    }


def _cell(task):
    idx, M_Lz, y = task
    cfg, nz, freqs, t0, h = _W["cfg"], _W["nz"], _W["freqs"], _W["t0"], _W["h"]
    psd, rho = _W["psd"], _W["rho_ref"]

    t_d_true = float(amp.time_delay(M_Lz, y))
    t_d_span = t_d_true
    if t0 + t_d_span > 0.9 * cfg.seg_dur:
        return idx, {"skipped": "delay too long", "t_d_s": t_d_true}

    # exact wave-optics lensed injection at fixed optimal SNR, noise-free.
    # NOTE: F_ML / F_ML_auto already return the PROJECT convention (they apply
    # to_project_convention internally), so it must NOT be applied again here.
    # This is the Phase 3 injection pattern verbatim.
    F = _F_inj(freqs[nz], M_Lz, y)
    data = np.zeros(freqs.size, dtype=complex)
    data[nz] = h[nz] * F
    data = wf.fd_delay(data, freqs, t0)
    data = data / wf.sigma(data, psd, cfg.delta_f) * rho

    # (1) unlensed template, chi-squared at its own trigger time
    z1, dt = wf.snr_series(data, h, psd, cfg, oversample=_W["oversample"])
    chi_unl, snr_unl, _k = _trigger_chisq(data, h)

    res = {
        "t_d_s": t_d_true,
        "a_true": float(amp.a_of_y(y)),
        "snr_unlensed": snr_unl,
        "chi2_unlensed": chi_unl,
    }

    fit = _best_fit(z1, dt, t_d_span)
    if fit is None:
        return idx, dict(res, skipped="no active grid point")
    c = cv.DEFAULT.image_coeff(fit["a_fit"])

    # (2) PRIMARY: the conventional veto -- the original data against the
    #     best-fit two-image template, which is the template that produced
    #     the trigger.  Same treatment as arm (1): the veto is evaluated at
    #     the base-grid trigger of that template's own SNR series.
    h_L = _lensed_template(fit["a_fit"], fit["t_d_fit"])
    chi_len, snr_len, _kl = _trigger_chisq(data, h_L)

    # (3) CROSS-CHECK: subtract the best-fit trailing image, re-test the
    #     residual with the unlensed template.
    second = (fit["amplitude"] * c
              * wf.fd_delay(h, freqs, fit["t_peak"] + cv.DEFAULT.lag(fit["t_d_fit"])))
    resid = data - second
    chi_res, snr_res, _kr = _trigger_chisq(resid, h)
    res.update(
        snr_bankfree=fit["snr"],
        t_d_fit_s=fit["t_d_fit"],
        a_fit=fit["a_fit"],
        snr_lensed=snr_len,
        chi2_lensed=chi_len,
        snr_residual=snr_res,
        chi2_residual=chi_res,
    )
    return idx, res


def _null_control():
    """An *unlensed* injection must give essentially zero chi-squared excess."""
    cfg, nz, freqs, t0, h = _W["cfg"], _W["nz"], _W["freqs"], _W["t0"], _W["h"]
    psd, rho = _W["psd"], _W["rho_ref"]
    data = np.zeros(freqs.size, dtype=complex)
    data[nz] = h[nz]
    data = wf.fd_delay(data, freqs, t0)
    data = data / wf.sigma(data, psd, cfg.delta_f) * rho
    return _trigger_chisq(data, h)[0]


def _null_control_lensed(a=0.5, t_d=0.05):
    """A *geometric-optics* two-image injection, vetoed by its own template.

    The wave-optics null control above isolates the veto machinery for a
    single chirp.  This one isolates it for the interference-fringed binning
    of a two-image template: injection and template are the same GO pair, so
    any excess is bin placement, not physics, and must be ~0.
    """
    cfg, freqs, t0, h = _W["cfg"], _W["freqs"], _W["t0"], _W["h"]
    psd, rho = _W["psd"], _W["rho_ref"]
    t_d = float(np.rint(t_d / _W["dt"]) * _W["dt"])
    c = cv.DEFAULT.image_coeff(a)
    sig = (wf.fd_delay(h, freqs, t0)
           + c * wf.fd_delay(h, freqs, t0 + cv.DEFAULT.lag(t_d)))
    sig = sig / wf.sigma(sig, psd, cfg.delta_f) * rho
    excess, snr, _k = _trigger_chisq(sig, _lensed_template(a, t_d))
    return {"a": a, "t_d_s": t_d, "excess": float(excess),
            "snr_recovered": float(snr), "rho_ref": rho}


def _noise_check(M_Lz, y, n_real, seed):
    """Verify ``E[chi2_r] = 1 + excess`` rather than asserting it."""
    cfg, nz, freqs, t0, h = _W["cfg"], _W["nz"], _W["freqs"], _W["t0"], _W["h"]
    psd, rho = _W["psd"], _W["rho_ref"]
    F = _F_inj(freqs[nz], M_Lz, y)
    sig = np.zeros(freqs.size, dtype=complex)
    sig[nz] = h[nz] * F
    sig = wf.fd_delay(sig, freqs, t0)
    sig = sig / wf.sigma(sig, psd, cfg.delta_f) * rho

    noiseless = _trigger_chisq(sig, h)[0]
    vals = []
    for r in range(n_real):
        d = sig + sim.noise_frequency_series(
            cfg, psd, np.random.default_rng([seed, r])
        )
        vals.append(_trigger_chisq(d, h)[0])
    return {
        "M_Lz": M_Lz, "y": y, "n_realizations": n_real,
        "noise_free_excess": noiseless,
        "mean_with_noise": float(np.mean(vals)),
        "std_with_noise": float(np.std(vals)),
        "predicted_mean": 1.0 + noiseless,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m-tot", type=float, default=50.0)
    ap.add_argument("--n-m", type=int, default=28)
    ap.add_argument("--n-y", type=int, default=20)
    ap.add_argument("--log-m-min", type=float, default=1.1)
    ap.add_argument("--log-m-max", type=float, default=5.0)
    ap.add_argument("--y-min", type=float, default=0.01)
    ap.add_argument("--y-max", type=float, default=1.5)
    ap.add_argument("--rho-ref", type=float, default=20.0)
    ap.add_argument("--nbins", type=int, default=16)
    ap.add_argument("--backend", default="auto", choices=["auto", "mpmath"])
    ap.add_argument("--oversample", type=int, default=4)
    ap.add_argument("--grid-n-td", type=int, default=1000)
    ap.add_argument("--grid-n-y", type=int, default=8)
    ap.add_argument("--seg-dur", type=float, default=0.0)
    ap.add_argument("--n-proc", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--check-noise", type=int, default=64,
                    help="noise realizations for the E[chi2]=1+excess check")
    ap.add_argument("--outdir", default="regenerated/chisq")
    args = ap.parse_args()

    base = wf.AnalysisConfig(mass1=args.m_tot / 2, mass2=args.m_tot / 2)
    seg = args.seg_dur or sim.suggested_seg_dur(base, t_d_max=1.0)
    cfg_kw = dict(
        seg_dur=seg, sample_rate=4096.0, f_lower=20.0, f_final=1024.0,
        psd_name="aLIGOZeroDetHighPower", approximant="IMRPhenomD",
        mass1=args.m_tot / 2, mass2=args.m_tot / 2,
    )
    cfg = wf.AnalysisConfig(**cfg_kw)
    seg_info = sim.validate_segment(cfg, t_d_max=1.0)
    t0 = max(min(2.0, 0.1 * seg), seg_info["chirp_time_s"] * 1.05)
    # snap the injection time to the BASE sample grid: the chi-squared series
    # lives there, and a sub-sample offset in the injection shows up as a
    # spurious excess (it put the null control at 3e-2 instead of ~1e-4).
    t0 = round(t0 * cfg.sample_rate) / cfg.sample_rate
    grid_kw = dict(
        n_td=args.grid_n_td, n_y=args.grid_n_y, t_d_min=1e-3, t_d_max=0.5,
        y_min=0.01, y_max=2.0, M_Lz_min=1e2, M_Lz_max=1e5,
    )
    probe = gr.snap_grid_to_samples(
        gr.build_grid(**grid_kw), cfg.delta_t / args.oversample
    )

    print("chi-squared consistency  M_tot=%g  rho_ref=%g  n_bins=%d (dof=%d)"
          % (args.m_tot, args.rho_ref, args.nbins, 2 * args.nbins - 2))
    print("  seg=%.0f s  chirp=%.2f s  t0=%.2f s  frozen grid %d x %d (%d active)"
          % (seg, seg_info["chirp_time_s"], t0, probe.n_td, probe.n_y,
             probe.n_points))

    _init(cfg_kw, t0, args.oversample, grid_kw, args.nbins, args.rho_ref,
          args.backend)
    null = _null_control()
    null_lensed = _null_control_lensed()
    print("  null control (unlensed injection): chi2_r excess = %.2e" % null)
    print("  null control (GO two-image injection, own template): "
          "chi2_r excess = %.2e, snr %.3f"
          % (null_lensed["excess"], null_lensed["snr_recovered"]))
    noise_checks = []
    if args.check_noise:
        for (M_Lz, y) in ((1e3, 0.3), (1e4, 0.1), (3e4, 0.3)):
            nc = _noise_check(M_Lz, y, args.check_noise, args.seed)
            noise_checks.append(nc)
            print("  noise check M_Lz=%.0e y=%.2f: excess %.3f -> predicted "
                  "mean %.3f, measured %.3f +- %.3f"
                  % (M_Lz, y, nc["noise_free_excess"], nc["predicted_mean"],
                     nc["mean_with_noise"],
                     nc["std_with_noise"] / np.sqrt(nc["n_realizations"])))

    M_grid = np.logspace(args.log_m_min, args.log_m_max, args.n_m)
    y_grid = np.geomspace(args.y_min, args.y_max, args.n_y)
    tasks = [((i, j), float(M_grid[i]), float(y_grid[j]))
             for i in range(args.n_m) for j in range(args.n_y)]

    shape = (args.n_m, args.n_y)
    keys = ("chi2_unlensed", "chi2_lensed", "chi2_residual", "snr_unlensed",
            "snr_bankfree", "snr_lensed", "snr_residual", "a_true", "a_fit",
            "t_d_s", "t_d_fit_s")
    arr = {k: np.full(shape, np.nan) for k in keys}
    n_skip = {}

    from multiprocessing import Pool

    t_start = time.time()
    with Pool(processes=args.n_proc, initializer=_init,
              initargs=(cfg_kw, t0, args.oversample, grid_kw, args.nbins,
                        args.rho_ref, args.backend)) as pool:
        for n, (idx, d) in enumerate(
            pool.imap_unordered(_cell, tasks, chunksize=1), start=1
        ):
            for k in keys:
                if k in d:
                    arr[k][idx] = d[k]
            if "skipped" in d:
                n_skip[d["skipped"]] = n_skip.get(d["skipped"], 0) + 1
            if n % 100 == 0 or n == len(tasks):
                el = time.time() - t_start
                print("  %5d/%d  %.0f s elapsed, %.0f s projected"
                      % (n, len(tasks), el, el * len(tasks) / n), flush=True)

    t_d = arr["t_d_s"]
    in_domain = (t_d >= 1e-3) & (t_d <= 0.5) & np.isfinite(arr["chi2_lensed"])

    def stats(mask, a):
        if not mask.any():
            return None
        v = a[mask]
        v = v[np.isfinite(v)]
        return {
            "n": int(v.size), "min": float(np.min(v)),
            "median": float(np.median(v)), "max": float(np.max(v)),
            "frac_gt_1": float(np.mean(v > 1.0)),
            "frac_gt_2": float(np.mean(v > 2.0)),
        }

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        commit = "unknown"

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "seed": args.seed,
        "config": cfg.as_dict(),
        "m_tot": args.m_tot,
        "rho_ref": args.rho_ref,
        "n_bins": args.nbins,
        "backend": args.backend,
        "dof": 2 * args.nbins - 2,
        "t0_s": t0,
        "segment": seg_info,
        "oversample": args.oversample,
        "search_grid": probe.as_dict(),
        "veto": "pycbc.vetoes.power_chisq, evaluated at the trigger time of "
                "the template in use, divided by dof = 2 n_bins - 2",
        "injection": "exact wave-optics point-mass F_ML, noise-free, scaled to "
                     "rho_ref; reported values are the deterministic excess, "
                     "so E[chi2_r] = 1 + excess once noise is added",
        "null_control_excess": null,
        "null_control_lensed": null_lensed,
        "noise_checks": noise_checks,
        "primary_veto": "conventional: power_chisq of the ORIGINAL data "
                        "against the best-fit two-image template "
                        "h_L = h + c(a) h(t+tau), the template that produced "
                        "the trigger; equal-power bins come from |h_L|^2 "
                        ".  chi2_residual is the earlier construction, "
                        "retained as a cross-check.",
        "subtraction": "A c h(t_peak + t_d) with (t_d, a) the two-image best "
                       "fit on the frozen grid and A = (d|h_L)/(h_L|h_L)",
        "wall_time_s": time.time() - t_start,
        "n_skipped": n_skip,
        "chi2_unlensed_stats": stats(in_domain, arr["chi2_unlensed"]),
        "chi2_lensed_stats": stats(in_domain, arr["chi2_lensed"]),
        "chi2_residual_stats": stats(in_domain, arr["chi2_residual"]),
        "chirp_time_s": seg_info["chirp_time_s"],
    }

    os.makedirs(args.outdir, exist_ok=True)
    tag = "chisq_M%g" % args.m_tot
    npz = os.path.join(args.outdir, f"{tag}.npz")
    np.savez_compressed(
        npz, M=M_grid, y=y_grid,
        grid_t_d=probe.t_d, grid_y=probe.y,
        meta=json.dumps(meta), **arr,
    )
    with open(os.path.join(args.outdir, f"{tag}.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print("\nskipped:", n_skip)
    for k in ("chi2_unlensed_stats", "chi2_lensed_stats",
              "chi2_residual_stats"):
        st = meta[k]
        if st:
            print("%-22s n=%4d median %.3f  max %.2f  frac>1 %.3f  frac>2 %.3f"
                  % (k, st["n"], st["median"], st["max"], st["frac_gt_1"],
                     st["frac_gt_2"]))
    print("wrote %s" % npz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
