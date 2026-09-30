#!/usr/bin/env python
"""Chi-squared on *background* triggers, so the volume comparison is
apples-to-apples at fixed false-alarm probability.

The volume ratio in ``scripts/plot_appendix_chisq.py`` is quoted at a fixed
threshold on the re-weighted SNR. That is not the same as a fixed false-alarm
probability: the two searches have different backgrounds, and the two-image
arm additionally fits two lens parameters to the noise before subtracting, so
its background could be re-weighted differently from the standard arm's. This
script measures both backgrounds under exactly the same re-weighting.

Per pure-noise segment it computes

    rho_UL,  chi2_UL     the standard search: unlensed template, its own veto
    rho_BF,  chi2_len    the two-image search: maximum over the frozen lens
                         grid, then the CONVENTIONAL veto -- the noise against
                         the best-fit two-image template that produced the
                         trigger.  This is the primary arm.
             chi2_res    the earlier construction, kept as a cross-check: the
                         veto on the residual left after the best-fit trailing
                         image is subtracted

and re-weights each with PyCBC's ``newsnr``. Thresholds at fixed FAP then
follow from the two background distributions, and the injection side
(``results/frozen/appendix_chisq``) supplies the detection efficiency.

Why this is affordable
----------------------
``newsnr`` never *increases* a statistic, so a segment whose raw ``rho`` is
below a threshold can never exceed it after re-weighting. Chi-squared is
therefore only needed for segments above a floor safely below any plausible
threshold, and the floor is validated after the fact against both measured
thresholds. Everything below it is recorded but never vetoed.

Statistics convention follows Phase 4: ``S = rho^2 / 2``, so a floor
``S = 12.5`` is ``rho = 5``.

    python scripts/simulate/appendix_chisq_background.py --n-seg 300000 --n-proc 16
"""

from __future__ import annotations

import os

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

from lensing import background as bg  # noqa: E402
from lensing import conventions as cv  # noqa: E402
from lensing import grids as gr  # noqa: E402
from lensing import priors as pr  # noqa: E402
from lensing import recombine as rc  # noqa: E402
from lensing import reweighted as rw  # noqa: E402
from lensing import simulation as sim  # noqa: E402
from lensing import waveforms as wf  # noqa: E402

_W = {}


def _init(cfg_kw, oversample, grid_kw, nbins, rho_floor, chi_floor, seed):
    from pycbc.types import FrequencySeries

    cfg = wf.AnalysisConfig(**cfg_kw)
    psd = wf.make_psd(cfg)
    h = wf.normalize(wf.make_waveform(cfg), psd, cfg.delta_f)
    grid = gr.snap_grid_to_samples(gr.build_grid(**grid_kw),
                                   cfg.delta_t / oversample)
    pr.assign_weights(grid, "P0")
    setup = bg.SearchSetup.build(cfg, psd, h, grid, oversample=oversample)
    safe = np.where(np.isfinite(psd) & (psd > 0), psd, 1e50)
    _W.update(
        cfg=cfg, psd=psd, h=h, grid=grid, setup=setup, oversample=oversample,
        rho_floor=rho_floor, chi_floor=chi_floor, seed=seed, nbins=nbins,
        dof=2 * nbins - 2, psd_fs=FrequencySeries(safe, delta_f=cfg.delta_f),
        freqs=cfg.frequencies(),
    )


def _chisq_at(data, template, k):
    from pycbc.types import FrequencySeries
    from pycbc.vetoes import power_chisq

    cfg = _W["cfg"]
    cs = power_chisq(
        FrequencySeries(np.asarray(template, dtype=complex), delta_f=cfg.delta_f),
        FrequencySeries(np.asarray(data, dtype=complex), delta_f=cfg.delta_f),
        _W["nbins"], _W["psd_fs"],
        low_frequency_cutoff=cfg.f_lower, high_frequency_cutoff=cfg.f_final,
    )
    return float(np.array(cs.data)[k] / _W["dof"])


def _bankfree_argmax(z1):
    """``max |z_L|`` over the frozen grid, with the location.

    Uses the same lossless per-delay candidate prefilter as Phase 4, so the
    maximum is exact; only the argmax bookkeeping is added.
    """
    setup, grid = _W["setup"], _W["grid"]
    n = z1.size
    loud = np.nonzero(np.abs(z1) >= setup.image_floor(_W["rho_floor"]))[0]
    if loud.size == 0:
        return None
    u = setup.conv.filter_coeff(1.0)
    best, out = -np.inf, None
    for i, n_sh in enumerate(setup.shifts):
        active = np.nonzero(grid.mask[i])[0]
        if active.size == 0:
            continue
        cand = bg.candidate_indices(loud, n_sh, n)
        z_w = z1[cand]
        z_sh = z1[(cand + n_sh) % n]
        a = grid.a[active]
        num = (np.abs(z_w) ** 2)[:, None] + (a * a)[None, :] * (
            np.abs(z_sh) ** 2)[:, None] + 2.0 * a[None, :] * np.real(
            np.conjugate(u) * z_w * np.conjugate(z_sh))[:, None]
        q = num / setup.norm[i, active][None, :]
        k = int(np.argmax(q))
        if float(q.flat[k]) > best:
            best = float(q.flat[k])
            it, ja = divmod(k, active.size)
            out = (i, int(active[ja]), int(cand[it]), int(n_sh))
    if out is None:
        return None
    i, j, k_time, n_sh = out
    a = float(grid.a[j])
    N = float(setup.norm[i, j])
    numerator = z1[k_time] + setup.conv.filter_coeff(a) * z1[(k_time + n_sh) % n]
    return {
        "rho": float(np.sqrt(best)), "t_d": float(grid.t_d[i]), "a": a,
        "k_time": k_time, "n_sh": n_sh, "amplitude": complex(numerator / N),
    }


def _one_segment(noise):
    """``(rho_UL, chi2_UL, rho_BF, chi2_len, chi2_res)`` for one segment."""
    cfg, psd, h, freqs = _W["cfg"], _W["psd"], _W["h"], _W["freqs"]
    ov = _W["oversample"]

    # Both arms use the OVERSAMPLED series for the statistic, matching Phase 4
    # and the rest of the paper, and the nearest base sample for the veto,
    # because the chi-squared series lives on the base grid. Identical
    # treatment in both arms is what the comparison requires.
    z_os, _ = wf.snr_series(noise, h, psd, cfg, oversample=ov)
    k_os = int(np.argmax(np.abs(z_os)))
    rho_ul = float(np.abs(z_os[k_os]))
    k_ul = int(round(k_os / ov)) % (z_os.size // ov)

    fit = _bankfree_argmax(z_os)
    rho_bf = fit["rho"] if fit else rho_ul

    # chi-squared only where it can matter: newsnr never raises a statistic
    if max(rho_ul, rho_bf) < _W["chi_floor"]:
        return rho_ul, np.nan, rho_bf, np.nan, np.nan, 0

    chi_ul = _chisq_at(noise, h, k_ul)
    if fit is None:
        return rho_ul, chi_ul, rho_bf, chi_ul, chi_ul, 1

    c = cv.DEFAULT.image_coeff(fit["a"])
    t_pk = fit["k_time"] * (cfg.delta_t / ov)

    # PRIMARY: the conventional veto, against the template that fired
    h_L = wf.normalize(
        h + c * wf.fd_delay(h, freqs, cv.DEFAULT.lag(fit["t_d"])),
        psd, cfg.delta_f)
    zl, _ = wf.snr_series(noise, h_L, psd, cfg, oversample=ov)
    kl = int(round(int(np.argmax(np.abs(zl))) / ov)) % (zl.size // ov)
    chi_len = _chisq_at(noise, h_L, kl)

    # CROSS-CHECK: the residual construction
    second = (fit["amplitude"] * c
              * wf.fd_delay(h, freqs, t_pk + cv.DEFAULT.lag(fit["t_d"])))
    resid = noise - second
    zr, _ = wf.snr_series(resid, h, psd, cfg, oversample=ov)
    kr = int(round(int(np.argmax(np.abs(zr))) / ov)) % (zr.size // ov)
    chi_res = _chisq_at(resid, h, kr)
    return rho_ul, chi_ul, rho_bf, chi_len, chi_res, 1


def _chunk(task):
    chunk_id, n_seg = task
    cfg, psd = _W["cfg"], _W["psd"]
    rng = np.random.default_rng([_W["seed"], chunk_id])
    out = np.full((n_seg, 5), np.nan, dtype=np.float32)
    n_chi = 0
    for i in range(n_seg):
        noise = sim.noise_frequency_series(cfg, psd, rng)
        a, b, c, d, e, used = _one_segment(noise)
        out[i] = (a, b, c, d, e)
        n_chi += used
    return chunk_id, out, n_chi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seg", type=int, default=300000)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--nbins", type=int, default=16)
    ap.add_argument("--rho-floor", type=float, default=5.0)
    ap.add_argument("--chi-floor", type=float, default=5.0,
                    help="compute chi-squared only above this raw rho")
    ap.add_argument("--fap", type=float, default=1e-3)
    ap.add_argument("--oversample", type=int, default=4)
    ap.add_argument("--grid-n-td", type=int, default=1000)
    ap.add_argument("--grid-n-y", type=int, default=8)
    ap.add_argument("--seg-dur", type=float, default=4.0)
    ap.add_argument("--m-tot", type=float, default=50.0)
    ap.add_argument("--n-proc", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--outdir", default="regenerated/chisq")
    args = ap.parse_args()

    cfg_kw = dict(
        seg_dur=args.seg_dur, sample_rate=4096.0, f_lower=20.0, f_final=1024.0,
        psd_name="aLIGOZeroDetHighPower", approximant="IMRPhenomD",
        mass1=args.m_tot / 2, mass2=args.m_tot / 2,
    )
    grid_kw = dict(
        n_td=args.grid_n_td, n_y=args.grid_n_y, t_d_min=1e-3, t_d_max=0.5,
        y_min=0.01, y_max=2.0, M_Lz_min=1e2, M_Lz_max=1e5,
    )
    print("background chi-squared: %d segments, seg_dur=%g s, %d bins, "
          "chi floor rho=%g" % (args.n_seg, args.seg_dur, args.nbins,
                                args.chi_floor))

    chunks = []
    left, cid = args.n_seg, 0
    while left > 0:
        m = min(args.chunk, left)
        chunks.append((cid, m))
        left -= m
        cid += 1

    from multiprocessing import Pool

    res = np.full((args.n_seg, 5), np.nan, dtype=np.float32)
    offs = np.cumsum([0] + [c[1] for c in chunks])
    n_chi_total = 0
    t0 = time.time()
    with Pool(processes=args.n_proc, initializer=_init,
              initargs=(cfg_kw, args.oversample, grid_kw, args.nbins,
                        args.rho_floor, args.chi_floor, args.seed)) as pool:
        done = 0
        for cid, out, n_chi in pool.imap_unordered(_chunk, chunks):
            res[offs[cid]:offs[cid] + out.shape[0]] = out
            n_chi_total += n_chi
            done += out.shape[0]
            if done % 10000 < args.chunk or done == args.n_seg:
                el = time.time() - t0
                print("  %7d/%d  %.0f s elapsed, %.0f s projected"
                      % (done, args.n_seg, el, el * args.n_seg / done),
                      flush=True)

    rho_ul, chi_ul, rho_bf, chi_len, chi_res = (
        res[:, i].astype(float) for i in range(5))
    # below the chi floor the statistic is un-re-weighted, which is the
    # correct treatment: newsnr leaves chi2_r <= 1 alone and those triggers
    # are far below threshold in any case
    hat_ul = np.where(np.isfinite(chi_ul), rw.new_snr(rho_ul, chi_ul), rho_ul)
    hat_bf = np.where(np.isfinite(chi_len), rw.new_snr(rho_bf, chi_len), rho_bf)
    hat_bf_res = np.where(np.isfinite(chi_res),
                          rw.new_snr(rho_bf, chi_res), rho_bf)

    def thr(v, fap):
        k = max(1, int(round(fap * v.size)))
        return float(np.partition(v, -k)[-k])

    out = {
        "raw": {"UL": thr(rho_ul, args.fap), "BF": thr(rho_bf, args.fap)},
        "reweighted": {"UL": thr(hat_ul, args.fap), "BF": thr(hat_bf, args.fap),
                       "BF_residual_crosscheck": thr(hat_bf_res, args.fap)},
    }

    def q(v):
        v = v[np.isfinite(v)]
        if v.size == 0:
            return None
        return {"n": int(v.size), "median": float(np.median(v)),
                "p90": float(np.percentile(v, 90)),
                "p99": float(np.percentile(v, 99)),
                "max": float(np.max(v)),
                "frac_gt_1": float(np.mean(v > 1.0))}

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
        "n_segments": args.n_seg,
        "chunk": args.chunk,     # seeds the per-chunk RNG streams; part of the config
        "n_chisq_evaluated": int(n_chi_total),
        "chisq_fraction": float(n_chi_total / args.n_seg),
        "config": cfg_kw,
        "n_bins": args.nbins,
        "dof": 2 * args.nbins - 2,
        "rho_floor": args.rho_floor,
        "chi_floor": args.chi_floor,
        "fap": args.fap,
        "thresholds": out,
        "primary_veto": "conventional: the noise against the best-fit "
                        "two-image template that produced the trigger; "
                        "chi2_res is the residual construction, retained as "
                        "a cross-check",
        "chi2_unlensed_background": q(chi_ul),
        "chi2_lensed_background": q(chi_len),
        "chi2_residual_background": q(chi_res),
        "reweight_loss": {
            "UL": out["raw"]["UL"] - out["reweighted"]["UL"],
            "BF": out["raw"]["BF"] - out["reweighted"]["BF"],
        },
        "floor_is_safe": bool(
            out["reweighted"]["UL"] > args.chi_floor
            and out["reweighted"]["BF"] > args.chi_floor
        ),
        "wall_time_s": time.time() - t0,
    }

    os.makedirs(args.outdir, exist_ok=True)
    npz = os.path.join(args.outdir, "background_chisq.npz")
    np.savez_compressed(npz, rho_ul=rho_ul, chi2_ul=chi_ul, rho_bf=rho_bf,
                        chi2_len=chi_len, chi2_res=chi_res,
                        newsnr_ul=hat_ul, newsnr_bf=hat_bf,
                        newsnr_bf_res=hat_bf_res, meta=json.dumps(meta))
    with open(os.path.join(args.outdir, "background_chisq.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print("\nchi-squared evaluated on %d/%d segments (%.3f)"
          % (n_chi_total, args.n_seg, n_chi_total / args.n_seg))
    print("background chi2_r  unlensed:", meta["chi2_unlensed_background"])
    print("background chi2_r  lensed  :", meta["chi2_lensed_background"])
    print("background chi2_r  residual:", meta["chi2_residual_background"])
    print("thresholds at FAP=%g:" % args.fap)
    for k in ("raw", "reweighted"):
            print("   %-11s UL %.4f   BF %.4f%s"
              % (k, out[k]["UL"], out[k]["BF"],
                 ("   (BF residual cross-check %.4f)"
                  % out[k]["BF_residual_crosscheck"]) if k == "reweighted"
                 else ""))
    print("re-weighting cost at threshold: UL %.4f, BF %.4f"
          % (meta["reweight_loss"]["UL"], meta["reweight_loss"]["BF"]))
    print("chi floor safely below both thresholds:", meta["floor_is_safe"])
    print("wrote %s" % npz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
