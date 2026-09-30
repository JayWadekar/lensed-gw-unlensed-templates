#!/usr/bin/env python
"""Lens-model generality: recovery maps for the SIS and the cored isothermal.

How far does the two-image statistic go for other strongly lensed systems? The
**search is held exactly as frozen** -- the same production grid of 1000 delays
by 8 amplitude rows, derived from the point-mass mapping -- and signals from a
different lens model are injected into it. Only the injection changes.

Injections use the **exact wave-optics** amplification factor from GLoW
(Villarrubia-Rojo et al. 2024, arXiv:2409.04606), whose convention was verified
against this project's frozen mpmath point-lens implementation to 1.5e-6
relative (``tests/test_lens_models.py``).

Three statistics per cell
-------------------------
``recovery_unlensed``
    the ordinary single-template search.
``recovery_lensed`` (also stored per ``q`` as ``recovery_q1``)
    the deployed two-image family, relative image coefficient ``c = ia``.
``recovery_general``
    the **generalized** family: the same grid and the same recombination
    identity, maximized over the four relative Morse phases ``c = a i^q``,
    ``q = 0..3``. This matters here for the same reason it matters for the
    Chang-Refsdal lens: a cored isothermal's third image is a **maximum**,
    whose relative Morse phase to the minimum is ``180 deg`` (``q = 2``), not
    the ``90 deg`` the deployed family supplies. The expensive steps are
    computed once and shared; only the cheap array algebra repeats
.

This is a variant of ``scripts/simulate/phase3_wave_optics.py``. ``_recover`` is
unchanged in substance -- it touches only ``grid.mask``, ``grid.a``, the sample
shifts, the autocorrelation and ``rc.lensed_norm_sq``, none of which know
anything about a lens model -- with one necessary change: the search window is
sized by the **maximum** image delay rather than the second image's, because a
cored lens has three images.

``--seg-dur`` must be given explicitly: the frozen products use **16 s** at
``M_tot = 50`` and ``100``, and 32 s at ``11``, whereas
``simulation.suggested_seg_dur`` returns 4 s and 2 s for the first two. Omitting
it silently produces a different ``delta_f`` and ``t0`` and will not reproduce
the stored maps.

    python scripts/simulate/appendix_lens_models.py --lens-model sis \\
        --m-tot 50 --seg-dur 16 --n-proc 16
    python scripts/simulate/appendix_lens_models.py --lens-model cored \\
        --core-radius 0.15 --m-tot 50 --seg-dur 16 --n-proc 16
"""

from __future__ import annotations

import os

# Single-threaded BLAS, set BEFORE numpy is imported: this driver forks a pool
# after the parent has run a large matmul, and a fork inheriting locked
# OpenMP/BLAS state deadlocks the children.
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
from lensing import glow_backend as gb  # noqa: E402
from lensing import grids as gr  # noqa: E402
from lensing import lens_models as lm  # noqa: E402
from lensing import priors as pr  # noqa: E402
from lensing import recombine as rc  # noqa: E402
from lensing import simulation as sim  # noqa: E402
from lensing import waveforms as wf  # noqa: E402

_W = {}

#: the four relative Morse phases the generalized statistic maximizes over
QS = (0, 1, 2, 3)

#: the deployed one
Q_FROZEN = 1


def _make_model(args):
    if args.lens_model == "cored":
        return lm.CoredIsothermal(rc=args.core_radius)
    return lm.build(args.lens_model)


def _init(cfg_kw, t0, oversample, grid_kw, model_kw):
    cfg = wf.AnalysisConfig(**cfg_kw)
    psd = wf.make_psd(cfg)
    h = wf.normalize(wf.make_waveform(cfg), psd, cfg.delta_f)
    dt = cfg.delta_t / oversample
    # THE FROZEN SEARCH GRID -- point-mass derived, deliberately not re-tuned
    grid = gr.snap_grid_to_samples(gr.build_grid(**grid_kw), dt)
    pr.assign_weights(grid, "P0")
    # the lag is q-independent, so one autocorrelation serves all four families
    C_vals = wf.autocorr_at_lags(h, psd, cfg, cv.DEFAULT.lag(grid.t_d))
    model = (
        lm.CoredIsothermal(rc=model_kw["core_radius"])
        if model_kw["name"] == "cored"
        else lm.build(model_kw["name"])
    )
    convs = {q: cv.DEFAULT.with_rel_morse(q) for q in QS}
    _W.update(
        cfg=cfg, psd=psd, h=h, freqs=cfg.frequencies(),
        nz=np.nonzero(np.abs(h) > 0)[0], t0=t0, oversample=oversample,
        grid=grid, C_vals=C_vals, dt=dt, model=model,
        shifts=np.rint(cv.DEFAULT.lag(grid.t_d) / dt).astype(int),
        convs=convs,
        us={q: convs[q].filter_coeff(1.0) for q in QS},
        # N(t_d, a) per family, precomputed once: shape (n_td, n_a)
        norms={q: rc.lensed_norm_sq(C_vals[:, None], grid.a[None, :],
                                    conv=convs[q]) for q in QS},
    )


def _recover(data, t_d_span=0.0):
    """(unlensed, per-q two-image, per-q best grid indices).

    Identical to the Phase 3 routine except that ``t_d_span`` is the *maximum*
    injection delay -- so a three-image injection's later pairings are inside
    the window -- and that the inner loop evaluates all four relative Morse
    phases off the *same* shifted SNR series.
    """
    cfg, psd, h = _W["cfg"], _W["psd"], _W["h"]
    grid, shifts, t0 = _W["grid"], _W["shifts"], _W["t0"]
    us, norms = _W["us"], _W["norms"]

    sig = wf.sigma(data, psd, cfg.delta_f)
    z1, dt = wf.snr_series(data, h, psd, cfg, oversample=_W["oversample"])
    n = z1.size

    t_d_max = float(grid.t_d[-1])
    lo = max(0, int(np.floor((t0 - 2.0 * t_d_max) / dt)))
    hi = min(n, int(np.ceil((t0 + t_d_span + 2.0 * t_d_max) / dt)) + 1)
    win = np.arange(lo, hi)

    z_w = z1[win]
    p_self = np.abs(z_w) ** 2

    best = {q: -np.inf for q in QS}
    best_ij = {q: (-1, -1) for q in QS}
    for i, n_sh in enumerate(shifts):
        active = np.nonzero(grid.mask[i])[0]
        if active.size == 0:
            continue
        z_sh = z1[(win + n_sh) % n]
        p_shift = np.abs(z_sh) ** 2
        a = grid.a[active]
        aa = (a * a)[None, :]
        # shared across families; only the projection of the cross term differs
        zz = z_w * np.conjugate(z_sh)
        for q in QS:
            cross = np.real(np.conjugate(us[q]) * zz)
            num = (p_self[:, None] + aa * p_shift[:, None]
                   + 2.0 * a[None, :] * cross[:, None])
            qq = num / norms[q][i, active][None, :]
            k = int(np.argmax(qq))
            if float(qq.flat[k]) > best[q]:
                best[q] = float(qq.flat[k])
                best_ij[q] = (i, int(active[k % active.size]))
    return (
        float(np.abs(z1).max() / sig),
        {q: float(np.sqrt(best[q]) / sig) for q in QS},
        best_ij,
    )


def _cell(task):
    idx, M, y = task
    cfg, nz, freqs, t0 = _W["cfg"], _W["nz"], _W["freqs"], _W["t0"]
    model = _W["model"]

    ims = model.images(y)
    n_img = len(ims)
    if n_img == 0:
        return idx, np.nan, None, None, {"n_images": 0}

    t_d_span = model.max_time_delay(M, y)
    # the latest image must fit inside the segment: the transform is circular
    if t0 + t_d_span > 0.9 * cfg.seg_dur:
        return idx, np.nan, None, None, {
            "n_images": n_img, "skipped": "delay too long"
        }

    try:
        F = model.F_wave(freqs[nz], M, y)
    except Exception as exc:                       # GLoW failure at this cell
        return idx, np.nan, None, None, {
            "n_images": n_img, "glow_error": type(exc).__name__
        }

    data = np.zeros(freqs.size, dtype=complex)
    data[nz] = _W["h"][nz] * F
    data = wf.fd_delay(data, freqs, t0)
    r_ul, r_q, ij = _recover(data, t_d_span=t_d_span)

    ratios = model.amplitude_ratios(y)
    morse = "".join(str(im.morse) for im in ims)
    return idx, r_ul, r_q, ij, {
        "n_images": n_img,
        "t_d_s": float(model.time_delay(M, y)),
        "t_d_max_s": float(t_d_span),
        "a2": float(ratios[0]) if ratios.size else 0.0,
        "a3": float(ratios[1]) if ratios.size > 1 else 0.0,
        "morse_seq": morse,
        "has_maximum": int("2" in morse),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens-model", default="sis",
                    choices=["sis", "cored", "pointmass"])
    ap.add_argument("--core-radius", type=float, default=0.15)
    ap.add_argument("--m-tot", type=float, default=50.0)
    ap.add_argument("--n-m", type=int, default=32)
    ap.add_argument("--n-y", type=int, default=22)
    ap.add_argument("--log-m-min", type=float, default=1.1)
    ap.add_argument("--log-m-max", type=float, default=5.0)
    ap.add_argument("--y-min", type=float, default=0.01)
    ap.add_argument("--y-max", type=float, default=1.5)
    ap.add_argument("--oversample", type=int, default=4)
    ap.add_argument("--grid-n-td", type=int, default=1000)
    ap.add_argument("--grid-n-y", type=int, default=8)
    ap.add_argument("--seg-dur", type=float, default=0.0)
    ap.add_argument("--n-proc", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--outdir", default="regenerated/isothermal")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    model = _make_model(args)
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

    grid_kw = dict(
        n_td=args.grid_n_td, n_y=args.grid_n_y, t_d_min=1e-3, t_d_max=0.5,
        y_min=0.01, y_max=2.0, M_Lz_min=1e2, M_Lz_max=1e5,
    )
    probe = gr.snap_grid_to_samples(
        gr.build_grid(**grid_kw), cfg.delta_t / args.oversample
    )
    model_kw = {"name": args.lens_model, "core_radius": args.core_radius}

    label = args.lens_model + (
        f" rc={args.core_radius:g}" if args.lens_model == "cored" else ""
    )
    print(
        "lens=%s  M_tot=%g  seg=%.0f s  chirp=%.2f s  t0=%.2f s  "
        "FROZEN search grid %d td x %d y (%d active)"
        % (label, args.m_tot, seg, seg_info["chirp_time_s"], t0,
           probe.n_td, probe.n_y, probe.n_points)
    )
    print("GLoW version: %s" % gb.glow_version())

    M_grid = np.logspace(args.log_m_min, args.log_m_max, args.n_m)
    y_grid = np.geomspace(args.y_min, args.y_max, args.n_y)
    tasks = [
        ((i, j), float(M_grid[i]), float(y_grid[j]))
        for i in range(args.n_m) for j in range(args.n_y)
    ]

    shape = (args.n_m, args.n_y)
    r_ul = np.full(shape, np.nan)
    r_q = {q: np.full(shape, np.nan) for q in QS}
    has_max = np.zeros(shape, dtype=bool)
    n_img = np.zeros(shape, dtype=int)
    a2 = np.full(shape, np.nan)
    a3 = np.full(shape, np.nan)
    t_d = np.full(shape, np.nan)
    best_td = np.full(shape, -1, dtype=int)
    best_y = np.full(shape, -1, dtype=int)
    n_glow_err = 0

    from multiprocessing import Pool

    t_start = time.time()
    with Pool(processes=args.n_proc, initializer=_init,
              initargs=(cfg_kw, t0, args.oversample, grid_kw, model_kw)) as pool:
        for n, (idx, v_ul, vq, ij, d) in enumerate(
            pool.imap_unordered(_cell, tasks, chunksize=1), start=1
        ):
            r_ul[idx] = v_ul
            n_img[idx] = d.get("n_images", 0)
            a2[idx] = d.get("a2", np.nan)
            a3[idx] = d.get("a3", np.nan)
            t_d[idx] = d.get("t_d_s", np.nan)
            has_max[idx] = bool(d.get("has_maximum", 0))
            n_glow_err += int("glow_error" in d)
            if vq is not None:
                for q in QS:
                    r_q[q][idx] = vq[q]
                best_td[idx], best_y[idx] = ij[Q_FROZEN]
            if n % 100 == 0 or n == len(tasks):
                el = time.time() - t_start
                print("  %5d/%d  %.0f s elapsed, %.0f s projected"
                      % (n, len(tasks), el, el * len(tasks) / n), flush=True)

    r_l = r_q[Q_FROZEN]
    r_general = np.nanmax(np.stack([r_q[q] for q in QS]), axis=0)
    best_q = np.full(shape, -1, dtype=int)
    ok = np.isfinite(r_general)
    if ok.any():
        best_q[ok] = np.argmax(
            np.stack([np.where(np.isfinite(r_q[q]), r_q[q], -1) for q in QS]),
            axis=0)[ok]

    # ---- statistics over the searched delay domain -----------------------
    in_domain = (t_d >= 1e-3) & (t_d <= 0.5) & np.isfinite(r_l)
    multi = in_domain & (n_img >= 3)
    two = in_domain & (n_img == 2)

    def stats(mask, arr):
        if not mask.any():
            return None
        v = arr[mask]
        return {
            "n": int(mask.sum()), "min": float(np.min(v)),
            "median": float(np.median(v)),
            "frac_ge_0.97": float(np.mean(v >= 0.97)),
            "frac_ge_0.95": float(np.mean(v >= 0.95)),
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
        "lens_model": args.lens_model,
        "core_radius": args.core_radius if args.lens_model == "cored" else None,
        "lens_label": label,
        "m_tot": args.m_tot,
        "t0_s": t0,
        "segment": seg_info,
        "oversample": args.oversample,
        "search_grid": probe.as_dict(),
        "search_grid_note": "frozen point-mass-derived grid, deliberately not "
                            "re-tuned for this lens model",
        "glow_version": gb.glow_version(),
        "wave_optics_source": "GLoW (Villarrubia-Rojo et al. 2024)",
        "quantity": "recovery = max over time and the frozen lens grid of "
                    "|z|/sigma(d); injections use the exact wave-optics F",
        "wall_time_s": time.time() - t_start,
        "n_skipped_delay_too_long": int(np.sum(~np.isfinite(r_l))),
        "n_glow_errors": int(n_glow_err),
        "searched_domain_t_d_1ms_to_500ms": {
            "lensed": stats(in_domain, r_l), "unlensed": stats(in_domain, r_ul),
            "generalized": stats(in_domain, r_general),
            **{f"q{q}": stats(in_domain, r_q[q]) for q in QS},
        },
        "two_image_cells": {
            "lensed": stats(two, r_l), "unlensed": stats(two, r_ul),
            "generalized": stats(two, r_general),
        },
        "three_or_more_image_cells": {
            "lensed": stats(multi, r_l), "unlensed": stats(multi, r_ul),
            "generalized": stats(multi, r_general),
        },
        "cells_with_a_maximum": {
            "lensed": stats(in_domain & has_max, r_l),
            "unlensed": stats(in_domain & has_max, r_ul),
            "generalized": stats(in_domain & has_max, r_general),
        },
        "morse_generalization_gain": {
            "median": float(np.nanmedian((r_general - r_l)[in_domain]))
            if in_domain.any() else None,
            "max": float(np.nanmax((r_general - r_l)[in_domain]))
            if in_domain.any() else None,
            "frac_cells_improved_by_gt_0.01":
                float(np.mean((r_general - r_l)[in_domain] > 0.01))
                if in_domain.any() else None,
        },
        "best_q_census": {str(q): int(np.sum(best_q[in_domain] == q))
                          for q in QS},
        "image_counts": {str(k): int(np.sum(n_img == k)) for k in (0, 1, 2, 3, 4)},
    }

    os.makedirs(args.outdir, exist_ok=True)
    tag = args.tag or (
        f"{args.lens_model}_rc{args.core_radius:g}_M{args.m_tot:g}"
        if args.lens_model == "cored"
        else f"{args.lens_model}_M{args.m_tot:g}"
    )
    npz = os.path.join(args.outdir, f"lensmodel_{tag}.npz")
    np.savez_compressed(
        npz, M=M_grid, y=y_grid, recovery_unlensed=r_ul, recovery_lensed=r_l,
        recovery_general=r_general,
        **{f"recovery_q{q}": r_q[q] for q in QS},
        best_q=best_q, has_maximum=has_max,
        n_images=n_img, a2=a2, a3=a3, t_d=t_d,
        best_td_index=best_td, best_y_index=best_y,
        grid_t_d=probe.t_d, grid_y=probe.y, meta=json.dumps(meta),
    )
    with open(os.path.join(args.outdir, f"lensmodel_{tag}.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print("\nimage-count census:", meta["image_counts"])
    for key in ("searched_domain_t_d_1ms_to_500ms", "two_image_cells",
                "three_or_more_image_cells"):
        print("-- %s --" % key)
        for k in ("unlensed", "lensed", "generalized"):
            st = meta[key].get(k)
            if st:
                print("   %-12s n=%4d  min %.4f  median %.4f  frac>=0.97 %.3f"
                      % (k, st["n"], st["min"], st["median"],
                         st["frac_ge_0.97"]))
    print("Morse-generalization gain:", meta["morse_generalization_gain"])
    print("best-q census:", meta["best_q_census"])
    if meta["cells_with_a_maximum"]["lensed"]:
        print("-- cells containing a maximum --")
        for k in ("unlensed", "lensed", "generalized"):
            st = meta["cells_with_a_maximum"][k]
            print("   %-12s n=%4d  min %.4f  median %.4f  frac>=0.97 %.3f"
                  % (k, st["n"], st["min"], st["median"], st["frac_ge_0.97"]))
    if n_glow_err:
        print("GLoW errors: %d cells" % n_glow_err)
    print("wrote %s" % npz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
