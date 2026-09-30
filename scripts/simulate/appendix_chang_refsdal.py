#!/usr/bin/env python
"""Chang-Refsdal scan: a microlens sitting on a strongly lensed macro-image.

The physically realistic microlensing configuration is not an isolated lens: it
is a point mass embedded in the local convergence and shear of a macromodel.
This script scans that two-parameter plane -- external convergence ``kappa`` and
shear ``gamma`` at fixed shear orientation -- and asks how far the two-image
statistic goes, with the **search held exactly as frozen** (the same production
grid of 1000 delays by 8 amplitude rows, derived from the point-mass mapping).
Only the injection changes.

Three statistics per cell
-------------------------
``recovery_unlensed``
    the ordinary single-template search.
``recovery_frozen``
    the deployed two-image family, relative image coefficient ``c = i a``.
``recovery_general``
    the **generalized** family: the same grid and the same recombination
    identity, maximized over the four possible relative Morse phases
    ``c = a i^q``, ``q = 0, 1, 2, 3``. The expensive steps (waveform, SNR
    series, autocorrelation, shifted stack) are computed once and shared; only
    the cheap array algebra repeats, which is why the generalization is nearly
    free.

``recovery_extended``     (only with ``--extend-a``)
    the generalized family on an amplitude axis **mirrored about one**,
    ``a -> 1/a``, so that the trailing image may be the brighter of the two.
    This is the configuration in which the *fainter* image arrives first, and
    it is the one case the deployed family cannot reach by any choice of
    ``q``: ``a`` is bounded in ``(0, 1)`` because a point-mass lens always
    makes the leading image the brighter one, and a Chang-Refsdal lens does
    not. Nothing in the recombination cares -- ``N(theta)`` and the candidate
    bound hold for any ``|c|`` -- so the extension costs grid points and no
    new filtering.

    **This is a probe, not a change of search.** It exists so that one
    appendix figure can say which half of the remaining loss is removable.
    Every other product, and the deployed search, keeps ``0 < a < 1``.

Which optics, and where
-----------------------
The Fermat potential goes as
``phi -> [(1-kappa-gamma) x1^2 + (1-kappa+gamma) x2^2] / 2`` at large radius, so
its level sets close **iff** ``kappa + gamma < 1`` -- exactly the macro-minimum
side of the critical line. Injections are therefore **hybrid**:

``kappa + gamma < 1``
    exact **wave optics**. ``I(tau)`` is GLoW's ``It_MultiContour_C``; the
    transform to ``F(w)`` is :mod:`lensing.wave_contour`, because every GLoW
    regularization stage assumes an isolated lens
    (``I(tau -> inf) -> 2 pi``) whereas an external convergence and shear make
    the asymptotic reference the macro-image, ``2 pi sqrt(mu_macro)``.
``kappa + gamma > 1``
    exact **geometric optics** from the Chang-Refsdal quartic. The contours are
    open hyperbolae and the contour method has nothing to integrate; this is a
    property of the lens, not a tooling limitation.

Every cell records which of the two produced it, in ``optics_wave``. GLoW
remains the authority on the lens potential and cross-checks the image census.

    # Fig. 1(b, d)
    python scripts/simulate/appendix_chang_refsdal.py --m-lz 3000 --n-proc 16 \\
        --outdir regenerated/chang_refsdal
    # appendix figure (amplitude axis mirrored about one)
    python scripts/simulate/appendix_chang_refsdal.py --m-lz 3000 --extend-a \\
        --n-proc 16 --outdir regenerated/chang_refsdal_extended_a
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
from lensing import glow_backend as gb  # noqa: E402
from lensing import grids as gr  # noqa: E402
from lensing import lens_models as lm  # noqa: E402
from lensing import priors as pr  # noqa: E402
from lensing import recombine as rc  # noqa: E402
from lensing import simulation as sim  # noqa: E402
from lensing import wave_contour as wc  # noqa: E402
from lensing import waveforms as wf  # noqa: E402

_W = {}

#: the four relative Morse phases the generalized statistic maximizes over
QS = (0, 1, 2, 3)

#: the deployed one
Q_FROZEN = 1


def _init(cfg_kw, t0, oversample, grid_kw, model_kw, extend_a=False):
    cfg = wf.AnalysisConfig(**cfg_kw)
    psd = wf.make_psd(cfg)
    h = wf.normalize(wf.make_waveform(cfg), psd, cfg.delta_f)
    dt = cfg.delta_t / oversample
    # THE FROZEN SEARCH GRID -- point-mass derived, deliberately not re-tuned
    grid = gr.snap_grid_to_samples(gr.build_grid(**grid_kw), dt)
    pr.assign_weights(grid, "P0")
    # the lag is q-independent, so one autocorrelation serves all four families
    C_vals = wf.autocorr_at_lags(h, psd, cfg, cv.DEFAULT.lag(grid.t_d))
    convs = {q: cv.DEFAULT.with_rel_morse(q) for q in QS}
    # The amplitude axis actually swept.  Without --extend-a this *is* the
    # frozen axis and the extended bookkeeping below collapses onto the
    # ordinary one, so there is a single code path and the a<1 answers of an
    # extended run are bit-identical to those of a plain one.
    n_base = grid.a.size
    a_all = np.concatenate([grid.a, 1.0 / grid.a]) if extend_a else grid.a
    mask_all = (np.concatenate([grid.mask, grid.mask], axis=1) if extend_a
                else grid.mask)
    _W.update(
        cfg=cfg, psd=psd, h=h, freqs=cfg.frequencies(),
        nz=np.nonzero(np.abs(h) > 0)[0], t0=t0, oversample=oversample,
        grid=grid, C_vals=C_vals, dt=dt, model_kw=model_kw,
        shifts=np.rint(cv.DEFAULT.lag(grid.t_d) / dt).astype(int),
        convs=convs, optics_mode=model_kw.get("optics_mode", "hybrid"),
        us={q: convs[q].filter_coeff(1.0) for q in QS},
        extend_a=bool(extend_a), n_base=n_base, a_all=a_all, mask_all=mask_all,
        # N(t_d, a) per family, precomputed once: shape (n_td, n_a_all)
        norms={q: rc.lensed_norm_sq(C_vals[:, None], a_all[None, :],
                                    conv=convs[q]) for q in QS},
    )


def _recover(data, t_d_span=0.0):
    """(unlensed, frozen-family, per-q) recovery for one injection.

    Structurally the Phase 3 routine, with two changes: the window is sized by
    the **maximum** image delay (a Chang-Refsdal lens has up to four images),
    and the inner loop evaluates all four relative Morse phases off the *same*
    shifted SNR series.
    """
    cfg, psd, h = _W["cfg"], _W["psd"], _W["h"]
    grid, shifts, t0 = _W["grid"], _W["shifts"], _W["t0"]
    us, norms = _W["us"], _W["norms"]
    ext, n_base = _W["extend_a"], _W["n_base"]
    a_all, mask_all = _W["a_all"], _W["mask_all"]

    sig = wf.sigma(data, psd, cfg.delta_f)
    z1, dt = wf.snr_series(data, h, psd, cfg, oversample=_W["oversample"])
    n = z1.size

    t_d_max = float(grid.t_d[-1])
    lo = max(0, int(np.floor((t0 - 2.0 * t_d_max) / dt)))
    hi = min(n, int(np.ceil((t0 + t_d_span + 2.0 * t_d_max) / dt)) + 1)
    win = np.arange(lo, hi)

    z_w = z1[win]
    p_self = np.abs(z_w) ** 2

    best = {q: -np.inf for q in QS}            # the deployed axis, a < 1
    best_ij = {q: (-1, -1) for q in QS}
    best_ext = {q: -np.inf for q in QS}        # that axis plus its mirror
    ext_used = {q: False for q in QS}
    for i, n_sh in enumerate(shifts):
        active = np.nonzero(mask_all[i])[0]
        if active.size == 0:
            continue
        z_sh = z1[(win + n_sh) % n]
        p_shift = np.abs(z_sh) ** 2
        a = a_all[active]
        aa = (a * a)[None, :]
        # shared across families; only the projection of the cross term differs
        zz = z_w * np.conjugate(z_sh)
        base = np.nonzero(active < n_base)[0] if ext else None
        for q in QS:
            cross = np.real(np.conjugate(us[q]) * zz)
            num = (p_self[:, None] + aa * p_shift[:, None]
                   + 2.0 * a[None, :] * cross[:, None])
            qq = num / norms[q][i, active][None, :]
            k = int(np.argmax(qq))
            v = float(qq.flat[k])
            col = int(active[k % active.size])
            if v > best_ext[q]:
                best_ext[q] = v
                ext_used[q] = col >= n_base
            if not ext:
                if v > best[q]:
                    best[q] = v
                    best_ij[q] = (i, col)
            elif base.size:
                sub_q = qq[:, base]
                kb = int(np.argmax(sub_q))
                vb = float(sub_q.flat[kb])
                if vb > best[q]:
                    best[q] = vb
                    best_ij[q] = (i, int(active[base[kb % base.size]]))

    return (
        float(np.abs(z1).max() / sig),
        {q: float(np.sqrt(best[q]) / sig) for q in QS},
        best_ij,
        ({q: float(np.sqrt(best_ext[q]) / sig) for q in QS}, dict(ext_used))
        if ext else None,
    )


def _model(kappa, gamma):
    mk = _W["model_kw"]
    return lm.ChangRefsdal(kappa=kappa, gamma=gamma,
                           phi_gamma=mk["phi_gamma"], psi0=mk["psi0"])


def _cell(task):
    idx, kappa, gamma = task
    cfg, nz, freqs, t0 = _W["cfg"], _W["nz"], _W["freqs"], _W["t0"]
    mk = _W["model_kw"]
    M_Lz, y = mk["m_lz"], mk["y"]

    model = _model(kappa, gamma)
    info = {
        "macro_type": model.macro_type,
        "d_crit": float(model.distance_to_critical),
        "mu_macro": float(model.macro_magnification),
    }

    if abs(model.distance_to_critical) < mk["exclude"]:
        return idx, np.nan, None, None, dict(info, skipped="near critical line")

    ims = model.images(y)
    info["n_images"] = len(ims)
    if len(ims) < 2:
        return idx, np.nan, None, None, dict(info, skipped="fewer than 2 images")

    t_d_span = model.max_time_delay(M_Lz, y)
    if t0 + t_d_span > 0.9 * cfg.seg_dur:
        return idx, np.nan, None, None, dict(info, skipped="delay too long")

    try:
        if _W.get("optics_mode") == "go":
            F, optics = model.F_go(freqs[nz], M_Lz, y), "go"
        else:
            F, optics = model.F_hybrid(freqs[nz], M_Lz, y)
    except Exception as exc:                       # a cell we cannot inject
        return idx, np.nan, None, None, dict(
            info, skipped="wave-optics failure: %s" % type(exc).__name__)
    if not np.all(np.isfinite(F)):
        return idx, np.nan, None, None, dict(info, skipped="non-finite F")
    info["optics"] = optics
    data = np.zeros(freqs.size, dtype=complex)
    data[nz] = _W["h"][nz] * F
    data = wf.fd_delay(data, freqs, t0)

    r_ul, r_q, ij, r_ext = _recover(data, t_d_span=t_d_span)
    if r_ext is not None:
        info["ext_q"], info["ext_used"] = r_ext[0], r_ext[1]

    ratios = model.amplitude_ratios(y)
    delays = model.delays_dimensionless(y)
    info.update(
        t_d_s=float(model.time_delay(M_Lz, y)),
        t_d_max_s=float(t_d_span),
        a2=float(ratios[0]),
        a_max=float(np.max(ratios)),
        dtau2=float(delays[0]),
        required_q=int(model.required_rel_morse(y)),
        morse_seq="".join(str(im.morse) for im in ims),
    )
    return idx, r_ul, r_q, ij, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m-lz", type=float, default=1e3,
                    help="redshifted microlens mass, Msun (sets the w band)")
    ap.add_argument("--y", type=float, default=0.3,
                    help="source position, microlens Einstein radii")
    ap.add_argument("--phi-gamma-deg", type=float, default=45.0,
                    help="shear orientation w.r.t. the source axis; 0 is "
                         "degenerate (mirror images, zero delay)")
    ap.add_argument("--psi0", type=float, default=1.0)
    ap.add_argument("--kappa-max", type=float, default=0.8)
    ap.add_argument("--gamma-max", type=float, default=0.8)
    ap.add_argument("--n-kappa", type=int, default=33)
    ap.add_argument("--n-gamma", type=int, default=33)
    ap.add_argument("--exclude", type=float, default=0.03,
                    help="half-width of the excluded band about gamma=1-kappa")
    ap.add_argument("--m-tot", type=float, default=50.0)
    ap.add_argument("--oversample", type=int, default=4)
    ap.add_argument("--grid-n-td", type=int, default=1000)
    ap.add_argument("--grid-n-y", type=int, default=8)
    ap.add_argument("--seg-dur", type=float, default=0.0)
    ap.add_argument("--n-proc", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--outdir", default="regenerated/chang_refsdal")
    ap.add_argument("--tag", default="")
    ap.add_argument("--extend-a", action="store_true",
                    help="additionally sweep the amplitude axis mirrored "
                         "about one, a -> 1/a, so the trailing image may be "
                         "the brighter: the fainter-image-first case the "
                         "deployed a<1 family cannot reach at any q. A probe "
                         "for one appendix figure; needs its own "
                         "--outdir, and does not change the a<1 outputs.")
    ap.add_argument("--optics", default="hybrid", choices=["hybrid", "go"],
                    help="'go' forces exact geometric optics everywhere, so "
                         "that a hybrid run and a go run differ ONLY in the "
                         "injected F on the closed-contour half")
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

    grid_kw = dict(
        n_td=args.grid_n_td, n_y=args.grid_n_y, t_d_min=1e-3, t_d_max=0.5,
        y_min=0.01, y_max=2.0, M_Lz_min=1e2, M_Lz_max=1e5,
    )
    probe = gr.snap_grid_to_samples(
        gr.build_grid(**grid_kw), cfg.delta_t / args.oversample
    )
    model_kw = {
        "m_lz": args.m_lz, "y": args.y, "psi0": args.psi0,
        "phi_gamma": float(np.deg2rad(args.phi_gamma_deg)),
        "exclude": args.exclude,
        "optics_mode": args.optics,
    }

    print("Chang-Refsdal  M_Lz=%g Msun  y=%g  phi_gamma=%g deg  M_tot=%g"
          % (args.m_lz, args.y, args.phi_gamma_deg, args.m_tot))
    print("  seg=%.0f s  chirp=%.2f s  t0=%.2f s  FROZEN grid %d td x %d y "
          "(%d active)" % (seg, seg_info["chirp_time_s"], t0,
                           probe.n_td, probe.n_y, probe.n_points))
    print("  injections: wave optics where kappa+gamma<1 (GLoW %s I(tau) + the "
          "wave_contour transform),\n              exact geometric optics where "
          "the contours are open" % gb.glow_version())
    print("  wave-optics band: w in [%.2f, %.1f]; validated to |dF|/|F| <~ 4e-3 "
          "for w <= %g" % (amp.w_of_f(cfg.f_lower, args.m_lz),
                           amp.w_of_f(cfg.f_final, args.m_lz), wc.W_MAX_TRUSTED))

    k_grid = np.linspace(0.0, args.kappa_max, args.n_kappa)
    g_grid = np.linspace(0.0, args.gamma_max, args.n_gamma)
    tasks = [((i, j), float(k_grid[i]), float(g_grid[j]))
             for i in range(args.n_kappa) for j in range(args.n_gamma)]

    shape = (args.n_kappa, args.n_gamma)
    r_ul = np.full(shape, np.nan)
    r_q = {q: np.full(shape, np.nan) for q in QS}
    r_qe = {q: np.full(shape, np.nan) for q in QS}
    ext_used_q = {q: np.zeros(shape, dtype=bool) for q in QS}
    n_img = np.zeros(shape, dtype=int)
    req_q = np.full(shape, -1, dtype=int)
    a2 = np.full(shape, np.nan)
    a_max = np.full(shape, np.nan)
    t_d = np.full(shape, np.nan)
    dtau2 = np.full(shape, np.nan)
    d_crit = np.full(shape, np.nan)
    mu_macro = np.full(shape, np.nan)
    macro_saddle = np.zeros(shape, dtype=bool)
    optics_wave = np.zeros(shape, dtype=bool)
    best_td = np.full(shape, -1, dtype=int)
    best_y = np.full(shape, -1, dtype=int)
    n_skip = {}

    from multiprocessing import Pool

    t_start = time.time()
    with Pool(processes=args.n_proc, initializer=_init,
              initargs=(cfg_kw, t0, args.oversample, grid_kw, model_kw,
                        args.extend_a)) as pool:
        for n, (idx, v_ul, vq, ij, d) in enumerate(
            pool.imap_unordered(_cell, tasks, chunksize=4), start=1
        ):
            r_ul[idx] = v_ul
            d_crit[idx] = d["d_crit"]
            mu_macro[idx] = d["mu_macro"]
            macro_saddle[idx] = d["macro_type"] == "saddle"
            optics_wave[idx] = d.get("optics") == "wave"
            n_img[idx] = d.get("n_images", 0)
            if "skipped" in d:
                n_skip[d["skipped"]] = n_skip.get(d["skipped"], 0) + 1
            if vq is not None:
                for q in QS:
                    r_q[q][idx] = vq[q]
                if "ext_q" in d:
                    for q in QS:
                        r_qe[q][idx] = d["ext_q"][q]
                        ext_used_q[q][idx] = d["ext_used"][q]
                best_td[idx], best_y[idx] = ij[Q_FROZEN]
                req_q[idx] = d["required_q"]
                a2[idx] = d["a2"]
                a_max[idx] = d["a_max"]
                t_d[idx] = d["t_d_s"]
                dtau2[idx] = d["dtau2"]
            if n % 200 == 0 or n == len(tasks):
                el = time.time() - t_start
                print("  %5d/%d  %.0f s elapsed, %.0f s projected"
                      % (n, len(tasks), el, el * len(tasks) / n), flush=True)

    r_frozen = r_q[Q_FROZEN]
    r_general = np.nanmax(np.stack([r_q[q] for q in QS]), axis=0)
    r_extended = ext_gt1 = None
    if args.extend_a:
        stack_e = np.stack([np.where(np.isfinite(r_qe[q]), r_qe[q], -1.0)
                            for q in QS])
        r_extended = np.nanmax(np.stack([r_qe[q] for q in QS]), axis=0)
        bq = np.argmax(stack_e, axis=0)
        # did the winning q reach its best on the mirrored half?
        ext_gt1 = np.choose(bq, [ext_used_q[q] for q in QS])
        ext_gt1 &= np.isfinite(r_extended)
    best_q = np.full(shape, -1, dtype=int)
    ok = np.isfinite(r_general)
    if ok.any():
        best_q[ok] = np.argmax(np.stack([np.where(np.isfinite(r_q[q]), r_q[q], -1)
                                         for q in QS]), axis=0)[ok]

    # ---- statistics ------------------------------------------------------
    in_domain = (t_d >= 1e-3) & (t_d <= 0.5) & np.isfinite(r_frozen)
    sad = in_domain & macro_saddle
    mn = in_domain & ~macro_saddle

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

    def block(mask):
        out = {
            "unlensed": stats(mask, r_ul),
            "frozen_q1": stats(mask, r_frozen),
            "generalized": stats(mask, r_general),
            **{f"q{q}": stats(mask, r_q[q]) for q in QS},
        }
        if r_extended is not None:
            out["extended"] = stats(mask, r_extended)
        return out

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
        "lens_model": "changrefsdal",
        "lens_label": "Chang-Refsdal M_Lz=%g" % args.m_lz,
        "m_lz": args.m_lz,
        "y": args.y,
        "phi_gamma_deg": args.phi_gamma_deg,
        "psi0": args.psi0,
        "m_tot": args.m_tot,
        "exclude_half_width": args.exclude,
        "t0_s": t0,
        "segment": seg_info,
        "oversample": args.oversample,
        "search_grid": probe.as_dict(),
        "search_grid_note": "frozen point-mass-derived grid, deliberately not "
                            "re-tuned for this lens model",
        "injection": "hybrid: exact wave optics where the Fermat contours "
                     "close (kappa+gamma<1) -- GLoW's I(tau) with the "
                     "lensing.wave_contour transform -- and exact geometric "
                     "optics from the Chang-Refsdal quartic where they are "
                     "open",
        "optics_census": {
            "wave": int(optics_wave.sum()),
            "geometric": int(np.sum(np.isfinite(r_frozen) & ~optics_wave)),
        },
        "wave_optics": {
            "w_min": float(amp.w_of_f(cfg.f_lower, args.m_lz)),
            "w_max": float(amp.w_of_f(cfg.f_final, args.m_lz)),
            "w_max_trusted": wc.W_MAX_TRUSTED,
            "tau_max": wc.TAU_MAX, "n_seg": wc.N_SEG, "T_saddle": wc.T_SADDLE,
            "validation": "results/frozen/cr_wave_validation/"
                          "cr_wave_validation.json",
        },
        "optics_mode": args.optics,
        "extend_a": bool(args.extend_a),
        "extended_amplitude_axis": (
            {"rule": "a -> 1/a, the frozen axis mirrored about one",
             "n_a_frozen": int(probe.a.size),
             "n_a_extended": int(2 * probe.a.size),
             "a_range_extended": [float(probe.a.min()),
                                  float(1.0 / probe.a.min())],
             "note": "a probe for one appendix figure; the deployed search "
                     "and every other product keep 0 < a < 1"}
            if args.extend_a else None),
        "glow_version": gb.glow_version(),
        "morse_phases": {str(q): cv.DEFAULT.with_rel_morse(q).rel_morse_label
                         for q in QS},
        "quantity": "recovery = max over time and the frozen lens grid of "
                    "|z|/sigma(d)",
        "wall_time_s": time.time() - t_start,
        "n_skipped": n_skip,
        "searched_domain": block(in_domain),
        "macro_minimum_cells": block(mn),
        "macro_saddle_cells": block(sad),
        "required_q_census": {
            str(q): int(np.sum(req_q == q)) for q in QS
        },
        "best_q_census_macro_saddle": {
            str(q): int(np.sum(best_q[sad] == q)) for q in QS
        },
        "image_counts": {str(k): int(np.sum(n_img == k)) for k in (0, 1, 2, 3, 4)},
    }
    if sad.any():
        at = a2[sad]
        meta["macro_saddle_analytic_floor"] = {
            "median_a2": float(np.median(at)),
            "predicted_1_over_sqrt_1_plus_a2sq_median":
                float(np.median(1.0 / np.sqrt(1.0 + at**2))),
        }

    os.makedirs(args.outdir, exist_ok=True)
    tag = args.tag or ("changrefsdal_MLz%g_M%g" % (args.m_lz, args.m_tot))
    npz = os.path.join(args.outdir, f"lensmodel_{tag}.npz")
    np.savez_compressed(
        npz, kappa=k_grid, gamma=g_grid,
        recovery_unlensed=r_ul, recovery_frozen=r_frozen,
        recovery_general=r_general,
        **{f"recovery_q{q}": r_q[q] for q in QS},
        **({"recovery_extended": r_extended,
            "ext_uses_a_gt_1": ext_gt1,
            **{f"recovery_ext_q{q}": r_qe[q] for q in QS}}
           if args.extend_a else {}),
        n_images=n_img, required_q=req_q, best_q=best_q,
        a2=a2, a_max=a_max, t_d=t_d, dtau2=dtau2,
        d_crit=d_crit, mu_macro=mu_macro, macro_saddle=macro_saddle,
        optics_wave=optics_wave,
        best_td_index=best_td, best_y_index=best_y,
        grid_t_d=probe.t_d, grid_y=probe.y, meta=json.dumps(meta),
    )
    with open(os.path.join(args.outdir, f"lensmodel_{tag}.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print("\noptics census:", meta["optics_census"])
    print("image-count census:", meta["image_counts"])
    print("required-q census:", meta["required_q_census"])
    print("skipped:", n_skip)
    for key in ("searched_domain", "macro_minimum_cells", "macro_saddle_cells"):
        print("-- %s --" % key)
        for k in ("unlensed", "frozen_q1", "generalized"):
            st = meta[key][k]
            if st:
                print("   %-12s n=%4d  min %.4f  median %.4f  frac>=0.97 %.3f"
                      % (k, st["n"], st["min"], st["median"], st["frac_ge_0.97"]))
    if sad.any():
        print("macro-saddle analytic floor:",
              meta["macro_saddle_analytic_floor"])
        print("best-q census in macro-saddle:",
              meta["best_q_census_macro_saddle"])
    print("wrote %s" % npz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
