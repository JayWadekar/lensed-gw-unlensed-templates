#!/usr/bin/env python
"""Fitting factors of the two-image family against wave-optics point-mass injections.

Inject with the **exact** point-lens wave-optics factor ``F_ML``; recover with
the geometric-optics recombination over a lens grid.  For each ``(M_Lz, y)``
this records

- ``recovery_unlensed``: ``max_t |z_1(t)| / sigma(d)``, the ordinary CBC search;
- ``recovery_lensed``:   ``max_{t,theta} |z_L(t;theta)| / sigma(d)``, the
  two-image family.

which is the pair of panels in G26 Fig. 9 (top row / bottom row).  G26 has no figure with ``f_ML`` on an axis, so the
reproduction is in the ``(log10 M_Lz, y)`` plane with ``f_ML`` contours
overlaid, and the ``>= 0.97`` claim is tested inside the banded region.

The quantity is a match/recovery maximized over
``(t_c, phi_c)`` and over the lens grid, at **fixed intrinsic CBC parameters**.

The defaults are a quick, coarse run.  The paper's maps (``data/``) use::

    # Fig. 1(a, c): delay grid extended to the chirp duration
    python scripts/simulate/phase3_wave_optics.py --m-tot 50 --seg-dur 16 \\
        --grid-n-td 2318 --grid-n-y 8 --t-d-max 0 --outdir regenerated/pointmass_M50
    # appendix figure at 11 and 100 Msun
    python scripts/simulate/phase3_wave_optics.py --m-tot 11 --seg-dur 32 \\
        --grid-n-td 1000 --grid-n-y 8 --outdir regenerated/pointmass_M11_M100
    python scripts/simulate/phase3_wave_optics.py --m-tot 100 --seg-dur 16 \\
        --grid-n-td 1000 --grid-n-y 8 --outdir regenerated/pointmass_M11_M100
"""

from __future__ import annotations

import os

# Single-threaded BLAS, set BEFORE numpy is imported.  Two reasons:
#  (1) this driver forks a process pool after the parent has already run a
#      large matmul (the 1000-lag autocorrelation), and a fork that inherits
#      locked OpenMP/BLAS state deadlocks the children -- observed as 16
#      workers sleeping at zero CPU indefinitely;
#  (2) we want N independent processes, not N processes each spawning threads,
#      which would oversubscribe the machine.
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

#: the four relative Morse phases the generalized statistic maximizes over
QS = (0, 1, 2, 3)

#: the deployed one
Q_FROZEN = 1


def _init(cfg_kw, t0, oversample, grid_kw):
    cfg = wf.AnalysisConfig(**cfg_kw)
    psd = wf.make_psd(cfg)
    h = wf.normalize(wf.make_waveform(cfg), psd, cfg.delta_f)
    dt = cfg.delta_t / oversample
    grid = gr.snap_grid_to_samples(gr.build_grid(**grid_kw), dt)
    pr.assign_weights(grid, "P0")
    # the lag is q-independent, so one autocorrelation serves all four families
    C_vals = wf.autocorr_at_lags(h, psd, cfg, cv.DEFAULT.lag(grid.t_d))
    convs = {q: cv.DEFAULT.with_rel_morse(q) for q in QS}
    _W.update(
        cfg=cfg, psd=psd, h=h, freqs=cfg.frequencies(),
        nz=np.nonzero(np.abs(h) > 0)[0], t0=t0, oversample=oversample,
        grid=grid, C_vals=C_vals, dt=dt,
        shifts=np.rint(cv.DEFAULT.lag(grid.t_d) / dt).astype(int),
        convs=convs,
        us={q: convs[q].filter_coeff(1.0) for q in QS},
        norms={q: rc.lensed_norm_sq(C_vals[:, None], grid.a[None, :],
                                    conv=convs[q]) for q in QS},
    )


def _recover(data, t_d_inj=0.0):
    """Return (unlensed recovery, lensed recovery, best grid indices).

    The time search is restricted to a window around the injection, which is
    exact rather than approximate.  A lensed template parameterized by time
    ``t`` places its images at ``t`` and ``t + t_d_tmpl`` with
    ``t_d_tmpl <= t_d_max``, and a nonzero overlap requires one template image
    to coincide with one injection image (at ``t0`` or ``t0 + t_d_inj``), so

        t in [t0 - t_d_max,  t0 + t_d_inj]

    covers every pairing.  A margin of ``t_d_max`` is added on each side.  With
    ``n_td = 1000`` this is what makes the dense-grid map affordable: the window
    is ~1 s of a 16 s segment over most of the plane.
    """
    cfg, psd, h = _W["cfg"], _W["psd"], _W["h"]
    grid, shifts, t0 = _W["grid"], _W["shifts"], _W["t0"]
    us, norms = _W["us"], _W["norms"]
    sig = wf.sigma(data, psd, cfg.delta_f)
    z1, dt = wf.snr_series(data, h, psd, cfg, oversample=_W["oversample"])
    n = z1.size

    t_d_max = float(grid.t_d[-1])
    lo = max(0, int(np.floor((t0 - 2.0 * t_d_max) / dt)))
    hi = min(n, int(np.ceil((t0 + t_d_inj + 2.0 * t_d_max) / dt)) + 1)
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
    idx, M_Lz, y, backend = task
    cfg, nz, freqs, t0 = _W["cfg"], _W["nz"], _W["freqs"], _W["t0"]
    t_d = float(amp.time_delay(M_Lz, y))
    # the trailing image must fit inside the segment: the FFT is circular
    if t0 + t_d > 0.9 * cfg.seg_dur:
        return idx, np.nan, None, None, {"skipped": "t_d too long"}

    f_nz = freqs[nz]
    if backend == "auto":
        F, _info = amp.F_ML_auto(f_nz, M_Lz, y)
    else:
        F = amp.F_ML(f_nz, M_Lz, y, backend=backend)

    data = np.zeros(freqs.size, dtype=complex)
    data[nz] = _W["h"][nz] * F
    data = wf.fd_delay(data, freqs, t0)
    r_ul, r_q, ij = _recover(data, t_d_inj=t_d)
    return idx, r_ul, r_q, ij, {"t_d_s": t_d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m-tot", type=float, default=50.0)
    ap.add_argument("--n-m", type=int, default=32)
    ap.add_argument("--n-y", type=int, default=22)
    ap.add_argument("--log-m-min", type=float, default=1.1)
    ap.add_argument("--log-m-max", type=float, default=5.0)
    ap.add_argument("--y-min", type=float, default=0.01)
    ap.add_argument("--y-max", type=float, default=2.0)
    ap.add_argument("--oversample", type=int, default=4)
    ap.add_argument("--grid-n-td", type=int, default=120)
    ap.add_argument("--t-d-max", type=float, default=0.5,
                    help="upper edge of the SEARCH delay grid, in seconds. "
                         "0 = the chirp duration, which is where the two "
                         "images stop overlapping inside the waveform and a "
                         "bank stops being the right instrument. The frozen "
                         "products use the default 0.5, G26's bank edge; any "
                         "other value needs its own --outdir.")
    ap.add_argument("--grid-n-y", type=int, default=24)
    ap.add_argument("--seg-dur", type=float, default=0.0,
                    help="0 = choose automatically from the chirp time")
    ap.add_argument("--backend", default="auto", choices=["auto", "mpmath"])
    ap.add_argument("--n-proc", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260831)
    ap.add_argument("--n-crosscheck", type=int, default=20)
    ap.add_argument("--outdir", default="regenerated/pointmass_M50")
    args = ap.parse_args()

    base = wf.AnalysisConfig(mass1=args.m_tot / 2, mass2=args.m_tot / 2)
    # `--t-d-max 0` means "stop the grid at the chirp duration"; resolving it
    # needs the chirp time, which needs a config, so do it before `seg`.
    t_d_max = args.t_d_max if args.t_d_max > 0 else sim.chirp_time(base)
    # The segment must hold the chirp plus the largest *searched* delay; the
    # margin below keeps the old behaviour at the default t_d_max = 0.5.
    seg_margin = max(1.0, 2.0 * t_d_max)
    seg = args.seg_dur or sim.suggested_seg_dur(base, t_d_max=seg_margin)
    cfg_kw = dict(
        seg_dur=seg, sample_rate=4096.0, f_lower=20.0, f_final=1024.0,
        psd_name="aLIGOZeroDetHighPower", approximant="IMRPhenomD",
        mass1=args.m_tot / 2, mass2=args.m_tot / 2,
    )
    cfg = wf.AnalysisConfig(**cfg_kw)
    seg_info = sim.validate_segment(cfg, t_d_max=seg_margin)
    t0 = min(2.0, 0.1 * seg)
    t0 = max(t0, seg_info["chirp_time_s"] * 1.05)

    grid_kw = dict(
        n_td=args.grid_n_td, n_y=args.grid_n_y,
        t_d_min=1e-3, t_d_max=t_d_max, y_min=0.01, y_max=2.0,
        M_Lz_min=1e2, M_Lz_max=1e5,
    )
    probe = gr.snap_grid_to_samples(
        gr.build_grid(**grid_kw), cfg.delta_t / args.oversample
    )
    print(
        "M_tot=%g  seg=%.0f s  chirp=%.2f s  t0=%.2f s  t_d_max=%.4f s  "
        "search grid %d td x %d y (%d active)"
        % (args.m_tot, seg, seg_info["chirp_time_s"], t0, t_d_max,
           probe.n_td, probe.n_y, probe.n_points)
    )

    M_grid = np.logspace(args.log_m_min, args.log_m_max, args.n_m)
    y_grid = np.geomspace(args.y_min, args.y_max, args.n_y)
    tasks = [
        ((i, j), float(M_grid[i]), float(y_grid[j]), args.backend)
        for i in range(args.n_m)
        for j in range(args.n_y)
    ]

    r_ul = np.full((args.n_m, args.n_y), np.nan)
    r_q = {q: np.full((args.n_m, args.n_y), np.nan) for q in QS}
    best_td = np.full((args.n_m, args.n_y), -1, dtype=int)
    best_y = np.full((args.n_m, args.n_y), -1, dtype=int)

    from multiprocessing import Pool

    t_start = time.time()
    with Pool(
        processes=args.n_proc, initializer=_init,
        initargs=(cfg_kw, t0, args.oversample, grid_kw),
    ) as pool:
        for n, (idx, v_ul, vq, ij, _d) in enumerate(
            pool.imap_unordered(_cell, tasks, chunksize=1), start=1
        ):
            r_ul[idx] = v_ul
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
    best_q = np.full((args.n_m, args.n_y), -1, dtype=int)
    _ok = np.isfinite(r_general)
    if _ok.any():
        best_q[_ok] = np.argmax(
            np.stack([np.where(np.isfinite(r_q[q]), r_q[q], -1) for q in QS]),
            axis=0)[_ok]

    # ---- pure-mpmath cross-check subsample -------------
    cross = []
    if args.backend == "auto" and args.n_crosscheck > 0:
        rng = np.random.default_rng(args.seed)
        finite = [t for t in tasks if np.isfinite(r_l[t[0]])]
        picks = rng.choice(
            len(finite), size=min(args.n_crosscheck, len(finite)), replace=False
        )
        sub = [(finite[k][0], finite[k][1], finite[k][2], "mpmath")
               for k in picks]
        with Pool(
            processes=args.n_proc, initializer=_init,
            initargs=(cfg_kw, t0, args.oversample, grid_kw),
        ) as pool:
            for idx, v_ul, vq, _ij, _d in pool.imap_unordered(_cell, sub):
                v_l = vq[Q_FROZEN]
                cross.append({
                    "M_Lz": float(M_grid[idx[0]]), "y": float(y_grid[idx[1]]),
                    "recovery_auto": float(r_l[idx]),
                    "recovery_mpmath": float(v_l),
                    "abs_diff": float(abs(r_l[idx] - v_l)),
                    "abs_diff_unlensed": float(abs(r_ul[idx] - v_ul)),
                })
        print("cross-check vs pure mpmath: max |drecovery| = %.3e over %d points"
              % (max(c["abs_diff"] for c in cross), len(cross)))

    # ---- the banded-region claim ------------------------------------------
    f_ml = 1.0 / amp.time_delay(M_grid[:, None], y_grid[None, :])
    f_ml_lo = 1.0 / t_d_max
    banded = (f_ml >= f_ml_lo) & (f_ml <= 1000.0) & np.isfinite(r_l)
    t_d_map = amp.time_delay(M_grid[:, None], y_grid[None, :])
    in_domain = (t_d_map >= 1e-3) & (t_d_map <= t_d_max) & np.isfinite(r_l)

    def stats(mask, arr):
        if not mask.any():
            return None
        v = arr[mask]
        return {
            "n": int(mask.sum()),
            "min": float(np.min(v)),
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
        "m_tot": args.m_tot,
        "t0_s": t0,
        "segment": seg_info,
        "oversample": args.oversample,
        "search_grid": probe.as_dict(),
        "backend": args.backend,
        "quantity": (
            "recovery = max over time and lens grid of |z| / sigma(d); "
            "injections use the exact wave-optics F_ML, recovery uses the "
            "geometric-optics two-image recombination"
        ),
        "wall_time_s": time.time() - t_start,
        "n_skipped_t_d_too_long": int(np.sum(~np.isfinite(r_l))),
        "crosscheck_vs_mpmath": cross,
        "crosscheck_max_abs_diff": (
            float(max(c["abs_diff"] for c in cross)) if cross else None
        ),
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
        "banded_region_f_ML_lo_Hz": float(f_ml_lo),
        "banded_region_f_ML_hi_Hz": 1000.0,
        "banded_region_t_d_max_s": float(t_d_max),
        "searched_domain_t_d_min_s": 1e-3,
        "searched_domain_t_d_max_s": float(t_d_max),
        "banded_region": {
            "lensed": stats(banded, r_l),
            "unlensed": stats(banded, r_ul),
            "generalized": stats(banded, r_general),
        },
        "searched_domain": {
            "lensed": stats(in_domain, r_l),
            "unlensed": stats(in_domain, r_ul),
            "generalized": stats(in_domain, r_general),
            **{f"q{q}": stats(in_domain, r_q[q]) for q in QS},
        },
    }

    os.makedirs(args.outdir, exist_ok=True)
    tag = f"M{args.m_tot:g}"
    npz = os.path.join(args.outdir, f"wave_optics_{tag}.npz")
    np.savez_compressed(
        npz, M_Lz=M_grid, y=y_grid, recovery_unlensed=r_ul,
        recovery_lensed=r_l, recovery_general=r_general,
        **{f"recovery_q{q}": r_q[q] for q in QS}, best_q=best_q,
        best_td_index=best_td, best_y_index=best_y,
        grid_t_d=probe.t_d, grid_y=probe.y, meta=json.dumps(meta),
    )
    with open(os.path.join(args.outdir, f"wave_optics_{tag}.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print("\n-- banded region (f_ML in [%.3f, 1000] Hz) --" % f_ml_lo)
    for k in ("unlensed", "lensed", "generalized"):
        st = meta["banded_region"][k]
        if st:
            print("  %-9s n=%4d  min %.4f  median %.4f  frac>=0.97 %.3f"
                  % (k, st["n"], st["min"], st["median"], st["frac_ge_0.97"]))
    print("-- searched domain (t_d in [1, %.0f] ms) --" % (1e3 * t_d_max))
    for k in ("unlensed", "lensed", "generalized"):
        st = meta["searched_domain_t_d_1ms_to_500ms"][k]
        if st:
            print("  %-12s n=%4d  min %.4f  median %.4f  frac>=0.97 %.3f"
                  % (k, st["n"], st["min"], st["median"], st["frac_ge_0.97"]))
    print("Morse-generalization gain:", meta["morse_generalization_gain"])
    print("best-q census:", meta["best_q_census"])
    print("wrote %s" % npz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
