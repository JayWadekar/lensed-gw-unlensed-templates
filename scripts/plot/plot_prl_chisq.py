#!/usr/bin/env python
"""Figure 2 of the paper: the signal-consistency test.

One CBC total mass, M_tot = 50 Msun, and three panels in order.

    (a)  an overlapping-image lensed signal filtered with the ordinary
         *unlensed* template fails the standard power chi-squared test, so a
         standard search down-weights a real astrophysical signal
    (b)  the same data against the best-fit two-image template -- the template
         that produced the two-image trigger, with equal-power bins that move
         with (t_d, a) -- is consistent again
    (c)  what that is worth, with each arm at its *own* measured
         FAP = 1e-3 threshold, so the lens-grid trials factor is paid for

Panels (a) and (b) share one colour scale.  Reads stored products only; the
volume ratio is the closed-form re-weighting of the stored maps, exactly as in
``plot_appendix_chisq.py``, and no simulation is repeated here.

    python scripts/plot/plot_prl_chisq.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src"))

import prl_style as st  # noqa: E402

st.use()

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from lensing import reweighted as rw  # noqa: E402

NAME = "prl_chisq"
CHI2_LEVEL = 2.0          # the value above which a search demotes a trigger
C_CHI2 = "#00e5ff"        # the chi2_r = 2 contour
VOL_TICKS = (0.1, 0.3, 1.0, 3.0, 10.0)
T_D_DOMAIN = (1e-3, 0.5)


def _chi2_panel(ax, x, y, arr, norm, t_d, t_chirp):
    im = ax.pcolormesh(x, y, arr.T, cmap="inferno", norm=norm,
                       shading="auto", rasterized=True)
    with np.errstate(invalid="ignore"):
        ax.contour(x, y, arr.T, levels=[CHI2_LEVEL], colors=C_CHI2,
                   linewidths=1.7, zorder=6)
        ax.contour(x, y, t_d.T, levels=[t_chirp], colors="white",
                   linewidths=1.8, linestyles="dashed", zorder=5)
    return im


def _volume_panel(ax, x, y, ratio, norm, t_d, t_chirp):
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log10(np.where(ratio > 0, ratio, np.nan))
    im = ax.pcolormesh(x, y, lr.T, cmap="RdBu_r", norm=norm, shading="auto",
                       rasterized=True)
    with np.errstate(invalid="ignore"):
        cs = ax.contour(x, y, lr.T, levels=[np.log10(2.0)], colors="black",
                        linewidths=1.6, zorder=6)
        ax.contour(x, y, t_d.T, levels=[t_chirp], colors="black",
                   linewidths=1.8, linestyles="dashed", zorder=5)
    return im, cs


def _label_flattest(ax, cs, level, text, fontsize=8):
    """Put one inline label on ``cs`` where the contour is flattest on screen.

    ``clabel`` left to itself lands on the steep right-hand branch of this
    particular contour and prints almost vertically at PRL column width.  The
    y axis is logarithmic and the panel is square, so the flattest point has to
    be found in *display* coordinates -- which means calling this only after
    the scale and limits are set.
    """
    best = None
    for path in cs.get_paths():
        v = path.vertices
        if len(v) < 3:
            continue
        seg = np.diff(ax.transData.transform(v), axis=0)
        tilt = np.abs(np.degrees(np.arctan2(seg[:, 1], seg[:, 0])))
        tilt = np.minimum(tilt, 180.0 - tilt)
        i = int(np.argmin(tilt))
        if best is None or tilt[i] < best[0]:
            best = (tilt[i], tuple(0.5 * (v[i] + v[i + 1])))
    if best is None:
        return
    lab = ax.clabel(cs, fmt={level: text}, inline=True, inline_spacing=3,
                    fontsize=fontsize, colors="black", manual=[best[1]])
    # Horizontal, not contour-following: a tilted "$\times$" is read as a plus
    # sign at this size.  The gap clabel carved stays where it is.
    for t in lab:
        t.set_rotation(0.0)
        t.set_va("center")
        t.set_ha("center")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",
                    default=st.data_path("chisq", "chisq_M50.npz"))
    ap.add_argument("--background",
                    default=st.data_path("chisq") + "/"
                            "background_chisq.npz")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    bgd = np.load(args.background, allow_pickle=False)
    bg_meta = json.loads(str(bgd["meta"]))
    th_ul = bg_meta["thresholds"]["reweighted"]["UL"]
    th_bf = bg_meta["thresholds"]["reweighted"]["BF"]

    M, y, t_d = d["M"], d["y"], d["t_d_s"]
    x = np.log10(M)
    t_chirp = meta["chirp_time_s"]
    rho_ref = meta["rho_ref"]
    chi2_ul = 1.0 + d["chi2_unlensed"]
    chi2_bf = 1.0 + d["chi2_lensed"]
    dom = (t_d >= T_D_DOMAIN[0]) & (t_d <= T_D_DOMAIN[1])

    ratio = rw.volume_ratio(
        d["chi2_lensed"], d["snr_bankfree"] / rho_ref,
        d["chi2_unlensed"], d["snr_unlensed"] / rho_ref,
        rho_ref, th_bf, rho_hat_th_b=th_ul,
    )

    hi = float(np.nanmax([np.nanmax(chi2_ul), np.nanmax(chi2_bf)]))
    norm = mcolors.LogNorm(vmin=0.9, vmax=np.ceil(2.0 * hi) / 2.0)
    # Colour clipped at a factor of 10 either way: cells just inside the
    # saturation boundary have formally divergent ratios and would otherwise
    # compress all the real structure to white.  Stated in the caption; the
    # unclipped statistics go to the summary JSON.
    vnorm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    fig, axes = plt.subplots(1, 3, figsize=(st.TEXT_WIDTH, 3.30),
                             sharex=True, sharey=True,
                             constrained_layout=True)
    for ax in axes:
        ax.set_box_aspect(1.0)

    im = _chi2_panel(axes[0], x, y, chi2_ul, norm, t_d, t_chirp)
    _chi2_panel(axes[1], x, y, chi2_bf, norm, t_d, t_chirp)
    imv, csv = _volume_panel(axes[2], x, y, ratio, vnorm, t_d, t_chirp)

    axes[0].set_title(r"\textbf{Unlensed search}", fontsize=11.5)
    axes[1].set_title(r"\textbf{Two-image search}", fontsize=11.5)
    axes[2].set_title(r"\textbf{Volume ratio at fixed FAP}", fontsize=11.5)

    for ax, lab in zip(axes, "abc"):
        st.panel_label(ax, lab)
        ax.set_yscale("log")
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])
        ax.set_xticks([2, 3, 4, 5])
        ax.set_xlabel(r"$\log_{10}\!\left(M_{\mathrm{L}z}/M_\odot\right)$")
    axes[0].set_ylabel(r"impact parameter $y$")

    # After the log scale and the limits, so the tilt is measured on screen.
    _label_flattest(axes[2], csv, np.log10(2.0), r"$\times 2$")

    cb = fig.colorbar(im, ax=[axes[0], axes[1]], location="bottom",
                      shrink=0.92, pad=0.035, aspect=42)
    cb.set_label(r"expected $\langle\chi^{2}_{r}\rangle$", labelpad=2)
    cb.set_ticks([1, 2, 3, 5])
    cb.set_ticklabels([r"$1$", r"$2$", r"$3$", r"$5$"])
    cb.minorticks_off()
    cb.ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    cb.ax.axvline(CHI2_LEVEL, color=C_CHI2, lw=2.2)
    cb.solids.set_rasterized(True)

    cbv = fig.colorbar(imv, ax=[axes[2]], location="bottom", shrink=0.92,
                       pad=0.035, aspect=21)
    cbv.set_label(r"$V_{\textrm{two-image}}/V_{\textrm{unlensed}}$", labelpad=2)
    cbv.set_ticks([np.log10(v) for v in VOL_TICKS])
    cbv.set_ticklabels([r"$%g$" % v for v in VOL_TICKS])
    cbv.solids.set_rasterized(True)

    out = st.save(fig, NAME)
    print("wrote %s" % out)

    def _stats(arr):
        m = dom & np.isfinite(arr)
        return {"n_cells": int(m.sum()), "median": float(np.median(arr[m])),
                "max": float(np.max(arr[m])),
                "frac_gt_2": float(np.mean(arr[m] > 2.0))}

    okv = dom & np.isfinite(ratio) & (ratio > 0)
    band = okv & (t_d < t_chirp)

    # The SNR-only comparison the chi2 term is contrasted against in Sec. IV B:
    # naive "mismatch cube" (FF_two-image/FF_unlensed)^3, and the same quantity
    # once each arm pays its own FAP threshold.  No chi^2 enters either.
    cube = (d["snr_bankfree"] / d["snr_unlensed"]) ** 3
    cube_fap = cube * (th_ul / th_bf) ** 3

    def _vstats(arr, m):
        a = arr[m]
        return {"n_cells": int(m.sum()), "median": float(np.median(a)),
                "p90": float(np.percentile(a, 90)), "max": float(np.max(a)),
                "frac_gt_2": float(np.mean(a > 2.0)),
                "frac_lt_1": float(np.mean(a < 1.0))}
    summary = {
        "sources": [args.data, args.background],
        "m_tot": meta["m_tot"], "rho_ref": rho_ref,
        "n_bins": meta["n_bins"], "dof": meta["dof"],
        "chirp_time_s": t_chirp,
        "null_control_excess": meta["null_control_excess"],
        "null_control_lensed": meta.get("null_control_lensed"),
        "thresholds": {"unlensed": th_ul, "bank_free": th_bf,
                       "fap": bg_meta["fap"],
                       "n_background_segments": bg_meta["n_segments"]},
        "chi2_unlensed": _stats(chi2_ul),
        "chi2_bankfree": _stats(chi2_bf),
        "volume_ratio": {
            "n_cells": int(okv.sum()),
            "median": float(np.median(ratio[okv])),
            "p90": float(np.percentile(ratio[okv], 90)),
            "frac_gt_2": float(np.mean(ratio[okv] > 2.0)),
            "frac_lt_1": float(np.mean(ratio[okv] < 1.0)),
            "overlap_band": {
                "definition": "searched domain with t_d < chirp duration",
                "n_cells": int(band.sum()),
                "median": float(np.median(ratio[band])),
                "p90": float(np.percentile(ratio[band], 90)),
                "frac_gt_2": float(np.mean(ratio[band] > 2.0)),
            },
            "threshold_only_factor": float((th_ul / th_bf) ** 3),
            "n_standard_undetectable": int(np.sum(dom & np.isinf(ratio))),
            # Conditioned on cells where the lensed signal clearly fails the
            # standard test.  The chi2_r > 4 row is what the abstract's
            # "median factor of 2.6" quotes.
            "conditional": {
                ("chi2_gt_%g" % lev): {
                    "full": _vstats(ratio, okv & (chi2_ul > lev)),
                    "mismatch_cube": _vstats(cube, okv & (chi2_ul > lev)),
                } for lev in (2.0, 4.0)
            },
        },
        "mismatch_cube": {
            "definition": "(snr_bankfree/snr_unlensed)^3, SNR only, no chi2",
            "raw": _vstats(cube, okv),
            "at_own_fap": _vstats(cube_fap, okv),
        },
        "fitting_factor": {
            "unlensed": {"median": float(np.median(d["snr_unlensed"][okv]
                                                   / rho_ref)),
                         "min": float(np.min(d["snr_unlensed"][okv]
                                             / rho_ref))},
            "two_image": {"median": float(np.median(d["snr_bankfree"][okv]
                                                    / rho_ref)),
                          "min": float(np.min(d["snr_bankfree"][okv]
                                              / rho_ref))},
        },
        "colour_clip": "volume ratio clipped at a factor of 10 either way",
    }
    for outdir in st.OUTDIRS:
        with open(os.path.join(outdir, NAME + "_summary.json"), "w") as fh:
            json.dump(st.relative(summary), fh, indent=2)

    caption = r"""An overlapping-image lensed signal fails the standard
signal-consistency test, and the two-image template that the search selects
--- found by recombination --- restores it. Exact wave-optics point-mass injections at
$M_{\rm tot}=50\,\msun$, noise-free and scaled to optimal SNR $\rho=%(rho)g$;
PyCBC \texttt{power\_chisq} with $%(nbins)d$ bins ($%(dof)d$ d.o.f.) evaluated
at the trigger time. Plotted is $\langle\chi^2_r\rangle = 1+\text{excess}$, the
value expected once Gaussian noise is added. The dashed contour is $t_d$ equal
to the chirp duration, $%(tchirp).2f$~s; to its left the two images overlap
inside the waveform, and at this mass the whole searched delay domain
$1~\mathrm{ms}\le t_d\le500$~ms lies to its left.
\textbf{(a)} The test applied to the lensed signal as an unlensed search sees it: median
$%(ul_med).2f$ over the searched domain, reaching $%(ul_max).1f$, and above
$%(lev)g$ over $%(ul_frac).0f\%%$ of it. \textbf{(b)} The same test for
the trigger the two-image search actually reports --- the same data against
the best-fit two-image template, whose equal-power bins move with $(t_d,a)$ ---
median $%(bf_med).2f$, maximum $%(bf_max).2f$, and no cell above $%(lev)g$.
\textbf{(c)} The ratio of sensitive volumes with each search at its own
$\mathrm{FAP}=10^{-3}$, using thresholds $%(th_bf).2f$ (two-image) and
$%(th_ul).2f$ (unlensed) measured from $%(nseg)s$ pure-noise segments. The
unequal thresholds are the lens-grid trials factor, worth $%(thonly).2f$ in
volume on their own. Red is where the consistency test more than repays it ---
median $\times%(v_med).2f$, reaching $\times%(v_p90).1f$ at the $90$th
percentile and exceeding $\times2$ over $%(v_frac2).0f\%%$ of the searched
cells, which the black contour encloses --- and blue is where it does not,
$%(v_frac_lt1).0f\%%$ of cells, which is the same conclusion as the
fixed-false-alarm comparison of the SNR alone. Colour is clipped at a factor of
$10$ either way."""
    caption = caption % {
        "rho": rho_ref, "nbins": meta["n_bins"], "dof": meta["dof"],
        "lev": CHI2_LEVEL,
        "ul_med": summary["chi2_unlensed"]["median"],
        "ul_max": summary["chi2_unlensed"]["max"],
        "ul_frac": 100.0 * summary["chi2_unlensed"]["frac_gt_2"],
        "bf_med": summary["chi2_bankfree"]["median"],
        "bf_max": summary["chi2_bankfree"]["max"],
        "th_bf": th_bf, "th_ul": th_ul,
        "nseg": r"3\times10^{5}" if bg_meta["n_segments"] == 300000
                else "%g" % bg_meta["n_segments"],
        "thonly": summary["volume_ratio"]["threshold_only_factor"],
        "v_med": summary["volume_ratio"]["median"],
        "tchirp": t_chirp,
        "v_frac2": 100.0 * summary["volume_ratio"]["frac_gt_2"],
        "v_p90": summary["volume_ratio"]["p90"],
        "v_frac_lt1": 100.0 * summary["volume_ratio"]["frac_lt_1"],
    }
    for outdir in st.OUTDIRS:
        with open(os.path.join(outdir, NAME + "_caption.txt"), "w") as fh:
            fh.write(caption.strip() + "\n")
    print(json.dumps({k: summary[k] for k in
                      ("chi2_unlensed", "chi2_bankfree", "volume_ratio")},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
