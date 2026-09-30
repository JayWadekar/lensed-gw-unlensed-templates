#!/usr/bin/env python
"""Figure 1 of the paper: fitting factors of the two-image templates.

A 2x2 figure supporting the claim that the
two-image family is adequate for real signals and is not specific to
an isolated point mass:

    columns   left   an isolated point-mass microlens, injections built from the
                     *exact* wave-optics amplification factor, in (M_Lz, y) at
                     M_tot = 50 Msun
              right  a microlens on a strongly lensed macro-image
                     (Chang-Refsdal), in (kappa, gamma) at M_Lz = 3000 Msun

    rows      top     the ordinary unlensed-template search
              bottom  the two-image family: c = ia on the left, and
                      maximized over the relative Morse phase c = a i^q on the
                      right, which is the statistic a search would deploy there

Both columns are recovered on the *same* frozen 1000 x 8 point-mass search
grid.  Reads stored products only; recomputes nothing.

    python scripts/plot/plot_prl_effectualness.py
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

import matplotlib.pyplot as plt  # noqa: E402

from lensing import amplification as amp  # noqa: E402

NAME = "prl_effectualness"
F_ML_CONTOURS = (1000.0,)

#: Colour for cells whose injected delay exceeds the searched grid.  Beyond
#: the chirp duration the two images no longer overlap inside the waveform:
#: they arrive as separate triggers, which is the pair regime of the
#: sub-threshold section and not something a bank should be asked to cover.
#: Masking is honest rather than cosmetic -- on the old 1-500 ms grid those
#: cells were still coloured, and in every one of them the two-image family
#: scored *below* the unlensed template, because the grid has no a = 0 point
#: and so is forced to add a second copy it cannot place.
C_OUTSIDE = "0.82"

# Manual anchors for the f_ML guide labels, in (log10 M_Lz, y).  Chosen to sit
# clear of the 0.97 contour in the panel where the labels are drawn.
F_ML_LABEL_AT = {1000.0: (2.01, 0.25)}
# Degrees to rotate that label clockwise from the contour tangent clabel picks.
F_ML_LABEL_TILT = 8.0


def _load(path):
    d = np.load(path, allow_pickle=False)
    return d, json.loads(str(d["meta"]))


def _mesh(ax, xx, yy, arr, vmin):
    """Rasterized heat map plus the single 0.97 minimal-match contour."""
    cmap = plt.get_cmap(st.CMAP).copy()
    cmap.set_bad(C_OUTSIDE)
    im = ax.pcolormesh(xx, yy, arr.T, cmap=cmap, vmin=vmin, vmax=1.0,
                       shading="auto", rasterized=True)
    with np.errstate(invalid="ignore"):
        ax.contour(xx, yy, arr.T, levels=[st.MINIMAL_MATCH], colors=st.C_MM,
                   linewidths=1.7, zorder=6)
    return im


def _searched_mask(d):
    """True where the injected delay lies outside the searched delay grid."""
    t_d = amp.time_delay(d["M_Lz"][:, None], d["y"][None, :])
    return t_d > float(d["grid_t_d"].max())


def _point_mass_panel(ax, d, arr, vmin, label_levels=()):
    x = np.log10(d["M_Lz"])
    yy = d["y"]
    arr = np.where(_searched_mask(d), np.nan, arr)
    im = _mesh(ax, x, yy, arr, vmin)
    fml = 1.0 / amp.time_delay(d["M_Lz"][:, None], yy[None, :])
    cf = ax.contour(x, yy, fml.T, levels=F_ML_CONTOURS, colors=st.C_GUIDE,
                    linestyles="dashed", linewidths=1.3, zorder=5)
    for lev in label_levels:
        # A second, invisible single-level contour set carries the label, so
        # that each panel draws both guides but labels only one of them.
        cl = ax.contour(x, yy, fml.T, levels=[lev], colors="none")
        texts = ax.clabel(
            cl, fmt=lambda v: r"$f_{\mathrm{ML}}\!=\!%g$ Hz" % v,
            fontsize=9, inline=False, manual=[F_ML_LABEL_AT[lev]],
        )
        for t in texts:
            t.set_zorder(11)
            t.set_color(st.C_GUIDE)
            # clabel aligns the label with the local contour tangent; nudge it
            # clockwise so it reads flatter against the dashed guide.
            t.set_rotation(t.get_rotation() - F_ML_LABEL_TILT)
            t.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.82,
                            boxstyle="round,pad=0.12"))
    ax.set_yscale("log")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(yy[0], yy[-1])
    ax.set_xticks([1, 2, 3, 4, 5])
    return im


def _chang_refsdal_panel(ax, d, arr, vmin, four_image=False, regions=False):
    kap, gam = d["kappa"], d["gamma"]
    im = _mesh(ax, kap, gam, arr, vmin)

    k = np.linspace(kap[0], kap[-1], 200)
    ax.plot(k, 1.0 - k, color=st.C_STRUCT, lw=2.6, zorder=7,
            solid_capstyle="butt")
    ax.plot(k, 1.0 - k, color="black", lw=1.0, zorder=8,
            solid_capstyle="butt")

    if four_image:
        n_img = d["n_images"]
        if (n_img >= 4).any():
            with np.errstate(invalid="ignore"):
                ax.contour(kap, gam, (n_img >= 4).astype(float).T,
                           levels=[0.5], colors=st.C_STRUCT, linewidths=1.8,
                           linestyles="dashed", zorder=9)
    if regions:
        # The critical line is also the boundary between the two injection
        # physics: the Fermat contours close only for kappa + gamma < 1, so the
        # macro-minimum half is exact wave optics and the macro-saddle half is
        # exact geometric optics.
        for xx, yy, txt, ha in (
                (0.035, 0.105, "macro-minimum\nwave optics", "left"),
                (0.505, 0.890, "macro-saddle\ngeometric optics", "left")):
            ax.text(xx, yy, txt, transform=ax.transAxes, fontsize=9,
                    color="white", ha=ha, va="center", zorder=11,
                    linespacing=1.25,
                    bbox=dict(facecolor="black", edgecolor="none", alpha=0.55,
                              boxstyle="round,pad=0.22"))
    ax.set_xlim(kap[0], kap[-1])
    ax.set_ylim(gam[0], gam[-1])
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    return im


def main():
    ap = argparse.ArgumentParser()
    # The point-mass column searches delays out to the chirp duration, where
    # the two images stop overlapping inside the waveform; results/frozen/
    # phase3/ is the 1-500 ms grid the long-form paper and Sec. V still use.
    ap.add_argument("--wave-optics",
                    default=st.data_path("pointmass_M50", "wave_optics_M50.npz"))
    ap.add_argument(
        "--chang-refsdal",
        default=st.data_path("chang_refsdal") + "/"
                "lensmodel_changrefsdal_MLz3000_M50.npz")
    args = ap.parse_args()

    wo, wo_meta = _load(args.wave_optics)
    cr, cr_meta = _load(args.chang_refsdal)

    keep = ~_searched_mask(wo)
    arrays = (wo["recovery_unlensed"][keep], cr["recovery_unlensed"],
              wo["recovery_lensed"][keep], cr["recovery_general"])
    vmin = np.floor(100.0 * min(float(np.nanmin(a)) for a in arrays)) / 100.0

    fig, axes = plt.subplots(2, 2, figsize=(st.TEXT_WIDTH, 5.4),
                             sharex="col", sharey="col",
                             constrained_layout=True)
    # Reserve a strip on the left for the rotated row labels.
    fig.get_layout_engine().set(rect=(0.052, 0.0, 0.948, 1.0))
    for ax in axes.ravel():
        ax.set_box_aspect(1.0)

    # Only one guide contour survives the mask (the 2 Hz one is now the edge
    # of the grey), so it is labelled once, in the bottom panel.
    im = _point_mass_panel(axes[0, 0], wo, wo["recovery_unlensed"], vmin)
    _point_mass_panel(axes[1, 0], wo, wo["recovery_lensed"], vmin,
                      label_levels=(1000.0,))
    _chang_refsdal_panel(axes[0, 1], cr, cr["recovery_unlensed"], vmin,
                         regions=True)
    _chang_refsdal_panel(axes[1, 1], cr, cr["recovery_general"], vmin,
                         four_image=True)

    # Column headers carry the lens scenario; the two row labels below carry
    # the statistic, and the bottom titles the relative Morse phase used.
    axes[0, 0].set_title(r"\textbf{isolated point-mass lens}", fontsize=11.5)
    axes[0, 1].set_title(r"\textbf{Chang--Refsdal microlens}", fontsize=11.5)
    # The bottom panels carry no title: the row label names the statistic and
    # the caption names each panel's relative Morse phase -- (c) is c = ia,
    # (d) is maximized over c = a i^q, which the caption states explicitly.

    for ax, lab in zip(axes.ravel(), "abcd"):
        st.panel_label(ax, lab)

    for row in (0, 1):
        axes[row, 0].set_ylabel(r"impact parameter $y$")
        axes[row, 1].set_ylabel(r"shear $\gamma$")
    axes[1, 0].set_xlabel(r"$\log_{10}\!\left(M_{\mathrm{L}z}/M_\odot\right)$")
    axes[1, 1].set_xlabel(r"convergence $\kappa$")

    cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.95,
                        pad=0.015, aspect=32)
    # "Fitting factor" = max over the search family, of which the unlensed
    # template is the one-element case.  The contract requires the reduced
    # calculation to be labelled explicitly, since no *intrinsic* CBC
    # parameters are maximized over here; the caption does that.
    cbar.set_label(r"fitting factor", labelpad=8)
    cbar.set_ticks([0.7, 0.8, 0.9, 1.0])
    cbar.ax.axhline(st.MINIMAL_MATCH, color=st.C_MM, lw=1.7)
    cbar.ax.text(1.9, st.MINIMAL_MATCH, r"$0.97$", color=st.C_MM,
                 fontsize=9.5, ha="left", va="center",
                 transform=cbar.ax.get_yaxis_transform())
    cbar.solids.set_rasterized(True)

    # Rotated row labels naming the statistic, outside the y-axis labels.
    # Their x is *measured* against each y-label rather than guessed at: with
    # rotation=90 the text's horizontal extent is its cap height, so a fixed
    # offset silently overlaps the y-label whenever the font size changes.
    fig.canvas.draw()
    fig.set_layout_engine("none")          # freeze the axes positions
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    GAP = 0.014
    for row, txt in ((0, "unlensed template"), (1, "two-image family")):
        ax = axes[row, 0]
        ybb = ax.yaxis.label.get_window_extent(rend).transformed(inv)
        abb = ax.get_tightbbox(rend).transformed(inv)
        t = fig.text(0.0, 0.5 * (abb.y0 + abb.y1), r"\textbf{%s}" % txt,
                     rotation=90, fontsize=11.5, ha="center", va="center")
        tbb = t.get_window_extent(rend).transformed(inv)
        t.set_x(ybb.x0 - GAP - 0.5 * tbb.width)
        left = ybb.x0 - GAP - tbb.width
        print("  row %d label: x span [%.4f, %.4f], y-label starts %.4f"
              % (row, left, left + tbb.width, ybb.x0))
        if left < 0.004:
            raise SystemExit(
                "row label would run off the canvas: widen the reserved "
                "strip in fig.get_layout_engine().set(rect=...)")

    out = st.save(fig, NAME)
    print("wrote %s" % out)

    # "banded_region" since 2026-09-21; the 500 ms products predate the
    # rename and carry the bounds in the key name instead.
    wo_band = wo_meta.get("banded_region",
                          wo_meta.get("banded_region_f_ML_2_to_1000Hz"))
    fin = np.isfinite(cr["recovery_frozen"])
    sad = cr["macro_saddle"].astype(bool) & fin      # geometric-optics half
    mn = (~cr["macro_saddle"].astype(bool)) & fin    # wave-optics half
    summary = {
        "sources": [args.wave_optics, args.chang_refsdal],
        "vmin": vmin,
        "point_mass": {
            "m_tot": wo_meta["m_tot"],
            "search_grid": wo_meta["search_grid"],
            "banded_region_f_ML_2_to_1000Hz": wo_band,
        },
        "chang_refsdal": {
            "m_lz": cr_meta["m_lz"], "y": cr_meta["y"],
            "phi_gamma_deg": cr_meta["phi_gamma_deg"],
            "m_tot": cr_meta["m_tot"],
            "exclude_half_width": cr_meta["exclude_half_width"],
            "searched_domain": cr_meta["searched_domain"],
            "injection": cr_meta["injection"],
            "optics_census": cr_meta.get("optics_census"),
            "macro_saddle_cells": int(sad.sum()),
            "macro_minimum_cells": int(mn.sum()),
            "macro_minimum": {
                key: {
                    "median": float(np.nanmedian(cr[key][mn])),
                    "min": float(np.nanmin(cr[key][mn])),
                    "frac_ge_0.97": float(np.nanmean(cr[key][mn] >= 0.97)),
                }
                for key in ("recovery_unlensed", "recovery_q1",
                            "recovery_general")
            },
            "macro_saddle": {
                key: {
                    "median": float(np.nanmedian(cr[key][sad])),
                    "min": float(np.nanmin(cr[key][sad])),
                    "frac_ge_0.97": float(np.nanmean(cr[key][sad] >= 0.97)),
                }
                for key in ("recovery_unlensed", "recovery_q1",
                            "recovery_general")
            },
        },
    }
    for d in st.OUTDIRS:
        with open(os.path.join(d, NAME + "_summary.json"), "w") as fh:
            json.dump(st.relative(summary), fh, indent=2)

    caption = r"""The two-image family, evaluated by recombination, is adequate for
real signals, and is not specific to an isolated point mass. Colour is the
fitting factor of each search: the largest fraction of the injection's optimal
signal-to-noise ratio that any template of the search family collects,
$\max_{t,\theta}|z(t;\theta)|/\rho_{\rm opt}$ with
$\rho_{\rm opt}=\sqrt{(d|d)}$. The maximization runs over coalescence time and
phase and, in the bottom row, over the frozen $1000\times8$ search grid of
$(t_d,a)$ ($5339$ active points) --- \emph{not} over the injection parameters
on the axes, and not over intrinsic CBC parameters, which are held fixed
throughout. In the top row the family is the single unlensed template, so the
fitting factor reduces to the ordinary match. Both columns are recovered on
the \emph{same} search grid, unmodified.
\textbf{Left column:} an isolated point-mass microlens, with injections built
from the \emph{exact} wave-optics amplification factor and
$M_{\rm tot}=50\,\msun$; dashed contours mark $f_\ML=2$ and $1000$~Hz, which
bound the region G26 bank. Between them the two-image family with the
minimum--saddle coefficient $c=ia$ (c) is everywhere above $0.97$ (worst case %(pm_lensed_min).4f, median %(pm_lensed_med).4f) while
the unlensed search (a) clears $0.97$ over only %(pm_unlensed_frac).1f\%% of it,
with a worst case of %(pm_unlensed_min).4f. \textbf{Right column:} a point-mass
microlens embedded in the convergence and shear of a macromodel
(Chang--Refsdal), $y=0.3$, $\phi_\gamma=45^\circ$, $\MLz=3000\,\msun$. The
solid black line is the critical line $\gamma=1-\kappa$, below which the macro-image
is a minimum and above which it is a saddle; the band
$|\gamma-(1-\kappa)|<0.03$ is excluded because the magnification diverges
there. That line is also the boundary between the two injection physics: the
Fermat potential is bounded below, and its contours therefore close, only for
$\kappa+\gamma<1$, so the injections are exact \emph{wave} optics over the
%(cr_wave_cells)d macro-minimum cells and exact \emph{geometric} optics over
the %(cr_cells)d macro-saddle cells. In the macro-saddle region both images
are saddles, so the required relative Morse phase is $0^\circ$ rather than
$90^\circ$: there a search fixed at $q=1$ reaches a median of
%(cr_q1_med).3f and clears $0.97$ nowhere, whereas maximizing over $q$ (d)
reaches %(cr_gen_med).3f and clears it everywhere. The macro-minimum half is
the harder half, and for a different reason: with wave-optics injections and a
four-image region --- whose boundary is the dashed contour, and which no
two-image family represents completely --- (d) reaches a median of
%(cr_min_med).3f there against %(cr_min_ul_med).3f for the unlensed
template. The red contour and the mark on the colour bar are the $0.97$
minimal match; the colour scale is common to all four panels."""
    caption = caption % {
        "pm_lensed_min": wo_band["lensed"]["min"],
        "pm_lensed_med": wo_band["lensed"]["median"],
        "pm_unlensed_frac": 100.0 * wo_band["unlensed"]["frac_ge_0.97"],
        "pm_unlensed_min": wo_band["unlensed"]["min"],
        "cr_cells": int(sad.sum()),
        "cr_wave_cells": int(mn.sum()),
        "cr_q1_med": float(np.nanmedian(cr["recovery_q1"][sad])),
        "cr_gen_med": float(np.nanmedian(cr["recovery_general"][sad])),
        "cr_min_med": float(np.nanmedian(cr["recovery_general"][mn])),
        "cr_min_ul_med": float(np.nanmedian(cr["recovery_unlensed"][mn])),
    }
    for d in st.OUTDIRS:
        with open(os.path.join(d, NAME + "_caption.txt"), "w") as fh:
            fh.write(caption.strip() + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
