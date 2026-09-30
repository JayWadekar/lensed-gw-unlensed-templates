#!/usr/bin/env python
"""Appendix figures of the paper (Figs. 4--7), in the main-text style.

One 0.97 contour, one shared colour bar, panel letters, bold column headers
and rotated row labels.  These re-plot the data products in ``data/``; they do
not compute anything new.

    --which pointmass   an isolated point mass at M_tot = 11 and 100 Msun,
                        the two CBC masses Fig. 1's left column omits
    --which cr          a Chang-Refsdal microlens at M_Lz = 3000 Msun, the
                        same lens as Fig. 1's right column. Three statistics
                        in order of what they remove -- the deployed q = 1,
                        then the phase bound, then the amplitude bound -- and
                        a map of which structural limitation applies where.
                        Reads the ``--extend-a`` product, which is the
                        only one carrying ``recovery_extended``; the unlensed
                        baseline is not plotted (it is Fig. 1b) but is still
                        summarized.
    --which sis         a singular isothermal sphere, M_tot = 11, 50, 100 Msun
    --which cored       a cored isothermal sphere, r_c = 0.05, 0.15, 0.30

Grey marks cells whose delay exceeds the searched grid, t_d > 500 ms.  That
is Fig. 1's convention and the user asked these to match it, so the short-delay
edge of the domain is *not* greyed even though the quoted statistics stop
there: cells below 1 ms are drawn but not counted, exactly as in Fig. 1.  The
two sets therefore differ, and ``_check_domain`` exists to keep that
deliberate rather than accidental -- it re-derives the counted domain and
fails if it no longer matches the ``n`` the stored product recorded.

    python scripts/plot/plot_prl_appendix_lenses.py --which sis
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
from lensing import simulation as sim  # noqa: E402
from lensing import waveforms as wf  # noqa: E402

#: Lower edge of the searched delay grid.  The upper edge is read from each
#: product's own ``grid_t_d``; both together are the "searched delay domain".
T_D_MIN = 1.0e-3

#: Cells the search grid cannot reach at all: the delay is longer than the
#: grid's last point, so no template of the family can place the second image.
#: Same grey and same meaning as Fig. 1.
C_OUTSIDE = "0.82"

_MASS_X = r"$\log_{10}\!\left(M_{\mathrm{L}z}/M_\odot\right)$"
_ME_X = r"$\log_{10}\!\left(M_E/M_\odot\right)$"

_UNLENSED = ("recovery_unlensed", "unlensed template")
_DEPLOYED = ("recovery_lensed", "two-image family")
_GENERAL = ("recovery_general", r"generalized, $\max_q$")

PRESETS = {
    "pointmass": dict(
        kind="mass",
        # phase3_morse rather than phase3: same grid, same products, but it
        # also carries recovery_general, which the third row needs.
        indir=st.data_path("pointmass_M11_M100"),
        files=["wave_optics_M11.npz", "wave_optics_M100.npz"],
        # No generalized row: it is visually identical to the deployed one
        # here (a minimum--saddle pair is what q = 1 is for), so it goes in
        # the caption as a number instead.  Still summarized, via extra_stats.
        rows=[_UNLENSED, _DEPLOYED],
        extra_stats=["recovery_general"],
        col=lambda m: r"$M_{\rm tot}=%g\,M_\odot$" % m["m_tot"],
        xlabel=_MASS_X, ylabel=r"impact parameter $y$",
        xticks=[1, 2, 3, 4, 5],
        title="Isolated point-mass lens",
        out="prl_app_pointmass", height=5.75,
        chirp=True,
    ),
    "cr": dict(
        kind="cr",
        indir=st.data_path("chang_refsdal_extended_a"),
        files=["lensmodel_changrefsdal_MLz3000_M50.npz"],
        # Two statistic panels, not three: lifting the amplitude bound is
        # measured below to move the median by 0.000 over the whole domain and
        # to clear no additional cell, so a third map would be a copy of the
        # second.  It goes in the caption as a number, as in the point-mass
        # figure.  The structure panel is what shows the fainter-first cells.
        panels=[("recovery_q1", r"deployed family, $c=ia$"),
                (_GENERAL[0], r"free relative phase, $\max_q$")],
        extra_stats=["recovery_unlensed", "recovery_extended"],
        xlabel=r"convergence $\kappa$", ylabel=r"shear $\gamma$",
        xticks=[0.0, 0.2, 0.4, 0.6, 0.8],
        title=r"Chang--Refsdal microlens, $M_{\mathrm{L}z}=3000\,M_\odot$",
        out="prl_app_cr", height=3.05,
    ),
    "sis": dict(
        kind="mass",
        indir=st.data_path("isothermal"),
        files=["lensmodel_sis_M11.npz", "lensmodel_sis_M50.npz",
               "lensmodel_sis_M100.npz"],
        rows=[_UNLENSED, _DEPLOYED],
        col=lambda m: r"$M_{\rm tot}=%g\,M_\odot$" % m["m_tot"],
        xlabel=_ME_X, ylabel=r"source position $y$",
        xticks=[2, 3, 4, 5],
        title="Singular isothermal sphere",
        out="prl_app_sis", height=4.75,
    ),
    "cored": dict(
        kind="mass",
        indir=st.data_path("isothermal"),
        files=["lensmodel_cored_rc0.05_M50.npz",
               "lensmodel_cored_rc0.15_M50.npz",
               "lensmodel_cored_rc0.3_M50.npz"],
        rows=[_UNLENSED,
              ("recovery_lensed", r"deployed family, $q=1$"), _GENERAL],
        col=lambda m: r"core radius $r_c=%.2f$" % m["core_radius"],
        xlabel=_ME_X, ylabel=r"source position $y$",
        xticks=[2, 3, 4, 5],
        title="Cored isothermal sphere",
        out="prl_app_cored", height=6.75,
        three_image=True,
    ),
}


def _load(indir, fn):
    d = np.load(os.path.join(indir, fn), allow_pickle=False)
    return d, json.loads(str(d["meta"]))


def _delay_map(d):
    """Injected time delay per cell, however the product stores it."""
    if "t_d" in d.files:
        return d["t_d"]
    return amp.time_delay(d["M_Lz"][:, None], d["y"][None, :])


def _searched(d):
    """True on cells inside the searched delay domain.

    What the captions and Table II count.  Narrower than what is drawn: the
    short-delay edge is a statistics cut, not a reason to hide a cell.
    """
    t_d = _delay_map(d)
    with np.errstate(invalid="ignore"):
        return (np.isfinite(t_d) & (t_d >= T_D_MIN)
                & (t_d <= float(d["grid_t_d"].max())))


def _reachable(d):
    """True on cells the search grid can reach at all: what is drawn.

    Only the long-delay edge, as in Fig. 1.  Beyond it the grid has no
    template that can place the second image, so a colour there would report
    on a configuration the family was never offered.
    """
    t_d = _delay_map(d)
    with np.errstate(invalid="ignore"):
        return np.isfinite(t_d) & (t_d <= float(d["grid_t_d"].max()))


def _check_domain(d, meta, fn):
    """Fail loudly if the plotted domain is not the counted one.

    The caption quotes fractions computed by the simulation script over its
    own notion of the searched domain.  If this script's mask disagreed, the
    figure and its numbers would describe different sets of cells and nothing
    in the build would notice.
    """
    for key in ("searched_domain_t_d_1ms_to_500ms", "searched_domain"):
        if key in meta:
            n = meta[key]["unlensed"]["n"]
            break
    else:
        return
    got = int(_searched(d).sum())
    if got != n:
        raise SystemExit(
            "%s: plotted domain has %d cells but the product counted %d; the "
            "mask here and the statistics in the caption would disagree"
            % (fn, got, n))


def _chirp_time(meta):
    # Stored by the simulation; recomputing it needs PyCBC.
    for m in (meta, meta.get("segment", {})):
        if "chirp_time_s" in m:
            return float(m["chirp_time_s"])
    keep = wf.AnalysisConfig.__dataclass_fields__
    cfg = {k: v for k, v in meta["config"].items() if k in keep}
    return sim.chirp_time(wf.AnalysisConfig(**cfg))


def _mesh(ax, xx, yy, arr, vmin):
    cmap = plt.get_cmap(st.CMAP).copy()
    cmap.set_bad(C_OUTSIDE)
    im = ax.pcolormesh(xx, yy, arr.T, cmap=cmap, vmin=vmin, vmax=1.0,
                       shading="auto", rasterized=True)
    with np.errstate(invalid="ignore"):
        ax.contour(xx, yy, arr.T, levels=[st.MINIMAL_MATCH], colors=st.C_MM,
                   linewidths=1.7, zorder=6)
    return im


def _mass_panel(ax, d, arr, vmin, cfgp, meta):
    """Recovery over (lens mass, source position), grey outside the domain."""
    x = np.log10(d["M_Lz"] if "M_Lz" in d.files else d["M"])
    yy = d["y"]
    im = _mesh(ax, x, yy, np.where(_reachable(d), arr, np.nan), vmin)

    if cfgp.get("three_image"):
        # Where the central maximum appears.  Worth keeping even though every
        # coloured cell is a three-image cell: the boundary runs on into the
        # grey, and it is the structure, not the delay cut, that puts it there.
        n_img = d["n_images"]
        if (n_img >= 3).any():
            with np.errstate(invalid="ignore"):
                for col, lw in ((st.C_STRUCT, 2.2), ("black", 0.8)):
                    ax.contour(x, yy, (n_img >= 3).astype(float).T,
                               levels=[0.5], colors=col, linewidths=lw,
                               zorder=7)
    if cfgp.get("chirp"):
        # Beyond the chirp duration the two images no longer overlap inside
        # the waveform.  For M_tot = 11 Msun this is off scale (16.4 s); for
        # 100 Msun it cuts the searched domain, which the caption states.
        t_c = _chirp_time(meta)
        t_d = _delay_map(d)
        if np.nanmin(t_d) < t_c < np.nanmax(t_d):
            # Double-stroked: the line crosses yellow, grey and dark blue in
            # one panel, and a single colour is invisible on one of them.
            with np.errstate(invalid="ignore"):
                for col, lw in ((st.C_STRUCT, 2.6), ("black", 1.1)):
                    ax.contour(x, yy, t_d.T, levels=[t_c], colors=col,
                               linewidths=lw, linestyles="dashed", zorder=8)
    ax.set_yscale("log")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(yy[0], yy[-1])
    ax.set_xticks(cfgp["xticks"])
    return im


def _cr_panel(ax, d, arr, vmin, cfgp, meta, four_image=False, regions=False):
    kap, gam = d["kappa"], d["gamma"]
    im = _mesh(ax, kap, gam, arr, vmin)

    k = np.linspace(kap[0], kap[-1], 200)
    ax.plot(k, 1.0 - k, color=st.C_STRUCT, lw=2.6, zorder=7,
            solid_capstyle="butt")
    ax.plot(k, 1.0 - k, color="black", lw=1.0, zorder=8, solid_capstyle="butt")

    if four_image and (d["n_images"] >= 4).any():
        with np.errstate(invalid="ignore"):
            ax.contour(kap, gam, (d["n_images"] >= 4).astype(float).T,
                       levels=[0.5], colors=st.C_STRUCT, linewidths=1.8,
                       linestyles="dashed", zorder=9)
    if regions:
        for xx, yy, txt in ((0.035, 0.105, "macro-minimum\nwave optics"),
                            (0.505, 0.890, "macro-saddle\ngeometric optics")):
            ax.text(xx, yy, txt, transform=ax.transAxes, fontsize=9,
                    color="white", ha="left", va="center", zorder=11,
                    linespacing=1.25,
                    bbox=dict(facecolor="black", edgecolor="none", alpha=0.55,
                              boxstyle="round,pad=0.22"))
    ax.set_xlim(kap[0], kap[-1])
    ax.set_ylim(gam[0], gam[-1])
    ax.set_xticks(cfgp["xticks"])
    ax.set_yticks(cfgp["xticks"])
    return im


#: Panel (d) of the Chang-Refsdal figure: which structural limitation applies.
#: Categorical rather than a fourth heat map, so the figure needs no second
#: colour bar -- and the three categories are what the caption has to separate
#: anyway.  ``a`` here is the amplitude ratio of the second-arriving image to
#: the first, so ``a > 1`` is exactly "the fainter image arrives first".
STRUCT_CATS = (
    (r"2 images", "#cfe0ee"),
    (r"4 images, brighter first", "#f2b880"),
    (r"4 images, \emph{fainter} first", "#a83232"),
)


def _structure_map(d):
    """0 = two images, 1 = four images with a<=1, 2 = four images with a>1."""
    fin = np.isfinite(d["recovery_unlensed"])
    cat = np.full(d["n_images"].shape, np.nan)
    with np.errstate(invalid="ignore"):
        four = d["n_images"] >= 4
        faint = d["a2"] > 1.0
    cat[fin & ~four] = 0.0
    cat[fin & four & ~faint] = 1.0
    cat[fin & four & faint] = 2.0
    return cat


def _build_cr(cfgp):
    """The Chang-Refsdal figure: three statistics plus the structure map."""
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    d, meta = _load(cfgp["indir"], cfgp["files"][0])
    _check_domain(d, meta, cfgp["files"][0])
    panels = cfgp["panels"]

    # vmin over the plotted panels *and* the unplotted unlensed baseline, so
    # the scale is the one Fig. 1 would have used on the same lens.
    keys = [k for k, _t in panels] + list(cfgp.get("extra_stats", ()))
    vmin = np.floor(100.0 * min(float(np.nanmin(d[k])) for k in keys)) / 100.0
    cat = _structure_map(d)

    fig, axes = plt.subplots(1, len(panels) + 1,
                             figsize=(st.TEXT_WIDTH, cfgp["height"]),
                             sharex=True, sharey=True,
                             constrained_layout=True)
    for ax in axes.ravel():
        ax.set_box_aspect(1.0)

    im = None
    for i, (ax, (key, tag)) in enumerate(zip(axes.ravel()[:len(panels)], panels)):
        # Structure overlays stay in (d).  Tried them on the statistic panels
        # and the fainter-first boundary reads as a second 0.97 contour: two
        # reds, one of which is a threshold and one of which is not.  The
        # four-image boundary is kept on (c) alone, as in Fig. 1(d).
        im = _cr_panel(ax, d, d[key], vmin, cfgp, meta,
                       four_image=(i == len(panels) - 1), regions=False)
        ax.set_title(tag, fontsize=10.5)

    axd = axes.ravel()[-1]
    cmap = mcolors.ListedColormap([c for _t, c in STRUCT_CATS])
    cmap.set_bad(C_OUTSIDE)
    axd.pcolormesh(d["kappa"], d["gamma"], np.ma.masked_invalid(cat).T,
                   cmap=cmap, norm=mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5],
                                                        cmap.N),
                   shading="auto", rasterized=True)
    k = np.linspace(d["kappa"][0], d["kappa"][-1], 200)
    axd.plot(k, 1.0 - k, color="black", lw=1.0, zorder=8, solid_capstyle="butt")
    axd.set_xlim(d["kappa"][0], d["kappa"][-1])
    axd.set_ylim(d["gamma"][0], d["gamma"][-1])
    axd.set_title(r"image structure", fontsize=10.5)
    axd.legend(handles=[Patch(facecolor=c, edgecolor="0.3", lw=0.5, label=t)
                        for t, c in STRUCT_CATS],
               loc="lower left", fontsize=6.6, frameon=True, framealpha=0.93,
               borderpad=0.28, handlelength=1.0, handleheight=0.8,
               handletextpad=0.45, labelspacing=0.28).set_zorder(12)

    for ax, lab in zip(axes.ravel(), "abcdef"):
        st.panel_label(ax, lab)
    axes.ravel()[0].set_ylabel(cfgp["ylabel"])
    for ax in axes.ravel():
        ax.set_xlabel(cfgp["xlabel"])

    cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.95, pad=0.015,
                        aspect=32)
    cbar.set_label(r"fitting factor", labelpad=8)
    cbar.set_ticks([0.7, 0.8, 0.9, 1.0])
    cbar.ax.axhline(st.MINIMAL_MATCH, color=st.C_MM, lw=1.7)
    cbar.ax.text(1.9, st.MINIMAL_MATCH, r"$0.97$", color=st.C_MM, fontsize=9.5,
                 ha="left", va="center",
                 transform=cbar.ax.get_yaxis_transform())
    cbar.solids.set_rasterized(True)
    fig.suptitle(r"{\boldmath %s}" % cfgp["title"], fontsize=13)

    out = st.save(fig, cfgp["out"])
    print("wrote %s" % out)

    fin = np.isfinite(d["recovery_unlensed"])
    groups = {"two_image": fin & (cat == 0.0),
              "four_image_brighter_first": fin & (cat == 1.0),
              "four_image_fainter_first": fin & (cat == 2.0),
              "macro_saddle": fin & d["macro_saddle"].astype(bool)}
    summary = {
        "which": "cr", "vmin": vmin,
        "sources": [os.path.join(cfgp["indir"], cfgp["files"][0])],
        "m_lz": meta["m_lz"], "y": meta["y"],
        "phi_gamma_deg": meta["phi_gamma_deg"], "m_tot": meta["m_tot"],
        "a2_over_fainter_first": {
            "median": float(np.median(d["a2"][groups["four_image_fainter_first"]])),
            "max": float(np.max(d["a2"][groups["four_image_fainter_first"]])),
        },
        "extend_a": meta.get("extend_a"),
        "extended_amplitude_axis": meta.get("extended_amplitude_axis"),
        "ext_uses_a_gt_1_cells": int(np.sum(d["ext_uses_a_gt_1"]))
        if "ext_uses_a_gt_1" in d.files else None,
        "groups": {
            g: dict({"n": int(m.sum())}, **{
                key: {"median": float(np.median(d[key][m])),
                      "min": float(np.min(d[key][m])),
                      "frac_ge_0.97": float(np.mean(d[key][m] >= 0.97))}
                for key in keys})
            for g, m in groups.items()
        },
    }
    for dd in st.OUTDIRS:
        with open(os.path.join(dd, cfgp["out"] + "_summary.json"), "w") as fh:
            json.dump(st.relative(summary), fh, indent=2)
    print(json.dumps(summary["groups"], indent=1))
    return 0


def _row_labels(fig, axes, texts, gap=0.014):
    """Rotated labels naming the statistic, left of the y-axis labels.

    Their x is measured against each y-label rather than guessed at: with
    rotation 90 the text's horizontal extent is its cap height, so a fixed
    offset silently overlaps the y-label when the font size changes.  Lifted
    from ``plot_prl_effectualness.py``, which bit that once.
    """
    fig.canvas.draw()
    fig.set_layout_engine("none")          # freeze the axes positions
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, txt in zip(axes[:, 0], texts):
        ybb = ax.yaxis.label.get_window_extent(rend).transformed(inv)
        abb = ax.get_tightbbox(rend).transformed(inv)
        t = fig.text(0.0, 0.5 * (abb.y0 + abb.y1), r"\textbf{%s}" % txt,
                     rotation=90, fontsize=11.5, ha="center", va="center")
        tbb = t.get_window_extent(rend).transformed(inv)
        t.set_x(ybb.x0 - gap - 0.5 * tbb.width)
        if ybb.x0 - gap - tbb.width < 0.004:
            raise SystemExit(
                "row label would run off the canvas: widen the reserved "
                "strip in fig.get_layout_engine().set(rect=...)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="sis", choices=sorted(PRESETS))
    args = ap.parse_args()
    cfgp = PRESETS[args.which]
    if args.which == "cr":
        return _build_cr(cfgp)

    data = []
    for fn in cfgp["files"]:
        d, meta = _load(cfgp["indir"], fn)
        _check_domain(d, meta, fn)
        data.append((fn, d, meta))

    rows, ncol, nrow = cfgp["rows"], len(data), len(cfgp["rows"])
    is_cr = cfgp["kind"] == "cr"

    def shown(d, key):
        return np.where(_reachable(d), d[key], np.nan)

    vmin = np.floor(100.0 * min(
        float(np.nanmin(shown(d, k))) for _f, d, _m in data for k, _t in rows
    )) / 100.0

    fig, axes = plt.subplots(nrow, ncol, figsize=(st.TEXT_WIDTH,
                                                  cfgp["height"]),
                             sharex=True, sharey=True,
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    fig.get_layout_engine().set(rect=(0.052, 0.0, 0.948, 1.0))
    for ax in axes.ravel():
        # Square unless the row count would push the figure off the page:
        # three rows of two columns at text width do not fit square.
        ax.set_box_aspect(cfgp.get("aspect", 1.0))

    im = None
    for col, (_fn, d, meta) in enumerate(data):
        for row, (key, _tag) in enumerate(rows):
            ax = axes[row, col]
            if is_cr:
                im = _cr_panel(ax, d, d[key], vmin, cfgp, meta,
                               four_image=(row == nrow - 1),
                               regions=(row == 0 and col == 0))
            else:
                im = _mass_panel(ax, d, d[key], vmin, cfgp, meta)
        # ``\textbf`` does not reach inside math mode, and every column
        # header here is math; ``\boldmath`` is what bolds the symbols.
        axes[0, col].set_title(r"{\boldmath %s}" % cfgp["col"](meta),
                               fontsize=11.5)

    for ax, lab in zip(axes.ravel(order="C"),
                       "abcdefghijkl"[:nrow * ncol]):
        st.panel_label(ax, lab)
    for row in range(nrow):
        axes[row, 0].set_ylabel(cfgp["ylabel"])
    for col in range(ncol):
        axes[nrow - 1, col].set_xlabel(cfgp["xlabel"])

    cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.95, pad=0.015,
                        aspect=32)
    cbar.set_label(r"fitting factor", labelpad=8)
    cbar.set_ticks([0.7, 0.8, 0.9, 1.0])
    cbar.ax.axhline(st.MINIMAL_MATCH, color=st.C_MM, lw=1.7)
    cbar.ax.text(1.9, st.MINIMAL_MATCH, r"$0.97$", color=st.C_MM,
                 fontsize=9.5, ha="left", va="center",
                 transform=cbar.ax.get_yaxis_transform())
    cbar.solids.set_rasterized(True)
    if cfgp.get("title"):
        fig.suptitle(r"\textbf{%s}" % cfgp["title"], fontsize=13)

    _row_labels(fig, axes, [t for _k, t in rows])

    out = st.save(fig, cfgp["out"])
    print("wrote %s" % out)

    summary = {
        "which": args.which,
        "sources": [os.path.join(cfgp["indir"], f) for f in cfgp["files"]],
        "rows": [k for k, _t in rows],
        "vmin": vmin,
        "t_d_domain_s": [T_D_MIN, float(data[0][1]["grid_t_d"].max())],
        "panels": {},
    }
    for fn, d, meta in data:
        ent = {"search_grid": meta["search_grid"]}
        for key in ("searched_domain_t_d_1ms_to_500ms", "searched_domain"):
            if key in meta:
                ent["searched_domain"] = meta[key]
        if cfgp.get("chirp"):
            ent["chirp_time_s"] = _chirp_time(meta)
            ent["cells_beyond_chirp_in_domain"] = int(
                (_searched(d) & (_delay_map(d) > _chirp_time(meta))).sum())
        for key in [k for k, _t in rows] + list(cfgp.get("extra_stats", ())):
            a = d[key][_searched(d)]
            ent[key] = {"median": float(np.median(a)), "min": float(np.min(a)),
                        "frac_ge_0.97": float(np.mean(a >= 0.97))}
        summary["panels"][fn] = ent
    for dd in st.OUTDIRS:
        with open(os.path.join(dd, cfgp["out"] + "_summary.json"), "w") as fh:
            json.dump(st.relative(summary), fh, indent=2)
    print(json.dumps(summary["panels"], indent=1)[:1800])
    return 0


if __name__ == "__main__":
    sys.exit(main())
