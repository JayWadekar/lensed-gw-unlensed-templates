#!/usr/bin/env python
"""Figure 3 of the paper: sensitive volume of the lensed-pair strategies.

The arms are compared at the *same* expected number of false alarms
(0.1 per observing run), so the trials factor each one pays is part of the
comparison rather than a footnote.  The figure of merit is the differential
sensitive volume dV/drho in the measure used throughout the paper
, normalized to the sphere out to the rho = 8 horizon, so the area
under each curve is that arm's volume and the legend's numbers are volume
ratios against requiring both images to be super-threshold.

Reads the stored product only and recomputes nothing.

    python scripts/plot/plot_prl_strategies.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prl_style as st  # noqa: E402

st.use()

import matplotlib.pyplot as plt  # noqa: E402

NAME = "prl_strategies"

# Arm order comes from the product's own ``plot_order``, so that a
# regenerated product with different arms drives the figure rather than being
# silently cropped to whatever this file happens to list.  An arm with no
# entry in STYLE/LABEL is a hard error, not a dropped curve.
#
# Only one deviation from ``plot_order`` is applied: the thick maximized curve
# is drawn *before* the dashed prior-weighted one, because the two differ by
# under 3% in volume and would otherwise hide one another.
#
# The MAXIMIZED arm is the headline (thick, solid, red) as of 2026-09-27: it is
# the larger volume, x4.15 against x4.04, because prior-weighting the delay
# grid costs x2.2 in trials in this regime rather than saving them.  The
# manuscript quotes it; the prior-weighted arm is the dashed green comparison.
UNDER, OVER = "long_max", "long_marg"

# Neither sub-threshold arm carries information the other lacks -- they lie
# within 3% and overlap on the page -- so the smaller one is dropped by default
# and its ratio is still recorded in the summary JSON and quoted in the
# caption.  ``--include-marg`` shows both.
EXCLUDE_BY_DEFAULT = ("long_marg",)

STYLE = {
    "both_catalog": dict(color="#1f77b4", lw=1.7, ls="-", zorder=3),
    "one_subthr":   dict(color="#ff7f0e", lw=1.7, ls="-", zorder=3),
    "long_max":     dict(color="#d62728", lw=2.1, ls="-", zorder=4),
    "long_marg":    dict(color="#2ca02c", lw=1.7, ls=(0, (4.5, 2.2)),
                         zorder=4),
}
# Deliberately verbose: the legend has to say what each strategy *is*, so the
# figure reads on its own without a pass through the text.  Any arm not listed
# falls back to the product's own ``meta["labels"]``.
#: The grid variable is a property of the PRODUCT, not of this script: a
#: product built with ``--scan intrinsic`` runs over the unlensed amplitude
#: that Eq. (10) of ``sec:faint`` is written in, and one built with
#: ``--scan observed`` runs over the brighter image's observed amplitude and
#: carries the magnification bias in its weights instead.  Labelling the
#: second as though it were the first is the error this table exists to stop.
AXIS_LABEL = {
    "intrinsic": r"unlensed amplitude $\rho_{\rm UL}$",
    "observed": r"observed amplitude of the brighter image $\rho_+$",
}

#: The three strategies are NESTED, not disjoint: every pair the first finds
#: the second finds, and every pair the second finds the third finds (the
#: per-image thresholds exceed the joint one in quadrature).  The labels are
#: therefore CUMULATIVE: each arm is the one above it plus what its own
#: "incl." names.  Read down the legend, the strategies build up.  Without
#: this the names read as three exclusive classes of source and the V/V_ref
#: column reads as a partition, which it is not.  Keep them short -- the
#: legend width is asserted below, and self-contained wordings such as
#: "both sub-threshold (incl. one super-, one sub)" do NOT fit (3.62 in
#: against 3.40 in available).
LABEL = {
    "both_catalog": "both super-threshold",
    "one_subthr":   "incl. one super, one sub",
    "long_max":     "incl. both sub-threshold",
    "long_marg":    "the same, prior-weighted",
}


def _magbias_note(rep):
    """Name the population weighting, and quote the stored alternative.

    The alternative ratio is read from the product's
    ``ratios_under_the_other_weighting``, which is the same Monte Carlo
    contracted with the other weights -- never a literal typed in here.
    """
    if not rep.get("magnification_bias"):
        return ""
    alt = (rep.get("ratios_under_the_other_weighting") or {}).get("long_marg")
    if rep.get("scan", "observed") == "intrinsic":
        # The bias is not a separate factor in this variable; saying that it
        # "is included" would be true of the physics and false of the
        # formula, which is exactly the confusion the change of variable was
        # made to remove.
        tail = ("" if alt is None else
                r" Weighting the impact-parameter average as though the "
                r"count were Euclidean in the \emph{observed} amplitude "
                r"instead would give $V/V_{\rm ref}=%.2f$" % alt)
        return (r" Because the horizons are written in the unlensed "
                r"amplitude, magnification bias is not a separate factor "
                r"here: it is absorbed by that choice of variable, and the "
                r"impact-parameter average carries the cross-section prior "
                r"$p(y)\propto y$ alone.%s." % tail)
    tail = ("" if alt is None else
            r" --- without which the plotted arm would read "
            r"$V/V_{\rm ref}=%.2f$" % alt)
    return (r" The impact-parameter average includes magnification bias: the "
            r"count is Euclidean in the \emph{intrinsic} amplitude, so a more "
            r"strongly magnified impact parameter supplies more lensed "
            r"sources%s." % tail)


def _slope_rise(rep):
    """How much the ratios grow between the shallowest and steepest slope.

    Derived from the stored sweep rather than asserted: the old caption said
    "about a factor of two", which the magnification-biased product no longer
    supports.
    """
    sw = rep.get("slope_robustness") or {}
    if not sw:
        return "an unrecorded amount"
    ks = sorted(sw, key=float, reverse=True)          # -4.0 ... -6.0
    lo, hi = sw[ks[0]]["ratios"], sw[ks[-1]]["ratios"]
    f = sorted(hi[k] / lo[k] for k in lo if k != "both_catalog")
    return r"$%d$--$%d\%%$" % (round(100 * (f[0] - 1)), round(100 * (f[-1] - 1)))


def _label(key, rep):
    return LABEL.get(key) or rep["labels"].get(key, key.replace("_", " "))


def _save(fig, name, outdirs):
    """The shared, overflow-checked writer with overridable directories."""
    return st.save(fig, name, outdirs=outdirs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz",
                    default=st.data_path("strategies", "lensed_strategies_rhoUL.npz"))
    ap.add_argument("--yscale", choices=("log", "linear"), default="linear")
    ap.add_argument("--unweighted", action="store_true",
                    help="plot dV/drho rather than the default rho dV/drho; "
                         "the weighting is what makes the area under a curve "
                         "proportional to the arm's volume on a log rho axis")
    ap.add_argument("--include-marg", action="store_true",
                    help="also show the prior-weighted (marginalized) arm")
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--outdir", action="append", default=None,
                    help="override the output directories (repeatable)")
    args = ap.parse_args()
    args.weight_by_rho = not args.unweighted
    name = args.name
    outdirs = args.outdir or list(st.OUTDIRS)

    if os.path.normpath(args.npz).startswith(
            os.path.join("results", "development")):
        print("NOTE: reading a results/development/ product, which the "
              "contract calls disposable. Freeze it before submission so "
              "this main-text figure has a content-hashed source.")

    z = np.load(args.npz, allow_pickle=True)
    rep = json.loads(str(z["meta"]))
    A = z["amplitude"]
    ratios = rep["ratios_vs_both_in_catalog"]
    vols = rep["volumes"]
    rho_ref = rep["rho_ref"]
    robust = rep["slope_robustness"]

    dropped = () if args.include_marg else EXCLUDE_BY_DEFAULT
    legend_order = [k for k in rep["plot_order"] if k not in dropped]
    unstyled = [k for k in legend_order if k not in STYLE]
    if unstyled:
        raise SystemExit(
            "the product carries arms this figure has no style for: %s. Add "
            "them to STYLE in %s rather than letting them be dropped from a "
            "main-text figure. (Labels fall back to the product's own "
            "meta['labels'], so only STYLE is mandatory.)"
            % (unstyled, os.path.basename(__file__)))
    draw_order = list(legend_order)
    if UNDER in draw_order and OVER in draw_order:
        draw_order.remove(UNDER)
        draw_order.insert(draw_order.index(OVER), UNDER)

    # A volume whose grid stopped before the arm saturated is a statement
    # about the grid edge, not about the search.  Keep the guard.
    for k in draw_order:
        d = rep["saturation_diagnostics"][k]
        assert d["eff_high"] > 0.999, (k, d)
        assert d["tail_residual_frac"] <= rep["max_truncation_allowed"], (k, d)

    fig, ax = plt.subplots(figsize=(st.COL_WIDTH, 4.0),
                           constrained_layout=True)

    handles = {}
    def curve(k):
        c = z["curve_%s" % k]
        return A * c if args.weight_by_rho else c

    for k in draw_order:
        handles[k], = ax.plot(A, curve(k), **STYLE[k])

    ax.set_xscale("log")
    ax.set_yscale(args.yscale)
    peak = max(float(curve(k).max()) for k in draw_order)
    if args.yscale == "log":
        ax.set_ylim(peak * 1e-4, peak * 4.0)
    else:
        ax.set_ylim(0.0, peak * 1.12)
    ax.set_xlim(A[0], A[-1])
    ticks = (4, 6, 8, 10, 15, 20, 30, 60)
    ax.set_xticks(ticks)
    ax.set_xticklabels([r"$%d$" % t for t in ticks])
    ax.minorticks_off()
    ax.tick_params(top=False)          # the reach axis owns the top spine
    ax.set_xlabel(AXIS_LABEL[rep.get("scan", "observed")])
    if args.weight_by_rho:
        ax.set_ylabel(r"$\rho\,\mathrm{d}V/\mathrm{d}\rho$ "
                      r"$\left[V_{\rho=%.0f}\right]$" % rho_ref)
    else:
        ax.set_ylabel(r"$\mathrm{d}V/\mathrm{d}\rho$ "
                      r"$\left[V_{\rho=%.0f}\,\rho^{-1}\right]$" % rho_ref)
    ax.grid(alpha=0.22, which="major", lw=0.5)
    ax.set_axisbelow(True)

    # Outside the axes: in the axes it covers the rising edges, which are
    # where each strategy's threshold shows.
    leg = fig.legend([handles[k] for k in legend_order],
                     [r"%s $\;V/V_{\rm ref}=%.2f$"
                      % (_label(k, rep), ratios[k])
                      for k in legend_order],
                     loc="outside lower center", ncol=1, fontsize=8.2,
                     handlelength=2.0, handletextpad=0.6, labelspacing=0.32,
                     frameon=False)

    # the physical reading of the amplitude axis
    top = ax.secondary_xaxis(
        "top", functions=(lambda a: rho_ref / np.clip(a, 1e-9, None),
                          lambda d: rho_ref / np.clip(d, 1e-9, None)))
    top.set_xlabel(r"reach $D/D_{\rho=%.0f}$" % rho_ref, labelpad=4)
    reach = (2.0, 1.0, 0.5, 0.3, 0.15)
    top.set_xticks(list(reach))
    top.set_xticklabels([r"$%g$" % t for t in reach])

    # Self-describing labels are long; check they actually fit rather than
    # discovering a clipped legend in the PDF.
    fig.canvas.draw()
    lw = leg.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    print("  legend width %.2f in of %.2f in available" % (lw, st.COL_WIDTH))
    if lw > st.COL_WIDTH - 0.02:
        raise SystemExit(
            "legend is %.2f in wide and would be clipped at %.2f in: shorten "
            "LABEL entries or drop the legend font size"
            % (lw, st.COL_WIDTH))

    out = _save(fig, name, outdirs)
    print("wrote %s" % out)

    gains = {sl: robust[sl]["marginalization_gain"] for sl in robust}
    summary = {
        "source": args.npz,
        "presentation": {
            "yscale": args.yscale,
            "ordinate": ("rho dV/drho" if args.weight_by_rho else "dV/drho"),
            "arms_shown": legend_order,
            "arms_not_shown": list(dropped),
        },
        "source_is_frozen": "results/frozen" in args.npz,
        "product_timestamp_utc": rep["timestamp_utc"],
        "product_git_commit": rep["git_commit"],
        "seed": rep["seed"],
        "rho_ref": rho_ref,
        "measure": rep["measure"],
        "a_ratio_prior_mean": rep["a_ratio_prior_mean"],
        "volumes": vols,
        "ratios_vs_both_in_catalog": ratios,
        # The per-image cuts, and the brighter image's implied horizon for the
        # catalogue-pair strategy: that cut is on the FAINTER image, so the
        # brighter one must reach it amplified by 1/a.  Derived here, never
        # typed in, because the main text quotes it.
        "per_image_cuts": rep["per_image_cuts"],
        "rho_bright_catalog_pair": (rep["per_image_cuts"]["both_catalog"]
                                    / rep["a_ratio_prior_mean"]),
        # Whether the impact-parameter average carries magnification bias, and
        # what the ratios would be without it -- the same Monte Carlo
        # contracted with the other weights.
        "scan": rep.get("scan", "observed"),
        "scan_variable": rep.get("scan_variable"),
        "magnification_bias": rep.get("magnification_bias"),
        "ratios_under_the_other_weighting":
            rep.get("ratios_under_the_other_weighting"),
        "marginalization_gain_long": rep["marginalization_gain_long"],
        "marginalization_gain_vs_slope": gains,
        "delay_trials_measured": rep["delay_trials_measured"],
        "marg_trials_saving_measured": rep["marg_trials_saving_measured"],
        "trials": rep["trials"],
        "thresholds_chi2": rep["thresholds_chi2"],
        "dof": rep["dof"],
        "derived_trials": rep.get("derived_trials"),
        "base_trials_inherited_not_measured":
            rep.get("base_trials_inherited_not_measured"),
        "changes": rep.get("changes"),
    }
    for d in outdirs:
        with open(os.path.join(d, name + "_summary.json"), "w") as fh:
            json.dump(st.relative(summary), fh, indent=2)

    caption = r"""Sensitive volume opened by %(nstrat)s lensed-pair detection
strategies, all at the same expected number of false alarms ($0.1$ per
observing run), so that each arm pays its own trials factor. The measure is the
one used throughout, $V\propto\int\varepsilon\,\rho^{-4}\dd\rho$ in the
amplitude of the horizontal axis, normalized to
the $\rho=%(rho).0f$ horizon. %(area)s The legend gives each arm's volume as a
fraction of $V_{\rm ref}$, the volume of the first arm --- requiring both
images to be super-threshold. The upper axis is the
corresponding reach. That both-super-threshold requirement is the most restrictive, because with a population-mean magnification ratio
$a=%(a).2f$ the fainter image forces the brighter one's observed amplitude to
$\gtrsim%(rho_bright).0f$. Retaining
one super-threshold event with a sub-threshold counterpart gains
$\times%(r_one).2f$. Searching for a pair with \emph{both} images
sub-threshold --- the regime no current pipeline covers, and which the
recombination makes affordable --- gains $\times%(r_marg).2f$ with the
delay marginalized over its prior, which is the search plotted. %(maxnote)s %(trials)s%(magbias)s Ratios depend
on the assumed source-count slope, rising by %(slope_rise)s as it steepens from
$-4$ to $-6$, and the amplitude grid is run to saturation at both edges so that no
ratio is set by where it was cut."""
    # Only on a *linear* ordinate with the rho weighting does visual area
    # equal the integral; claiming it otherwise is simply wrong, and the
    # log-log version of this figure did claim it.
    if args.weight_by_rho and args.yscale == "linear":
        area = (r"Curves are $\rho\,\dd V/\dd\rho$ against $\log\rho$, so "
                r"the area under each is proportional to that arm's volume.")
    else:
        area = (r"Curves are the differential contribution $\dd V/\dd\rho$; "
                r"the axes are logarithmic, so visual area is not the "
                r"integral and the volumes are quoted rather than read off.")
    # Whatever is not drawn still has to be on the record.
    if "long_marg" in dropped:
        maxnote = (r"Weighting the delay grid by its prior and summing over it "
                   r"rather than maximizing gives $\times%.2f$, within "
                   r"%.1f\%% of the curve shown; the two are "
                   r"indistinguishable at this scale and it is omitted for "
                   r"that reason."
                   % (ratios["long_marg"],
                      100.0 * abs(1.0 - rep["marginalization_gain_long"])))
    else:
        maxnote = (r"The dashed curve weights the delay grid by its prior and "
                   r"sums over it rather than maximizing, giving "
                   r"$\times%.2f$." % ratios["long_marg"])
    # The product states its own trials provenance, and it has changed once
    # already: an inherited order-of-magnitude template/time/sky estimate was
    # replaced by trials derived from T_obs, f_rms and the measured
    # delay-window N_eff.  Follow the product rather than restating the old
    # account.
    dt = rep.get("derived_trials")
    if dt:
        trials = (r"Every trials factor is derived from the observing time, "
                  r"the template's root-mean-square frequency "
                  r"($%.1f$~Hz) and the \emph{measured} delay-window trials "
                  r"count $N_{\rm eff}=%.1f\times10^{5}$, rather than "
                  r"inheriting an order-of-magnitude time and sky estimate. "
                  r"Two inputs are assumed and both are swept in the stored "
                  r"product: the number of super-threshold events, taken as $%d$, and "
                  r"the number of effectively \emph{independent} high-mass "
                  r"templates, taken as $%g$ --- charged to every strategy "
                  r"except the conditional one, whose bright event pins the "
                  r"template."
                  % (dt["f_rms_hz"], dt["N_eff_window_measured"] / 1e5,
                     dt["N_events_assumed"],
                     dt.get("N_templates_blind", 1)))
    else:
        trials = (r"The lens-dimension trials are measured rather than "
                  r"asserted; the template, time and sky trials are "
                  r"inherited.")
    caption = caption % {
        "trials": trials,
        "nstrat": {2: "two", 3: "three", 4: "four"}.get(
            len(legend_order), "%d" % len(legend_order)),
        "area": area,
        "maxnote": maxnote,
        "rho": rho_ref, "a": rep["a_ratio_prior_mean"],
        "slope_rise": _slope_rise(rep),
        "magbias": _magbias_note(rep),
        # the catalogue cut is on the FAINTER image, so the brighter one must
        # reach it amplified by 1/a -- derived, never hard-coded
        "rho_bright": (rep["per_image_cuts"]["both_catalog"]
                       / rep["a_ratio_prior_mean"]),
        "r_one": ratios["one_subthr"],
        "r_marg": ratios["long_marg"],
    }
    for d in outdirs:
        with open(os.path.join(d, name + "_caption.txt"), "w") as fh:
            fh.write(caption.strip() + "\n")

    print(json.dumps({"ratios": ratios,
                      "marginalization_gain": rep["marginalization_gain_long"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
