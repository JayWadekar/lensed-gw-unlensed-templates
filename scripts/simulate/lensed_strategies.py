#!/usr/bin/env python
"""Sensitive volume of three strategies for detecting separated lensed pairs.

Produces the data for Fig. 3 and the simulated column of Table I of the paper
(Sec. V).  The strategy-comparison framework follows B. Zackay: each strategy
has its own trials factor, every threshold is solved at the same expected
number of false alarms (0.1 per observing run), and the figure of merit is the
pair-detection efficiency weighted by a Euclidean source count.

Strategies (nested: each detects everything the one above it does)
------------------------------------------------------------------
``both_catalog``  both images super-threshold, found by the ordinary search;
``one_subthr``    one super-threshold image plus a targeted search for its
                  sub-threshold counterpart;
``long_max``      the two-image search of this paper: no per-image
                  requirement, only ``|z_1|^2 + |z_2|^2`` above threshold,
                  maximized over the unknown delay (the arm quoted in Sec. V);
``long_marg``     the same, marginalized over the delay prior (not in the
                  paper's figures).

Model
-----
* The lens population is an isolated point mass whose 10--60 minute delay
  window fixes ``M_Lz`` (``lensing.strong_pair.MonoMassPopulation``), with the
  area prior ``p(y) ~ y`` sampled at ``--n-y`` quadrature nodes.  The two images
  have amplitudes ``rho_UL sqrt(mu_+)`` and ``rho_UL sqrt(|mu_-|)``.
* Each image is drawn as a *complex* SNR (unit complex Gaussian noise), so a
  single image carries 2 degrees of freedom and the two-image sum 4; every
  threshold uses its own statistic's dof.
* Trials: ``T_obs f_rms N_t`` for the single-image search, times the measured
  number of independent delays ``N_delay = 1.73e5`` in the window for the
  two-image search, and ``N_events N_delay`` for the targeted search.
* Volume: ``V ~ int eps(rho_UL) rho_UL^-4 drho_UL`` on a grid that must reach
  saturation at both ends; the script refuses to write a product whose
  truncated tail exceeds 0.1%.
* Sweeps of the source-count slope, antenna response, number of events and
  number of templates are stored alongside the main result.

Run (about 10 minutes on one core)::

    python scripts/simulate/lensed_strategies.py --scan intrinsic --a-min 2.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np                                                # noqa: E402
from scipy import stats                                           # noqa: E402
from scipy.optimize import brentq                                 # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from lensing import provenance as prov                            # noqa: E402
from lensing import strong_pair as sp                             # noqa: E402

#: Volumes are quoted in units of the sphere reached by a perfect search
#: thresholded at ``RHO_REF``, so ``V = 1`` means "as sensitive as eps = 1
#: above ``RHO_REF``".
RHO_REF = 8.0

#: A strategy whose unmodelled off-grid tail exceeds this fraction of its
#: volume is refused rather than reported.
MAX_TRUNCATION = 1e-3


def threshold_chi2(n_trials, budget, dof=4):
    """Statistic value whose expected false-alarm count is ``budget``.

    ``dof`` is the statistic's own degree-of-freedom count: two for a single
    image's ``|z|^2`` (a complex SNR) and four for the two-image
    incoherent sum.
    """
    return brentq(lambda z: stats.chi2.sf(z + dof, dof) * n_trials - budget,
                  1.0, 4000.0)


#: --- every trials factor below is derived from this project's own numbers ---
#:
#: The earlier version inherited order-of-magnitude estimates for the
#: template, time and sky dimensions.  For the strategy that conditions on an
#: already detected bright image those estimates were wrong by four orders of
#: magnitude, because conditioning **pins** the template, the sky position and
#: the arrival time -- only the delay remains free.  Everything is now built
#: from three measured or declared quantities.

#: Observing time.  One year, the unit in which a false-alarm *rate* is quoted.
T_OBS = 3.156e7

#: Zero-up-crossing rate of the matched-filter output, i.e. the rate at which
#: the SNR series produces independent samples.  Measured for the fiducial
#: template (``strong_pair.f_moments``).
F_RMS = 121.88

#: Effective independent trials of the 600-3600 s delay window, MEASURED by
#: the faint-image study: 1.73e5, or 0.47 of ``T_window f_rms``, the deficit
#: being the ``sigma(y)`` weighting which localizes the search.
N_EFF_WINDOW = 1.73e5

#: Independent templates a blind search must cover.  This work injects one
#: fixed intrinsic template, but a real search scans a bank, and the bank is
#: what a blind strategy pays for.  A high-mass-only bank is sparse -- the
#: waveforms are short and the parameter space small -- so ``1e4`` is the
#: scale for the triggers this comparison is about.
#:
#: It does **not** apply uniformly, which is the point.  A search conditioned on
#: a detected bright image already knows the intrinsic parameters, so it pays
#: ``1e4`` is the scale quoted for binary-black-hole banks in the search
#: literature.  Only ``2 ln N`` enters, so a decade is worth ``4.6`` in
#: ``rho^2``; the sweep below brackets 1 to 1e5, so the reported value is not
#: at an edge.
#:
#: ``N_TEMPLATES_CONDITIONAL`` instead: a handful at most, to allow for the
#: bright event's own parameter uncertainty, and set to one here because a
#: factor of a few is invisible against the logarithm.
N_TEMPLATES = 10000
N_TEMPLATES_CONDITIONAL = 1

#: Detected events per observing year that a conditional search can be run on.
#: An O4-scale assumption, and the one input that is neither measured nor
#: derived; the sensitivity to it is swept and stored.
N_EVENTS = 100

#: Single-image and (t_+, t_d) pair trials, including the bank a blind search
#: must scan.
N_SINGLE = T_OBS * F_RMS * N_TEMPLATES
N_PAIRS = N_SINGLE * N_EFF_WINDOW

#: Trials saved by weighting the lens grid by its prior rather than maximizing
#: over it, measured in this regime against a 3e6-block rich pool
#:.  Below one means weighting is worse; see the
#: discussion there for why this regime differs from the compact overlapping
#: prior of the frozen Phase 4 calibration.
MARG_TRIALS_SAVING = 0.445
MARG_SAVING_MEASURED_AT_FAP = 1e-3

#: Each strategy is calibrated on **its own dominant search**, which is what
#: makes the comparison fair without pretending the strategies search the same
#: space:
#:
#: ``both_catalog``  both images must be independently confident detections,
#:      so the relevant search is the ordinary single-image one over
#:      ``N_SINGLE`` trials.
#: ``one_subthr``    the bright image is such a detection; the faint one is
#:      then sought over the delay window of each detected event, i.e.
#:      ``N_EVENTS * N_EFF_WINDOW`` trials -- with no template or sky freedom,
#:      because the bright event fixes both.
#: ``long_max``      neither image is independently significant, so the search
#:      runs over ``N_PAIRS`` and the statistic carries four degrees of freedom.
#: ``long_marg``     the same, with the lens grid prior-weighted.
STRATEGIES = {
    "both_catalog":   dict(trials=N_SINGLE, dof=2,
                           response="unit", rank="both_above",
                           label="Both in catalogue"),
    "one_subthr":     dict(trials=N_TEMPLATES_CONDITIONAL * N_EVENTS
                           * N_EFF_WINDOW, dof=2,
                           response="unit", rank="one_plus_one",
                           label="One in catalogue, one sub-threshold"),
    "long_max":       dict(trials=N_PAIRS, dof=4,
                           response="unit", rank="joint",
                           label="Both sub-threshold, unweighted max"),
    "long_marg":      dict(trials=N_PAIRS / MARG_TRIALS_SAVING, dof=4,
                           response="unit", rank="joint",
                           label="Both sub-threshold, prior-weighted"),
}

ORDER = ("both_catalog", "one_subthr", "long_max", "long_marg")


def draw_pair(rng, amp, a_ratio, n, response="unit"):
    """The two images' observed ``|z|^2``, ``(n, 2)``.

    Each image's SNR is complex, so ``|z|^2`` carries two degrees of freedom
: the signal enters one quadrature, and both are noisy.

    ``response`` scales each image's amplitude by an antenna factor.  The
    default is ``unit`` -- **no antenna factor at all** -- because the factor
    multiplies every strategy's horizon identically, scales all volumes by
    ``u^3``, and therefore cancels in every ratio reported here.  The stored
    ``response_robustness_ratios`` shows the ratios moving by under ``6%``
    across the alternatives while the absolute volume moves by ``2.3x``.
    Carrying it would complicate the closed form of the manuscript's
    ``sec:envelope`` for nothing, so ``amp`` is the observed amplitude.

    ``indep``   ``U(0.7,1)`` per image -- the Earth has rotated between them
    ``common``  one ``U(0.7,1)`` draw shared by both, for a short delay
    ``mean``    ``0.85 x U(0.99,1)``, i.e. pinned at the mean of ``U(0.7,1)``
    """
    if response == "unit":
        u = np.ones((n, 2))
    elif response == "indep":
        u = rng.uniform(0.7, 1.0, size=(n, 2))
    elif response == "common":
        u = np.repeat(rng.uniform(0.7, 1.0, size=(n, 1)), 2, axis=1)
    elif response == "mean":
        u = 0.85 * rng.uniform(0.99, 1.0, size=(n, 2))
    else:
        raise ValueError(response)
    mean = np.column_stack([amp * u[:, 0], a_ratio * amp * u[:, 1]])
    re = mean + rng.normal(0.0, 1.0, size=(n, 2))
    im = rng.normal(0.0, 1.0, size=(n, 2))
    return re ** 2 + im ** 2


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-samples", type=int, default=200000)
    ap.add_argument("--budget", type=float, default=0.1,
                    help="expected false alarms, identical for every strategy")
    ap.add_argument("--slope", type=float, default=-4.0,
                    help="source-count slope dN/dA.  -4 is Euclidean, the "
                         "measure used throughout the paper")
    ap.add_argument("--slope-robustness", type=float, nargs="*",
                    default=(-4.0, -4.5, -5.0, -5.5, -6.0),
                    help="the ratios depend on the slope; record the sweep")
    ap.add_argument("--n-y", type=int, default=8)
    ap.add_argument("--a-min", type=float, default=4.0,
                    help="low edge; efficiency must be negligible here")
    ap.add_argument("--a-max", type=float, default=60.0,
                    help="high edge; every arm must be saturated here, which "
                         "for the both-in-catalogue reference needs A > 32 "
                         "")
    ap.add_argument("--a-step", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--scan", choices=("intrinsic", "observed"),
                    default="intrinsic",
                    help="which amplitude the grid runs over.  `intrinsic` "
                         "scans rho_UL, the SNR the source would have "
                         "unlensed, and scales each image by sqrt(mu_+) and "
                         "sqrt(|mu_-|); the impact-parameter average then "
                         "carries p(y) ~ y alone.  `observed` scans the "
                         "brighter image's rho_+ = sqrt(mu_+) rho_UL and "
                         "carries the magnification bias mu_+^{(s-1)/2} in "
                         "the weights instead.  The two are the SAME integral "
                         "under rho_+ = sqrt(mu_+) rho_UL "
                         "and reproduce each other's ratios; `intrinsic` is "
                         "the manuscript's convention because it is the one "
                         "Eq. (10) of sec:faint is written in.")
    ap.add_argument("--no-mu-weight", dest="mu_weight", action="store_false",
                    help="drop magnification bias, i.e. take the observed "
                         "brighter-image count to be the same at every impact "
                         "parameter.  The default INCLUDES the bias; see "
                         "_pop_weights.")
    ap.set_defaults(mu_weight=True)
    ap.add_argument("--out", default="regenerated/strategies/lensed_strategies_rhoUL.npz")
    args = ap.parse_args(argv)

    thr = {k: threshold_chi2(v["trials"], args.budget, dof=v["dof"])
           for k, v in STRATEGIES.items()}

    pop = sp.MonoMassPopulation.solve()
    y_nodes, y_w = sp.prior_nodes(pop, args.n_y, kind="area")
    a_ratio = pop.a(y_nodes)
    mu_plus = np.asarray(pop.mu(y_nodes))[0]
    amp_scale = (np.sqrt(mu_plus) if args.scan == "intrinsic"
                 else np.ones_like(mu_plus))

    A = np.arange(args.a_min, args.a_max, args.a_step)
    a_end = float(A[-1] + args.a_step)
    weight = (A / RHO_REF) ** args.slope

    def _bias_exponent(slope, mu_weight):
        """Power of ``mu_+`` in the impact-parameter weight.

        The physically correct weight depends on which amplitude the grid
        scans, and the two choices are the same integral: under
        ``rho_+ = sqrt(mu_+) rho_UL`` the Jacobian and the ``rho^-s`` measure
        convert ``mu_+^{(s-1)/2} drho_+`` into ``drho_UL`` exactly.  Dropping
        the bias (``--no-mu-weight``) is the naive convention, and it is the
        same subtraction in either variable.
        """
        e = (abs(float(slope)) - 1.0) / 2.0
        return (e if args.scan == "observed" else 0.0) - (0.0 if mu_weight
                                                          else e)

    def _pop_weights(slope):
        """Normalized population weight per impact-parameter node.

        The lensing cross-section prior ``p(y) ~ y`` times **magnification
        bias**.  The simulation scans the *observed* brighter-image amplitude
        ``rho_+ = sqrt(mu_+) rho_0``; for a source count ``dN/drho_0 ~
        rho_0^-s`` the observed count at fixed ``rho_+`` is
        ``mu_+^{(s-1)/2} rho_+^-s``, so a more strongly magnified impact
        parameter contributes proportionally more lensed sources.  Equivalent
        to writing every horizon in the intrinsic amplitude instead.

        The exponent carries the slope, which is why efficiencies are kept per
        node and only collapsed at integration time: the robustness sweep must
        re-form these weights, not reuse the ones built at ``-4``.
        """
        e = _bias_exponent(slope, args.mu_weight)
        if e == 0.0:
            return y_w
        w = y_w * mu_plus ** e
        return w / w.sum()

    def _collapse(e2d, slope):
        """Population-average a per-node efficiency, ``(n_y, n_A) -> (n_A,)``.

        Collapsing before integrating is exact: both the grid integral and the
        analytic tail continuation are linear in the efficiency.
        """
        return np.tensordot(_pop_weights(slope), e2d, axes=(0, 0))

    def _dvol(e, slope):
        """Differential volume ``dV/drho`` on the grid, in ``RHO_REF`` units."""
        s = abs(float(slope))
        return (s - 1.0) / RHO_REF * e * (A / RHO_REF) ** (-s)

    def _volume(e, slope):
        """Sensitive volume: the grid integral plus the continued tail.

        Above the grid the efficiency has saturated (enforced below), so the
        rest of the integral is analytic rather than an uncertainty.
        """
        s = abs(float(slope))
        return float(np.trapezoid(_dvol(e, slope), A)
                     + float(e[-1]) * (RHO_REF / a_end) ** (s - 1.0))

    def _tail_residual(e, slope):
        """What that continuation could still be wrong by, at ``eps <= 1``."""
        s = abs(float(slope))
        return float((1.0 - e[-1]) * (RHO_REF / a_end) ** (s - 1.0))
    n = args.n_samples
    rng = np.random.default_rng(args.seed)

    keys = tuple(STRATEGIES)

    def _hit(sq, spec, th_k, th_cat):
        """Does this pair pass ``spec``?  ``sq`` is ``|z|^2``, ``(n, 2)``.

        Thresholds are on ``chi2``, so a per-image cut compares the SNR
        *magnitude* against ``sqrt(th)`` and the joint cut compares the
        two-image sum ``|z_1|^2 + |z_2|^2`` against ``th`` directly.
        """
        mag = np.sqrt(sq)
        lo, hi = mag.min(axis=1), mag.max(axis=1)
        if spec["rank"] == "both_above":
            return lo > np.sqrt(th_k)
        if spec["rank"] == "one_plus_one":
            return (hi > np.sqrt(th_cat)) & (lo > np.sqrt(th_k))
        return sq.sum(axis=1) > th_k            # joint |z_1|^2 + |z_2|^2

    def _efficiency(th, rng_, n_draw, response=None):
        """``eps(A)`` for every strategy at the chi2 thresholds ``th``.

        Every sweep below goes through this one function.  Three hand-copied
        versions of the ranking had drifted apart -- one of them comparing
        ``|z|^2`` against ``sqrt(th)`` -- which is why the sweeps disagreed
        with the main run.
        """
        e = {k: np.zeros((a_ratio.size, A.size)) for k in keys}
        for j, a_r in enumerate(a_ratio):
            for i, amp_v in enumerate(A):
                # `amp_scale` turns the grid variable into the BRIGHTER
                # image's observed amplitude, which is what draw_pair means
                # by `amp`: sqrt(mu_+) when the grid is rho_UL, 1 when the
                # grid already is rho_+.  The fainter image then comes out at
                # a * sqrt(mu_+) rho_UL = sqrt(|mu_-|) rho_UL either way.
                amp_b = amp_v * amp_scale[j]
                for k, spec in STRATEGIES.items():
                    resp = spec["response"] if response is None else response
                    sq = draw_pair(rng_, amp_b, a_r, n_draw, resp)
                    e[k][j, i] = np.mean(
                        _hit(sq, spec, th[k], th["both_catalog"]))
        return e

    eff2d = _efficiency(thr, rng, n)
    eff = {k: _collapse(eff2d[k], args.slope) for k in keys}

    def _other_weighting_ratios():
        """Volume ratios under the population weighting we did *not* report.

        Free: the per-node efficiencies are already in hand, so this is a
        second contraction of the same Monte Carlo rather than a second run.
        """
        w = y_w * mu_plus ** _bias_exponent(args.slope, not args.mu_weight)
        w = w / w.sum()
        v = {k: _volume(np.tensordot(w, eff2d[k], axes=(0, 0)), args.slope)
             for k in keys}
        out = {k: float(v[k] / v["both_catalog"]) for k in keys}
        out["weighting"] = ("cross-section prior only" if args.mu_weight
                            else "magnification-biased")
        return out

    curves = {k: _dvol(eff[k], args.slope) for k in keys}
    volumes = {k: _volume(eff[k], args.slope) for k in keys}
    base = volumes["both_catalog"]
    ratios = {k: float(volumes[k] / base) for k in keys}

    # --- the grid must not be the thing setting the answer ----------
    resid = {k: _tail_residual(eff[k], args.slope) / volumes[k] for k in keys}
    edges = {k: {"eff_low": float(eff[k][0]), "eff_high": float(eff[k][-1]),
                 "tail_residual_frac": float(resid[k]),
                 "first_bin_frac": float(np.trapezoid(curves[k][:2], A[:2])
                                         / volumes[k])} for k in keys}
    bad = {k: v for k, v in resid.items() if v > MAX_TRUNCATION}
    if bad:
        raise SystemExit(
            "refusing to write a truncated volume: residual tail is "
            + ", ".join("%s %.2g" % (k, v) for k, v in bad.items())
            + " of the volume, above the %.g limit; eff at the top edge is "
            % MAX_TRUNCATION
            + ", ".join("%s %.4f" % (k, eff[k][-1]) for k in bad)
            + ".  Raise --a-max.")
    low = {k: v for k, v in edges.items() if v["eff_low"] > 1e-3}
    if low:
        raise SystemExit(
            "refusing to write a volume whose low edge is not empty: "
            + ", ".join("%s eff=%.4g" % (k, v["eff_low"])
                        for k, v in low.items())
            + ".  A rho^-4 measure weights the region below the grid heavily "
              "and it is assumed empty.  Lower --a-min.")

    robustness = {}
    for sl in args.slope_robustness:
        vs = {k: _volume(_collapse(eff2d[k], sl), sl) for k in keys}
        robustness["%.1f" % sl] = {
            "volumes": {k: float(vs[k]) for k in keys},
            "ratios": {k: float(vs[k] / vs["both_catalog"]) for k in keys},
            "marginalization_gain": float(vs["long_marg"] / vs["long_max"]),
        }

    rep = prov.block(seed=args.seed, **{
        "script": os.path.basename(__file__),
        "extends": "strategy-comparison framework of B. Zackay",
        "changes": [
            "fainter image scaled by the measured a(y); prior mean 0.52",
            "lens/delay trials MEASURED: N_eff = 1.73e5 for the 10-60 min "
            "window (0.47 of T f_rms) and 1.04 marginalized; Phase 4's 5339 "
            "grid points measure as N_eff = 572",
            f"all thresholds re-solved at the same {args.budget} false alarms",
            "joint strategies ranked by z1^2+z2^2, the quantity the chi2 refers to",
            "added the delay-marginalized both-sub-threshold strategy",
            "the marginalized arm uses the value measured IN THIS REGIME "
            "against a 3e6-block rich pool: x0.445 at FAP 1e-3, i.e. a PENALTY "
            "not a saving.  The earlier x14.97 is withdrawn.",
            "SNRs are complex and each ranking's threshold uses the dof its "
            "own statistic has (2 per image, 4 for the incoherent two-image "
            "sum); real gaussians against 4-dof thresholds ran the joint arms "
            "15-20x inside the budget",
            "figure of merit is sensitive volume, int eps rho^-4 drho, "
            "not an arbitrary-unit A^-5.5 phase space, over a grid that "
            "reaches saturation -- without which every ratio against the "
            "reference arm is set by the grid edge",
        ],
        "budget": args.budget, "slope": args.slope, "n_samples": n,
        "delay_trials_measured": N_EFF_WINDOW,
        "delay_window_measured_s": 3000.0,
        "marg_trials_saving_measured": MARG_TRIALS_SAVING,
        "marg_saving_measured_at_fap": MARG_SAVING_MEASURED_AT_FAP,
        "marg_trials_saving_note": (
            "x0.445 at FAP 1e-3, i.e. prior weighting COSTS trials in this "
            "regime rather than saving them.  The earlier x1.54 / x3.08 / "
            "x14.97 at FAP 1e-2 / 1e-3 / 1e-4 were measured against a pool "
            "too shallow to resolve the marginalized arm's tail and are "
            "WITHDRAWN; the value here is measured against "
            "a 3e6-block rich pool.  Phase 4's +13% for the compact "
            "overlapping prior is a different regime and is unaffected."),
        "trials": {k: v["trials"] for k, v in STRATEGIES.items()},
        "thresholds_chi2": thr,
        "per_image_cuts": {
            k: float(np.sqrt(thr[k])) if STRATEGIES[k]["rank"] != "joint"
            else float(np.sqrt(thr[k] / 2)) for k in keys},
        "a_ratio_nodes": a_ratio.tolist(), "a_ratio_weights": y_w.tolist(),
        "mu_plus_nodes": mu_plus.tolist(),
        "scan_variable": ("rho_UL, the unlensed (intrinsic) amplitude"
                          if args.scan == "intrinsic"
                          else "rho_+, the observed brighter-image amplitude"),
        "scan": args.scan,
        "magnification_bias": bool(args.mu_weight),
        "population_weights": _pop_weights(args.slope).tolist(),
        "a_ratio_prior_mean": float(np.sum(_pop_weights(args.slope) * a_ratio)),
        "a_ratio_prior_mean_cross_section": float(np.sum(y_w * a_ratio)),
        "magnification_bias_note": (
            "two equivalent conventions, and --scan picks which one is run.  "
            "OBSERVED: the grid is rho_+ = sqrt(mu_+) rho_UL and the "
            "impact-parameter average carries p(y) ~ y times mu_+^{(s-1)/2}, "
            "because a Euclidean population magnified by mu_+ yields that "
            "many more lensed sources at a given observed amplitude.  "
            "INTRINSIC: the grid is rho_UL, each image is scaled by "
            "sqrt(mu_+) and sqrt(|mu_-|), and the average carries p(y) ~ y "
            "ALONE -- the bias is not a separate factor, it is absorbed by "
            "the change of variable, which is exact.  "
            "Ratios agree between the two to sampling noise.  --no-mu-weight "
            "is the naive convention in either variable and is what this "
            "product used before 2026-09-12."),
        "population": pop.as_dict(),
        "labels": {k: v["label"] for k, v in STRATEGIES.items()},
        "plot_order": list(ORDER),
        "measure": "sensitive volume, V = ((s-1)/rho_ref) int eps "
                   "(rho/rho_ref)^-s drho plus the saturated tail, in units "
                   "of the sphere out to the rho_ref horizon, averaged over "
                   "the impact-parameter nodes with the magnification-biased "
                   "weights above",
        "rho_ref": RHO_REF,
        "dof": {k: v["dof"] for k, v in STRATEGIES.items()},
        "derived_trials": {
            "T_obs_s": T_OBS, "f_rms_hz": F_RMS,
            "N_eff_window_measured": N_EFF_WINDOW,
            "N_events_assumed": N_EVENTS,
            "N_templates_blind": N_TEMPLATES,
            "N_templates_conditional": N_TEMPLATES_CONDITIONAL,
            "N_templates_is_a_bank_size": True,
            "N_single": N_SINGLE, "N_pairs": N_PAIRS,
            "assumptions_swept": ["N_events_assumed", "N_templates_blind"],
            "note": ("every trials factor is derived from T_obs, f_rms and the "
                     "MEASURED delay-window N_eff; no order-of-magnitude "
                     "time/sky estimate is inherited.  TWO inputs are assumed "
                     "rather than measured and both are swept below: N_events, "
                     "and N_templates -- taken as 1e4, the scale quoted for "
                     "binary-black-hole banks in the search literature, and "
                     "charged to every strategy except the conditional one, "
                     "whose bright event pins the template.  Only "
                     "2 ln N_templates enters, so a decade in it is worth 4.6 "
                     "in rho^2 against thresholds of 60-90.")},
        "amplitude_grid": {"min": float(A[0]), "max": a_end,
                           "step": args.a_step, "n": int(A.size)},
        "volumes": {k: float(volumes[k]) for k in keys},
        "ratios_vs_both_in_catalog": ratios,
        # the same Monte Carlo under the other population weighting, so the
        # manuscript's "without magnification bias it would read ..." is a
        # stored number rather than a literal typed into a caption
        "ratios_under_the_other_weighting": _other_weighting_ratios(),
        "saturation_diagnostics": edges,
        "max_truncation_allowed": MAX_TRUNCATION,
        "slope_robustness": robustness,
        "marginalization_gain_long": float(ratios["long_marg"] / ratios["long_max"]),
    })
    # response robustness: the antenna factor cancels in the ratios, and the
    # claim is stored rather than asserted
    resp_sweep = {}
    n_sweep = max(n // 8, 20000)
    for mode in ("unit", "indep", "common", "mean"):
        eff2 = _efficiency(thr, np.random.default_rng(args.seed + 77),
                           n_sweep, response=mode)
        cv = {k: _volume(_collapse(eff2[k], args.slope), args.slope)
              for k in keys}
        base2 = cv["both_catalog"]
        resp_sweep[mode] = {k: cv[k] / base2 for k in keys}
    rep["response_robustness_ratios"] = resp_sweep

    # N_events is the one input that is assumed rather than measured; sweep it
    ev_sweep = {}
    for nev in (30, 100, 300, 1000):
        t2 = dict((k, v["trials"]) for k, v in STRATEGIES.items())
        t2["one_subthr"] = nev * N_EFF_WINDOW
        th2 = {k: threshold_chi2(t2[k], args.budget, dof=STRATEGIES[k]["dof"])
               for k in STRATEGIES}
        e2 = _efficiency(th2, np.random.default_rng(args.seed + 991), n_sweep)
        cv = {k: _volume(_collapse(e2[k], args.slope), args.slope)
              for k in keys}
        ev_sweep[str(nev)] = {"threshold_one_subthr": float(np.sqrt(th2["one_subthr"])),
                              "ratios": {k: cv[k] / cv["both_catalog"] for k in cv}}
    rep["n_events_sweep"] = ev_sweep

    # the bank size raises the blind strategies' thresholds but not the
    # conditional one, so it changes the ordering rather than a normalization
    tmpl_sweep = {}
    for nt in (1, 10, 100, 1000, 10000, 100000):
        t3 = {"both_catalog": T_OBS * F_RMS * nt,
              "one_subthr": N_TEMPLATES_CONDITIONAL * N_EVENTS * N_EFF_WINDOW,
              "long_max": T_OBS * F_RMS * nt * N_EFF_WINDOW,
              "long_marg": T_OBS * F_RMS * nt * N_EFF_WINDOW
                           / MARG_TRIALS_SAVING}
        th3 = {k: threshold_chi2(t3[k], args.budget, dof=STRATEGIES[k]["dof"])
               for k in STRATEGIES}
        e3 = _efficiency(th3, np.random.default_rng(args.seed + 4242), n_sweep)
        cv = {k: _volume(_collapse(e3[k], args.slope), args.slope)
              for k in keys}
        a_switch = float(np.sqrt(th3["one_subthr"])
                         / np.sqrt(th3["both_catalog"]))
        tmpl_sweep[str(nt)] = {
            "cut_catalogue": float(np.sqrt(th3["both_catalog"])),
            "cut_conditional": float(np.sqrt(th3["one_subthr"])),
            "a_switch": a_switch,
            "ratios": {k: cv[k] / cv["both_catalog"] for k in cv}}
    rep["n_templates_sweep"] = tmpl_sweep
    rep["response_note"] = (
        "the antenna factor multiplies every strategy's horizon identically, "
        "so it scales all volumes by u^3 and cancels in the ratios; the sweep "
        "above shows the ratios moving by <6% while the absolute volume moves "
        "by 2.3x.  It is therefore set to unity in the reported run.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, meta=json.dumps(rep, default=float),
                        amplitude=A, weight=weight,
                        **{f"curve_{k}": curves[k] for k in keys},
                        y_nodes=y_nodes, mu_plus=mu_plus,
                        pop_weights=_pop_weights(args.slope),
                        **{f"eff_{k}": eff[k] for k in keys},
                        # per-node efficiencies, so the population weighting
                        # can be changed without re-running the Monte Carlo
                        **{f"effnode_{k}": eff2d[k] for k in keys})
    with open(args.out.replace(".npz", ".json"), "w") as fh:
        json.dump(rep, fh, indent=2, default=float)

    print("thresholds at %.3g expected false alarms:" % args.budget)
    for k in keys:
        print("   %-22s trials %.2e  dof %d  chi2 %7.2f  per-image %.3f"
              % (k, STRATEGIES[k]["trials"], STRATEGIES[k]["dof"],
                 thr[k], rep["per_image_cuts"][k]))
    print("\nsensitive volume (units of the rho=%.0f sphere), slope %.1f:"
          % (RHO_REF, args.slope))
    print("   %-22s %10s %8s %10s %10s"
          % ("strategy", "V", "V/V_ref", "eff(top)", "tail/V"))
    for k in keys:
        print("   %-22s %10.4f %8.2f %10.4f %10.1e"
              % (k, volumes[k], ratios[k], eff[k][-1], resid[k]))
    print("\n   marginalizing the delay: x%.2f"
          % rep["marginalization_gain_long"])
    print("   (measured IN THIS REGIME against a 3e6-block rich pool: x%.2f"
          % MARG_TRIALS_SAVING)
    print("    at FAP %.0e, i.e. a PENALTY not a saving.  Phase 4's"
          % MARG_SAVING_MEASURED_AT_FAP)
    print("    +13%% for the compact overlapping prior is unaffected.)")
    print("   slope robustness of that gain: "
          + ", ".join("%s: x%.2f" % (sl, robustness[sl]["marginalization_gain"])
                      for sl in robustness))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
