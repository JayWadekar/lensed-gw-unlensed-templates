#!/usr/bin/env python
"""How the strategy comparison of ``sec:faint`` depends on the caustic.

``sec:faint`` fixes the lens population to an isolated point mass, whose
caustic is a single point.  The configuration thought to matter most for
gravitational-wave lensing is not that: it is a compact object embedded in the
convergence and shear of a macro-image (``sec:changrefsdal``), whose caustic is
extended and bounded by **folds**.  This script asks what the three strategies
of ``eq:horizons`` are worth there.

What the comparison actually depends on
---------------------------------------
Written in the unlensed amplitude, the horizons of ``eq:horizons`` are

    both super :  rho_UL sqrt(|mu_-|)        > rho_th
    one + one  :  rho_UL sqrt(mu_+)          > rho_th   AND
                  rho_UL sqrt(|mu_-|)        > rho_th_sub
    both sub   :  rho_UL sqrt(mu_+ + |mu_-|) > rho_th_comb

and a Euclidean count gives ``V ~ <rho_UL_min^-3>``.  Hold the amplitude ratio
``a = sqrt(|mu_-|/mu_+)`` fixed and ``mu_+`` cancels from every ratio: the
reward factor of ``eq:envratio`` is exactly ``((1+a^2)/a^2)^{3/2}``.  The
*total* magnification distribution therefore does not enter the comparison at
all -- which is just as well, since ``p(mu) ~ mu^-3`` holds both for a point
lens with ``p(y) ~ y`` and near a fold with a flat source distribution, and so
could not have distinguished them.

Because ``|mu_-| <= mu_+`` by construction, the reward factor is bounded below
by ``2^{3/2}``, and that bound is saturated exactly where two images merge with
equal magnification and opposite parity -- a fold.

Thresholds are held at the values ``lensed_strategies.py`` calibrates for the
10--60 minute window, so that what varies between rows is the magnification
distribution and nothing else.  The Chang--Refsdal case is a microlensing
configuration with far shorter delays; the point of evaluating it here is to
isolate the effect of the amplitude ratio, not to re-calibrate the trials.

Everything below is deterministic: no RNG, and the seed recorded in the
provenance block is ``None`` for that reason.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from lensing import provenance as prov                      # noqa: E402
from lensing.lens_models import ChangRefsdal                # noqa: E402

#: The calibrated thresholds of ``results/.../lensed_strategies_rhoUL.npz``,
#: read from the product at start-up.  Held fixed across every row, on purpose.
STRATEGIES = os.environ.get("LENSING_STRATEGIES", "data/strategies/lensed_strategies_rhoUL.npz")
RHO_TH = RHO_TH_SUB = RHO_TH_COMB = None

#: Trials bookkeeping, for the delay-window sweep only.
T_OBS, F_RMS, N_TEMPLATES = 3.156e7, 121.88, 1e4
N_EFF_WINDOW, WINDOW_S = 1.73e5, 3000.0
#: Trials multiplier of the joint arm relative to the unweighted maximum:
#: 1 for ``long_max``; 1/0.445 for the prior-weighted ``long_marg``.
TRIALS_FACTOR = {"long_max": 1.0, "long_marg": 1.0 / 0.445}


def load_thresholds(path, arm):
    """Thresholds of the three strategies, from the strategies product."""
    global RHO_TH, RHO_TH_SUB, RHO_TH_COMB
    meta = json.loads(str(np.load(path, allow_pickle=True)["meta"]))
    RHO_TH = float(np.sqrt(meta["thresholds_chi2"]["both_catalog"]))
    RHO_TH_SUB = float(np.sqrt(meta["thresholds_chi2"]["one_subthr"]))
    RHO_TH_COMB = float(np.sqrt(meta["thresholds_chi2"][arm]))
    return meta


def volumes(mu_p, mu_m, w):
    """``(V_sup, V_one, V_sub)`` for a population of pairs.

    ``mu_m`` is ``|mu_-|``; ``w`` are normalized population weights.
    """
    mu_p, mu_m, w = map(np.asarray, (mu_p, mu_m, w))
    r_sup = RHO_TH / np.sqrt(mu_m)
    r_one = np.maximum(RHO_TH / np.sqrt(mu_p), RHO_TH_SUB / np.sqrt(mu_m))
    r_sub = RHO_TH_COMB / np.sqrt(mu_p + mu_m)
    V = lambda r: float(np.sum(w * r ** -3.0))               # noqa: E731
    return V(r_sup), V(r_one), V(r_sub)


def _row(mu_p, mu_m, w):
    Vs, Vo, Vb = volumes(mu_p, mu_m, w)
    a = np.sqrt(np.asarray(mu_m) / np.asarray(mu_p))
    # the single fixed `a` that would reproduce the measured V_sub/V_sup
    a_eff = 1.0 / np.sqrt(((Vb / Vs) / (RHO_TH / RHO_TH_COMB) ** 3) ** (2 / 3.) - 1)
    return {"n_cells": int(np.size(a)),
            "a_mean": float(np.sum(w * a)),
            "a_median": float(np.interp(0.5, np.cumsum(np.asarray(w)[np.argsort(a)]),
                                        np.sort(a))),
            "a_max": float(np.max(a)),
            "a_effective": float(a_eff),
            "V_one_over_V_sup": Vo / Vs,
            "V_sub_over_V_sup": Vb / Vs,
            "V_sub_over_V_one": Vb / Vo}


def point_mass_population(y_min, y_max, n=400001):
    """The population of ``sec:faint``: an isolated point mass, ``p(y) ~ y``."""
    y = np.linspace(y_min, y_max, n)
    s = (y ** 2 + 2) / (2 * y * np.sqrt(y ** 2 + 4))
    return 0.5 + s, s - 0.5, y / np.sum(y)


def chang_refsdal_plane(kappa, gamma, y_max=2.0, n_y=600, n_phi=31):
    """Area-weighted sample of the Chang--Refsdal source plane.

    ``images()`` places the source on the real axis, so sweeping the shear
    angle ``phi_gamma`` over a quadrant covers the plane; the astroid caustic
    has that symmetry.  The pair is the two brightest images, which is what a
    search would find.
    """
    ys = np.linspace(0.004, y_max, n_y)
    Y, N, MP, MM = [], [], [], []
    for phi in np.linspace(0.0, 90.0, n_phi):
        lens = ChangRefsdal(psi0=1.0, kappa=kappa, gamma=gamma,
                            phi_gamma=float(phi))
        for y in ys:
            im = lens.images(float(y))
            if len(im) < 2:
                continue
            mu = np.sort(np.abs([i.mu for i in im]))[::-1][:2]
            Y.append(y), N.append(len(im)), MP.append(mu[0]), MM.append(mu[1])
    return (np.array(Y), np.array(N), np.array(MP), np.array(MM))


def _threshold_chi2(n_trials, budget=0.1, dof=4):
    return brentq(lambda z: stats.chi2.sf(z + dof, dof) * n_trials - budget,
                  1.0, 4000.0)


def delay_window_sweep(windows_s, arm):
    """At the fold (``a = 1``), what a shorter delay window is worth.

    Only the joint arm's trials scale with the window, and only ``2 ln N``
    enters the threshold, so the relief is real but slow.  At ``a = 1`` the
    conditional arm's binding cut is the *bright* image at ``rho_th``, the
    same cut the both-super strategy applies, so ``V_one = V_sup`` and the two
    gain columns coincide.
    """
    n_single = T_OBS * F_RMS * N_TEMPLATES
    out = []
    for W in windows_s:
        n_joint = n_single * N_EFF_WINDOW * (W / WINDOW_S) * TRIALS_FACTOR[arm]
        rho_comb = float(np.sqrt(_threshold_chi2(n_joint)))
        f = (RHO_TH / rho_comb) ** 3
        out.append({"window_s": float(W), "n_trials_joint": float(n_joint),
                    "rho_th_comb": rho_comb, "threshold_factor": float(f),
                    "V_sub_over_V_sup_at_a_equals_1": float(f * 2 ** 1.5)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="regenerated/caustics/appendix_caustics.json")
    ap.add_argument("--arm", default="long_max", choices=sorted(TRIALS_FACTOR),
                    help="which joint two-image arm sets rho_th_comb")
    args = ap.parse_args(argv)
    meta = load_thresholds(STRATEGIES, args.arm)

    report = {"joint_arm": args.arm,
              "thresholds_source": {"path": STRATEGIES,
                                    "timestamp_utc": meta["timestamp_utc"],
                                    "git_commit": meta["git_commit"]},
              "fixed_thresholds": {"rho_th": RHO_TH, "rho_th_sub": RHO_TH_SUB,
                                   "rho_th_comb": RHO_TH_COMB},
              "reward_factor_formula": "((1+a^2)/a^2)^{3/2}, exact when a is fixed",
              "floor": {"a": 1.0,
                        "reward": float(2 ** 1.5),
                        "V_sub_over_V_sup": float((RHO_TH / RHO_TH_COMB) ** 3
                                                  * 2 ** 1.5),
                        "note": "|mu_-| <= mu_+ always, so this is a lower "
                                "bound for ANY population, saturated at a fold"}}

    # --- the section's own population, as a check on the closed form -------
    report["point_mass_sec_faint"] = _row(*point_mass_population(0.17315539240441197, 1.0))

    # --- eq:envpaironevol treats the fainter image as binding everywhere ---
    mp, mm, w = point_mass_population(0.17315539240441197, 1.0)
    a_pm = np.sqrt(mm / mp)
    a_sw = RHO_TH_SUB / RHO_TH
    Vs, Vo, Vb = volumes(mp, mm, w)
    Vo_faint = float(np.sum(w * (RHO_TH_SUB / np.sqrt(mm)) ** -3.0))
    report["targeted_closed_form"] = {
        "note": "V_one with only the faint-image cut (eq:envpaironevol) vs both cuts; "
                "the bright-image cut binds for a > rho_th_sub/rho_th",
        "a_switch": float(a_sw),
        "prior_weight_above_switch": float(w[a_pm > a_sw].sum()),
        "V_sub_over_V_one_faint_cut_only": Vb / Vo_faint,
        "V_sub_over_V_one_both_cuts": Vb / Vo}

    # --- fixed amplitude ratio: mu_+ cancels, so one cell suffices ---------
    report["fixed_a"] = {}
    for a in (1.0, 0.95, 0.9, 0.8, 0.7, 0.604, 0.524):
        report["fixed_a"]["%.3f" % a] = _row(np.array([1.0]), np.array([a * a]),
                                             np.array([1.0]))

    # --- Chang-Refsdal, area-weighted over the source plane ---------------
    report["chang_refsdal"] = {}
    for kappa, gamma in ((0.3, 0.3), (0.0, 0.3), (0.45, 0.3),
                         (0.3, 0.15), (0.3, 0.5), (0.3, 0.9), (0.6, 0.6)):
        Y, N, MP, MM = chang_refsdal_plane(kappa, gamma)
        entry = {"macro_image": "minimum" if gamma < 1 - kappa else "saddle",
                 "y_max": 2.0}
        for label, sel in (("caustic_interior", N >= 4),
                           ("caustic_exterior", N == 2)):
            if sel.sum() < 50:
                continue
            entry[label] = _row(MP[sel], MM[sel], Y[sel] / Y[sel].sum())
        report["chang_refsdal"]["kappa%.2f_gamma%.2f" % (kappa, gamma)] = entry

    # --- Table III of the paper: the cost of WIDENING the window ---------
    n_fid = T_OBS * F_RMS * N_TEMPLATES * N_EFF_WINDOW * TRIALS_FACTOR[args.arm]
    r_fid = float(np.sqrt(_threshold_chi2(n_fid)))
    report["window_widening_table"] = [
        {"window": lab, "N_over_N_fid": f,
         "rho_th_comb": float(np.sqrt(_threshold_chi2(n_fid * f))),
         "V_over_V_fid": float((r_fid / np.sqrt(_threshold_chi2(n_fid * f))) ** 3)}
        for lab, f in (("10-60 min (fiducial)", 1.0), ("1 day", 24.0),
                       ("1 week", 168.0), ("1 month", 720.0),
                       ("1 year", 8766.0))]

    report["delay_window_sweep"] = delay_window_sweep(
        (3000.0, 1000.0, 300.0, 100.0, 30.0, 10.0, 3.0, 1.0), args.arm)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(prov.block(seed=None, script=os.path.basename(__file__),
                             **report), fh, indent=2)
    print("wrote", args.out)

    interior = [v["caustic_interior"]["V_sub_over_V_sup"]
                for v in report["chang_refsdal"].values() if "caustic_interior" in v]
    print("  point mass (sec:faint)      V_sub/V_sup = %.2f"
          % report["point_mass_sec_faint"]["V_sub_over_V_sup"])
    print("  Chang-Refsdal, interior     V_sub/V_sup = %.2f - %.2f over %d (kappa,gamma)"
          % (min(interior), max(interior), len(interior)))
    print("  absolute floor (a = 1)      V_sub/V_sup = %.2f"
          % report["floor"]["V_sub_over_V_sup"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
