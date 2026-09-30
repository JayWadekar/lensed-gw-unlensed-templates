"""Threshold-triggered evaluation of the lensed statistics.

Why a prefilter is needed
-------------------------

Calibrating a false-alarm probability of ``1e-4`` with at least 100 background
exceedances needs of order ``1e6`` noise segments.  Evaluating the full lens
grid at every time sample of every segment is far too slow, and the user's
compute budget caps the job at 20 workers.

The lossless bound
------------------

For any lens point ``(tau, a)``, with ``|A + B|^2 <= 2(|A|^2 + |B|^2)``,

    |z_L(t)|^2 = |z_1(t) + conj(c) z_1(t+tau)|^2 / N
              <= 2 ( |z_1(t)|^2 + a^2 |z_1(t+tau)|^2 ) / N
              <= 2 (1 + a^2) / N * max( |z_1(t)|^2, |z_1(t+tau)|^2 ).

So a lensed statistic can only reach ``rho_floor`` if

    max( |z_1(t)|, |z_1(t+tau)| )  >=  rho_floor / K,
    K = max_k sqrt( 2 (1 + a_k^2) / N_k )

with the maximum taken over the actual grid, computed exactly rather than
estimated.  In words: **a threshold crossing requires at least one of the two
images to be loud.**  The candidate times are therefore the loud samples
themselves together with their translates by each grid delay -- a set of size
``|S| * (1 + n_td)`` instead of ``n_time * n_lens``.

The same bound covers every statistic used here, because with ``sum_k w_k = 1``

    S_soft(t) = log sum_k w_k exp(q_k) <= max_k q_k,
    S_mix(t)  <= max( S_UL(t), S_soft(t) ),

so one floor censors all four consistently.

Censoring, not discarding
-------------------------

A segment whose candidate set is empty has *all* its statistics provably below
``rho_floor^2 / 2``.  Such segments are recorded as **censored at the floor**
rather than dropped, so the total segment count -- and hence the false-alarm
probability -- stays exact.  The floor must be placed below the smallest
quoted threshold; :func:`validate_floor` checks this after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import conventions as cv
from . import kernels as kn
from . import recombine as rc
from . import waveforms as wf


# --------------------------------------------------------------------------
# Precomputed, grid-dependent quantities
# --------------------------------------------------------------------------


@dataclass
class SearchSetup:
    """Everything about the search that does not change between segments."""

    cfg: object
    psd: np.ndarray
    template: np.ndarray
    grid: object
    oversample: int
    delta_t: float
    shifts: np.ndarray          # (n_td,) integer sample offsets
    C_vals: np.ndarray          # (n_td,) autocorrelation at each signed lag
    norm: np.ndarray            # (n_td, n_y) template norm squared N
    log_w: np.ndarray           # (n_td, n_y) log prior weight, -inf if masked
    bound_K: float              # the exact prefilter constant
    conv: object

    @classmethod
    def build(cls, cfg, psd, template, grid, oversample=4, conv=None):
        if conv is None:
            conv = cv.DEFAULT
        if grid.weights is None:
            raise ValueError("grid has no weights; call priors.assign_weights")
        dt = cfg.delta_t / oversample
        lags = conv.lag(grid.t_d)
        shifts = np.rint(lags / dt).astype(int)
        C_vals = wf.autocorr_at_lags(template, psd, cfg, lags)
        norm = rc.lensed_norm_sq(C_vals[:, None], grid.a[None, :], conv=conv)

        with np.errstate(divide="ignore"):
            log_w = np.where(grid.mask, np.log(grid.weights), -np.inf)

        bound_K = prefilter_bound_K(norm, grid.a, mask=grid.mask)
        return cls(
            cfg=cfg, psd=psd, template=template, grid=grid,
            oversample=oversample, delta_t=dt, shifts=shifts, C_vals=C_vals,
            norm=norm, log_w=log_w, bound_K=bound_K, conv=conv,
        )

    def image_floor(self, rho_floor):
        """The per-image amplitude below which no crossing is possible."""
        return rho_floor / self.bound_K


def prefilter_bound_K(norm, a, mask=None) -> float:
    """``K = max_k sqrt( 2 (1 + a_k^2) / N_k )`` over the active grid points.

    Since ``|A + B|^2 <= 2(|A|^2 + |B|^2)``, the expanded form of the lensed
    SNR obeys ``|z_L(t;theta)| <= K * max{|z1(t)|, |z1(t+tau)|}``, so a crossing
    of ``rho_floor`` requires one image sample of at least ``rho_floor / K``.
    That is what makes the candidate prefilter lossless rather than heuristic.

    Factored out of :meth:`SearchSetup.build` so that code which does not carry
    a full :class:`lensing.grids.LensGrid` -- the cost benchmark, for one --
    uses the identical constant rather than a re-derived lookalike.

    Parameters
    ----------
    norm : array
        ``N = 1 + a^2 + 2 a Re[conj(c) C(tau)]``, broadcast to the grid shape.
    a : array
        Amplitude ratios, broadcastable against ``norm``.
    mask : bool array, optional
        Active points. Inactive points cannot be reached by the search and must
        not inflate the bound.
    """
    a = np.asarray(a, dtype=float)
    norm = np.asarray(norm, dtype=float)
    if a.ndim == 1 and norm.ndim == 2:
        a = a[None, :]
    k2 = 2.0 * (1.0 + a**2) / norm
    k2 = np.broadcast_to(k2, norm.shape)
    if mask is not None:
        k2 = np.where(mask, k2, -np.inf)
    return float(np.sqrt(np.max(k2)))


def candidate_indices(loud, n_sh, n):
    """The per-delay candidate set ``loud U (loud - n_sh) mod n``, deduplicated.

    Equivalent to ``np.unique(np.concatenate([loud, (loud - n_sh) % n]))`` as a
    *set*, but roughly six times cheaper, and the returned order is arbitrary
    rather than sorted.

    Why not just sort. The obvious construction sorts ~2|S| indices once per
    delay, and profiling showed that single ``np.unique`` call to be about half
    the cost of the whole candidate-restricted evaluation -- far more than the
    gather it exists to enable. Since ``loud`` is already sorted, membership
    can be tested with a binary search instead, which is what this does.

    Why the result is still deduplicated, when a maximum would not care. The
    same candidate set feeds the sparse accumulation of ``S_soft`` and
    ``S_mix``, where a repeated index is not obviously harmless. Duplicates
    happen to survive ``acc[cand] += ...`` only because of how NumPy buffers
    scattered writes, which is far too subtle a thing to rely on. Exact
    deduplication keeps every statistic correct for a reason that can be stated
    in one line.

    Callers must not assume sorted output. Every consumer here either takes a
    maximum or scatters into a per-time accumulator, and both are insensitive to
    the order of distinct indices, so results are bit-for-bit unchanged.
    """
    shifted = (loud - n_sh) % n
    pos = np.searchsorted(loud, shifted)
    pos_clipped = np.minimum(pos, loud.size - 1)
    already_present = loud[pos_clipped] == shifted
    return np.concatenate([loud, shifted[~already_present]])


# --------------------------------------------------------------------------
# Candidate selection
# --------------------------------------------------------------------------


def candidate_times(z1, setup, rho_floor):
    """Times at which some lens point could reach ``rho_floor``.

    Returns a sorted array of sample indices.  A crossing at time ``t`` via
    delay ``tau_k`` needs a loud sample at ``t`` **or** at ``t + tau_k``, so
    the candidates are the loud set ``S`` together with ``S - shift_k`` for
    every grid delay.  Indices wrap modulo the segment length, matching the
    circular convention of the SNR series itself.
    """
    n = z1.size
    loud = np.nonzero(np.abs(z1) >= setup.image_floor(rho_floor))[0]
    if loud.size == 0:
        return np.empty(0, dtype=np.int64)
    offsets = np.concatenate(([0], setup.shifts))
    cand = (loud[None, :] - offsets[:, None]) % n
    return np.unique(cand.ravel())


def candidate_times_with_ul_peak(z1, setup, rho_floor):
    """:func:`candidate_times` plus the time of the unlensed maximum.

    Including that one sample makes ``S_mix`` **exact** rather than a lower
    bound: the mixture at the unlensed peak needs the lensed term there too,
    and that time is not guaranteed to be a candidate.  The cost is one extra
    column.
    """
    cand = candidate_times(z1, setup, rho_floor)
    i_ul = int(np.argmax(np.abs(z1)))
    if cand.size == 0:
        return np.array([i_ul], dtype=np.int64), True
    return np.unique(np.append(cand, i_ul)), False


# --------------------------------------------------------------------------
# Statistic evaluation on a candidate set
# --------------------------------------------------------------------------


def evaluate_statistics(z1, setup, rho_floor, etas=(0.01, 0.1, 0.5),
                        return_details=False):
    """All four statistics for one segment, using the lossless prefilter.

    Returns a dict.  ``censored`` is True when the candidate set was empty, in
    which case every lensed statistic is reported at the floor value
    ``rho_floor^2 / 2`` and flagged; ``S_UL`` is always exact because it costs
    nothing.
    """
    from scipy.special import logsumexp

    grid, u = setup.grid, setup.conv.filter_coeff(1.0)
    floor_stat = 0.5 * rho_floor * rho_floor

    s_ul = 0.5 * float(np.max(np.abs(z1) ** 2))
    cand, censored = candidate_times_with_ul_peak(z1, setup, rho_floor)

    out = {
        "S_UL": s_ul,
        "n_candidates": int(cand.size),
        "censored": bool(censored),
        "rho_floor": float(rho_floor),
        "floor_stat": float(floor_stat),
    }

    n = z1.size
    # half_q[t_index, i_td, j_y] = |z_L|^2 / 2 on the candidate times
    p_self = np.abs(z1[cand]) ** 2
    running_max = np.full(cand.size, -np.inf)
    lse_terms = []

    for i, n_sh in enumerate(setup.shifts):
        active = np.nonzero(grid.mask[i])[0]
        if active.size == 0:
            continue
        idx_sh = (cand + n_sh) % n
        a = grid.a[active]
        # One fused pass, gathering z_1[cand] and z_1[cand + n_sh] inside the
        # kernel: the broadcast form allocated five (n_cand, n_a) temporaries
        # per delay and cost 5x this (kernels.py).
        half_q = kn.half_q_from_indices(
            z1, cand, idx_sh, u, a, setup.norm[i, active]
        )
        running_max = np.maximum(running_max, half_q.max(axis=1))
        lse_terms.append(half_q + setup.log_w[i, active][None, :])

    terms = np.concatenate(lse_terms, axis=1)
    lse = logsumexp(terms, axis=1)

    # When the segment is censored the candidate set holds only the unlensed
    # peak, so the lensed maxima are bounded by -- and reported at -- the floor.
    if censored:
        out["S_max"] = floor_stat
        out["S_soft"] = floor_stat
    else:
        out["S_max"] = float(np.max(running_max))
        out["S_soft"] = float(np.max(lse))

    half_ul_cand = 0.5 * p_self
    for eta in etas:
        if eta <= 0.0:
            out[f"S_mix_eta{eta:g}"] = s_ul
            continue
        if eta >= 1.0:
            out[f"S_mix_eta{eta:g}"] = out["S_soft"]
            continue
        # The unlensed peak is always in the candidate set (see
        # candidate_times_with_ul_peak), so this maximum is exact, not a bound.
        mixed = logsumexp(
            np.stack(
                [np.log1p(-eta) + half_ul_cand, np.log(eta) + lse], axis=1
            ),
            axis=1,
        )
        out[f"S_mix_eta{eta:g}"] = float(np.max(mixed))
    if return_details:
        out["_candidates"] = cand
        out["_running_max"] = running_max
        out["_lse"] = lse
    return out


def evaluate_statistics_bruteforce(z1, setup, etas=(0.01, 0.1, 0.5)):
    """Reference implementation with **no** prefilter, for validating the prefilter.

    Evaluates every lens point at every time sample.  Only for small segments.
    """
    from scipy.special import logsumexp

    grid, u = setup.grid, setup.conv.filter_coeff(1.0)
    p_self = np.abs(z1) ** 2
    running_max = np.full(z1.size, -np.inf)
    lse_terms = []
    for i, n_sh in enumerate(setup.shifts):
        active = np.nonzero(grid.mask[i])[0]
        if active.size == 0:
            continue
        z_sh = np.roll(z1, -n_sh)
        cross = np.real(np.conjugate(u) * z1 * np.conjugate(z_sh))
        a = grid.a[active]
        num = (
            p_self[:, None]
            + (a * a)[None, :] * (np.abs(z_sh) ** 2)[:, None]
            + 2.0 * a[None, :] * cross[:, None]
        )
        half_q = 0.5 * num / setup.norm[i, active][None, :]
        running_max = np.maximum(running_max, half_q.max(axis=1))
        lse_terms.append(half_q + setup.log_w[i, active][None, :])
    terms = np.concatenate(lse_terms, axis=1)
    lse = logsumexp(terms, axis=1)

    s_ul = 0.5 * float(np.max(p_self))
    out = {
        "S_UL": s_ul,
        "S_max": float(np.max(running_max)),
        "S_soft": float(np.max(lse)),
    }
    for eta in etas:
        mixed = logsumexp(
            np.stack(
                [np.log1p(-eta) + 0.5 * p_self, np.log(eta) + lse], axis=1
            ),
            axis=1,
        )
        out[f"S_mix_eta{eta:g}"] = float(np.max(mixed))
    return out


# --------------------------------------------------------------------------
# Threshold calibration
# --------------------------------------------------------------------------


def validate_floor(values, floor_stat, faps):
    """Confirm the censoring floor lies below every quoted threshold."""
    v = np.asarray(values, dtype=float)
    out = {}
    for fap in faps:
        thr = float(np.quantile(v, 1.0 - fap))
        out[str(fap)] = {
            "threshold": thr,
            "above_floor": bool(thr > floor_stat),
            "n_exceedances": int(np.sum(v > thr)),
        }
    out["floor_stat"] = float(floor_stat)
    out["n_censored"] = int(np.sum(v <= floor_stat))
    out["all_thresholds_above_floor"] = bool(
        all(o["above_floor"] for k, o in out.items() if k.startswith("0")
            or k.startswith("1e") or isinstance(o, dict) and "threshold" in o)
    )
    return out


def threshold_at_fap(values, fap):
    """Background threshold at a per-segment false-alarm probability.

    Uses the empirical order statistic: the threshold is the value exceeded by
    a fraction ``fap`` of segments.  Also returns the exceedance count, which
    we require to be at least 100 at any quoted threshold, and a
    bootstrap uncertainty.
    """
    v = np.sort(np.asarray(values, dtype=float))
    n = v.size
    k = int(np.floor(fap * n))
    if k < 1:
        return {
            "fap": float(fap),
            "threshold": float("nan"),
            "n_exceedances": 0,
            "n_segments": int(n),
            "sufficient": False,
            "reason": f"fap*n = {fap * n:.2f} < 1 segment",
        }
    thr = float(v[n - k])
    # binomial (Wilson-free, exact-ish) uncertainty on the threshold via
    # neighbouring order statistics
    sd_k = np.sqrt(n * fap * (1.0 - fap))
    lo_i = max(0, int(n - k - 2 * sd_k))
    hi_i = min(n - 1, int(n - k + 2 * sd_k))
    return {
        "fap": float(fap),
        "threshold": thr,
        "threshold_lo_2sigma": float(v[lo_i]),
        "threshold_hi_2sigma": float(v[hi_i]),
        "n_exceedances": int(k),
        "n_segments": int(n),
        "sufficient": bool(k >= 100),
    }


def efficiency_at_threshold(values, threshold):
    """Fraction of injections whose statistic exceeds ``threshold``.

    Returns the efficiency with a binomial (Wilson) 1-sigma interval.
    """
    v = np.asarray(values, dtype=float)
    n = v.size
    k = int(np.sum(v > threshold))
    p = k / n if n else float("nan")
    # Wilson interval, z = 1
    z = 1.0
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {
        "n": int(n),
        "n_above": k,
        "efficiency": float(p),
        "eff_lo": float(max(0.0, centre - half)),
        "eff_hi": float(min(1.0, centre + half)),
    }


# --------------------------------------------------------------------------
# Multi-prior evaluation (Phase 5)
# --------------------------------------------------------------------------


def log_weights_for_priors(grid, prior_names):
    """``{name: (n_td, n_y) log-weight array}`` for several priors on one grid.

    The grid's own ``weights`` attribute is left untouched.  Masked points get
    ``-inf``.  Building these together is what makes the Phase 5 prior study
    nearly free: the expensive part of the statistic -- the ``|z_L|^2`` block --
    does not depend on the prior at all, only the log-sum-exp weighting does.
    """
    from copy import deepcopy

    from . import priors as _pr

    out = {}
    for name in prior_names:
        g = deepcopy(grid)
        _pr.assign_weights(g, name)
        with np.errstate(divide="ignore"):
            out[name] = np.where(g.mask, np.log(g.weights), -np.inf)
    return out


def evaluate_statistics_multi(z1, setup, rho_floor, log_w_by_prior,
                              etas=(0.01, 0.1, 0.5)):
    """All statistics for every prior in one pass over the lens grid.

    ``S_UL`` and ``S_max`` are prior-independent and appear once.  ``S_soft``
    and ``S_mix`` appear once per prior, keyed ``S_soft_<prior>`` and
    ``S_mix_<prior>_eta<eta>``.
    """
    from scipy.special import logsumexp

    grid, u = setup.grid, setup.conv.filter_coeff(1.0)
    floor_stat = 0.5 * rho_floor * rho_floor
    s_ul = 0.5 * float(np.max(np.abs(z1) ** 2))
    cand, censored = candidate_times_with_ul_peak(z1, setup, rho_floor)
    n = z1.size

    p_self = np.abs(z1[cand]) ** 2
    running_max = np.full(cand.size, -np.inf)
    blocks = []      # list of (i_td, active, half_q)

    for i, n_sh in enumerate(setup.shifts):
        active = np.nonzero(grid.mask[i])[0]
        if active.size == 0:
            continue
        z_sh = z1[(cand + n_sh) % n]
        a = grid.a[active]
        num = (
            p_self[:, None]
            + (a * a)[None, :] * (np.abs(z_sh) ** 2)[:, None]
            + 2.0
            * a[None, :]
            * np.real(np.conjugate(u) * z1[cand] * np.conjugate(z_sh))[:, None]
        )
        half_q = 0.5 * num / setup.norm[i, active][None, :]
        running_max = np.maximum(running_max, half_q.max(axis=1))
        blocks.append((i, active, half_q))

    out = {
        "S_UL": s_ul,
        "S_max": floor_stat if censored else float(np.max(running_max)),
        "n_candidates": int(cand.size),
        "censored": bool(censored),
        "floor_stat": float(floor_stat),
    }
    half_ul_cand = 0.5 * p_self
    for name, log_w in log_w_by_prior.items():
        terms = np.concatenate(
            [hq + log_w[i, act][None, :] for i, act, hq in blocks], axis=1
        )
        lse = logsumexp(terms, axis=1)
        out[f"S_soft_{name}"] = (
            floor_stat if censored else float(np.max(lse))
        )
        for eta in etas:
            mixed = logsumexp(
                np.stack(
                    [np.log1p(-eta) + half_ul_cand, np.log(eta) + lse], axis=1
                ),
                axis=1,
            )
            out[f"S_mix_{name}_eta{eta:g}"] = float(np.max(mixed))
    return out


# --------------------------------------------------------------------------
# Per-delay candidate restriction (the fast path for dense grids)
# --------------------------------------------------------------------------


def evaluate_statistics_fast(z1, setup, rho_floor, etas=(0.01, 0.1, 0.5)):
    """Same statistics as :func:`evaluate_statistics`, restricted **per delay**.

    Why this exists
    ---------------

    :func:`evaluate_statistics` forms one candidate set -- the union over delays
    of the translated loud samples -- and evaluates *every* delay at *every*
    candidate time.  For a dense grid that union saturates the whole segment
    (with 1000 delays and ~280 loud samples out of 65536, the union is
    everything), so the prefilter stops saving anything and the cost returns to
    ``n_time * n_lens``.

    But delay ``k`` can only produce a crossing at times in
    ``S union (S - tau_k)``, a set of at most ``2|S|`` samples -- not at the
    translates belonging to *other* delays.  Restricting each delay to its own
    candidates reduces the work from ``n_time * n_td`` to ``2|S| * n_td``, which
    for the production grid is a factor of ~100.

    Exactness
    ---------

    ``S_UL`` and ``S_max`` are **exact**: a maximum decomposes over delays, so
    per-delay candidate sets lose nothing.

    ``S_soft`` and ``S_mix`` couple all delays at a *common* time, and a delay
    whose own candidates exclude time ``t`` still contributes
    ``w_k exp(q_k(t))`` to the sum at ``t``.  Every such contribution is bounded
    by ``exp(floor_stat)``, so the statistic is bracketed:

        lower = log( sum_{k in K(t)} w_k e^{q_k} )
        upper = log( lower_sum + (1 - W(t)) e^{floor_stat} )

    with ``W(t)`` the total prior weight of the delays that *are* evaluated at
    ``t``.  Both ends are returned, together with the bracket width.  Near a
    threshold of ~20 with a floor of 12.5 the width is of order
    ``e^{12.5-20} ~ 5e-4`` in statistic value, i.e. thousands of times smaller
    than the separation between the false-alarm probabilities being calibrated.
    The reported value is the **lower** end, which is the conservative choice
    for a detection statistic, and the width is recorded so the calibration can
    be shown to be insensitive to it.
    """
    grid, u = setup.grid, setup.conv.filter_coeff(1.0)
    floor_stat = 0.5 * rho_floor * rho_floor
    n = z1.size
    abs_z = np.abs(z1)
    s_ul = 0.5 * float(np.max(abs_z) ** 2)

    loud = np.nonzero(abs_z >= setup.image_floor(rho_floor))[0]
    i_ul = int(np.argmax(abs_z))

    out = {
        "S_UL": s_ul,
        "n_loud": int(loud.size),
        "rho_floor": float(rho_floor),
        "floor_stat": float(floor_stat),
    }
    if loud.size == 0:
        out.update(
            S_max=floor_stat, S_soft=floor_stat, S_soft_upper=floor_stat,
            S_soft_bracket=0.0, censored=True, n_evaluated=0,
        )
        for eta in etas:
            out[f"S_mix_eta{eta:g}"] = max(
                s_ul, np.log1p(-eta) + s_ul if eta < 1 else floor_stat
            )
        return out
    out["censored"] = False

    # ---- pass 1: exact S_max, and the shift for the log-sum-exp ----------
    per_delay = []
    s_max = -np.inf
    for i, n_sh in enumerate(setup.shifts):
        active = np.nonzero(grid.mask[i])[0]
        if active.size == 0:
            continue
        cand = candidate_indices(loud, n_sh, n)
        z_sh = z1[(cand + n_sh) % n]
        a = grid.a[active]
        num = (
            (abs_z[cand] ** 2)[:, None]
            + (a * a)[None, :] * (np.abs(z_sh) ** 2)[:, None]
            + 2.0
            * a[None, :]
            * np.real(np.conjugate(u) * z1[cand] * np.conjugate(z_sh))[:, None]
        )
        half_q = 0.5 * num / setup.norm[i, active][None, :]
        s_max = max(s_max, float(half_q.max()))
        per_delay.append((i, active, cand, half_q))
    out["S_max"] = s_max
    out["n_evaluated"] = int(sum(p[3].size for p in per_delay))

    # ---- pass 2: sparse accumulation of the weighted exponential sum -----
    shift = s_max
    acc = np.zeros(n)
    wsum = np.zeros(n)
    for i, active, cand, half_q in per_delay:
        w = np.exp(setup.log_w[i, active])
        acc[cand] += (np.exp(half_q - shift) * w[None, :]).sum(axis=1)
        wsum[cand] += w.sum()

    touched = np.nonzero(acc > 0)[0]
    lower = shift + np.log(acc[touched])

    # Bound the un-evaluated contributions.  If delay k is not evaluated at t
    # then neither |z_1(t)| nor |z_1(t+tau_k)| reaches the image floor, so
    #   q_k(t) <= [ |z_1(t)|^2 + a_k^2 f_img^2 ] / N_k
    # using the *actual* |z_1(t)|, which at quiet times is far below the floor.
    # This is much tighter than the flat `floor_stat` bound and makes the
    # bracket negligible wherever the statistic is large enough to matter.
    f_img = setup.image_floor(rho_floor)
    inv_n_max = float(
        np.max(np.where(grid.mask, 1.0 / setup.norm, -np.inf))
    )
    a2_max = float(np.max(grid.a**2))
    q_bound = inv_n_max * (abs_z[touched] ** 2 + a2_max * f_img * f_img)
    q_bound = np.minimum(q_bound, floor_stat)
    missing = np.clip(1.0 - wsum[touched], 0.0, 1.0) * np.exp(q_bound - shift)
    upper = shift + np.log(acc[touched] + missing)

    out["S_soft"] = float(lower.max())
    out["S_soft_upper"] = float(upper.max())
    out["S_soft_bracket"] = float(out["S_soft_upper"] - out["S_soft"])

    # ---- mixture: the unlensed peak must be included explicitly ----------
    half_ul = 0.5 * abs_z[touched] ** 2
    for eta in etas:
        if eta <= 0.0:
            out[f"S_mix_eta{eta:g}"] = s_ul
            continue
        if eta >= 1.0:
            out[f"S_mix_eta{eta:g}"] = out["S_soft"]
            continue
        m = np.maximum(np.log1p(-eta) + half_ul, np.log(eta) + lower)
        vals = m + np.log(
            np.exp(np.log1p(-eta) + half_ul - m)
            + np.exp(np.log(eta) + lower - m)
        )
        # the unlensed maximum may lie outside `touched`; there the lensed term
        # is below the floor, so this is the correct conservative value
        cand_ul = np.log1p(-eta) + s_ul
        out[f"S_mix_eta{eta:g}"] = float(max(vals.max(), cand_ul))
    return out


def evaluate_statistics_multi_fast(z1, setup, rho_floor, log_w_by_prior,
                                   etas=(0.01, 0.1, 0.5)):
    """Per-delay-restricted evaluation for **several priors** in one pass.

    Combines the cost reduction of :func:`evaluate_statistics_fast` with the
    prior sharing of :func:`evaluate_statistics_multi`.  ``S_UL`` and ``S_max``
    are prior-independent and exact; ``S_soft`` and ``S_mix`` are computed per
    prior and carry the same conservative bracket, which vanishes identically
    wherever the maximum falls at a loud sample -- i.e. throughout the
    calibrated tail.

    This is the evaluator used for the Phase 5 prior study, where the expensive
    ``|z_L|^2`` blocks are computed once and reweighted per prior.
    """
    grid, u = setup.grid, setup.conv.filter_coeff(1.0)
    floor_stat = 0.5 * rho_floor * rho_floor
    n = z1.size
    abs_z = np.abs(z1)
    s_ul = 0.5 * float(np.max(abs_z) ** 2)
    loud = np.nonzero(abs_z >= setup.image_floor(rho_floor))[0]

    out = {"S_UL": s_ul, "n_loud": int(loud.size),
           "floor_stat": float(floor_stat)}
    if loud.size == 0:
        out["S_max"] = floor_stat
        out["censored"] = True
        for name in log_w_by_prior:
            out[f"S_soft_{name}"] = floor_stat
            out[f"S_soft_{name}_bracket"] = 0.0
            for eta in etas:
                out[f"S_mix_{name}_eta{eta:g}"] = max(
                    s_ul, np.log1p(-eta) + s_ul
                )
        return out
    out["censored"] = False

    blocks = []
    s_max = -np.inf
    for i, n_sh in enumerate(setup.shifts):
        active = np.nonzero(grid.mask[i])[0]
        if active.size == 0:
            continue
        cand = candidate_indices(loud, n_sh, n)
        z_sh = z1[(cand + n_sh) % n]
        a = grid.a[active]
        num = (
            (abs_z[cand] ** 2)[:, None]
            + (a * a)[None, :] * (np.abs(z_sh) ** 2)[:, None]
            + 2.0
            * a[None, :]
            * np.real(np.conjugate(u) * z1[cand] * np.conjugate(z_sh))[:, None]
        )
        half_q = 0.5 * num / setup.norm[i, active][None, :]
        s_max = max(s_max, float(half_q.max()))
        blocks.append((i, active, cand, half_q))
    out["S_max"] = s_max

    f_img = setup.image_floor(rho_floor)
    inv_n_max = float(np.max(np.where(grid.mask, 1.0 / setup.norm, -np.inf)))
    a2_max = float(np.max(grid.a**2))

    for name, log_w in log_w_by_prior.items():
        acc = np.zeros(n)
        wsum = np.zeros(n)
        for i, active, cand, half_q in blocks:
            w = np.exp(log_w[i, active])
            acc[cand] += (np.exp(half_q - s_max) * w[None, :]).sum(axis=1)
            wsum[cand] += w.sum()
        touched = np.nonzero(acc > 0)[0]
        lower = s_max + np.log(acc[touched])
        q_bound = np.minimum(
            inv_n_max * (abs_z[touched] ** 2 + a2_max * f_img * f_img),
            floor_stat,
        )
        missing = np.clip(1.0 - wsum[touched], 0.0, 1.0) * np.exp(
            q_bound - s_max
        )
        upper = s_max + np.log(acc[touched] + missing)

        j = int(np.argmax(lower))
        out[f"S_soft_{name}"] = float(lower[j])
        out[f"S_soft_{name}_bracket"] = float(upper.max() - lower.max())

        half_ul = 0.5 * abs_z[touched] ** 2
        for eta in etas:
            m = np.maximum(np.log1p(-eta) + half_ul, np.log(eta) + lower)
            vals = m + np.log(
                np.exp(np.log1p(-eta) + half_ul - m)
                + np.exp(np.log(eta) + lower - m)
            )
            out[f"S_mix_{name}_eta{eta:g}"] = float(
                max(vals.max(), np.log1p(-eta) + s_ul)
            )
    return out
