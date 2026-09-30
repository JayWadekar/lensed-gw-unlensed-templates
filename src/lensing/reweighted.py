"""Re-weighted SNR and the sensitive volume it implies, at fixed threshold.

A search does not rank triggers by SNR but by a chi-squared-**re-weighted** SNR,
so a signal that fails the signal-consistency test is demoted even when it is
loud. This module turns the measured chi-squared excess of a lensed injection
into the quantity that actually matters -- sensitive volume -- without needing
a background distribution: the re-weighting is a deterministic function of
``(rho, chi2_r)``, and comparing two searches *at the same threshold* on the
same re-weighted statistic needs nothing else.

The one physical input beyond the veto formula is how the excess scales with
distance. Noise-free, the chi-squared excess of a mismatched signal is
``rho^2`` times a mismatch that does not depend on distance, so

    chi2_r(rho) = 1 + excess_ref * (rho / rho_ref)^2

exactly, with ``excess_ref`` measured once at a reference SNR. That is what
:func:`chi2_at_snr` encodes, and it is verified against a direct recomputation
in ``tests/test_reweighted.py``.

The consequence worth stating plainly: because the excess grows as ``rho^2``
while the numerator grows as ``rho``, the re-weighted SNR of a mismatched
signal **saturates**,

    rho_hat -> 2^(1/6) * rho_ref / sqrt(excess_ref)   as rho -> infinity,

so a sufficiently mismatched signal is undetectable *at any distance*, and its
sensitive volume is zero rather than merely reduced. :func:`saturation_newsnr`
returns that ceiling.

Caveat, stated rather than buried: a volume ratio computed here holds at a
**fixed threshold on the re-weighted statistic**. It does not include the extra
trials factor incurred by searching a lens grid (handled separately for the
SNR-only statistic), nor the four-fold Morse-phase maximization. It isolates the effect of the consistency test.
"""

from __future__ import annotations

import numpy as np

#: PyCBC's ``newsnr`` exponents (``pycbc.events.ranking.newsnr``).
Q_DEFAULT = 6.0
N_DEFAULT = 2.0


def new_snr(rho, chi2_r, q: float = Q_DEFAULT, n: float = N_DEFAULT):
    """Re-weighted SNR, identical to :func:`pycbc.events.ranking.newsnr`.

    ``rho_hat = rho * [ (1 + chi2_r^(q/n)) / 2 ]^(-1/q)`` when ``chi2_r > 1``,
    and ``rho`` otherwise.
    """
    rho = np.asarray(rho, dtype=float)
    chi2_r = np.asarray(chi2_r, dtype=float)
    out = np.array(np.broadcast_to(rho, np.broadcast(rho, chi2_r).shape),
                   dtype=float, copy=True)
    hot = chi2_r > 1.0
    if np.any(hot):
        pen = (0.5 * (1.0 + chi2_r ** (q / n))) ** (-1.0 / q)
        out = np.where(hot, out * pen, out)
    return out


def chi2_at_snr(excess_ref, rho_opt, rho_ref):
    """``chi2_r`` at optimal SNR ``rho_opt``, from the excess measured at
    ``rho_ref``.

    Exact for a noise-free waveform mismatch, whose chi-squared contribution is
    proportional to ``rho^2``.
    """
    excess_ref = np.asarray(excess_ref, dtype=float)
    scale = (np.asarray(rho_opt, dtype=float) / float(rho_ref)) ** 2
    return 1.0 + excess_ref * scale


def saturation_newsnr(excess_ref, rho_ref, q: float = Q_DEFAULT,
                      n: float = N_DEFAULT):
    """The ceiling of ``rho_hat`` as ``rho_opt -> infinity``.

    ``rho_hat -> 2^(1/q) * rho_ref / sqrt(excess_ref)`` for ``q/n = 3``. Infinite
    when the excess vanishes.
    """
    e = np.asarray(excess_ref, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = 2.0 ** (1.0 / q) * float(rho_ref) / np.sqrt(e)
    return np.where(e > 0, val, np.inf)


def threshold_optimal_snr(excess_ref, recovery, rho_ref, rho_hat_th,
                          q: float = Q_DEFAULT, n: float = N_DEFAULT,
                          rho_max=1e6):
    """Smallest injected optimal SNR that reaches ``rho_hat_th``.

    ``recovery`` is the fraction of the optimal SNR the template actually
    collects (``rho_recovered / rho_optimal``), measured per cell. Since source
    amplitude scales as ``1/D``, this quantity is inversely proportional to the
    threshold distance, so ``V ~ rho_th^-3``.

    Returns ``inf`` where the re-weighted SNR saturates below the threshold --
    i.e. where the signal is undetectable at *any* distance.
    """
    excess_ref = np.atleast_1d(np.asarray(excess_ref, dtype=float))
    recovery = np.atleast_1d(np.asarray(recovery, dtype=float))
    excess_ref, recovery = np.broadcast_arrays(excess_ref, recovery)
    out = np.full(excess_ref.shape, np.inf)

    def rho_hat(rho_opt, e, m):
        return new_snr(rho_opt * m, chi2_at_snr(e, rho_opt, rho_ref), q=q, n=n)

    ceil = saturation_newsnr(excess_ref, rho_ref, q=q, n=n) * recovery
    for k in np.ndindex(excess_ref.shape):
        e, m = float(excess_ref[k]), float(recovery[k])
        if not np.isfinite(e) or not np.isfinite(m) or m <= 0:
            continue
        if np.isfinite(ceil[k]) and ceil[k] <= rho_hat_th:
            continue                      # saturates below threshold: V = 0
        # rho_hat is monotonically increasing in rho_opt, so bisect
        lo, hi = 1e-3, rho_max
        if float(rho_hat(hi, e, m)) < rho_hat_th:
            continue
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if float(rho_hat(mid, e, m)) < rho_hat_th:
                lo = mid
            else:
                hi = mid
        out[k] = 0.5 * (lo + hi)
    return out


def volume_ratio(excess_a, recovery_a, excess_b, recovery_b, rho_ref,
                 rho_hat_th, rho_hat_th_b=None, **kw):
    """``V_a / V_b``: ``(rho_th,b / rho_th,a)^3``.

    ``rho_hat_th`` applies to search ``a``; ``rho_hat_th_b`` to search ``b``,
    defaulting to the same value. Passing **different** thresholds is what makes
    the comparison fixed-false-alarm rather than fixed-threshold: each search is
    run at its own FAP, and a search that examines more templates pays for it
    through a higher threshold. Those thresholds come from measured background
    distributions (``scripts/simulate/appendix_chisq_background.py``), not from a model.

    ``0`` where search ``a`` saturates below its threshold, ``inf`` where ``b``
    does.
    """
    if rho_hat_th_b is None:
        rho_hat_th_b = rho_hat_th
    ra = threshold_optimal_snr(excess_a, recovery_a, rho_ref, rho_hat_th, **kw)
    rb = threshold_optimal_snr(excess_b, recovery_b, rho_ref, rho_hat_th_b, **kw)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (rb / ra) ** 3
    out = np.where(np.isinf(ra) & np.isinf(rb), np.nan, out)
    out = np.where(np.isinf(ra) & ~np.isinf(rb), 0.0, out)
    return out
