"""Noise generation, injections, and segment-length validation.

Noise normalization
-------------------

For a one-sided PSD ``S`` and the project's inner product
``(a|b) = 4 df sum a conj(b) / S``, stationary Gaussian noise must satisfy

    E[ |ntilde_k|^2 ] = S_k / (2 df)

for the complex SNR of a *unit-norm* template to have unit variance **per
quadrature**, i.e. ``E|z|^2 = 2`` with ``|z|^2`` distributed as chi-squared with
two degrees of freedom.  Derivation: with ``E[ntilde_k conj(ntilde_k')] =
delta_kk' sigma_k^2``,

    E|z|^2 = 16 df^2 sum_k sigma_k^2 |h_k|^2 / S_k^2

and requiring this to equal ``2 * (h|h) = 2 * 4 df sum_k |h_k|^2 / S_k``
term by term gives ``sigma_k^2 = S_k / (2 df)``.

Getting this factor wrong by ``sqrt(2)`` would rescale every threshold and
efficiency curve in Phase 4, so it is defined once here and pinned by
``tests/test_simulation.py::test_snr_variance_is_unity_per_quadrature``.
"""

from __future__ import annotations

import numpy as np

from . import amplification as amp
from . import waveforms as wf


# --------------------------------------------------------------------------
# Segment-length validation
# --------------------------------------------------------------------------


def chirp_time(cfg, f_lower=None, order=7) -> float:
    """Inspiral duration from ``f_lower`` to coalescence, in seconds."""
    from pycbc.waveform.spa_tmplt import findchirp_chirptime

    return float(
        findchirp_chirptime(
            cfg.mass1, cfg.mass2, f_lower or cfg.f_lower, order
        )
    )


def validate_segment(cfg, t_d_max=0.5, safety=1.25):
    """Raise if the segment is too short for the signal plus the largest delay.

    The FFTs here are circular, so a signal longer than the segment wraps
    around and corrupts both the SNR series and the injection.  Equal-mass
    ``M_tot = 11 Msun`` from 20 Hz lasts **16.43 s**, which overflows a 16 s
    segment -- hence this check rather than a rule of thumb.
    """
    t_chirp = chirp_time(cfg)
    needed = safety * (t_chirp + t_d_max)
    if cfg.seg_dur < needed:
        raise ValueError(
            f"seg_dur={cfg.seg_dur:g} s too short: chirp_time="
            f"{t_chirp:.3f} s at f_lower={cfg.f_lower:g} Hz plus t_d_max="
            f"{t_d_max:g} s needs >= {needed:.3f} s "
            f"(safety factor {safety:g}). Use seg_dur="
            f"{2 ** int(np.ceil(np.log2(needed))):g} s."
        )
    return {"chirp_time_s": t_chirp, "seg_dur_s": cfg.seg_dur, "required_s": needed}


def suggested_seg_dur(cfg, t_d_max=0.5, safety=1.25) -> float:
    """Smallest power-of-two segment duration that passes :func:`validate_segment`."""
    needed = safety * (chirp_time(cfg) + t_d_max)
    return float(2 ** int(np.ceil(np.log2(needed))))


# --------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------


def noise_frequency_series(cfg, psd, rng):
    """One realization of stationary Gaussian noise, frequency domain.

    ``E[|n_k|^2] = S_k / (2 df)``; out-of-band bins (``S = inf``) are exactly
    zero.  ``rng`` must be a ``numpy.random.Generator`` so realizations are
    reproducible from a recorded seed.
    """
    sigma_k = np.sqrt(psd / (2.0 * cfg.delta_f))
    sigma_k = np.where(np.isfinite(sigma_k), sigma_k, 0.0)
    # Var(Re) = Var(Im) = sigma_k^2 / 2  =>  E|n|^2 = sigma_k^2
    scale = sigma_k / np.sqrt(2.0)
    return scale * (
        rng.normal(size=cfg.n_freq) + 1j * rng.normal(size=cfg.n_freq)
    )


def noise_snr_series(cfg, psd, template, rng, oversample=1):
    """Complex SNR series of a pure-noise realization (unit variance/quadrature)."""
    n = noise_frequency_series(cfg, psd, rng)
    return wf.snr_series(n, template, psd, cfg, oversample=oversample)


# --------------------------------------------------------------------------
# Injections
# --------------------------------------------------------------------------


def lensed_injection(
    cfg,
    psd,
    template,
    M_Lz,
    y,
    t0,
    optimal_snr=None,
    wave_optics=True,
    backend="mpmath",
    cache_dir="cache/F_ML",
    n_proc=1,
):
    """A lensed CBC injection in the frequency domain.

    ``template`` is the unit-norm unlensed template.  The amplification factor
    is the exact wave-optics ``F_ML`` when ``wave_optics`` is true, else
    ``F_GO``.  If ``optimal_snr`` is given the injection is rescaled so that
    ``sqrt((d|d)) == optimal_snr``.

    Returns ``(data, info)``.
    """
    freqs = cfg.frequencies()
    if wave_optics:
        F = amp.F_ML_on_grid(
            freqs, M_Lz, y, backend=backend, cache_dir=cache_dir, n_proc=n_proc
        )
    else:
        F = amp.F_GO(freqs, M_Lz, y)

    data = wf.fd_delay(template * F, freqs, t0)
    sigma_raw = wf.sigma(data, psd, cfg.delta_f)
    scale = 1.0 if optimal_snr is None else optimal_snr / sigma_raw
    data = data * scale

    return data, {
        "M_Lz": float(M_Lz),
        "y": float(y),
        "t0_s": float(t0),
        "t_d_s": float(amp.time_delay(M_Lz, y)),
        "a": float(amp.a_of_y(y)),
        "wave_optics": bool(wave_optics),
        "optimal_snr": float(sigma_raw * scale),
        "amplitude_scale": float(scale),
    }


def unlensed_injection(cfg, psd, template, t0, optimal_snr=None):
    """An unlensed CBC injection, for the ``S_mix`` baseline-safety test."""
    data = wf.fd_delay(template, cfg.frequencies(), t0)
    sigma_raw = wf.sigma(data, psd, cfg.delta_f)
    scale = 1.0 if optimal_snr is None else optimal_snr / sigma_raw
    return data * scale, {
        "t0_s": float(t0),
        "optimal_snr": float(sigma_raw * scale),
    }
