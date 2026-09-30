"""Analysis configuration, PSD, templates, and matched-filter primitives.

Everything here is written directly on top of ``numpy`` arrays living on a
one-sided frequency grid ``f_k = k / T``, ``k = 0 ... N/2``, with ``N = T * fs``.
The matched filter is implemented explicitly rather than called from PyCBC so
that the conventions of :mod:`lensing.conventions` are enforced in one place;
:func:`snr_series` is cross-checked against ``pycbc.filter.matched_filter`` in
the Phase 0 tests.

Definitions used throughout::

    (a|b)    = 4 df sum_k  a_k conj(b_k) / S_k                 (complex)
    sigma(h) = sqrt(Re (h|h))
    z(t)     = (d | h_t) / sigma(h),   h_t delayed to coalesce at t
             = 4 df sum_k [d_k conj(h_k) / S_k] e^{2 pi i f_k t} / sigma(h)

so ``z`` is the *analytic* (complex) SNR time series: only positive
frequencies contribute, and its modulus is the phase-maximized SNR.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
import hashlib
import json

import numpy as np


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisConfig:
    """Frozen description of a single-detector analysis segment.

    ``seg_dur`` is the segment duration in seconds and ``sample_rate`` its
    sampling rate, so the frequency spacing is ``delta_f = 1 / seg_dur`` and
    the one-sided grid has ``N / 2 + 1`` points with ``N = seg_dur *
    sample_rate``.
    """

    seg_dur: float = 64.0
    sample_rate: float = 4096.0
    f_lower: float = 20.0
    f_final: float = 1024.0
    psd_name: str = "aLIGOZeroDetHighPower"
    approximant: str = "IMRPhenomD"
    mass1: float = 25.0
    mass2: float = 25.0
    spin1z: float = 0.0
    spin2z: float = 0.0
    distance: float = 1000.0
    extra: dict = field(default_factory=dict)

    # -- derived quantities ------------------------------------------------

    @property
    def n_time(self) -> int:
        return int(round(self.seg_dur * self.sample_rate))

    @property
    def n_freq(self) -> int:
        return self.n_time // 2 + 1

    @property
    def delta_f(self) -> float:
        return 1.0 / self.seg_dur

    @property
    def delta_t(self) -> float:
        return 1.0 / self.sample_rate

    @property
    def total_mass(self) -> float:
        return self.mass1 + self.mass2

    def frequencies(self) -> np.ndarray:
        return np.arange(self.n_freq) * self.delta_f

    def band_mask(self) -> np.ndarray:
        f = self.frequencies()
        return (f >= self.f_lower) & (f <= self.f_final)

    def hash(self) -> str:
        """Stable content hash of the configuration, for result provenance."""
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["config_hash"] = self.hash()
        return d


# --------------------------------------------------------------------------
# PSD and templates
# --------------------------------------------------------------------------


def make_psd(cfg: AnalysisConfig) -> np.ndarray:
    """One-sided PSD on ``cfg.frequencies()``, ``inf`` outside the band.

    Out-of-band bins are set to ``inf`` (not zero) so that ``1 / S`` is exactly
    zero there and no band-edge bookkeeping is needed downstream.
    """
    from pycbc.psd import analytical

    fn = getattr(analytical, cfg.psd_name)
    psd = np.asarray(
        fn(cfg.n_freq, cfg.delta_f, cfg.f_lower).numpy(), dtype=float
    ).copy()
    mask = cfg.band_mask()
    psd[~mask] = np.inf
    psd[mask & (psd <= 0)] = np.inf
    return psd


def make_waveform(cfg: AnalysisConfig, **overrides) -> np.ndarray:
    """Frequency-domain CBC waveform on ``cfg.frequencies()`` (complex array).

    The coalescence sits at ``t = 0`` of the segment, i.e. PyCBC's native
    frequency-domain convention with no extra time shift applied.
    """
    from pycbc.waveform import get_fd_waveform

    params = dict(
        approximant=cfg.approximant,
        mass1=cfg.mass1,
        mass2=cfg.mass2,
        spin1z=cfg.spin1z,
        spin2z=cfg.spin2z,
        distance=cfg.distance,
        delta_f=cfg.delta_f,
        f_lower=cfg.f_lower,
        f_final=cfg.f_final,
    )
    params.update(overrides)
    hp, _ = get_fd_waveform(**params)
    out = np.zeros(cfg.n_freq, dtype=complex)
    n = min(cfg.n_freq, len(hp))
    out[:n] = np.asarray(hp.numpy()[:n], dtype=complex)
    out[~cfg.band_mask()] = 0.0
    return out


# --------------------------------------------------------------------------
# Inner products
# --------------------------------------------------------------------------


def inner(a: np.ndarray, b: np.ndarray, psd: np.ndarray, delta_f: float) -> complex:
    """Complex inner product ``(a|b) = 4 df sum a conj(b) / S``."""
    return 4.0 * delta_f * np.sum(a * np.conjugate(b) / psd)


def sigma(h: np.ndarray, psd: np.ndarray, delta_f: float) -> float:
    """Template norm ``sqrt(Re (h|h))``."""
    return float(np.sqrt(np.real(inner(h, h, psd, delta_f))))


def normalize(h: np.ndarray, psd: np.ndarray, delta_f: float) -> np.ndarray:
    """Return ``h / sigma(h)``, so that ``(h|h) = 1``."""
    return h / sigma(h, psd, delta_f)


def fd_delay(h: np.ndarray, freqs: np.ndarray, tau: float, conv=None) -> np.ndarray:
    """Delay a frequency-domain waveform by ``tau`` seconds.

    Uses ``exp(fd_delay_sign * 2 pi i f tau)`` with the sign taken from the
    convention object (default ``-1``, matching ``numpy.fft.rfft``).
    """
    from . import conventions

    if conv is None:
        conv = conventions.DEFAULT
    return h * np.exp(conv.fd_delay_sign * 2j * np.pi * freqs * tau)


# --------------------------------------------------------------------------
# Matched filter
# --------------------------------------------------------------------------


def snr_series(
    data: np.ndarray,
    template: np.ndarray,
    psd: np.ndarray,
    cfg: AnalysisConfig,
    oversample: int = 1,
    normalized: bool = True,
):
    """Complex SNR time series ``z(t) = (d|h_t) / sigma(h)``.

    Returns ``(z, delta_t)`` where ``z`` has ``oversample * n_time`` samples
    spaced by ``delta_t = 1 / (oversample * sample_rate)``.  Oversampling is
    performed by zero-padding the one-sided integrand in the *frequency*
    domain, which is the band-limited fractional-delay interpolation required
    by design -- no interpolation of ``|z|`` is ever performed.
    """
    integrand = 4.0 * cfg.delta_f * data * np.conjugate(template) / psd
    if normalized:
        integrand = integrand / sigma(template, psd, cfg.delta_f)

    n_full = cfg.n_time * int(oversample)
    spectrum = np.zeros(n_full, dtype=complex)
    spectrum[: cfg.n_freq] = integrand
    z = np.fft.ifft(spectrum) * n_full
    return z, cfg.delta_t / int(oversample)


def snr_at_times(
    data: np.ndarray,
    template: np.ndarray,
    psd: np.ndarray,
    cfg: AnalysisConfig,
    times,
    normalized: bool = True,
) -> np.ndarray:
    """Exact ``z(t)`` at arbitrary times by direct summation (no interpolation).

    O(n_freq * n_times); intended for gate tests and reference values, not for
    production searches.
    """
    integrand = 4.0 * cfg.delta_f * data * np.conjugate(template) / psd
    if normalized:
        integrand = integrand / sigma(template, psd, cfg.delta_f)
    integrand = np.where(np.isfinite(integrand), integrand, 0.0)
    f = cfg.frequencies()
    times = np.atleast_1d(np.asarray(times, dtype=float))
    phase = np.exp(2j * np.pi * np.outer(times, f))
    return phase @ integrand


def autocorr_at_lags(
    template: np.ndarray,
    psd: np.ndarray,
    cfg: AnalysisConfig,
    lags,
    normalized: bool = True,
) -> np.ndarray:
    """Exact template autocorrelation ``C(tau) = (h_0 | h_tau)``.

    With ``h_tau`` delayed by ``tau``, the antilinear inner product gives

        C(tau) = 4 df sum_k |h_k|^2 / S_k * exp(-fd_delay_sign * 2 pi i f tau)

    which for the default ``fd_delay_sign = -1`` is
    ``4 df sum |h|^2 / S e^{+2 pi i f tau}``.  ``C(0) = 1`` for a normalized
    template and ``C(-tau) = conj(C(tau))``.
    """
    from . import conventions

    power = 4.0 * cfg.delta_f * np.abs(template) ** 2 / psd
    power = np.where(np.isfinite(power), power, 0.0)
    if normalized:
        power = power / np.sum(power)
    f = cfg.frequencies()
    lags = np.atleast_1d(np.asarray(lags, dtype=float))
    sign = -conventions.DEFAULT.fd_delay_sign
    phase = np.exp(sign * 2j * np.pi * np.outer(lags, f))
    return phase @ power


def autocorr_series(
    template: np.ndarray,
    psd: np.ndarray,
    cfg: AnalysisConfig,
    oversample: int = 1,
    normalized: bool = True,
):
    """``C(tau)`` on a regular oversampled lag grid.

    Returns ``(C, lags)`` with ``lags`` the *signed*, ``fftshift``-ordered lag
    axis in seconds, monotonically increasing.
    """
    from . import conventions

    power = 4.0 * cfg.delta_f * np.abs(template) ** 2 / psd
    power = np.where(np.isfinite(power), power, 0.0)
    if normalized:
        power = power / np.sum(power)

    n_full = cfg.n_time * int(oversample)
    spectrum = np.zeros(n_full, dtype=complex)
    spectrum[: cfg.n_freq] = power
    c = np.fft.ifft(spectrum) * n_full
    if -conventions.DEFAULT.fd_delay_sign < 0:  # pragma: no cover
        c = np.conjugate(c)
    dt = cfg.delta_t / int(oversample)
    lags = np.fft.fftfreq(n_full, d=1.0 / (n_full * dt))
    order = np.argsort(lags)
    return c[order], lags[order]


def autocorr_at_sample_lags(
    template: np.ndarray,
    psd: np.ndarray,
    cfg: AnalysisConfig,
    lags,
    oversample: int = 1,
    normalized: bool = True,
) -> np.ndarray:
    """``C(tau)`` at lags lying exactly on the oversampled sample grid.

    Same quantity as :func:`autocorr_at_lags`, but obtained from a single
    inverse FFT (:func:`autocorr_series`) plus integer indexing instead of the
    ``O(n_lags * n_freq)`` direct sum.  The two agree to floating-point
    roundoff whenever every lag is an integer multiple of
    ``cfg.delta_t / oversample``, which is guaranteed for grids passed through
    :func:`lensing.grids.snap_grid_to_samples`.

    Off-grid lags raise rather than round, because silently snapping here would
    move the delay without the caller knowing.  Use :func:`autocorr_at_lags`
    for arbitrary lags.
    """
    dt = cfg.delta_t / int(oversample)
    lags = np.atleast_1d(np.asarray(lags, dtype=float))
    k = np.rint(lags / dt)
    off = np.max(np.abs(lags - k * dt)) if lags.size else 0.0
    if off > 1e-9 * dt:
        raise ValueError(
            "lags are not on the oversampled sample grid: worst offset "
            f"{off:.3e} s exceeds 1e-9 * dt = {1e-9 * dt:.3e} s"
        )

    c, _ = autocorr_series(
        template, psd, cfg, oversample=oversample, normalized=normalized
    )
    n_full = c.size
    idx = k.astype(int) + n_full // 2
    if idx.min() < 0 or idx.max() >= n_full:
        raise ValueError("lag outside the representable range of the segment")
    return c[idx]
