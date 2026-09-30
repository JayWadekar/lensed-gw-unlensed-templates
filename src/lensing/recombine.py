"""The core algebra: evaluating every two-image lensed matched filter from the
ordinary complex CBC SNR series and the template autocorrelation.

For a unit-norm unlensed template ``h`` the geometric-optics family
is, up to an irrelevant absolute magnification,

    h_L,t = h_t + c h_{t + tau},     c = s_M i a,   tau = s_T t_d

and filtering data against it gives, *exactly*,

    z_L(t; t_d, a) = [ z_1(t) + conj(c) z_1(t + tau) ] / sqrt(N),
    N(t_d, a)      = 1 + a^2 + 2 Re[ conj(c) C(tau) ],
    C(tau)         = (h_0 | h_tau),   C(0) = 1.

Three implementations are provided and must agree:

``explicit_lensed_snr``
    builds ``h_L`` in the frequency domain and filters the data against it.
    This is the *source of truth*.

``recombine_direct``
    the readable complex recombination above.

``recombine_expanded``
    the same statistic expanded as an explicit quadratic in ``a``, vectorized
    over a lens grid.  Both its numerator and denominator are generated from
    the single complex coefficient ``u = conv.filter_coeff(1)``, so no sign can
    be hard-coded independently.

Writing ``conj(c) = a u`` with ``u = conj(s_M i) = -s_M i`` gives the expanded
form used in production::

    |numerator|^2 = |z_1|^2 + a^2 |z_1s|^2 + 2 a Re[ conj(u) z_1 conj(z_1s) ]
    N             = 1 + a^2 + 2 a Re[ u C(tau) ]

with ``z_1s(t) = z_1(t + tau)``.
"""

from __future__ import annotations

import numpy as np

from . import conventions, waveforms


# --------------------------------------------------------------------------
# Explicit reference: build the lensed template and filter with it
# --------------------------------------------------------------------------


def explicit_lensed_template(h, freqs, t_d, a, conv=None):
    """Frequency-domain two-image template ``h + c * h_delayed(tau)``.

    Not normalized; ``h`` may be any (e.g. already unit-norm) template.
    """
    if conv is None:
        conv = conventions.DEFAULT
    tau = conv.lag(t_d)
    c = conv.image_coeff(a)
    return h + c * waveforms.fd_delay(h, freqs, tau, conv=conv)


def explicit_lensed_norm(h, psd, cfg, t_d, a, conv=None) -> float:
    """``sqrt(Re (h_L|h_L))`` from the explicitly constructed template."""
    hl = explicit_lensed_template(h, cfg.frequencies(), t_d, a, conv=conv)
    return waveforms.sigma(hl, psd, cfg.delta_f)


def explicit_lensed_snr(data, h, psd, cfg, t_d, a, conv=None, oversample=1):
    """Normalized complex SNR from explicit filtering against ``h_L``.

    Returns ``(z_L, delta_t)``.  This is the reference the recombined forms
    are tested against.
    """
    hl = explicit_lensed_template(h, cfg.frequencies(), t_d, a, conv=conv)
    return waveforms.snr_series(data, hl, psd, cfg, oversample=oversample)


def explicit_lensed_snr_at_times(data, h, psd, cfg, t_d, a, times, conv=None):
    """Exact ``z_L(t)`` at arbitrary times, by direct summation."""
    hl = explicit_lensed_template(h, cfg.frequencies(), t_d, a, conv=conv)
    return waveforms.snr_at_times(data, hl, psd, cfg, times)


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------


def lensed_norm_sq(C_tau, a, conv=None):
    """``N = 1 + a^2 + 2 Re[conj(c) C(tau)]``, broadcasting over ``C_tau``/``a``.

    ``C_tau`` must be the autocorrelation evaluated at the *signed* lag
    ``tau = conv.lag(t_d)``, not at ``t_d``.
    """
    if conv is None:
        conv = conventions.DEFAULT
    u = conv.filter_coeff(1.0)
    a = np.asarray(a, dtype=float)
    C_tau = np.asarray(C_tau, dtype=complex)
    return 1.0 + a * a + 2.0 * a * np.real(u * C_tau)


def norm_is_positive(C_tau, a, conv=None, floor=0.0) -> bool:
    """Phase 0 item 7: template norm positive over the whole grid."""
    return bool(np.all(lensed_norm_sq(C_tau, a, conv=conv) > floor))


# --------------------------------------------------------------------------
# Recombination
# --------------------------------------------------------------------------


def recombine_direct(z1, z1_shifted, C_tau, a, conv=None):
    """Readable complex recombination ``(z_1 + conj(c) z_1s) / sqrt(N)``."""
    if conv is None:
        conv = conventions.DEFAULT
    c_conj = conv.filter_coeff(a)
    num = z1 + c_conj * z1_shifted
    return num / np.sqrt(lensed_norm_sq(C_tau, a, conv=conv))


def recombine_expanded_sq(z1, z1_shifted, C_tau, a, conv=None):
    """``|z_L|^2`` as an explicit quadratic in ``a`` (production form).

    ``z1`` and ``z1_shifted`` broadcast against ``a`` and ``C_tau``; the usual
    production call has ``z1`` of shape ``(n_time, 1)`` and ``a``, ``C_tau`` of
    shape ``(1, n_lens)``.
    """
    if conv is None:
        conv = conventions.DEFAULT
    u = conv.filter_coeff(1.0)
    a = np.asarray(a, dtype=float)

    p_self = np.abs(z1) ** 2
    p_shift = np.abs(z1_shifted) ** 2
    cross = np.real(np.conjugate(u) * z1 * np.conjugate(z1_shifted))

    num = p_self + a * a * p_shift + 2.0 * a * cross
    return num / lensed_norm_sq(C_tau, a, conv=conv)


# --------------------------------------------------------------------------
# Shifted SNR series on an oversampled grid
# --------------------------------------------------------------------------


def shift_snr_series(z, delta_t, tau):
    """Return ``z(t + tau)`` on the same grid, by integer-sample rolling.

    ``tau`` is rounded to the nearest available sample; the caller is
    responsible for choosing ``delta_t`` (i.e. the oversampling factor) fine
    enough that the residual fractional delay is negligible, which is what the
    convergence study measures.  The shift is circular, matching the
    periodic FFT convention of the SNR series itself.
    """
    n_shift = int(np.rint(tau / delta_t))
    return np.roll(z, -n_shift), n_shift * delta_t


def snr_pair_on_grid(z, delta_t, taus):
    """Stack ``z(t + tau)`` for each ``tau``: shape ``(n_time, n_tau)``.

    Returns ``(z_shifted, tau_realized)`` where ``tau_realized`` are the
    lags actually used after rounding to the sample grid.
    """
    taus = np.atleast_1d(np.asarray(taus, dtype=float))
    out = np.empty((z.size, taus.size), dtype=complex)
    realized = np.empty(taus.size, dtype=float)
    for j, tau in enumerate(taus):
        shifted, tr = shift_snr_series(z, delta_t, tau)
        out[:, j] = shifted
        realized[j] = tr
    return out, realized
