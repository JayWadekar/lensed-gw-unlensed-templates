"""The generalized relative Morse phase.

The claim the paper makes is that the recombination never used the
fact that the relative image coefficient is *pure imaginary*: it is generated
from one complex number, so any relative phase works with no new algebra.

That is a testable statement, not a remark, and this is where it is tested:
``test_recombination_is_exact_for_every_morse_phase`` runs the Phase 0 item-5
exactness check -- recombined statistic versus an explicitly constructed and
filtered frequency-domain template -- for all four phases.

The other load-bearing test here is
``test_default_convention_is_bit_identical``, which pins that generalizing the
convention object did not perturb the frozen ``q = 1`` configuration that
``results/frozen/`` depends on.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "checks"))

from lensing import conventions as cv
from lensing import recombine as rc
from lensing import waveforms as wf

GATE_TOL = 1e-10


@pytest.fixture(scope="module")
def setup():
    cfg = wf.AnalysisConfig(seg_dur=8.0, sample_rate=4096.0, mass1=25.0, mass2=25.0)
    psd = wf.make_psd(cfg)
    h = wf.normalize(wf.make_waveform(cfg), psd, cfg.delta_f)
    return cfg, psd, h


# --------------------------------------------------------------------------
# Nothing frozen moved
# --------------------------------------------------------------------------


def test_default_convention_is_bit_identical():
    """The default must still be exactly ``c = s_M i a``.

    Every frozen product was generated with that coefficient; if this drifts,
    results/frozen/ is silently invalidated.
    """
    d = cv.DEFAULT
    assert d.rel_morse_quarters == 1
    for a in (0.0, 0.172, 0.5, 0.9899, 1.0):
        assert d.image_coeff(a) == d.morse_sign * 1j * a
    assert d.filter_coeff(1.0) == -1j
    assert d.norm_cross_sign() == d.morse_sign


def test_quarter_turns_are_exact_complex_literals():
    """No floating-point ``pow`` in the generator of every downstream sign."""
    assert cv.QUARTER_TURN == {0: 1 + 0j, 1: 1j, 2: -1 + 0j, 3: -1j}
    for q, w in cv.QUARTER_TURN.items():
        assert abs(w) == 1.0                      # exactly, not approximately
        assert w.real in (-1.0, 0.0, 1.0) and w.imag in (-1.0, 0.0, 1.0)


def test_coefficient_is_linear_in_a_for_every_phase():
    """The expanded quadratic form requires ``filter_coeff(a) = a u``."""
    for conv in cv.all_rel_morse():
        u = conv.filter_coeff(1.0)
        for a in (0.0, 0.37, 1.0):
            assert conv.filter_coeff(a) == a * u


def test_norm_cross_sign_refuses_even_phases():
    """For an even ``q`` the cross term carries ``Re C``; no single sign
    describes it, so the integer accessor must refuse rather than mislead."""
    for q in (0, 2):
        with pytest.raises(ValueError, match="pure-imaginary"):
            cv.DEFAULT.with_rel_morse(q).norm_cross_sign()
    for q in (1, 3):
        assert cv.DEFAULT.with_rel_morse(q).norm_cross_sign() in (-1, +1)


def test_with_rel_morse_rejects_nonsense():
    with pytest.raises(ValueError, match="rel_morse_quarters"):
        cv.DEFAULT.with_rel_morse(4)


# --------------------------------------------------------------------------
# THE gate: exactness for every relative Morse phase
# --------------------------------------------------------------------------


@pytest.mark.parametrize("q", [0, 1, 2, 3])
@pytest.mark.parametrize("with_noise", [False, True])
def test_recombination_is_exact_for_every_morse_phase(setup, q, with_noise):
    """Phase 0 item 5, run for all four relative Morse phases.

    This is the evidence for the paper's generalization claim: the recombined
    statistic equals an explicitly built-and-filtered two-image template to
    machine precision for *every* relative phase, not just the deployed one.
    """
    cfg, psd, h = setup
    from phase0_conventions import item_5_pointwise

    conv = cv.DEFAULT.with_rel_morse(q)
    r = item_5_pointwise(
        cfg, psd, h, 2000.0, 0.3, 0.5 * cfg.seg_dur,
        with_noise=with_noise, conv=conv,
    )
    assert r["rel_morse_quarters"] == q
    assert r["max_rel_err_direct"] < GATE_TOL, (q, r)
    assert r["max_rel_err_expanded"] < GATE_TOL, (q, r)


@pytest.mark.parametrize("q", [0, 1, 2, 3])
def test_norm_matches_explicit_template_norm(setup, q):
    """``lensed_norm_sq`` must equal the norm of the explicit template for
    every phase -- the numerator and denominator cannot drift apart."""
    cfg, psd, h = setup
    conv = cv.DEFAULT.with_rel_morse(q)
    for t_d in (0.005, 0.05, 0.3):
        for a in (0.172, 0.5, 0.99):
            C = wf.autocorr_at_lags(h, psd, cfg, [conv.lag(t_d)])[0]
            n_alg = rc.lensed_norm_sq(C, a, conv=conv)
            n_exp = rc.explicit_lensed_norm(h, psd, cfg, t_d, a, conv=conv) ** 2
            assert n_alg == pytest.approx(n_exp, rel=1e-10), (q, t_d, a)


def test_even_phases_use_the_real_part_of_the_autocorrelation(setup):
    """Structural check on *why* the generalization is free: the cross term is
    ``2 a Re[u C]``, which for even ``q`` is ``+-2 a Re C`` and for odd ``q``
    is ``+-2 a Im C``.  Same precomputed ``C``, different projection."""
    cfg, psd, h = setup
    C = wf.autocorr_at_lags(h, psd, cfg, [cv.DEFAULT.lag(0.05)])[0]
    a = 0.6
    got = {
        q: rc.lensed_norm_sq(C, a, conv=cv.DEFAULT.with_rel_morse(q)) - (1 + a * a)
        for q in range(4)
    }
    assert got[0] == pytest.approx(+2 * a * C.real, rel=1e-12)
    assert got[2] == pytest.approx(-2 * a * C.real, rel=1e-12)
    assert got[1] == pytest.approx(+2 * a * C.imag, rel=1e-12)
    assert got[3] == pytest.approx(-2 * a * C.imag, rel=1e-12)


def test_same_type_pair_needs_the_q0_family(setup):
    """A pair of same-Morse-type images is recovered by ``q = 0`` and badly
    mismatched by the deployed ``q = 1`` family.

    This is the macro-saddle finding in miniature, with no lens model involved: build
    data whose two copies have a *real* relative coefficient and show which
    family matches it.
    """
    cfg, psd, h = setup
    freqs = cfg.frequencies()
    t0, t_d, a_true = 0.25 * cfg.seg_dur, 0.2, 0.75

    # two images of the same Morse type: relative coefficient is real
    data = wf.fd_delay(h, freqs, t0) + a_true * wf.fd_delay(h, freqs, t0 + t_d)
    sig = wf.sigma(data, psd, cfg.delta_f)

    best = {}
    for q in (0, 1):
        conv = cv.DEFAULT.with_rel_morse(q)
        peak = 0.0
        for a in np.linspace(0.02, 0.999, 60):
            z, _ = rc.explicit_lensed_snr(data, h, psd, cfg, t_d, a, conv=conv)
            peak = max(peak, float(np.abs(z).max() / sig))
        best[q] = peak

    z_ul, _ = wf.snr_series(data, h, psd, cfg)
    unlensed = float(np.abs(z_ul).max() / sig)

    assert best[0] > 0.999                      # q=0 is the right family
    assert best[1] < 0.9                        # q=1 is badly mismatched
    # and the mismatched family gains nothing over the unlensed template
    assert best[1] <= unlensed + 0.02
    # the analytic orthogonal-copies floor for the wrong phase
    assert best[1] == pytest.approx(1.0 / np.sqrt(1.0 + a_true**2), abs=0.05)
