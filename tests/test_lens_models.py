"""Lens models beyond the point mass, and the GLoW backend.

These are the gates for the lens-generality study. The load-bearing ones are:

* ``test_glow_point_lens_matches_frozen_amplification`` -- fixes the convention
  mapping between GLoW and this project. If it fails, every map produced from
  GLoW has a sign or phase error.
* ``test_images_match_glow`` -- the geometry layer against GLoW's own finder.
* ``test_wave_optics_approaches_geometric_optics`` -- the physical limit.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lensing import amplification as amp
from lensing import glow_backend as gb
from lensing import lens_models as lm

glow = pytest.importorskip("glow", reason="GLoW is required for the lens-model study")


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def test_sis_closed_forms():
    """SIS: a = sqrt((1-y)/(1+y)) and Dtau = 2y, against the lens equation."""
    sis = lm.SIS()
    for y in (0.05, 0.1, 0.3, 0.5, 0.7, 0.9):
        assert sis.a_of_y(y) == pytest.approx(float(sis.a_analytic(y)), abs=1e-12)
        assert sis.delay_dimensionless(y) == pytest.approx(2.0 * y, rel=1e-12)
        ims = sis.images(y)
        assert [im.name for im in ims] == ["minimum", "saddle"]


def test_sis_single_image_beyond_einstein_radius():
    """For y > 1 the SIS has only one image, so there is nothing to recombine."""
    sis = lm.SIS()
    assert sis.n_images(1.2) == 1
    assert sis.a_of_y(1.2) == 0.0
    assert sis.delay_dimensionless(1.2) == 0.0


def test_pointmass_reproduces_frozen_amplification():
    """The generic machinery must reproduce the frozen point-mass module.

    This is what makes the new geometry layer trustworthy: it is checked
    against code that all of results/frozen/ already depends on.
    """
    pm = lm.PointMass()
    for y in (0.05, 0.1, 0.3, 0.5, 1.0, 2.0):
        assert pm.n_images(y) == 2
        assert pm.a_of_y(y) == pytest.approx(float(amp.a_of_y(y)), abs=1e-8)
        assert pm.delay_dimensionless(y) == pytest.approx(
            float(amp.delay_dimensionless(y)), rel=1e-10
        )


def test_cored_has_three_images_with_a_maximum():
    """The cored lens's third image is a MAXIMUM with a real negative
    coefficient -- which the family's ``c = i a`` cannot represent."""
    c = lm.CoredIsothermal(rc=0.15)
    ims = c.images(0.1)
    assert len(ims) == 3
    assert [im.name for im in ims] == ["minimum", "saddle", "maximum"]
    # coefficients: +1, -i, -1 times sqrt|mu|
    assert ims[0].coeff.imag == pytest.approx(0.0, abs=1e-15)
    assert ims[0].coeff.real > 0
    assert ims[1].coeff.real == pytest.approx(0.0, abs=1e-15)
    assert ims[1].coeff.imag < 0
    assert ims[2].coeff.imag == pytest.approx(0.0, abs=1e-15)
    assert ims[2].coeff.real < 0          # the unrepresentable one


def test_images_sorted_by_arrival_time():
    """Ordering is by arrival time, not brightness -- the GO phase reference
    and the family's leading image both depend on it."""
    for mdl in (lm.SIS(), lm.CoredIsothermal(rc=0.30), lm.PointMass()):
        for y in (0.1, 0.3):
            ims = mdl.images(y)
            taus = [im.tau for im in ims]
            assert taus == sorted(taus)
            assert all(d >= 0 for d in mdl.delays_dimensionless(y))


def test_cored_brightest_image_can_arrive_second():
    """At small y a cored lens's brightest image is the saddle, so the
    amplitude ratio relative to the FIRST image exceeds 1 -- outside the
    family's a in (0,1). Pin this, since it is a stated limitation."""
    c = lm.CoredIsothermal(rc=0.30)
    ims = c.images(0.1)
    assert len(ims) == 3
    assert abs(ims[1].mu) > abs(ims[0].mu)
    assert c.a_of_y(0.1) > 1.0


# --------------------------------------------------------------------------
# GLoW backend
# --------------------------------------------------------------------------


def test_glow_point_lens_matches_frozen_amplification():
    """THE convention gate.

    GLoW returns F in the literature convention, the same one as
    ``amp.F_ML_from_w``. GLoW's analytic point lens is documented as accurate to
    roughly single precision, so agreement at the 1e-5 level is the expected
    outcome; the conjugated comparison must be wrong by O(1), which is what
    makes the identification unambiguous.
    """
    from glow import freq_domain_c

    ws = np.array([0.5, 2.0, 10.0, 50.0, 200.0])
    worst_same, worst_conj = 0.0, 0.0
    for y in (0.1, 0.3, 0.5, 1.0, 2.0):
        g = freq_domain_c.Fw_AnalyticPointLens_C(y=y)(ws)
        mine = amp.F_ML_from_w(ws, y, backend="mpmath")
        worst_same = max(worst_same, np.max(np.abs(g - mine) / np.abs(mine)))
        worst_conj = max(
            worst_conj, np.max(np.abs(g - np.conj(mine)) / np.abs(mine))
        )
    assert worst_same < 1e-4, worst_same
    assert worst_conj > 0.1, worst_conj


def test_images_match_glow():
    """The geometry layer against GLoW's own critical-point finder."""
    for mdl in (lm.SIS(), lm.CoredIsothermal(rc=0.05), lm.CoredIsothermal(rc=0.15)):
        for y in (0.1, 0.3, 0.8):
            mine = mdl.images(y)
            theirs = gb.images_from_glow(mdl, y)
            assert len(mine) == len(theirs), (mdl.name, y)
            tau_ref = min(im.tau for im in mine)
            mt = sorted((im.tau - tau_ref, abs(im.mu), im.morse) for im in mine)
            gt = sorted((d["tau"], d["mag_abs"], d["morse"]) for d in theirs)
            for a, b in zip(mt, gt):
                assert a[0] == pytest.approx(b[0], abs=1e-9)
                assert a[1] == pytest.approx(b[1], rel=1e-6)
                assert a[2] == b[2]          # same Morse classification


@pytest.mark.parametrize(
    "model,y,tol",
    [
        (lm.SIS(), 0.3, 3e-3),
        (lm.SIS(), 0.5, 3e-3),
        (lm.CoredIsothermal(rc=0.05), 0.3, 5e-3),
        (lm.PointMass(), 0.3, 3e-3),
    ],
)
def test_wave_optics_approaches_geometric_optics(model, y, tol):
    """F_wave -> F_GO at large w, both in the literature convention.

    Tolerances are per-case because convergence near a caustic is genuinely
    slower; the cored lens at rc=0.15, y=0.3 (a bright third image close to the
    fold) only reaches ~4e-2 by w=3000 and is deliberately not asserted here.
    """
    ws = np.array([1000.0, 3000.0])
    Fw = gb.F_of_w(model, y, ws)
    ims = model.images(y)
    tau_ref = min(im.tau for im in ims)
    Fgo = sum(im.coeff * np.exp(1j * ws * (im.tau - tau_ref)) for im in ims)
    assert np.max(np.abs(Fw - Fgo) / np.abs(Fgo)) < tol


def test_wave_optics_tends_to_unity_at_low_frequency():
    """F -> 1 as w -> 0. For the SIS the approach is ~sqrt(w) (GLoW's
    asymp_index = 0.5), so we check the trend rather than a tight bound."""
    sis = lm.SIS()
    ws = np.array([1e-4, 1e-3, 1e-2])
    dev = np.abs(np.abs(gb.F_of_w(sis, 0.3, ws)) - 1.0)
    assert dev[0] < dev[1] < dev[2]
    assert dev[0] < 0.02
    # ratio consistent with a sqrt(w) approach within a factor of two
    assert 0.15 < dev[0] / dev[1] < 0.65


def test_project_convention_round_trip():
    """F_wave is returned in the project convention: the trailing image is
    delayed, so it must be the conjugate of GLoW's literature-convention F."""
    sis = lm.SIS()
    M, y = 1e3, 0.3
    f = np.array([50.0, 150.0])
    w = amp.w_of_f(f, M)
    lit = gb.F_of_w(sis, y, w)
    proj = sis.F_wave(f, M, y)
    assert np.allclose(proj, np.conj(lit), rtol=1e-12)


def test_model_registry():
    assert set(lm.MODELS) == {"sis", "cored", "pointmass", "changrefsdal"}
    assert isinstance(lm.build("cored", rc=0.2), lm.CoredIsothermal)
    assert lm.build("cored", rc=0.2).rc == 0.2
    with pytest.raises(ValueError, match="unknown lens model"):
        lm.build("nope")


# --------------------------------------------------------------------------
# Chang-Refsdal: point mass in an external field
# --------------------------------------------------------------------------


def test_chang_refsdal_reduces_to_frozen_point_mass():
    """At kappa = gamma = 0 the quartic must reproduce the frozen point-mass
    module exactly. This is the anchor for the whole 2-D image finder."""
    cr = lm.ChangRefsdal(kappa=0.0, gamma=0.0)
    for y in (0.1, 0.3, 0.5, 1.0, 2.0):
        ims = cr.images(y)
        assert len(ims) == 2
        assert [im.morse for im in ims] == [0, 1]        # minimum then saddle
        assert cr.a_of_y(y) == pytest.approx(float(amp.a_of_y(y)), abs=1e-12)
        assert cr.delay_dimensionless(y) == pytest.approx(
            float(amp.delay_dimensionless(y)), rel=1e-12
        )
        assert cr.required_rel_morse(y) == 1                # the deployed family


def test_chang_refsdal_pure_convergence_is_a_rescaled_point_lens():
    """With gamma = 0 the lens is a point mass in a convergence sheet, i.e. a
    point lens with psi0 -> psi0/(1-kappa) and y -> y/(1-kappa).

    A degenerate case of the quartic (its leading coefficients vanish), so it
    is worth pinning that the root finder still returns the right two images.
    """
    for kappa in (0.2, 0.5, 0.8):
        for y in (0.15, 0.4):
            cr = lm.ChangRefsdal(kappa=kappa, gamma=0.0)
            ims = cr.images(y)
            assert len(ims) == 2
            assert [im.morse for im in ims] == [0, 1]
            # equivalent point lens, in its own (rescaled) length unit
            a_eff = 1.0 - kappa
            y_eq = y / (a_eff * np.sqrt(1.0 / a_eff))
            pm = lm.PointMass()
            assert cr.a_of_y(y) == pytest.approx(pm.a_of_y(y_eq), rel=1e-9)


@pytest.mark.parametrize(
    "kappa,gamma,macro",
    [(0.0, 0.0, "minimum"), (0.2, 0.2, "minimum"), (0.55, 0.4, "minimum"),
     (0.2, 0.9, "saddle"), (0.4, 0.8, "saddle"), (0.6, 0.6, "saddle")],
)
def test_chang_refsdal_macro_classification(kappa, gamma, macro):
    """The critical line is gamma = 1 - kappa, independent of orientation."""
    for phi in (0.0, np.pi / 6, np.pi / 4):
        cr = lm.ChangRefsdal(kappa, gamma, phi_gamma=phi)
        assert cr.macro_type == macro
        l1, l2 = cr.macro_eigenvalues
        assert l1 == pytest.approx(1 - kappa - gamma)
        assert l2 == pytest.approx(1 - kappa + gamma)
        assert np.sign(cr.distance_to_critical) == (1 if macro == "saddle" else -1)


def test_chang_refsdal_aligned_shear_is_degenerate():
    """Why phi_gamma defaults to 45 deg, not 0.

    With the shear aligned to the source the configuration is symmetric in
    x2 -> -x2, so images come in mirror pairs with equal magnification and
    equal arrival time -- one image with a rescaled amplitude, not two. If this
    ever stops holding, the reason for the 45 deg choice has changed.
    """
    for kappa, gamma in [(0.2, 0.9), (0.4, 0.8), (0.6, 0.6)]:
        cr = lm.ChangRefsdal(kappa, gamma, phi_gamma=0.0)
        ims = cr.images(0.3)
        assert len(ims) == 2
        assert abs(ims[1].tau - ims[0].tau) < 1e-12       # coincident
        assert cr.a_of_y(0.3) == pytest.approx(1.0, abs=1e-12)
        assert ims[0].x2 == pytest.approx(-ims[1].x2, abs=1e-12)

    # and 45 deg breaks it, giving two genuinely separated images
    for kappa, gamma in [(0.2, 0.9), (0.4, 0.8), (0.6, 0.6)]:
        cr = lm.ChangRefsdal(kappa, gamma, phi_gamma=np.pi / 4)
        assert cr.delay_dimensionless(0.3) > 0.1
        assert 0.7 < cr.a_of_y(0.3) < 0.8


def test_chang_refsdal_macro_saddle_pairs_are_the_same_morse_type():
    """The key finding: near a macro-saddle both images are saddles, so the
    required relative Morse phase is 0 -- not the deployed 90 deg."""
    for kappa, gamma in [(0.2, 0.9), (0.4, 0.8), (0.6, 0.6), (0.7, 0.5)]:
        cr = lm.ChangRefsdal(kappa, gamma, phi_gamma=np.pi / 4)
        ims = cr.images(0.3)
        assert len(ims) == 2
        assert [im.morse for im in ims] == [1, 1]         # saddle + saddle
        assert cr.required_rel_morse(0.3) == 0
        # the relative geometric-optics coefficient is real, not imaginary
        ratio = ims[1].coeff / ims[0].coeff
        assert abs(ratio.imag) < 1e-12
        assert ratio.real > 0


def test_chang_refsdal_four_images_in_the_macro_minimum_region():
    """Stronger shear inside the macro-minimum region adds a second pair."""
    cr = lm.ChangRefsdal(0.55, 0.4, phi_gamma=np.pi / 4)
    ims = cr.images(0.3)
    assert len(ims) == 4
    assert sorted(im.morse for im in ims) == [0, 0, 1, 1]
    assert np.all(np.diff([im.tau for im in ims]) >= 0)


def test_chang_refsdal_images_match_glow():
    """Against GLoW's own 2-D critical-point finder.

    GLoW's finder is *incomplete* in the macro-saddle region (it drops one of
    the two saddles), so the comparison is made only where its census matches;
    this incompleteness is a known limitation of the finder.
    """
    n_compared = 0
    for kappa, gamma in [(0.2, 0.2), (0.4, 0.3), (0.55, 0.4), (0.2, 0.9)]:
        cr = lm.ChangRefsdal(kappa, gamma, phi_gamma=np.pi / 4)
        mine = cr.images(0.3)
        theirs = gb.images_from_glow(cr, 0.3)
        if len(theirs) != len(mine):
            continue                                  # GLoW dropped an image
        n_compared += 1
        tau_ref = min(im.tau for im in mine)
        mt = sorted((im.tau - tau_ref, abs(im.mu), im.morse) for im in mine)
        gt = sorted((d["tau"], d["mag_abs"], d["morse"]) for d in theirs)
        for a, b in zip(mt, gt):
            assert a[0] == pytest.approx(b[0], abs=1e-6)
            assert a[1] == pytest.approx(b[1], rel=1e-5)
            assert a[2] == b[2]
    assert n_compared >= 3, "GLoW cross-check never ran"


@pytest.mark.parametrize("kappa,gamma,closes", [
    (0.0, 0.0, True), (0.2, 0.3, True), (0.45, 0.5, True),
    (0.6, 0.6, False), (0.4, 0.8, False), (0.2, 0.9, False),
])
def test_chang_refsdal_contour_closure_matches_the_critical_line(kappa, gamma, closes):
    """Closed contours iff ``kappa + gamma < 1``, i.e. iff the macro-image is a
    minimum. This is the boundary of where wave optics is available at all."""
    cr = lm.ChangRefsdal(kappa, gamma)
    assert cr.contours_close is closes
    assert closes == (cr.macro_type == "minimum")


def test_chang_refsdal_refuses_wave_optics_on_open_contours():
    """Where the Fermat potential is unbounded below along one axis the contour
    method has nothing to integrate; F_wave must refuse rather than return a
    plausible wrong number."""
    cr = lm.ChangRefsdal(0.6, 0.6)
    assert not cr.contours_close
    with pytest.raises(ValueError, match="do not close"):
        cr.F_wave(np.array([50.0]), 1e3, 0.3)
    F, mode = cr.F_hybrid(np.array([50.0]), 1e3, 0.3)
    assert mode == "go"
    assert np.allclose(F, cr.F_go(np.array([50.0]), 1e3, 0.3))


def test_chang_refsdal_wave_optics_reproduces_the_frozen_point_lens():
    """A check that an earlier implementation failed at 43%.

    At ``kappa = gamma = 0`` the whole closed-contour path -- CombinedLens,
    GLoW's multicontour I(tau), and the transform in ``wave_contour`` -- must
    reproduce the project's frozen mpmath point-lens amplification.
    """
    from lensing import wave_contour as wc

    y = 0.3
    w = np.array([0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    cr = lm.ChangRefsdal(0.0, 0.0)
    F = wc.F_of_w_contour(wc.build_It(cr, y), w)
    ref = amp.F_ML_from_w(w, y)
    assert np.max(np.abs(F - ref) / np.abs(ref)) < 2e-3


def test_chang_refsdal_wave_optics_tends_to_geometric_optics():
    """The physical limit, on a genuinely four-image macro-minimum cell."""
    from lensing import wave_contour as wc

    y, M = 0.3, 1e3
    cr = lm.ChangRefsdal(kappa=0.2, gamma=0.3, phi_gamma=np.pi / 4)
    assert len(cr.images(y)) == 4
    f_hi = amp.f_of_w(np.array([1000.0, 2000.0, 3000.0]), M)
    F_w, mode = cr.F_hybrid(f_hi, M, y)
    assert mode == "wave"
    F_g = cr.F_go(f_hi, M, y)
    assert np.max(np.abs(F_w - F_g) / np.abs(F_g)) < 2e-2


def test_wave_contour_asymptotic_bookkeeping():
    """``c_inf / 2 pi + sum_minima sqrt(mu) == sqrt(mu_macro)``.

    An analytic identity the numerics never sees: it ties the constant left in
    ``I(tau -> inf)`` by the external field to the macro-magnification.
    """
    from lensing import wave_contour as wc

    for kappa, gamma in ((0.2, 0.3), (0.45, 0.5), (0.5, 0.4)):
        cr = lm.ChangRefsdal(kappa=kappa, gamma=gamma, phi_gamma=np.pi / 4)
        _, d = wc.F_of_w_contour(wc.build_It(cr, 0.3), np.array([100.0]),
                                 return_diagnostics=True)
        expected = 1.0 / np.sqrt(abs((1 - kappa) ** 2 - gamma ** 2))
        assert d["sqrt_mu_asym"] == pytest.approx(expected, abs=1e-3)


def test_wave_contour_refuses_a_true_maximum():
    """A maximum's singular step reaches tau = -inf; the truncated grid cannot
    represent it, so the transform must refuse rather than guess."""
    from lensing import wave_contour as wc

    class _FakeIt:
        tmin = 0.0
        p_crits = [
            {"type": "min", "t": 0.0, "mag": 2.0},
            {"type": "saddle", "t": 0.5, "mag": 1.0},
            {"type": "max", "t": 1.0, "mag": 0.4},
        ]

    with pytest.raises(NotImplementedError, match="true maximum"):
        wc.F_of_w_contour(_FakeIt(), np.array([10.0]))


def test_filon_transform_is_exact_in_the_oscillatory_factor():
    """``int_0^inf e^{-t} e^{i w t} dt = 1 / (1 - i w)`` at large ``w``.

    Guards the one piece of quadrature written in this project rather than
    taken from GLoW.
    """
    from lensing import wave_contour as wc

    tau = wc._graded_left(0.0, 60.0, 8000)
    for w in (10.0, 100.0, 1000.0, 3000.0):
        got = wc.filon_ft(tau, np.exp(-tau), np.array([w]))[0]
        assert got == pytest.approx(1.0 / (1.0 - 1j * w), rel=1e-4)


def test_chang_refsdal_go_equals_the_explicit_two_image_template():
    """The convention chain, end to end.

    A hand-built geometric-optics injection is exactly where the conjugation
    can silently re-enter with the wrong sign. So: for a two-image cell,
    ``F_go`` divided by the leading image's amplitude must reproduce
    ``recombine.explicit_lensed_template`` at the corresponding ``(t_d, a)``
    and the *required* relative Morse phase -- ``q = 1`` for the
    minimum+saddle pair in the macro-minimum region, ``q = 0`` for the
    saddle+saddle pair in the macro-saddle region.
    """
    from lensing import conventions as cv
    from lensing import recombine as rcb
    from lensing import waveforms as wf

    cfg = wf.AnalysisConfig(seg_dur=8.0, sample_rate=4096.0,
                            mass1=25.0, mass2=25.0)
    freqs = cfg.frequencies()
    h = np.ones(freqs.size, dtype=complex)
    M_Lz, y = 1e3, 0.3

    for kappa, gamma, expect_q in [(0.2, 0.2, 1), (0.6, 0.6, 0), (0.4, 0.8, 0)]:
        cr = lm.ChangRefsdal(kappa, gamma, phi_gamma=np.pi / 4)
        ims = cr.images(y)
        assert len(ims) == 2
        q = cr.required_rel_morse(y)
        assert q == expect_q

        # F is in the PROJECT convention, i.e. the conjugate of the literature
        # sum, so the leading image's factor there is conj(coeff) -- dividing by
        # coeff itself would flip the sign whenever the leading image is a
        # saddle, which is exactly the macro-saddle case.
        F = cr.F_go(freqs, M_Lz, y)
        got = F / np.conj(ims[0].coeff)

        conv = cv.DEFAULT.with_rel_morse(q)
        want = rcb.explicit_lensed_template(
            h, freqs, cr.time_delay(M_Lz, y), cr.a_of_y(y), conv=conv
        )
        assert np.allclose(got, want, rtol=1e-9, atol=1e-9), (kappa, gamma, q)
