"""Lens models beyond the isolated point mass, for the lens-generality study.

Why this module exists
----------------------

The paper's recombination identity is written entirely in terms of the
*observable* pair ``(t_d, a)`` -- see :mod:`lensing.recombine`, where no
``M_Lz`` or ``y`` appears anywhere. What is specific to the point mass is only

1. the **map** from lens parameters to ``(t_d, a)``, and
2. the assumption that the second image is a **saddle**, so that the relative
   coefficient is ``c = i a`` (:meth:`lensing.conventions.Convention.image_coeff`).

This module supplies (1) for other lens models and exposes (2) explicitly, so
that the question "how far does the two-image statistic go for other strongly
lensed systems?" can be answered by injection rather than by argument.

:mod:`lensing.amplification` is left untouched: it is the frozen, validated
point-mass implementation and every result in `results/frozen/` depends on it.

Models
------

``SIS``
    Singular isothermal sphere, ``psi(x) = x``. Two images for ``y < 1``
    (minimum + saddle), one for ``y > 1``. Analytically,

        a(y) = sqrt((1 - y)/(1 + y)),      Dtau(y) = 2 y,

    both verified against the lens equation in :mod:`tests.test_lens_models`.
    The Morse structure is *identical* to the point mass, so the search family
    applies verbatim -- this model tests the mapping, not the family.

``CoredIsothermal``
    GLoW's ``Psi_CIS``. **Three** images over part of the plane: minimum,
    saddle, and a central **maximum** whose geometric-optics coefficient is real
    and negative (``-1``), not ``+-i``. The two-image family therefore cannot
    represent the third image at all, whatever its amplitude. This model tests
    the family's limit.

``PointMass``
    Present only so the generic machinery here can be validated against the
    frozen :mod:`lensing.amplification`.

Conventions
-----------

Time is measured in units of ``T = 4 G M / c^3`` with ``M`` the mass scale of
the model (for the point mass, the redshifted lens mass; for the isothermal
models, the mass inside the Einstein radius). Then

    t_d = 4 * T_SUN * M * Dtau,      w = 8 pi T_SUN M f,

which is exactly :func:`lensing.amplification.w_of_f`, so ``f_ML = 1/t_d`` and
the whole ``t_d``-linearity in the mass scale carry over unchanged.

Amplification factors are returned in the **project** convention (the trailing
image delayed, i.e. ``exp(-2 pi i f t_d)``), via
:func:`lensing.amplification.to_project_convention`. GLoW returns ``F`` in the
*literature* convention; this was confirmed rather than assumed by checking
GLoW's analytic point lens against the frozen mpmath implementation to
1.5e-6 relative (``tests/test_lens_models.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from . import amplification as amp

#: Morse index -> geometric-optics coefficient, literature convention
#: ``exp(-i n pi / 2)``: minimum 0 -> +1, saddle 1 -> -i, maximum 2 -> -1.
MORSE_COEFF = {0: 1.0 + 0.0j, 1: -1.0j, 2: -1.0 + 0.0j}
MORSE_NAME = {0: "minimum", 1: "saddle", 2: "maximum"}


class _MorseMixin:
    """The geometric-optics coefficient of one image, shared by 1-D and 2-D."""

    @property
    def coeff(self) -> complex:
        """Geometric-optics coefficient ``sqrt(|mu|) * exp(-i n pi/2)``.

        ``MORSE_COEFF[n] == (-i)**n``, so the *ratio* of a trailing to a
        leading image is ``(-i)**dn`` in the literature convention, hence
        ``i**dn`` in the project convention -- which is exactly
        ``conventions.Convention.image_coeff`` at
        ``rel_morse_quarters = dn % 4``. See :func:`required_rel_morse`.
        """
        return np.sqrt(abs(self.mu)) * MORSE_COEFF[self.morse]

    @property
    def name(self) -> str:
        return MORSE_NAME[self.morse]


@dataclass(frozen=True)
class Image(_MorseMixin):
    """One geometric-optics image of an axisymmetric lens."""

    x: float          #: radial position (signed; sign selects the side)
    mu: float         #: signed magnification
    tau: float        #: Fermat potential at the image (unshifted)
    morse: int        #: Morse index: 0 minimum, 1 saddle, 2 maximum


@dataclass(frozen=True)
class Image2D(_MorseMixin):
    """One geometric-optics image of a lens with no axial symmetry."""

    x1: float         #: position along the shear's first axis
    x2: float         #: position along the second axis
    mu: float         #: signed magnification, ``1/det A``
    tau: float        #: Fermat potential at the image (unshifted)
    morse: int        #: number of negative eigenvalues of ``A``


def required_rel_morse(morse_leading: int, morse_trailing: int) -> int:
    """``rel_morse_quarters`` that represents this ordered pair of images.

    The literature Morse factor is ``(-i)**n``, so a trailing image relative to
    a leading one carries ``(-i)**dn``; conjugating into the project convention
 gives ``i**dn``, i.e. ``q = dn mod 4``. The deployed
    family is ``q = 1``, the minimum-then-saddle pair, which is the
    empirically fixed Phase 0 configuration -- so this mapping is anchored on
    that case rather than re-derived from scratch.
    """
    return (int(morse_trailing) - int(morse_leading)) % 4


# --------------------------------------------------------------------------
# Base
# --------------------------------------------------------------------------


class LensCommon:
    """Everything that depends only on :meth:`images`, and so is shared by the
    axisymmetric models and the (2-D) Chang-Refsdal lens.

    Subclasses must provide ``images(y) -> list`` sorted by arrival time, and a
    ``name`` attribute.
    """

    def images(self, y):                                  # pragma: no cover
        raise NotImplementedError

    def n_images(self, y) -> int:
        return len(self.images(y))

    def amplitude_ratios(self, y) -> np.ndarray:
        """``sqrt(|mu_k| / |mu_1|)`` relative to the **first-arriving** image.

        May exceed 1 if a later image is brighter than the first -- which the
        two-image family, whose ``a`` is bounded in ``(0, 1)``, cannot
        represent. That is a genuine limit and is reported, not clipped.
        """
        ims = self.images(y)
        if len(ims) < 2:
            return np.array([])
        m1 = abs(ims[0].mu)
        return np.array([np.sqrt(abs(im.mu) / m1) for im in ims[1:]])

    def delays_dimensionless(self, y) -> np.ndarray:
        """``tau_k - tau_1`` relative to the first-arriving image (all >= 0)."""
        ims = self.images(y)
        if len(ims) < 2:
            return np.array([])
        return np.array([im.tau - ims[0].tau for im in ims[1:]])

    def a_of_y(self, y) -> float:
        """Amplitude ratio of the second-arriving image: the family's ``a``."""
        r = self.amplitude_ratios(y)
        return float(r[0]) if r.size else 0.0

    def delay_dimensionless(self, y) -> float:
        """Delay of the second image, in units of ``4 G M / c^3``."""
        d = self.delays_dimensionless(y)
        return float(d[0]) if d.size else 0.0

    def time_delay(self, M, y) -> float:
        """Physical delay of the second image, seconds."""
        return 4.0 * amp.T_SUN * float(M) * self.delay_dimensionless(y)

    def max_time_delay(self, M, y) -> float:
        """Physical delay of the *latest* image -- what sizes the search window."""
        d = self.delays_dimensionless(y)
        return 4.0 * amp.T_SUN * float(M) * (float(np.max(d)) if d.size else 0.0)

    def required_rel_morse(self, y) -> int:
        """``rel_morse_quarters`` for the first two arriving images.

        ``0`` when they are the same Morse type -- which is what the deployed
        ``q = 1`` family cannot represent.
        """
        ims = self.images(y)
        if len(ims) < 2:
            return 1
        return required_rel_morse(ims[0].morse, ims[1].morse)

    def F_go(self, f, M, y, conv=None):
        """Geometric-optics ``F``, all images, project convention.

        ``F_lit = sum_k sqrt(|mu_k|) exp(-i n_k pi/2) exp(+i w Dtau_k)``,
        with the phase referenced to the **first-arriving** image, which is
        the global minimum of the Fermat potential and the reference GLoW
        uses. Built from :meth:`images`, so it carries the correct Morse
        coefficient for a maximum (``-1``), which no two-image family can
        reproduce.
        """
        ims = self.images(y)
        if not ims:
            return np.zeros_like(np.asarray(f, dtype=float), dtype=complex)
        w = amp.w_of_f(np.asarray(f, dtype=float), M)
        tau_ref = min(im.tau for im in ims)   # the global minimum, as GLoW uses
        F = np.zeros(np.shape(w), dtype=complex)
        for im in ims:
            F = F + im.coeff * np.exp(1j * w * (im.tau - tau_ref))
        return amp.to_project_convention(F, conv=conv)


class AxisymLens(LensCommon):
    """An axisymmetric lens specified by its radial potential ``psi(x)``.

    Subclasses provide ``psi``, ``dpsi``, ``ddpsi`` and a GLoW lens object.
    Everything else -- image finding, Morse classification, the amplitude
    ratios, the delays and the geometric-optics factor -- is generic.
    """

    name = "base"

    # -- to be provided by subclasses ------------------------------------
    def psi(self, x):
        raise NotImplementedError

    def dpsi(self, x):
        raise NotImplementedError

    def ddpsi(self, x):
        raise NotImplementedError

    def glow_lens(self):
        """The corresponding ``glow.lenses`` object."""
        raise NotImplementedError

    # -- generic geometry -------------------------------------------------

    def fermat(self, x, y):
        """Fermat potential ``|x - y|^2 / 2 - psi(|x|)``, unshifted."""
        x = np.asarray(x, dtype=float)
        return 0.5 * (x - y) ** 2 - self.psi(np.abs(x))

    def _dfermat(self, x, y):
        """``d tau / dx``; its roots are the images."""
        return x - y - np.sign(x) * self.dpsi(np.abs(x))

    def images(self, y, span=None, n_scan=200001) -> list[Image]:
        """All geometric-optics images, **earliest-arriving first**.

        Ordering by arrival time rather than by brightness matters. The search
        family is ``h_t + c h_{t+t_d}`` with ``t_d > 0``, so its reference image
        is the one that arrives *first*; and GLoW measures ``tau`` from the
        global minimum of the Fermat potential, so the geometric-optics phase
        reference must be that same image. For the point mass and the SIS the
        earliest image is also the brightest and the distinction is invisible,
        but for a cored lens at small ``y`` the brightest image can be the
        saddle -- and referencing the phase to it puts ``F_GO`` an O(1) phase
        away from GLoW's ``F``.

        Roots of ``x - y - sign(x) psi'(|x|) = 0`` are bracketed on a dense
        scan and polished with Brent's method. The magnification of an
        axisymmetric lens is

            mu = 1 / [ (1 - psi'/x) (1 - psi'') ],

        and the Morse index is the number of negative Jacobian eigenvalues,
        i.e. of the two factors above. This is the classification that decides
        whether the search family can represent the image at all.
        """
        y = float(y)
        span = span if span is not None else max(6.0, 4.0 * (1.0 + y))
        xs = np.linspace(-span, span, n_scan)
        xs = xs[np.abs(xs) > 1e-12]
        f = self._dfermat(xs, y)
        sign_change = np.nonzero(np.diff(np.sign(f)) != 0)[0]

        roots = []
        for i in sign_change:
            a, b = xs[i], xs[i + 1]
            if a * b < 0:      # never bracket across the singular origin
                continue
            try:
                roots.append(brentq(self._dfermat, a, b, args=(y,), xtol=1e-14))
            except (ValueError, RuntimeError):
                continue

        out = []
        for x in sorted({round(r, 11) for r in roots}):
            ax = abs(x)
            tang = 1.0 - self.dpsi(ax) / ax
            radial = 1.0 - self.ddpsi(ax)
            det = tang * radial
            if not np.isfinite(det) or det == 0.0:
                continue
            morse = int(tang < 0) + int(radial < 0)
            out.append(
                Image(x=float(x), mu=float(1.0 / det),
                      tau=float(self.fermat(x, y)), morse=morse)
            )
        out.sort(key=lambda im: im.tau)
        return out

    # -- amplification factors -------------------------------------------

    def F_wave(self, f, M, y, conv=None, **kw):
        """Exact wave-optics ``F`` from GLoW, project convention."""
        from . import glow_backend

        f_arr = np.asarray(f, dtype=float)
        out = np.ones(f_arr.shape, dtype=complex)
        good = f_arr > 0
        if np.any(good):
            w = amp.w_of_f(f_arr[good], M)
            F_lit = glow_backend.F_of_w(self, float(y), w, **kw)
            out[good] = amp.to_project_convention(F_lit, conv=conv)
        return out


# --------------------------------------------------------------------------
# Concrete models
# --------------------------------------------------------------------------


class SIS(AxisymLens):
    """Singular isothermal sphere, ``psi(x) = psi0 * x`` with ``psi0 = 1``.

    Two images (minimum + saddle) for ``y < 1``; a single image for ``y > 1``.
    """

    name = "sis"

    def __init__(self, psi0: float = 1.0):
        self.psi0 = float(psi0)

    def psi(self, x):
        return self.psi0 * np.asarray(x, dtype=float)

    def dpsi(self, x):
        return np.full_like(np.asarray(x, dtype=float), self.psi0)

    def ddpsi(self, x):
        return np.zeros_like(np.asarray(x, dtype=float))

    def glow_lens(self):
        from glow import lenses

        return lenses.Psi_SIS({"psi0": self.psi0})

    # closed forms, for tests and for speed
    def a_analytic(self, y):
        y = np.asarray(y, dtype=float)
        return np.where(y < 1.0, np.sqrt(np.clip((1 - y) / (1 + y), 0, None)), 0.0)

    def delay_analytic(self, y):
        y = np.asarray(y, dtype=float)
        return np.where(y < 1.0, 2.0 * y, 0.0)


class CoredIsothermal(AxisymLens):
    """Cored isothermal sphere -- GLoW's ``Psi_CIS``,

        r = sqrt(x^2 + rc^2),   psi(x) = psi0 [ r + rc log(2 rc / (r + rc)) ].

    Three images over much of the plane: minimum, saddle and a central
    **maximum**, whose geometric-optics coefficient is real and negative.

    The potential and its two derivatives are **delegated to the GLoW lens
    object** rather than transcribed. That is deliberate: the geometric-optics
    images computed here and the wave-optics ``F`` computed by GLoW must refer
    to the same lens, and an independent transcription is exactly where they
    would silently diverge. (An earlier version of this class used the ad-hoc
    softened SIS ``psi = psi0 (r - rc)``, which is *not* GLoW's CIS; the
    geometric-optics limit then disagreed with GLoW's ``F`` by O(1) at every
    frequency. Delegation makes that failure mode impossible. GLoW's own
    ``ddpsi_ddx`` docstring also disagrees with its code -- the code is
    authoritative -- which is a second reason not to transcribe.)
    """

    name = "cored"

    def __init__(self, rc: float = 0.15, psi0: float = 1.0):
        self.rc = float(rc)
        self.psi0 = float(psi0)
        self._lens = None

    def _g(self):
        if self._lens is None:
            from glow import lenses

            self._lens = lenses.Psi_CIS({"psi0": self.psi0, "rc": self.rc})
        return self._lens

    def psi(self, x):
        return np.asarray(self._g().psi_x(np.asarray(x, dtype=float)), dtype=float)

    def dpsi(self, x):
        return np.asarray(self._g().dpsi_dx(np.asarray(x, dtype=float)), dtype=float)

    def ddpsi(self, x):
        return np.asarray(self._g().ddpsi_ddx(np.asarray(x, dtype=float)), dtype=float)

    def glow_lens(self):
        return self._g()


class PointMass(AxisymLens):
    """Isolated point mass, ``psi(x) = log|x|`` -- for validation only.

    Exists so the generic machinery in :class:`AxisymLens` can be checked
    against the frozen :mod:`lensing.amplification`. Production point-mass work
    must keep using that module.
    """

    name = "pointmass"

    def __init__(self, psi0: float = 1.0):
        self.psi0 = float(psi0)

    def psi(self, x):
        return self.psi0 * np.log(np.asarray(x, dtype=float))

    def dpsi(self, x):
        return self.psi0 / np.asarray(x, dtype=float)

    def ddpsi(self, x):
        x = np.asarray(x, dtype=float)
        return -self.psi0 / (x * x)

    def glow_lens(self):
        from glow import lenses

        return lenses.Psi_PointLens({"psi0": self.psi0})


# --------------------------------------------------------------------------
# Chang-Refsdal: a point mass in an external field (no axial symmetry)
# --------------------------------------------------------------------------


class ChangRefsdal(LensCommon):
    r"""Point mass plus external convergence and shear.

    .. math::
        \psi(x) = \psi_0 \ln|x| + \frac{\kappa}{2}|x|^2
                 + \frac{\gamma_1}{2}(x_1^2 - x_2^2) + \gamma_2 x_1 x_2

    with :math:`\gamma_1 + i\gamma_2 = \gamma e^{2i\phi_\gamma}`. This is the
    local model of a **microlens sitting on a macro-image**: the eigenvalues of
    the smooth deflection are ``1 - kappa -+ gamma``, so the macro-image is a
    minimum for ``gamma < 1 - kappa`` and a **saddle** for ``gamma > 1 - kappa``,
    the two regions separated by the critical line ``gamma = 1 - kappa``.

    Images: exactly, not iteratively
    --------------------------------
    In complex coordinates ``z = x1 + i x2`` the lens equation is

        ``y = (1 - kappa) z - Gamma zbar - psi0 / zbar``

    and eliminating ``zbar`` gives a **quartic**

        ``Gamma Q^2 + a z Q (y - a z) + psi0 a^2 z^2 = 0``,
        ``Q = conj(Gamma) z^2 + y z + psi0``,  ``a = 1 - kappa``,

    solved algebraically. Roots that do not satisfy the original (non-analytic)
    lens equation are spurious and are rejected on the residual. Magnification
    and Morse index come from the analytic Hessian.

    Wave optics: where it is available, and where it cannot be
    ----------------------------------------------------------
    GLoW's only non-axisymmetric integrator, ``It_MultiContour_C``, requires the
    Fermat potential's contours to close at large radius. Here

        ``phi -> [(1-kappa-gamma) x1^2 + (1-kappa+gamma) x2^2] / 2``,

    so the contours close **iff** ``kappa + gamma < 1`` -- which is exactly the
    macro-minimum side of the critical line (:attr:`contours_close`). For
    ``kappa + gamma > 1`` they are open hyperbolae and the method structurally
    cannot run; there :meth:`F_wave` raises and the caller must use
    :meth:`F_go`.

    Where the contours do close, ``I(tau)`` is GLoW's and is correct -- it
    reproduces the quartic image census to machine precision and converges to
    the analytic asymptote ``2 pi sqrt(mu_macro)``. What GLoW cannot do is the
    *transform* to ``F(w)``: all of its regularization stages assume an
    isolated lens, ``I(tau -> inf) -> 2 pi``, whereas an external convergence
    and shear make the asymptotic reference the macro-image itself. That is
    handled in :mod:`lensing.wave_contour`, which owns the transform and only
    the transform.

    GLoW is still the authority on the potential (:meth:`glow_lens`) and
    cross-checks the image census.

    The aligned configuration is degenerate
    ---------------------------------------
    At ``phi_gamma = 0`` the whole configuration is symmetric under
    ``x2 -> -x2`` and the images come in mirror pairs with *equal* magnification
    and *equal* arrival time -- one image with a rescaled amplitude, not two.
    ``phi_gamma`` therefore defaults to ``pi/4``.
    """

    name = "changrefsdal"

    def __init__(self, kappa: float = 0.0, gamma: float = 0.0,
                 phi_gamma: float = np.pi / 4.0, psi0: float = 1.0):
        self.kappa = float(kappa)
        self.gamma = float(gamma)
        self.phi_gamma = float(phi_gamma)
        self.psi0 = float(psi0)
        G = self.gamma * np.exp(2j * self.phi_gamma)
        self.gamma1, self.gamma2 = float(G.real), float(G.imag)

    def __repr__(self):
        return ("ChangRefsdal(kappa=%g, gamma=%g, phi_gamma=%g, psi0=%g)"
                % (self.kappa, self.gamma, self.phi_gamma, self.psi0))

    # -- macro classification --------------------------------------------

    @property
    def macro_eigenvalues(self) -> tuple:
        """``(1 - kappa - gamma, 1 - kappa + gamma)``: the smooth-field
        eigenvalues, independent of the shear orientation."""
        return (1.0 - self.kappa - self.gamma, 1.0 - self.kappa + self.gamma)

    @property
    def macro_type(self) -> str:
        """``'minimum'``, ``'saddle'`` or ``'maximum'`` for the macro-image."""
        l1, l2 = self.macro_eigenvalues
        if l1 > 0 and l2 > 0:
            return "minimum"
        if l1 < 0 and l2 < 0:
            return "maximum"
        return "saddle"

    @property
    def distance_to_critical(self) -> float:
        """``gamma - (1 - kappa)``: signed distance to the critical line.

        Negative in the macro-minimum region, positive in the macro-saddle
        region. Magnification diverges as it goes to zero, which is why a band
        around it is excluded from the scan.
        """
        return self.gamma - (1.0 - self.kappa)

    @property
    def macro_magnification(self) -> float:
        """``1 / |(1-kappa)^2 - gamma^2|``: the smooth-field magnification.

        An overall amplitude factor only. The recovery statistic is a
        *normalized* overlap, so it does not enter -- but it is recorded,
        since it is what diverges at the critical line.
        """
        l1, l2 = self.macro_eigenvalues
        return float("inf") if l1 * l2 == 0.0 else 1.0 / abs(l1 * l2)

    # -- geometry ---------------------------------------------------------

    def psi(self, x1, x2):
        r2 = np.asarray(x1) ** 2 + np.asarray(x2) ** 2
        return (0.5 * self.psi0 * np.log(r2)
                + 0.5 * self.kappa * r2
                + 0.5 * self.gamma1 * (np.asarray(x1) ** 2 - np.asarray(x2) ** 2)
                + self.gamma2 * np.asarray(x1) * np.asarray(x2))

    def fermat(self, x1, x2, y):
        return 0.5 * ((x1 - y) ** 2 + x2 ** 2) - self.psi(x1, x2)

    def glow_lens(self):
        """GLoW's own object for this lens: ``Psi_PointLens + Psi_Ext``.

        Used for the independent image-census cross-check; the potential and
        both derivative sets are GLoW's, not reimplemented here.
        """
        from glow import lenses

        return lenses.CombinedLens({"lenses": [
            lenses.Psi_PointLens({"psi0": self.psi0}),
            lenses.Psi_Ext({"kappa": self.kappa,
                            "gamma1": self.gamma1,
                            "gamma2": self.gamma2}),
        ]})

    def _quartic(self, y):
        """Coefficients of the image quartic, ascending powers of ``z``."""
        from numpy.polynomial import polynomial as P

        a = 1.0 - self.kappa
        G = self.gamma1 + 1j * self.gamma2
        Q = np.array([self.psi0 + 0j, y + 0j, np.conj(G)])
        return P.polyadd(
            P.polyadd(G * P.polymul(Q, Q),
                      a * P.polymul(P.polymul([0j, 1 + 0j], Q), [y + 0j, -a + 0j])),
            self.psi0 * a * a * np.array([0j, 0j, 1 + 0j]),
        )

    def images(self, y, residual_tol: float = 1e-7) -> list:
        """All images, sorted by arrival time.

        The point-mass singularity at the origin is *not* an image (the Fermat
        potential diverges there); GLoW labels it ``sing/cusp max`` and it is
        excluded here by construction, since ``z = 0`` is not a root.
        """
        y = float(y)
        a = 1.0 - self.kappa
        G = self.gamma1 + 1j * self.gamma2
        c = self._quartic(y)
        if np.all(c == 0):                                # pragma: no cover
            return []
        roots = np.roots(c[::-1])

        out = []
        for z in roots:
            if abs(z) < 1e-12:
                continue
            # the quartic was obtained by eliminating zbar, which can introduce
            # roots of the *analytic* equation that do not solve the original
            if abs(a * z - G * np.conj(z) - self.psi0 / np.conj(z) - y) > residual_tol:
                continue
            x1, x2 = float(z.real), float(z.imag)
            r2 = x1 * x1 + x2 * x2
            p11 = self.psi0 * (x2 * x2 - x1 * x1) / r2**2 + self.kappa + self.gamma1
            p22 = self.psi0 * (x1 * x1 - x2 * x2) / r2**2 + self.kappa - self.gamma1
            p12 = -2.0 * self.psi0 * x1 * x2 / r2**2 + self.gamma2
            A = np.array([[1.0 - p11, -p12], [-p12, 1.0 - p22]])
            det = float(np.linalg.det(A))
            if det == 0.0:                                # pragma: no cover
                continue
            out.append(Image2D(
                x1=x1, x2=x2, mu=1.0 / det,
                tau=float(self.fermat(x1, x2, y)),
                morse=int(np.sum(np.linalg.eigvalsh(A) < 0.0)),
            ))
        out.sort(key=lambda im: im.tau)
        return out

    @property
    def contours_close(self) -> bool:
        """Whether the Fermat potential's level sets close at large radius.

        ``phi -> [(1-kappa-gamma) x1^2 + (1-kappa+gamma) x2^2] / 2``, so the
        level sets are ellipses for ``kappa + gamma < 1`` and hyperbolae
        otherwise. This is the precondition of GLoW's contour integrator, and
        it coincides with the macro-image being a minimum.
        """
        return (self.kappa + self.gamma) < 1.0

    def F_wave(self, f, M, y, conv=None, **kw):
        """Exact wave-optics ``F``, project convention -- closed contours only.

        Raises :class:`ValueError` when ``kappa + gamma > 1``: there the Fermat
        potential is unbounded below along one axis, the contour method has no
        closed curves to integrate, and no setting changes that. Callers that
        must cover the whole plane use :meth:`F_hybrid`.

        ``I(tau)`` is GLoW's; the transform is
        :func:`lensing.wave_contour.F_of_w_contour`.
        """
        from . import wave_contour as wc

        f_arr = np.asarray(f, dtype=float)
        out = np.ones(f_arr.shape, dtype=complex)
        good = f_arr > 0
        if np.any(good):
            It = kw.pop("It", None) or wc.build_It(self, float(y))
            w = amp.w_of_f(f_arr[good], M)
            F_lit = wc.F_of_w_contour(It, w, **kw)
            out[good] = amp.to_project_convention(F_lit, conv=conv)
        return out

    def F_hybrid(self, f, M, y, conv=None, **kw):
        """``F_wave`` where the contours close, ``F_go`` where they cannot.

        Returns ``(F, mode)`` with ``mode`` one of ``"wave"`` / ``"go"``, so
        that every cell of a scan records which physics produced it rather than
        leaving the reader to infer it from ``kappa + gamma``.
        """
        if self.contours_close:
            return self.F_wave(f, M, y, conv=conv, **kw), "wave"
        return self.F_go(f, M, y, conv=conv), "go"


MODELS = {
    "sis": SIS,
    "cored": CoredIsothermal,
    "pointmass": PointMass,
    "changrefsdal": ChangRefsdal,
}


def build(name: str, **kw) -> LensCommon:
    """Construct a model by name, e.g. ``build("cored", rc=0.15)``."""
    if name not in MODELS:
        raise ValueError(f"unknown lens model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](**kw)
