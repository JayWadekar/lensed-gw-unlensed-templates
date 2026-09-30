"""Single source of truth for Fourier / matched-filter sign conventions.

The signs of the Morse factor, the time shift, and the imaginary
autocorrelation term are *convention dependent*, so they are fixed empirically
(``scripts/checks/phase0_conventions.py``, ``tests/test_phase0_gate.py``) by
comparing against an explicit frequency-domain lensed template.  They are therefore stored here in
one place, and every downstream module derives its signs from this object
rather than hard-coding them.  In particular the numerator coefficient and the
norm cross term are both generated from the single complex number
:meth:`Convention.image_coeff`, so they cannot drift apart.

Adopted definitions
-------------------

Frequency domain (PyCBC / ``numpy.fft.rfft``)::

    htilde(f) = \\int h(t) e^{-2 pi i f t} dt

so a template whose coalescence is *delayed* by ``tau`` picks up
``exp(-2 pi i f tau)``.

Complex inner product, single detector, one-sided PSD ``S_n``::

    (a|b) = 4 Re-free \\int_0^\\infty atilde(f) conj(btilde(f)) / S_n(f) df

i.e. **linear in the first argument and antilinear in the second**.  This is
the convention for which PyCBC's ``matched_filter`` returns
``z(t) = (d | h_t) / sigma(h)``, with ``h_t`` the template whose coalescence
sits at time ``t``.

Two-image geometric-optics template
-----------------------------------

The lensed template with its first image at time ``t`` is

    h_L,t = h_t + c * h_{t + tau},        c = s_M * i * a,   tau = s_T * t_d

with ``s_M = morse_sign`` and ``s_T = shift_sign``.  Because the inner product
is antilinear in its second argument, filtering data against it gives

    (d | h_L,t) = z_1(t) + conj(c) * z_1(t + tau)

and the squared norm is

    N = 1 + a^2 + 2 Re[ conj(c) * C(tau) ],     C(tau) = (h_0 | h_tau).

With ``c = s_M i a`` one has ``Re[conj(c) C] = s_M a Im C``, so

    N = 1 + a^2 + 2 * s_M * a * Im C(tau),

i.e. the *derived* norm cross-sign is ``s_N = +s_M``.

Relative Morse phase
--------------------

Nothing in the algebra above requires ``c`` to be *pure imaginary*.  The
recombination identity and the norm are generated from the single complex
number ``u = conj(c)/a``, so an arbitrary relative phase costs no new algebra
(see :mod:`lensing.recombine`).  In geometric optics the relative phase between
two images is not arbitrary anyway: with Morse factors ``exp(-i n pi/2)``,
``n in {0, 1, 2}`` for a minimum, saddle and maximum, the *ratio* of the
trailing to the leading factor is one of only **four** values, ``i^q`` with
``q = 0, 1, 2, 3``.  ``rel_morse_quarters`` selects ``q``:

===  ===================  =====================================================
``q``  ``c``              image pair (leading -> trailing)
===  ===================  =====================================================
0     ``s_M a``           same Morse type: min->min, saddle->saddle, max->max
1     ``s_M i a``         min->saddle, saddle->max   **(the deployed family)**
2     ``-s_M a``          min->max
3     ``-s_M i a``        saddle->min, max->saddle
===  ===================  =====================================================

``q = 1`` is the default and reproduces the frozen configuration exactly.  The
``q = 0`` family is needed for a pair of images of the *same* type, which is
what a microlens near a macro-saddle produces.

Note that the initial derivation
(``Relevant_papers/Lensing_template_theory.tex``, Eqs. 5 and 9) writes the
numerator coefficient as ``c`` rather than ``conj(c)`` and the cross term as
``2 Re[c C]``, which corresponds to an inner product linear in its *second*
argument.  Relative to the convention above that amounts to relabelling
``s_M -> -s_M``; only the relative sign between numerator and denominator is
physical.  This is why the signs are not settled by argument:
the Phase 0 check enumerates all four ``(s_M, s_T)`` combinations and picks the
one that reproduces explicit frequency-domain filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace

#: ``i^q`` as exact complex literals, so no floating-point ``pow`` enters the
#: coefficient that every downstream sign is generated from.
QUARTER_TURN = {0: 1 + 0j, 1: 1j, 2: -1 + 0j, 3: -1j}

#: Human-readable name of each relative Morse phase, for provenance records.
REL_MORSE_LABEL = {
    0: "same type (0 deg)",
    1: "min->saddle (90 deg)",
    2: "min->max (180 deg)",
    3: "saddle->min (270 deg)",
}


@dataclass(frozen=True)
class Convention:
    """Frozen sign/normalization convention for the lensed-template algebra."""

    name: str = "default"

    #: sign s_M in  h_L = h_0 + s_M * i * a * h_shift.
    morse_sign: int = +1

    #: sign s_T of the time shift of the second image.  s_T = +1 means the
    #: saddle image arrives *later* by t_d.
    shift_sign: int = +1

    #: exp(-2 pi i f tau) is the Fourier factor for a delay of +tau.
    fd_delay_sign: int = -1

    #: True if the matched-filter output is (d|h) with the inner product
    #: antilinear in its second argument, as PyCBC's matched_filter returns.
    conjugate_template: bool = True

    #: ``q`` in ``c = s_M a i^q``: the relative Morse phase of the trailing
    #: image in quarter turns.  ``q = 1`` is the deployed minimum+saddle
    #: family and the frozen default; see the module docstring.
    rel_morse_quarters: int = 1

    # -- the single generator of every downstream sign ---------------------

    def image_coeff(self, a):
        """``c``: the complex factor multiplying the second image.

        ``c = s_M a i^q`` with ``q = rel_morse_quarters``.  At the default
        ``q = 1`` this is identically ``s_M i a``, the frozen configuration.
        """
        return self.morse_sign * a * QUARTER_TURN[self.rel_morse_quarters]

    def filter_coeff(self, a):
        """Coefficient multiplying ``z_1(t + tau)`` in the numerator.

        ``conj(c)`` for an inner product antilinear in the template, ``c``
        otherwise.
        """
        c = self.image_coeff(a)
        return c.conjugate() if self.conjugate_template else c

    def norm_cross_sign(self) -> int:
        """``s_N`` in ``N = 1 + a^2 + 2 s_N a Im C(tau)``.

        Derived, not assumed: ``2 Re[filter_coeff(a) * C] = 2 s_N a Im C``.

        This *integer* form exists only when ``c`` is pure imaginary, i.e. for
        odd ``rel_morse_quarters``; for an even ``q`` the cross term picks up
        ``Re C`` instead and no single sign describes it.  Raises rather than
        returning a misleading number -- ``recombine.lensed_norm_sq`` is
        general and should be used instead.
        """
        if self.rel_morse_quarters % 2 == 0:
            raise ValueError(
                "norm_cross_sign() is defined only for a pure-imaginary c "
                "(odd rel_morse_quarters); q=%d gives a Re C cross term. "
                "Use recombine.lensed_norm_sq, which is general."
                % self.rel_morse_quarters
            )
        s = self.morse_sign * (1 if self.rel_morse_quarters == 1 else -1)
        return s if self.conjugate_template else -s

    def lag(self, t_d):
        """Signed time offset ``tau = s_T t_d`` of the second image."""
        return self.shift_sign * t_d

    # -- variants and bookkeeping -----------------------------------------

    def with_signs(self, morse_sign: int, shift_sign: int) -> "Convention":
        return replace(
            self,
            name=f"m{morse_sign:+d}_t{shift_sign:+d}",
            morse_sign=morse_sign,
            shift_sign=shift_sign,
        )

    def with_rel_morse(self, q: int) -> "Convention":
        """Same convention, different relative Morse phase ``q``."""
        if q not in QUARTER_TURN:
            raise ValueError("rel_morse_quarters must be one of %s, got %r"
                             % (sorted(QUARTER_TURN), q))
        return replace(self, name=f"{self.name}_q{q}", rel_morse_quarters=q)

    @property
    def rel_morse_label(self) -> str:
        """Human-readable name of the relative Morse phase."""
        return REL_MORSE_LABEL[self.rel_morse_quarters]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["rel_morse_label"] = self.rel_morse_label
        if self.rel_morse_quarters % 2:
            d["norm_cross_sign"] = self.norm_cross_sign()
        return d


#: The adopted project convention.  Phase 0 must confirm or replace this.
DEFAULT = Convention()


def all_rel_morse() -> list[Convention]:
    """The four relative-Morse-phase families, ``q = 0, 1, 2, 3``.

    Maximizing the statistic over these is the *generalized* two-image search:
    it reuses the same grid and the same recombination identity, and only the
    cheap array algebra repeats.
    """
    return [DEFAULT.with_rel_morse(q) for q in sorted(QUARTER_TURN)]


def all_candidates() -> list[Convention]:
    """The four ``(s_M, s_T)`` combinations tested by the Phase 0 gate."""
    return [DEFAULT.with_signs(m, t) for m in (+1, -1) for t in (+1, -1)]
