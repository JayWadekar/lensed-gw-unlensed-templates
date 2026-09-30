"""Isolation layer over GLoW, the only module that knows GLoW's API.

GLoW (Villarrubia-Rojo et al. 2024, arXiv:2409.04606) computes the exact
wave-optics amplification factor for general lens models. This wrapper exists so
that nothing else in the project depends on its interface, and so that the two
facts we rely on are asserted in one place:

**Convention.** GLoW returns ``F(w) = (-i w / 2 pi) \\int I(tau) e^{+i w tau}
d tau`` -- the *literature* convention -- with ``tau`` measured from the global
minimum of the Fermat potential, ``F -> 1`` as ``w -> 0``, and the
geometric-optics prefactor included. That is the same convention as this
project's :func:`lensing.amplification.F_ML_from_w`, so callers convert with
:func:`lensing.amplification.to_project_convention`. This was **verified, not
assumed**: GLoW's analytic point lens reproduces the frozen mpmath
implementation to 1.5e-6 relative in the same convention, while the conjugated
comparison is wrong by O(1). GLoW's own ``waveform.py`` likewise conjugates
``F`` before multiplying a PyCBC-convention waveform.

**Dimensionless frequency.** GLoW's ``w = 8 pi G M_Lz f / c^3``, identical to
:func:`lensing.amplification.w_of_f`.

Method selection follows GLoW's own guidance:

======================  ==================================================
model                   GLoW route
======================  ==================================================
point mass              ``Fw_AnalyticPointLens_C(y)``
SIS                     ``Fw_SemiAnalyticSIS_C(y, psi0)``, ``method='osc'``
axisymmetric, general   ``Fw_DirectFT_C(It_SingleIntegral_C(Psi, y))``
======================  ==================================================

``Fw_DirectFT_C`` is used for the general route rather than ``Fw_FFT_C``
because the latter precomputes an interpolation grid over ``[wmin, wmax]``
(default ``[1e-2, 1e2]``) and *raises* on extrapolation, whereas ``DirectFT``
evaluates exactly at each requested ``w``. Our maps need ``w`` up to ~1e4.
"""

from __future__ import annotations

import numpy as np


def glow_version() -> str:
    """Installed GLoW version, for provenance."""
    try:
        from importlib.metadata import version

        return version("glow")
    except Exception:  # pragma: no cover
        return "unknown"


def _as_array(w):
    w = np.atleast_1d(np.asarray(w, dtype=float))
    return w


def F_of_w(model, y: float, w, **kw) -> np.ndarray:
    """Exact wave-optics ``F(w)`` for ``model`` at impact parameter ``y``.

    Returned in GLoW's (literature) convention; the caller applies
    :func:`lensing.amplification.to_project_convention`.

    ``w`` may be any shape; the result matches it. Values are computed in a
    single GLoW call, which is why the maps are affordable: GLoW builds
    ``I(tau)`` once per ``(model, y)`` and then transforms.
    """
    from glow import freq_domain_c, time_domain_c

    w_arr = _as_array(w)
    name = getattr(model, "name", "")

    if name == "pointmass":
        Fw = freq_domain_c.Fw_AnalyticPointLens_C(y=float(y))
    elif name == "sis":
        p_prec = {"method": "osc"}
        p_prec.update(kw.pop("p_prec", {}))
        Fw = freq_domain_c.Fw_SemiAnalyticSIS_C(
            y=float(y), p_prec=p_prec, psi0=float(model.psi0)
        )
    else:
        # general axisymmetric route: radial integral -> direct Fourier transform
        Psi = model.glow_lens()
        it_prec = {"eval_mode": "interpolate"}
        it_prec.update(kw.pop("it_prec", {}))
        It = time_domain_c.It_SingleIntegral_C(Psi, float(y), p_prec=it_prec)
        Fw = freq_domain_c.Fw_DirectFT_C(It, p_prec=kw.pop("fw_prec", {}))

    out = np.asarray(Fw(w_arr), dtype=complex)
    return out.reshape(np.shape(w)) if np.ndim(w) else out[0]


class _BareIt:
    """Builds GLoW's C-side lens handle without running any contour machinery.

    ``It_MultiContour_C`` finds the critical points first and *then* initializes
    the contour families; for a Chang-Refsdal lens the second step fails (open
    contours) while the first is perfectly usable. This
    reaches the finder without triggering the part that cannot work.
    """

    def __new__(cls, Lens, y):
        from glow import time_domain_c

        class _B(time_domain_c.ItGeneral_C):
            def default_params(self):
                return {}

            def find_all_images(self):
                return []

            def compute_all(self):
                return None, None, None

        return _B(Lens, float(y))


def images_from_glow_2d(model, y: float) -> list[dict]:
    """GLoW's 2-D critical-point finder, for a lens with no axial symmetry.

    Known limitation, measured rather than assumed: in the macro-saddle region
    this finder is **incomplete** -- it can return only one of the two saddle
    points that the Chang-Refsdal quartic finds algebraically. Callers must
    compare censuses before comparing values.
    """
    from glow import wrapper

    bare = _BareIt(model.glow_lens(), float(y))
    p_crits = wrapper.pyFind_all_CritPoints_2D(float(y), bare.lens_to_c)

    keep = {"min", "saddle", "max"}
    morse_of = {"min": 0, "saddle": 1, "max": 2}
    rows = [p for p in p_crits if p["type"] in keep]
    if not rows:
        return []
    tmin = min(float(p["t"]) for p in rows)
    out = [
        {
            "type": p["type"],
            "morse": morse_of[p["type"]],
            "tau": float(p["t"]) - tmin,
            "x1": float(p["x1"]),
            "x2": float(p.get("x2", 0.0)),
            "mag_abs": float(p["mag"]),
        }
        for p in rows
    ]
    out.sort(key=lambda d: -d["mag_abs"])
    return out


def images_from_glow(model, y: float) -> list[dict]:
    """GLoW's own image list, for cross-checking :meth:`AxisymLens.images`.

    GLoW stores ``{'type', 't', 'x1', 'x2', 'mag'}`` per image, with ``mag``
    **unsigned** (``1/|det A|``) and the parity carried by ``type``. Entries of
    type ``'sing/cusp min'`` / ``'sing/cusp max'`` are *not* true images -- they
    are singular centres included for the contour algorithm -- and are dropped
    here.
    """
    from glow import time_domain_c

    Psi = model.glow_lens()
    if not getattr(Psi, "isAxisym", True):
        return images_from_glow_2d(model, y)
    It = time_domain_c.It_SingleIntegral_C(Psi, float(y),
                                           p_prec={"eval_mode": "interpolate"})
    keep = {"min", "saddle", "max"}
    morse_of = {"min": 0, "saddle": 1, "max": 2}
    out = []
    for p in It.p_crits:
        if p["type"] not in keep:
            continue
        out.append(
            {
                "type": p["type"],
                "morse": morse_of[p["type"]],
                "tau": float(p["t"]) - float(It.tmin),
                "x1": float(p["x1"]),
                "x2": float(p.get("x2", 0.0)),
                "mag_abs": float(p["mag"]),
            }
        )
    out.sort(key=lambda d: -d["mag_abs"])
    return out
