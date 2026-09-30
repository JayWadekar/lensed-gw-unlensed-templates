"""Journal figure style for the figures of the paper.

One place for every rcParam, size and colour used by the main-text figures, so
that no individual plot script sets them.  Import and call ``use()`` first:

    from prl_style import use, COL_WIDTH, TEXT_WIDTH, OUTDIRS
    use()

Text is rendered by LaTeX so that figure labels use the same Computer Modern
faces as a ``revtex4-2`` body, and labels are set deliberately large (axis
labels 12 pt, ticks 10 pt) to stay readable in a two-column layout.
"""

from __future__ import annotations

import os
import shutil

import matplotlib

# REVTeX 4-2, prd, twocolumn: \columnwidth = 246 pt, \textwidth = 510 pt.
COL_WIDTH = 3.404
TEXT_WIDTH = 7.055

# Where PRL/PRD-short main-text figures are written, and mirrored to.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
#: Data products the figures are drawn from.  Point ``LENSING_DATA`` at another
#: directory (e.g. one filled by ``make simulate``) to plot regenerated data.
DATA_ROOT = os.path.abspath(os.environ.get("LENSING_DATA",
                                           os.path.join(REPO_ROOT, "data")))
OUTDIRS = (os.path.abspath(os.environ.get("LENSING_FIGURES",
                                          os.path.join(REPO_ROOT, "figures"))),)


def relative(obj):
    """Copy of ``obj`` with absolute paths inside the repository made relative.

    Summaries record which data file a figure was drawn from; storing that path
    relative to the repository keeps them machine-independent.
    """
    if isinstance(obj, dict):
        return {k: relative(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(relative(v) for v in obj)
    if isinstance(obj, str) and os.path.isabs(obj) and obj.startswith(REPO_ROOT):
        return os.path.relpath(obj, REPO_ROOT)
    return obj


def data_path(*parts):
    """Path of a data product under ``DATA_ROOT``."""
    return os.path.join(DATA_ROOT, *parts)

# Recovery/match figures share one scale and one accent colour.
CMAP = "viridis"
MINIMAL_MATCH = 0.97
C_MM = "#d62728"        # the 0.97 minimal-match contour
C_GUIDE = "black"       # f_ML guide contours
C_STRUCT = "white"      # lens-structure boundaries (critical line, n_img)

#: LaTeX text rendering, as in the paper.  Set ``LENSING_USETEX=0`` (or run
#: without a ``latex`` executable) to fall back to matplotlib's mathtext.
USETEX = (os.environ.get("LENSING_USETEX", "1") != "0"
          and shutil.which("latex") is not None)

_RC = {
    "text.usetex": USETEX,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 11,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "legend.frameon": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.6,
    "ytick.major.size": 3.6,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.top": True,
    "ytick.right": True,
    "lines.linewidth": 1.4,
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "savefig.bbox": None,          # constrained_layout owns the margins
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "pdf.compression": 6,
}


def _bold(m):
    body = m.group(1)
    if "$" in body:                      # mixed text and math: drop the bold
        return body
    return r"$\mathbf{%s}$" % body.replace(" ", r"\ ").replace("--", "-").replace("-", r"\text{-}")


def _to_mathtext(s):
    """Translate the few LaTeX text macros the figures use into mathtext.

    Only used when LaTeX is unavailable (``USETEX`` false): mathtext has no
    ``\\textbf``, ``\\emph`` or ``\\textrm``, and would print them verbatim.
    """
    import re
    s = re.sub(r"\\textrm\{([^{}]*)\}",
               lambda m: r"\mathrm{%s}" % m.group(1).replace("-", r"\text{-}"), s)
    s = s.replace("--", "\u2013")
    s = re.sub(r"\\emph\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\textbf\{([^{}]*)\}", _bold, s)
    s = s.replace(r"\!", "").replace("~", " ")
    return s


def use():
    """Activate the style.  Safe to call more than once."""
    # Fixed PDF timestamps, so that regenerating a figure from the same data
    # gives a byte-identical file (``git status`` then shows no change).
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    matplotlib.use("Agg")
    matplotlib.rcParams.update(_RC)
    if not USETEX:
        import matplotlib.text as mtext
        if not getattr(mtext.Text.set_text, "_lensing_patched", False):
            _orig = mtext.Text.set_text

            def set_text(self, s):
                if isinstance(s, str):
                    s = _to_mathtext(s)
                return _orig(self, s)
            set_text._lensing_patched = True
            mtext.Text.set_text = set_text


def panel_label(ax, text, loc="upper left", color="black", fontsize=12):
    """Put a bold (a)/(b)/... label inside ``ax`` on an opaque patch."""
    x, ha = (0.035, "left") if "left" in loc else (0.965, "right")
    y, va = (0.965, "top") if "upper" in loc else (0.035, "bottom")
    return ax.text(
        x, y, r"\textbf{(%s)}" % text, transform=ax.transAxes,
        fontsize=fontsize, color=color, ha=ha, va=va, zorder=12,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                  boxstyle="round,pad=0.22"),
    )


def assert_fits(fig, tol=0.01):
    """Raise if any artist extends past the figure canvas.

    ``savefig`` is called with ``bbox=None`` so that ``constrained_layout``
    owns the margins; the cost is that an overflowing title is *clipped*
    rather than growing the canvas.  That bit once: Figure 2's panel titles
    were cut in the manuscript PDF while the PNG looked correct, because
    ``constrained_layout`` is solved at draw time and the PNG was written by
    the second ``savefig`` call, after the layout had re-solved.  Checking the
    tight bounding box against the canvas catches it at build time instead of
    in the compiled paper.
    """
    renderer = fig.canvas.get_renderer()
    tb = fig.get_tightbbox(renderer)          # inches
    w, h = fig.get_size_inches()
    over = {"left": -tb.x0, "bottom": -tb.y0,
            "right": tb.x1 - w, "top": tb.y1 - h}
    bad = {k: v for k, v in over.items() if v > tol}
    if bad and not USETEX:
        # mathtext glyphs are slightly larger than LaTeX's; the layout is tuned
        # for LaTeX, so only warn in the fallback rather than refuse.
        import warnings
        warnings.warn("mathtext fallback: figure content overflows the canvas "
                      "by %s in; install LaTeX for the paper's exact layout"
                      % ", ".join("%s %.3f" % kv for kv in bad.items()))
        return
    if bad:
        raise SystemExit(
            "figure content overflows the canvas and would be clipped: "
            + ", ".join("%s by %.3f in" % (k, v) for k, v in bad.items())
            + "; increase figsize (or relax box_aspect) rather than shipping "
              "a clipped figure")


def save(fig, name, outdirs=None):
    """Write ``name``.pdf (and .png) into every output directory.

    Returns the primary path.  The second directory is the Overleaf project,
    which must carry the binary itself in order to compile.

    The figure is drawn once before saving so that ``constrained_layout`` is
    settled and every output format gets the same, checked layout.
    """
    fig.canvas.draw()
    assert_fits(fig)
    paths = []
    for d in (OUTDIRS if outdirs is None else outdirs):
        os.makedirs(d, exist_ok=True)
        paths.append(os.path.join(d, name + ".pdf"))
    fig.savefig(paths[0])
    fig.savefig(paths[0].replace(".pdf", ".png"), dpi=200)
    for p in paths[1:]:
        shutil.copyfile(paths[0], p)
    return paths[0]
