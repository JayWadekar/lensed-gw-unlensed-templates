"""Provenance block stored in every result file.

Factored out of the phase scripts, which each carried their own copy, so the
faint-image study records exactly the same fields without duplicating them a
fourth time.  Existing scripts are left untouched.
"""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:                                       # pragma: no cover
        return "unknown"


def versions() -> dict:
    import numpy as np
    import scipy
    out = {"python": platform.python_version(), "numpy": np.__version__,
           "scipy": scipy.__version__}
    try:
        import pycbc
        out["pycbc"] = pycbc.__version__
    except Exception:                                       # pragma: no cover
        pass
    return out


def block(seed=None, **extra) -> dict:
    """Timestamp, commit, seed, dependency versions, plus anything passed in."""
    from . import conventions
    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "seed": seed,
        "convention": conventions.DEFAULT.as_dict(),
        "versions": versions(),
    }
    out.update(extra)
    return out
