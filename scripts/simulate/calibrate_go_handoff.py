#!/usr/bin/env python
"""Calibrate the geometric-optics handoff used by ``F_ML(backend="auto")``.

The exact point-lens factor is evaluated with mpmath, whose cost is driven by
``|z| = w y^2 / 2``.  The expensive corner (large ``w`` *and* large ``y``) is
precisely where geometric optics is accurate, so a measured handoff keeps the
maps affordable without approximating anywhere it matters.

For each ``y`` this script finds ``w_switch(y)``: the smallest ``w`` beyond
which the measured relative error ``|F_ML - F_GO| / |F_GO|`` stays below a
predeclared tolerance for **all** larger ``w`` sampled.  The answer is written
to ``results/frozen/go_handoff.json`` and consumed by
``lensing.amplification.go_handoff_table``.

    python scripts/simulate/calibrate_go_handoff.py

The scan is deliberately dense in ``w``, because the error oscillates: a single
lucky sample can sit near a zero crossing and grossly understate the envelope.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from lensing import amplification as amp  # noqa: E402


def error_envelope(y, w_grid, n_proc=1):
    """``|F_ML - F_GO| / |F_GO|`` on ``w_grid`` at fixed ``y``."""
    F_ml = amp.F_ML_from_w(w_grid, y, backend="mpmath")
    F_ml = amp.to_project_convention(F_ml)
    mu_p, mu_m = amp.magnifications(y)
    dt = float(amp.delay_dimensionless(y))
    F_go = np.conjugate(
        np.sqrt(mu_p) - 1j * np.sqrt(np.abs(mu_m)) * np.exp(1j * w_grid * dt)
    )
    return np.abs(F_ml - F_go) / np.abs(F_go)


def w_switch_for_y(y, tol, w_max=3.0e4, n_w=600):
    """Smallest ``w`` past which the error envelope stays below ``tol``."""
    w_grid = np.geomspace(1.0, w_max, n_w)
    err = error_envelope(y, w_grid)
    # walk backwards: the switch is just after the last violation
    bad = np.nonzero(err > tol)[0]
    if bad.size == 0:
        w_sw = float(w_grid[0])
    elif bad[-1] == w_grid.size - 1:
        w_sw = np.inf  # never converges within w_max
    else:
        w_sw = float(w_grid[bad[-1] + 1])
    return w_sw, w_grid, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--w-max", type=float, default=3.0e4)
    ap.add_argument("--n-w", type=int, default=600)
    ap.add_argument("--n-y", type=int, default=25)
    ap.add_argument("--out", default="regenerated/go_handoff.json")
    args = ap.parse_args()

    ys = np.geomspace(0.01, 2.0, args.n_y)
    rows = []
    print("      y     w_switch   |z|_switch   max_err(w>w_sw)")
    for y in ys:
        w_sw, w_grid, err = w_switch_for_y(y, args.tol, args.w_max, args.n_w)
        past = w_grid >= w_sw
        max_past = float(err[past].max()) if past.any() else float("nan")
        z_sw = 0.5 * w_sw * y * y
        rows.append(
            {
                "y": float(y),
                "w_switch": float(w_sw),
                "z_switch": float(z_sw),
                "max_err_past_switch": max_past,
                "err_at_w1e4": float(
                    np.interp(1e4, w_grid, err) if w_grid[-1] >= 1e4 else np.nan
                ),
            }
        )
        print(
            "  %6.4f  %10.1f  %10.2f   %.3e"
            % (y, w_sw, z_sw, max_past)
        )

    finite = [r for r in rows if np.isfinite(r["w_switch"])]
    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tolerance": args.tol,
        "w_max_scanned": args.w_max,
        "n_w": args.n_w,
        "definition": (
            "w_switch(y) is the smallest w beyond which "
            "|F_ML - F_GO|/|F_GO| <= tolerance for every larger w sampled. "
            "Geometric optics may replace the exact factor for w >= w_switch."
        ),
        "y": [r["y"] for r in rows],
        "w_switch": [r["w_switch"] for r in rows],
        "rows": rows,
        "n_converged": len(finite),
        "n_total": len(rows),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote %s (%d/%d y values converged within w<=%g)"
          % (args.out, len(finite), len(rows), args.w_max))
    return 0


if __name__ == "__main__":
    sys.exit(main())
