#!/usr/bin/env python
"""90% credible contour of the GW231123 point-mass lens posterior.

Reads the reweighted nested samples of the PM / NRSur7dq4 run from the
reproducibility package of the GW231123 lensing paper and writes the 90%
highest-density contour in (y, M_Lz) for Fig. 2(c).

The contour is computed exactly as the paper's ``y_MLz_cornerplot`` draws it
(``make_corner_plots_paper.py::fast_y_MLz_contour``): ``weights_agnostic``
weights, a 300 x 300 weighted histogram over y in [0.1, 1.5] and
M_Lz in [100, 2500] Msun, Gaussian smoothing of 7 bins, and the level that
encloses 90% of the smoothed histogram.

The samples are in the GW231123 reproducibility package; pass its root:

    python scripts/simulate/gw231123_pm_contour.py --package <package root>
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from contourpy import contour_generator
from scipy.ndimage import gaussian_filter

SAMPLES = ("data/pe_results/PM_NRSur_real_GW231123/"
           "nested_samples_all_bbh_parameters_reweighted.pkl")
WEIGHTS = "weights_agnostic"
Y_RANGE = (0.1, 1.5)
MLZ_RANGE = (100.0, 2500.0)
BINS = 300
SMOOTH = 7.0
LEVEL = 0.9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True,
                    help="root of the GW231123 reproducibility package")
    ap.add_argument("--out", default=os.path.join("data", "gw231123",
                                                  "pm_nrsur_y_MLz_90.json"))
    args = ap.parse_args()

    df = pd.read_pickle(os.path.join(args.package, SAMPLES))
    y, mlz, w = (df[k].to_numpy() for k in ("y", "MLz", WEIGHTS))

    H, ye, me = np.histogram2d(y, mlz, bins=BINS, range=[Y_RANGE, MLZ_RANGE],
                               weights=w)
    H = gaussian_filter(H, SMOOTH).T
    H = H / H.max()
    hs = np.sort(H.ravel())[::-1]
    cum = np.cumsum(hs)
    cum /= cum[-1]
    level_val = hs[np.searchsorted(cum, LEVEL)]

    yc = 0.5 * (ye[:-1] + ye[1:])
    mc = 0.5 * (me[:-1] + me[1:])
    lines = contour_generator(yc, mc, H).lines(level_val)

    in_range = ((y >= Y_RANGE[0]) & (y <= Y_RANGE[1])
                & (mlz >= MLZ_RANGE[0]) & (mlz <= MLZ_RANGE[1]))
    out = {
        "source": "GW231123 reproducibility package: " + SAMPLES,
        "event": "GW231123", "lens_model": "point mass",
        "waveform": "NRSur7dq4", "weights": WEIGHTS,
        "credible_level": LEVEL,
        "method": "HPD level of a %dx%d weighted histogram, gaussian_filter "
                  "sigma=%g bins, y in %s, M_Lz in %s (as in the paper's "
                  "y_MLz_cornerplot)" % (BINS, BINS, SMOOTH, list(Y_RANGE),
                                         list(MLZ_RANGE)),
        "weight_fraction_in_histogram_range": float(w[in_range].sum()
                                                    / w.sum()),
        "contours": [{"y": np.round(seg[:, 0], 5).tolist(),
                      "MLz": np.round(seg[:, 1], 2).tolist()}
                     for seg in lines],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote %s: %d contour segment(s), %.4f of the weight in range"
          % (args.out, len(lines), out["weight_fraction_in_histogram_range"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
