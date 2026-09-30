#!/usr/bin/env python
"""Compare regenerated data products against the shipped ones in ``data/``.

    python scripts/checks/compare_data.py data regenerated

For every ``.npz`` present in both trees, each array is compared exactly; for
every ``.json``, each number is compared, ignoring provenance fields
(timestamps, git commits, wall times, package versions).  Files present in only
one tree are listed, not treated as failures, so a partial regeneration can be
checked too.  Exit status is 1 if any shared file differs.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

#: Provenance fields that legitimately differ between runs.
IGNORE = ("timestamp", "git_commit", "wall_time", "versions", "hostname",
          "platform", "elapsed", "path", "source", "thresholds_source")


def _ignored(key):
    return any(tag in key for tag in IGNORE)


def _flatten(obj, prefix=""):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if not _ignored(str(k)):
                out.update(_flatten(v, "%s.%s" % (prefix, k)))
        return out
    if isinstance(obj, list):
        out = {}
        for i, v in enumerate(obj):
            out.update(_flatten(v, "%s[%d]" % (prefix, i)))
        return out
    return {prefix: obj}


def _same(a, b, rtol):
    if isinstance(a, float) and isinstance(b, float):
        if np.isnan(a) and np.isnan(b):
            return True
        return a == b or abs(a - b) <= rtol * max(abs(a), abs(b))
    return a == b


def compare_npz(a, b):
    x, y = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    bad = []
    for k in sorted(set(x.files) & set(y.files)):
        if k == "meta":
            continue
        u, v = x[k], y[k]
        if u.shape != v.shape:
            bad.append("%s: shape %s vs %s" % (k, u.shape, v.shape))
        elif u.dtype.kind in "fc":
            if not np.array_equal(u, v, equal_nan=True):
                bad.append("%s: max |diff| %.3g" % (k, np.nanmax(np.abs(u - v))))
        elif not np.array_equal(u, v):
            bad.append("%s: differs" % k)
    for k in sorted(set(x.files) ^ set(y.files)):
        bad.append("%s: only in one file" % k)
    return bad


def compare_json(a, b, rtol=1e-12):
    x = _flatten(json.load(open(a)))
    y = _flatten(json.load(open(b)))
    return ["%s: %r vs %r" % (k, x.get(k), y.get(k))
            for k in sorted(set(x) | set(y)) if not _same(x.get(k), y.get(k), rtol)]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit(__doc__)
    ref, new = argv
    n_ok = n_bad = 0
    only = []
    for root, _, files in sorted(os.walk(ref)):
        for f in sorted(files):
            if not f.endswith((".npz", ".json")):
                continue
            rel = os.path.relpath(os.path.join(root, f), ref)
            other = os.path.join(new, rel)
            if not os.path.exists(other):
                only.append(rel)
                continue
            fn = compare_npz if f.endswith(".npz") else compare_json
            bad = fn(os.path.join(ref, rel), other)
            if bad:
                n_bad += 1
                print("DIFFERS  %s" % rel)
                for line in bad[:10]:
                    print("         %s" % line)
            else:
                n_ok += 1
                print("same     %s" % rel)
    print("\n%d identical, %d differ, %d not regenerated" % (n_ok, n_bad, len(only)))
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
