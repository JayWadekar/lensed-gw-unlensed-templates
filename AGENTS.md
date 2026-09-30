# Notes for coding agents

Read `README.md` first; it maps every figure to its data and its commands.

- Run everything from the repository root. Scripts find `src/` themselves, so
  `pip install -e .` is optional.
- `make figures` rebuilds all seven figures from `data/` in about 30 s, and
  must leave `git status` clean: the committed figures are byte-identical to
  what the scripts produce. If a change you make alters a figure, say so.
- `make test` runs the test suite (about 2 minutes).
- The simulations in `scripts/simulate/` are expensive (see the README table).
  They write to `regenerated/` by default; never point them at `data/`. Check
  their output with `python scripts/checks/compare_data.py data regenerated`.
- Sign conventions (Morse phase, time shift, autocorrelation) live only in
  `src/lensing/conventions.py`. Do not hard-code a sign elsewhere; the tests in
  `tests/test_phase0_gate.py` and `tests/test_morse_phase.py` check them.
- Plots read data and never recompute it; simulation and plotting are separate.
