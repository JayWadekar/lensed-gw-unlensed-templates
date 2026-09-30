# Searching for lensed gravitational waves with unlensed templates

Code and data to reproduce the figures of

> D. Wadekar, B. Zackay, M. H.-Y. Cheung, T. Islam, J. Roulet, A. K. Mehta,
> T. Venumadhav, J. Mushkin, M. Zumalacárregui and M. Zaldarriaga,
> *Searching for lensed gravitational waves with unlensed templates* (2026).

Gravitational lensing can split a binary-merger signal into two images. The
paper shows that the matched-filter output of every two-image template
follows directly from the complex SNR time series that a search already
computes with its unlensed template:

$$
|z_L(t;\theta)|^2 = \frac{|z_1|^2 + a^2|z_{1s}|^2 + 2a\,\mathrm{Re}\!\left[i^{\,q} z_1 z_{1s}^*\right]}{N(\theta)},
\qquad z_{1s}(t) \equiv z_1(t+t_d),
$$

with $N(\theta) = 1 + a^2 + 2a\,\mathrm{Re}[i^{-q} C(t_d)]$ and $C$ the template
autocorrelation. No lensed templates are built and no extra matched filtering
is done. This repository contains the implementation (`src/lensing/`), the
scripts that produced every data product in the paper, those data products
(11 MB), and the scripts that turn them into the paper's figures.

## Quick start

```bash
git clone https://github.com/JayWadekar/lensed-gw-unlensed-templates.git
cd lensed-gw-unlensed-templates
conda env create -f environment.yml      # or see "Installation" below
conda activate lensing
pip install -e .
make figures                             # all seven figures, ~30 s
```

The figures are written to `figures/`. The repository already contains the
paper's versions there, and regenerating them from `data/` reproduces those
files byte for byte, so `git status` should show no change afterwards.

## Figures and the data behind them

Every command is run from the repository root.

| Figure | What it shows | Plot command | Data (`data/…`) |
| --- | --- | --- | --- |
| 1 | Fitting factor of unlensed vs two-image templates; point mass and Chang–Refsdal lens | `make fig1` | `pointmass_M50/`, `chang_refsdal/` |
| 2 | Signal-consistency ($\chi^2$) test and sensitive-volume ratio | `make fig2` | `chisq/` |
| 3 | Sensitive volume of three strategies for separated lensed pairs | `make fig3` | `strategies/` |
| 4 | Point mass at $M_{\rm tot}=11$ and $100\,M_\odot$ | `make fig4` | `pointmass_M11_M100/` |
| 5 | Chang–Refsdal lens, by image structure | `make fig5` | `chang_refsdal_extended_a/` |
| 6 | Singular isothermal sphere | `make fig6` | `isothermal/` |
| 7 | Cored isothermal sphere | `make fig7` | `isothermal/` |

Each plot script also writes `figures/<name>_summary.json` with every number
quoted in the corresponding caption (fractions of cells above the minimal
match, medians, thresholds, volume ratios), and the data file it read. Some
scripts also write a draft `*_caption.txt`; the paper's captions are
hand-written and are not reproduced here.

Tables: Table I is in `figures/prl_strategies_summary.json`; Table II
(other lens models) in `figures/prl_app_sis_summary.json` and
`figures/prl_app_cored_summary.json`; Tables III and IV (widening the delay
window; folds and the Chang–Refsdal lens) in `data/caustics/appendix_caustics.json`.

## Regenerating the data

The shipped `data/` lets you check the figures without any computing. To
regenerate a product yourself, run its simulation. Simulations write to
`regenerated/` by default, never to `data/`. `make simulate` runs all of them, `make compare` checks the result against `data/`,
and `make figures DATA=regenerated` plots from it.

| Data | Command (see `Makefile` for the exact arguments) | Wall time | Needs |
| --- | --- | --- | --- |
| `strategies/` | `scripts/simulate/lensed_strategies.py` | 10 min, 1 core | numpy, scipy |
| `caustics/` | `scripts/simulate/appendix_caustics.py` | 20 s | numpy, scipy |
| `chisq/chisq_M50` | `scripts/simulate/appendix_chisq_consistency.py` | 3 min, 12 cores | PyCBC |
| `chisq/background_chisq` | `scripts/simulate/appendix_chisq_background.py` | 22 min, 16 cores | PyCBC |
| `pointmass_M50/` | `scripts/simulate/phase3_wave_optics.py` | 28 min, 20 cores | PyCBC |
| `pointmass_M11_M100/` | `scripts/simulate/phase3_wave_optics.py` (×2) | 45 min, 16 cores | PyCBC |
| `chang_refsdal/` | `scripts/simulate/appendix_chang_refsdal.py` | 18 min, 16 cores | PyCBC, GLoW |
| `chang_refsdal_extended_a/` | `scripts/simulate/appendix_chang_refsdal.py --extend-a` | 30 min, 16 cores | PyCBC, GLoW |
| `isothermal/` | `scripts/simulate/appendix_lens_models.py` (×6) | 14 min each, 16 cores | PyCBC, GLoW |

In total this is about 60 CPU-hours. `make simulate-fast` regenerates only the
two quick products. Set the number of worker processes with `NPROC` (e.g.
`make simulate NPROC=32`); it does not change the results. Every run is seeded
explicitly, and every product records its seed, configuration, package
versions and creation time alongside the arrays (`meta` in each `.npz`, and the
`.json` next to it).

We checked that the scripts in this repository reproduce the shipped products:
`strategies/`, `caustics/`, `chisq/chisq_M50` and `isothermal/lensmodel_sis_M50`
regenerate bit for bit.

## Installation

The code needs Python 3.10 or later with numpy, scipy, matplotlib and mpmath.
The simulations also need [PyCBC](https://pycbc.org) (waveforms, the detector
noise PSD and the power $\chi^2$ test). Regenerating the data for the right
column of Fig. 1 and for Figs. 5–7 also needs
[GLoW](https://github.com/glow-astro/GLoW); plotting any figure from the shipped
data does not. Numba is optional and only speeds
up the background calibration.

With conda, `environment.yml` pins the versions every product was made with.
Without conda:

```bash
python -m pip install -e ".[gw,fast,test]"
```

**Plotting only.** `make figures` needs only numpy, scipy and matplotlib:
`python -m pip install -e .` is enough. The figures are typeset with LaTeX, as in
the paper. Without a `latex` executable, or with `LENSING_USETEX=0`, they
fall back to matplotlib's own math rendering: the content is the same, but the
fonts and spacing differ slightly from the paper.

**Installing GLoW.** GLoW (Villarrubia-Rojo et al., arXiv:2409.04606) computes
the exact wave-optics amplification factor for lenses other than the point
mass. It is not on PyPI; we used commit `3b66796` of its `main` branch:

```bash
git clone https://github.com/glow-astro/GLoW.git && cd GLoW
git checkout 3b667962ce5d01c799b88378795a123208db34b5
python configure.py -gsl $CONDA_PREFIX      # GSL from the conda environment
pip install --no-deps --no-build-isolation .
```

`--no-deps` stops pip from changing numpy, scipy or PyCBC; the parts of GLoW we
use need nothing else. On some systems two build problems need a local fix:
with setuptools ≥ 80, `configure.py`'s `from setuptools import sandbox` must
be replaced by a `subprocess` call to `setup.py build_ext --inplace`; and if
linking GSL fails with `undefined reference to ...@GLIBC_PRIVATE`, configure
with the system compiler, `python configure.py -gsl $CONDA_PREFIX -cc /usr/bin/gcc`.
Neither changes GLoW's numerics. The tests in `tests/test_lens_models.py` that
need GLoW are skipped if it is not installed.

## Tests

```bash
make test        # 139 tests, about 2 minutes
```

The tests check, among other things, that the recombined SNR agrees with
explicit filtering by the two-image templates to machine precision for all four
relative Morse phases (`test_morse_phase.py`, `test_phase0_gate.py`); that the
signs of the Morse phase, the time shift and the autocorrelation are fixed
by experiment rather than assumed (`test_phase0_gate.py`); the point-lens
geometry against standard benchmarks (`test_geometry.py`); and GLoW's
amplification factor against our independent point-lens implementation
(`test_lens_models.py`).

## Repository layout

```
src/lensing/          the package
  recombine.py          Eq. (5): the two-image SNR from z_1 and C(t_d)
  conventions.py        the Fourier, Morse-phase and time-shift sign conventions
  amplification.py      point-lens geometry and the exact wave-optics F(w)
  waveforms.py          templates, PSD, SNR time series (PyCBC)
  grids.py, priors.py   the lens grid (t_d, a) and its prior weights
  background.py         detection statistics and their noise background
  kernels.py            fast evaluation of Eq. (5) over the grid
  reweighted.py         the power chi^2 re-weighted SNR
  lens_models.py        singular / cored isothermal spheres and Chang-Refsdal
  glow_backend.py,      interface to GLoW
  wave_contour.py
  strong_pair.py        the separated-image lens population of Sec. V
  simulation.py         noise and injections
  provenance.py         seed, configuration and version records
scripts/simulate/     one script per data product (table above)
scripts/plot/         one script per figure; prl_style.py sets the style
scripts/checks/       convention checks, and compare_data.py
data/                 the data products the figures are drawn from
figures/              the paper's figures, and where make figures writes
tests/
```

A note on names. Code comments sometimes refer to the development stages in
which a piece was written: "Phase 0" is the convention checks
(`scripts/checks/phase0_conventions.py`), "Phase 3" the point-mass fitting
factors of Fig. 1(a, c), "Phase 4" the calibration of the detection statistics
on Gaussian noise (Appendix B) and "Phase 5" its robustness to the prior.
"G26" is Gholap et al. (2026), whose lensed template bank the paper compares
against. In the code, "two-image" statistics or searches are those evaluated
by recombination.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`; the arXiv
number will be added on submission) and, if you use the non-point-mass lens
models, GLoW.

## License

MIT; see `LICENSE`.
