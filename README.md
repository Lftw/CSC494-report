# Mixed-Polarization Imaging for the Event Horizon Telescope

**An interactive report for CSC494. Alex Worms, Summer 2026**
Supervised by Aviad Levis, with Rohan Dahale and Andrew Chael.

Radio telescopes measure polarized light in two different "languages": older
dishes use **circular** feeds (right/left), newer ones like ALMA use **linear**
feeds (X/Y). Existing EHT imaging software assumes every station in the array
speaks the same language. This report explains why that assumption breaks when
you mix them, works through the math needed to fix it, and shows the
implementation: a mixed-polarization data model, a Jones-matrix forward model,
and uvfits I/O for [eht-imaging](https://github.com/achael/eht-imaging).

Instead of a written report, this is a sequence of notebooks you can *touch*.
Every plot has sliders. Every equation has a picture next to it.

---

## The notebooks

To be read in order.

| # | Notebook | The question it answers |
|---|----------|------------------------|
| 01 | [Polarization](notebooks/01_polarization.ipynb) | Which way is the light wiggling, and why do we care? |
| 02 | [The Event Horizon Telescope](notebooks/02_the_eht.ipynb) | How do you build a telescope the size of the Earth, and what does it actually record? |
| 03 | [Jones matrices and mixed polarization](notebooks/03_jones_and_mixed_polarization.ipynb) | What happens between the sky and the recorded number, and what breaks when two telescopes disagree about what they're recording? |
| 04 | Results *(in progress)* | A polarized image from an array that speaks two languages. |

---

## Running it

```bash
git clone <this repo> && cd CSC494-report
micromamba create -y -n csc494 python=3.11 pip     # or conda / venv
micromamba run -n csc494 pip install -r requirements.txt
micromamba run -n csc494 python -m ipykernel install --user \
    --name csc494 --display-name "Python (csc494)"
micromamba run -n csc494 jupyter lab               # or open in VS Code
```

There is **no install step for this code**. Each notebook's first cell adds
`scripts/` to `sys.path`, so `import polarization` just works.

Notebooks 01–03 need nothing but the packages in
[requirements.txt](requirements.txt): numpy, scipy, matplotlib, plotly,
ipywidgets, astropy. Notebook 03's last cell additionally compares its algebra
against [eht-imaging](https://github.com/achael/eht-imaging) where that is
installed, and says so and skips when it is not; notebook 04 will use it
throughout, with its outputs committed so the notebook still reads without it.

The sliders need a live kernel (Jupyter Lab, VS Code, Binder). GitHub's static
preview shows the text and figures but nothing will move; every interactive also
takes `static=True` to return a plain figure.

In VS Code, pick the **Python (csc494)** kernel and open *this* folder as the
workspace: `.vscode/settings.json` then tells Pylance where the modules live and
runs notebooks with their own folder as the working directory. The setup cell
prints the path it imported from; if that is not the folder being edited, the
kernel is holding an older copy of the code and needs restarting.

---

## Layout

```
CSC494-report/
├── notebooks/                     the notebooks themselves
└── scripts/                       every function they call, one file per job
    ├── polarization.py            waves, Stokes parameters, polarization ellipses
    ├── images.py                  synthetic polarized sky images, EVPA ticks
    ├── arrays.py                  EHT stations, sky geometry, uv coverage
    ├── interferometry.py          fringes, visibilities, dirty beams and images
    ├── style.py                   one shared plotly look
    ├── jones.py                   Jones matrices, the RIME, mixed feed bases
    ├── ehtim_reference.py         eht-imaging's conventions, captured as data
    ├── figures_polarization.py    the figures for notebook 01
    ├── figures_telescope.py       the figures for notebook 02
    ├── figures_jones.py           the figures for notebook 03
    ├── widgets.py                 slider plumbing, shared by both
    ├── interactive_polarization.py  notebook 01's interactive figures
    ├── interactive_telescope.py     notebook 02's interactive figures
    ├── interactive_jones.py         notebook 03's interactive figures
    ├── data/                      the EHT station table, the ehtim capture
    └── tests/                     pytest
```

Flat on purpose: a notebook needs three imports and no package paths.

```python
import polarization as pol
import images
import interactive_polarization as explore
```

The split is by job, not by topic. `polarization.py`, `images.py`, `arrays.py`,
`interferometry.py` and `jones.py` are physics: pure functions, numpy in and numpy
out, no plotting and no widgets, so they can be tested headless and read on their
own.
The `figures_*` modules turn arrays into plotly figures and contain no physics.
The `interactive_*` modules only wire sliders to the first group and hand the
result to the second.

Every notebook cell is prose, a short call, or a figure. If a cell gets long, the
code belongs in a module.

```bash
micromamba run -n csc494 python -m pytest -q                    # 268 tests
```

The tests check the physics against closed-form answers rather than eyeballed
ones: hand-derived polarization states, sidereal time against astropy, the
discrete Fourier transform against the analytic Gaussian, closure phase
invariance under random station gains, and the Jones/mixed-basis algebra against
`eht-imaging`'s own `pol_conventions` module for all four feed pairings (those
ten checks skip where eht-imaging is not installed).

---

## Conventions

Sign conventions in polarimetry are a minefield; this report follows
eht-imaging exactly, which is the IAU / Hamaker-Bregman-Sault convention with
the engineering time dependence $e^{+i\omega t}$:

$$R = \frac{X + iY}{\sqrt 2}, \qquad L = \frac{X - iY}{\sqrt 2}$$

$$I = \tfrac{1}{2}(RR + LL), \quad Q = \tfrac{1}{2}(RL + LR), \quad
  U = \tfrac{i}{2}(LR - RL), \quad V = \tfrac{1}{2}(RR - LL)$$

The full derivation, the linear-feed equivalents, and the Jones factorization
$J = G\,(I + D)\,\Phi$ are documented in
`eht-imaging/docs/polarization_conventions.md`, which this report was written
against. The sky-geometry functions (`hour_angle`, `elevation`,
`parallactic_angle`, the $uv$ projection, the thermal-noise formula) are
transcriptions of the same ones in `ehtim/observing/obs_helpers.py`, so the
self-contained notebooks cannot silently drift from the code they describe.

Station coordinates, SEFDs and mount parameters in
`scripts/data/eht_sites.csv` are derived from `eht-imaging/arrays/`
(GPL-3); the feed-basis column, ALMA and NOEMA linear and the rest circular, is a
modelling choice of this report.

Notebook 03 ends by comparing its own algebra with eht-imaging's, term by term.
The notebooks run without eht-imaging, so the conventions are captured once, with
`micromamba run -n jax-ehtim python scripts/export_ehtim_reference.py`, into
`scripts/data/ehtim_pol_conventions.json`, and the notebook compares against
that. `scripts/tests/test_ehtim_reference.py` re-captures from the live module
wherever eht-imaging is installed, so the committed file cannot quietly go
stale.
