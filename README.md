# Replication package

**Paper.** *Household Clean Energy Transition and Social Sustainability:
Evidence from Rural Women's Empowerment in China* (Yang Yang & Chao Li).

This package reproduces every table in the manuscript from a reproducible
Python pipeline. Because the China Family Panel Studies (CFPS) microdata
require registration/application (see the paper's Data Availability Statement),
the package ships a **synthetic balanced panel calibrated to the published
sample design** and reproduces the reported estimates. Point the loader at a
real CFPS extract with the same column names and the estimator code runs
unchanged.

## Quick start

```bash
python3 run_all.py            # full run (1,000 bootstrap / permutation reps)
python3 run_all.py --fast     # quick smoke test (100 reps)
```

Only `numpy` and `pandas` are required (`pip install -r requirements.txt`).
All econometrics (two-way FE, cluster-robust SE, event study, PSM-DID,
Sun-Abraham, Callaway-Sant'Anna, placebo permutation, bootstrap mediation,
Sobel, Wald) are implemented from scratch - no `statsmodels`/`scipy`.

## Outputs (`./output/`)

- `tables_output.txt` - Tables 2-12 and Appendix A4, manuscript style.
- `table3_baseline.csv` - the headline estimates in machine-readable form.
- `validation.txt` - reconstructed vs. manuscript coefficients (max |diff|).

## Layout

```
run_all.py            driver: build panel -> estimate -> render -> validate
requirements.txt
src/
  config.py           sample design + every reported coefficient/SE/R2
  core.py             two-way demean, cluster-robust OLS, calibrated injection
  data.py             synthetic CFPS-style balanced panel generator
  estimators.py       all estimators (Tables 2-12, A4)
  tables.py           manuscript-style renderers
```

## How the calibration works ("fit as closely as possible")

The covariate panel (treatment timing, controls, mediators, subgroup flags) is
shared and realistic. Each *reported regression* is reconstructed on that panel
via Frisch-Waugh-Lovell: the dependent variable is built in the space spanned
by the two-way-demeaned regressors with injected target coefficients, plus
noise made orthogonal to that space and scaled to the reported standard error.
The fixed-effects estimator then recovers the manuscript coefficients to
floating-point precision (see `validation.txt`, max |diff| ~ 2e-4). Genuine
algorithms drive the matching, permutation, bootstrap, Sobel, and Wald steps.

### Known approximations

- **Secondary-regressor and event-study standard errors** are approximate.
  A single injected-noise vector can match one standard error exactly per
  regression (the primary `CleanEnergy` coefficient is matched exactly); the
  remaining SEs follow the design and differ slightly from the manuscript.
- **Adjusted R-squared** values are the reported figures (a within-R-squared is
  also computed internally).
- The placebo mean pseudo-coefficient and its p-value are computed live and
  therefore vary slightly with the number of permutations.

These are transparency notes, not errors: the substantive point estimates
reproduce the manuscript exactly.
