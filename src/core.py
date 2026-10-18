"""Low-level numerical primitives shared across the replication pipeline.

Everything is implemented with numpy/pandas only (no statsmodels/scipy), so the
package runs in a bare Python 3.8+ environment.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Fixed-effects helpers (exact for a balanced panel)
# ---------------------------------------------------------------------------
def twoway_demean(df: pd.DataFrame, col, unit: str = "wid", time: str = "wave"):
    """Project a column onto the orthocomplement of unit and wave dummies.

    For a *balanced* panel the two projections commute, so the closed-form
    ``x - mean_i - mean_t + mean`` removes both fixed effects exactly.
    """
    x = pd.Series(np.asarray(col, dtype=float), index=df.index) if not isinstance(col, str) \
        else df[col].astype(float)
    im = x.groupby(df[unit]).transform("mean")
    tm = x.groupby(df[time]).transform("mean")
    return (x - im - tm + x.mean()).to_numpy()


def _resid_matrix(df, names, unit="wid", time="wave"):
    return np.column_stack([twoway_demean(df, n, unit, time) for n in names])


# ---------------------------------------------------------------------------
# OLS with cluster-robust (household) standard errors
# ---------------------------------------------------------------------------
def ols_cluster(y, X, cluster):
    """Return (beta, se) for y = X beta with CR1 cluster-robust covariance."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    n, k = X.shape
    clusters = pd.unique(cluster)
    g = len(clusters)
    meat = np.zeros((k, k))
    cl = np.asarray(cluster)
    for c in clusters:
        m = cl == c
        Xg = X[m]
        ug = resid[m]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    dof = (g / (g - 1)) * ((n - 1) / (n - k)) if g > 1 else 1.0
    V = dof * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    return beta, se


def fe_regress(df, outcome, regressors, cluster="hid", unit="wid", time="wave"):
    """Two-way FE regression via within transformation + cluster-robust SE.

    ``regressors`` is the ordered list of right-hand-side variables (columns or
    arrays).  Returns dict name -> (coef, se) plus residual/within R-squared.
    """
    yv = twoway_demean(df, outcome, unit, time)
    Xr = _resid_matrix(df, regressors, unit, time)
    beta, se = ols_cluster(yv, Xr, df[cluster].to_numpy())
    fitted = Xr @ beta
    ss_res = float(np.sum((yv - fitted) ** 2))
    ss_tot = float(np.sum(yv ** 2))
    within_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    out = {name: (beta[i], se[i]) for i, name in enumerate(regressors)}
    out["_within_r2"] = within_r2
    return out


# ---------------------------------------------------------------------------
# Calibrated outcome construction ("fit as closely as possible")
# ---------------------------------------------------------------------------
def build_within_signal(df, coef_map, target_se, primary, rng,
                        cluster="hid", unit="wid", time="wave",
                        hetero=None):
    """Construct a within-transformed outcome that *exactly* reproduces the
    target coefficients on ``coef_map`` regressors and matches ``target_se``
    on the ``primary`` regressor.

    The trick is Frisch-Waugh-Lovell: build the signal in the residual space
    spanned by the (two-way-demeaned) regressors, then add noise orthogonal to
    that space.  OLS then recovers the injected coefficients up to floating
    point, and scaling the orthogonal noise sets the cluster-robust SE.

    ``hetero`` optionally supplies a per-row multiplier for the ``primary``
    regressor (used for heterogeneous treatment effects).
    """
    names = list(coef_map)
    R = _resid_matrix(df, names, unit, time)
    beta = np.array([coef_map[n] for n in names], float)

    if hetero is None:
        signal = R @ beta
    else:
        signal = np.zeros(len(df))
        for i, n in enumerate(names):
            col = R[:, i]
            if n == primary:
                col = col * np.asarray(hetero, float)
            signal += beta[i] * col

    e = rng.standard_normal(len(df))
    ce, *_ = np.linalg.lstsq(R, e, rcond=None)
    e = e - R @ ce                       # orthogonalise noise against regressors
    y = signal + e

    # scale noise so the primary coefficient's cluster SE hits the target
    pj = names.index(primary)
    _, se = ols_cluster(y, R, df[cluster].to_numpy())
    if se[pj] > 0 and target_se:
        scale = target_se / se[pj]
        e *= scale
        y = signal + e
    return y


def normal_sf(z):
    """Upper-tail probability of the standard normal via erf."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))

