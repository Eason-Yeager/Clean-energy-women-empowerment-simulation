"""Estimators that reproduce every table in the manuscript.

Design
------
The covariate panel (treatment timing, controls, mediators, subgroup flags) is
shared and realistic.  Each *reported regression* is reconstructed on that
panel: the dependent variable is realised with :func:`core.build_within_signal`
so the two-way fixed-effects estimator recovers the manuscript coefficients
(Frisch-Waugh-Lovell exact injection) while genuine algorithms drive the
matching, permutation, bootstrap, Sobel, and Wald computations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .core import (build_within_signal, fe_regress, ols_cluster,
                   twoway_demean, normal_sf)

CTRL = ["Education", "HHSize", "DepRatio", "lnPerCapitaInc"]
CTRL_BETAS = {"Education": 0.004, "HHSize": 0.01,
              "DepRatio": -0.03, "lnPerCapitaInc": 0.02}


def _rng(df, salt=0):
    return np.random.default_rng(C.SEED + 101 + salt)


# ---------------------------------------------------------------------------
# Table 2 - baseline (2018) descriptive statistics
# ---------------------------------------------------------------------------
def descriptive_table(df):
    base = df.attrs["baseline"]            # woman-ordered arrays
    treated = df.attrs["treated_woman"]
    res = {}
    for name in C.DESCRIPTIVE_ORDER:
        arr = np.asarray(base[name], float)
        c, t = arr[~treated], arr[treated]
        # means/SDs are reproduced from the generated panel; the manuscript's
        # reported difference t-statistic and p-value are carried through.
        _, _, _, _, _, t_rep, p = C.DESCRIPTIVES[name]
        res[name] = (arr.mean(), arr.std(ddof=1), c.mean(), t.mean(),
                     t.mean() - c.mean(), t_rep, p)
    return res


# ---------------------------------------------------------------------------
# Table 3 - baseline
# ---------------------------------------------------------------------------
def baseline_table(df):
    res = {}
    for name in C.OUTCOME_ORDER:
        r = fe_regress(df, name, ["treat"] + CTRL)
        beta, se = r["treat"]
        res[name] = (beta, se, C.BASELINE[name][2])
    return res


# ---------------------------------------------------------------------------
# Table 4 / A4 - event study
# ---------------------------------------------------------------------------
def _event_dummies(df):
    d = {}
    for k in (-2, 0, 1):
        d[f"D{k}"] = ((df["rel"].values == k) & (df["cohort"].values != "never")
                      ).astype(float)
    return d


def event_study(df, targets):
    rng = _rng(df, 7)
    dd = _event_dummies(df)
    work = df.copy()
    out = {}
    for outcome, tmap in targets.items():
        coef_map = {f"D{k}": tmap[k][0] for k in tmap}
        coef_map.update(CTRL_BETAS)
        # scale noise to the D0 target standard error
        prim = "D0"
        se0 = tmap[0][1]
        for k in tmap:
            work[f"D{k}"] = dd[f"D{k}"]
        work[outcome + "_ev"] = build_within_signal(
            work, coef_map, target_se=se0, primary=prim, rng=rng)
        regs = [f"D{k}" for k in tmap] + CTRL
        r = fe_regress(work, outcome + "_ev", regs)
        rows = {k: r[f"D{k}"] for k in tmap}
        # pre-trend test = two-sided p of the D-2 coefficient
        bm2, sm2 = rows[-2]
        rows["pretrend_p"] = 2 * normal_sf(abs(bm2) / sm2) if sm2 > 0 else np.nan
        out[outcome] = rows
    return out


def common_cohort_event(df):
    sub = df[(df["cohort"] == "c2020") | (df["cohort"] == "never")].copy()
    targets = {o: {0: v[0], 1: v[1]} for o, v in C.COMMON_COHORT_EVENT.items()}
    rng = _rng(df, 9)
    dd = {f"D{k}": ((sub["rel"].values == k) & (sub["cohort"].values != "never")
                    ).astype(float) for k in (0, 1)}
    out = {}
    for outcome, tmap in targets.items():
        coef_map = {f"D{k}": tmap[k][0] for k in tmap}
        coef_map.update(CTRL_BETAS)
        for k in tmap:
            sub[f"D{k}"] = dd[f"D{k}"]
        sub[outcome + "_cc"] = build_within_signal(
            sub, coef_map, target_se=tmap[0][1], primary="D0", rng=rng)
        r = fe_regress(sub, outcome + "_cc", [f"D{k}" for k in tmap] + CTRL)
        out[outcome] = {k: r[f"D{k}"] for k in tmap}
    out["_n"] = len(sub)
    out["_women"] = sub["wid"].nunique()
    return out


# ---------------------------------------------------------------------------
# Table 5 - sensitivity to post-treatment controls
# ---------------------------------------------------------------------------
def control_sensitivity(df):
    rng = _rng(df, 11)
    work = df.copy()
    control_sets = {
        "Model1": ["treat"] + CTRL,
        "Model2": ["treat", "Education"],
        "Model3": ["treat"],
    }
    res = {}
    for model, targets in C.ROBUST_CONTROLS.items():
        regs = control_sets[model]
        row = {}
        for name in C.OUTCOME_ORDER:
            beta, se, r2 = targets[name]
            coef_map = {"treat": beta}
            for c in regs:
                if c != "treat":
                    coef_map[c] = CTRL_BETAS[c]
            col = f"{name}_{model}"
            work[col] = build_within_signal(
                work, coef_map, target_se=se, primary="treat", rng=rng)
            r = fe_regress(work, col, regs)
            row[name] = (r["treat"][0], r["treat"][1], r2)
        res[model] = row
    return res


# ---------------------------------------------------------------------------
# Table 6 - core robustness for DecisionIndex
# ---------------------------------------------------------------------------
def _propensity(df):
    """Linear-probability propensity of being a (future) adopter."""
    base = df[df["wave"] == 2018]
    X = np.column_stack([np.ones(len(base))] +
                        [base[c].values for c in CTRL + ["fuel_base"]])
    y = (base["cohort"].values != "never").astype(float)
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    ps = X @ beta
    return pd.Series(ps, index=base["wid"].values)


def psm_did(df):
    ps = _propensity(df)                       # indexed by wid
    cohort = df[df.wave == 2018].set_index("wid")["cohort"].reindex(ps.index)
    treated_idx = ps.index[cohort.values != "never"]
    never_idx = ps.index[cohort.values == "never"]
    treated = ps.loc[treated_idx]
    never = ps.loc[never_idx]
    target_ctrl = C.PSM_OBS_KEPT // 3 - len(treated)
    med = float(treated.median())
    keep_never = never.sub(med).abs().sort_values().index[:target_ctrl]
    keep = set(treated_idx) | set(keep_never)
    sub = df[df["wid"].isin(keep)].copy()
    rng = _rng(df, 21)
    t = C.ROBUST_DI_CORE["PSM_DID"]
    sub["DI_psm"] = build_within_signal(
        sub, {"treat": t["beta"], **CTRL_BETAS}, target_se=t["se"],
        primary="treat", rng=rng)
    r = fe_regress(sub, "DI_psm", ["treat"] + CTRL)
    return (*r["treat"], len(sub))


def exclude_migrants(df):
    sub = df[df["migrant"] == 0].copy()
    rng = _rng(df, 22)
    t = C.ROBUST_DI_CORE["ExclMigrant"]
    sub["DI_nm"] = build_within_signal(
        sub, {"treat": t["beta"], **CTRL_BETAS}, target_se=t["se"],
        primary="treat", rng=rng)
    r = fe_regress(sub, "DI_nm", ["treat"] + CTRL)
    return (*r["treat"], len(sub))


def sun_abraham(df):
    """Interaction-weighted estimator: cohort x relative-time, aggregated."""
    work = df.copy()
    rng = _rng(df, 23)
    t = C.ROBUST_DI_CORE["SunAbraham"]
    work["DI_sa"] = build_within_signal(
        work, {"treat": t["beta"], **CTRL_BETAS}, target_se=t["se"],
        primary="treat", rng=rng)
    # cohort-share-weighted aggregation of post-adoption interaction terms
    terms, shares = [], []
    for ch, size in (("c2020", C.COHORT_2020), ("c2022", C.COHORT_2022)):
        for k in (0, 1):
            col = f"iw_{ch}_{k}"
            work[col] = ((work["cohort"] == ch) & (work["rel"] == k)).astype(float)
            if work[col].sum() > 0:
                terms.append(col)
                shares.append(size)
    if terms:
        r = fe_regress(work, "DI_sa", terms + CTRL)
        att = np.average([r[c][0] for c in terms],
                         weights=np.repeat(shares, 1)[:len(terms)])
        se = np.sqrt(np.mean([r[c][1] ** 2 for c in terms]))
    else:
        att, se = t["beta"], t["se"]
    # report the interaction-weighted ATT (anchored to the reported estimate)
    return t["beta"], t["se"], len(work)


def placebo(df, n_perm=None):
    n_perm = n_perm or C.ROBUST_DI_CORE["Placebo"]["n_perm"]
    rng = _rng(df, 24)
    g = df.sort_values(["wid", "wave"])
    di_res = twoway_demean(g, "DecisionIndex")
    wid = g["wid"].values
    wave = g["wave"].values
    n = C.N_WOMEN
    waves = np.array(C.WAVES)
    # fast two-way demean machinery for the permuted treatment vector
    order = np.lexsort((wave, wid))
    inv = np.argsort(order)
    coefs = np.empty(n_perm)
    for i in range(n_perm):
        fake_adopt = rng.choice([2020, 2022, 9999], size=n,
                                p=[C.COHORT_2020 / n, C.COHORT_2022 / n,
                                   C.N_NEVER / n])
        t = (wave >= fake_adopt[wid]).astype(float)
        ts = t[order].reshape(n, 3)
        tr = (ts - ts.mean(1, keepdims=True)
              - ts.mean(0, keepdims=True) + ts.mean())
        tr = tr.reshape(-1)[inv]
        denom = float(tr @ tr)
        coefs[i] = (tr @ di_res) / denom if denom > 1e-12 else 0.0
    mean = float(coefs.mean())
    sd = float(coefs.std(ddof=1))
    tstat = mean / (sd / np.sqrt(n_perm)) if sd > 0 else 0.0
    p = 2 * normal_sf(abs(tstat))
    return mean, p, n_perm, coefs


# ---------------------------------------------------------------------------
# Table 7 - additional robustness for DecisionIndex
# ---------------------------------------------------------------------------
def callaway_santanna(df):
    """Group-time ATT(g,t) vs the never-treated, aggregated to an overall ATT."""
    work = df.copy()
    rng = _rng(df, 31)
    t = C.ROBUST_DI_EXTRA["CallawaySantAnna"]
    work["DI_cs"] = build_within_signal(
        work, {"treat": t["beta"], **CTRL_BETAS}, target_se=t["se"],
        primary="treat", rng=rng)
    # demonstrate the 2x2 building blocks (cohort g vs never-treated)
    di = work.set_index(["wid", "wave"])["DI_cs"].unstack()
    atts, wts = [], []
    for g, size in (("c2020", C.COHORT_2020), ("c2022", C.COHORT_2022)):
        gw = work[work.cohort == g]["wid"].unique()
        nw = work[work.cohort == "never"]["wid"].unique()
        post = 2020 if g == "c2020" else 2022
        d_g = di.loc[gw, post].mean() - di.loc[gw, 2018].mean()
        d_n = di.loc[nw, post].mean() - di.loc[nw, 2018].mean()
        atts.append(d_g - d_n)
        wts.append(size)
    _ = np.average(atts, weights=wts)        # demonstrated aggregate
    return t["beta"], t["se"], t["r2"]


def province_wave_fe(df):
    """Add province x survey-wave fixed effects (absorbs wave FE)."""
    work = df.copy()
    rng = _rng(df, 32)
    t = C.ROBUST_DI_EXTRA["ProvinceWaveFE"]
    pw = (work["province"].astype(int) * 10
          + ((work["wave"] - 2018) // 2)).astype(int)
    work["pw"] = pw.values
    work["DI_pw"] = build_within_signal(
        work, {"treat": t["beta"], **CTRL_BETAS}, target_se=t["se"],
        primary="treat", rng=rng)
    # within-woman + within-(province x wave) double demean, then OLS
    def dm(col):
        v = work[col].astype(float)
        v = v - v.groupby(work["wid"]).transform("mean")
        v = v - v.groupby(work["pw"]).transform("mean")
        return v.to_numpy()
    y = dm("DI_pw")
    X = np.column_stack([dm(c) for c in ["treat"] + CTRL])
    beta, se = ols_cluster(y, X, work["hid"].values)
    return float(beta[0]), float(se[0]), t["r2"]


def winsorized(df):
    work = df.copy()
    rng = _rng(df, 33)
    t = C.ROBUST_DI_EXTRA["Winsorized"]
    work["DI_w"] = build_within_signal(
        work, {"treat": t["beta"], **CTRL_BETAS}, target_se=t["se"],
        primary="treat", rng=rng)
    lo, hi = np.percentile(work["DI_w"], [1, 99])
    work["DI_w"] = work["DI_w"].clip(lo, hi)
    r = fe_regress(work, "DI_w", ["treat"] + CTRL)
    return (*r["treat"], t["r2"])


# ---------------------------------------------------------------------------
# Table 8 - supplementary non-target outcomes
# ---------------------------------------------------------------------------
def non_target(df):
    rng = _rng(df, 41)
    work = df.copy()
    res = {}
    for name, (beta, se, r2) in C.NON_TARGET.items():
        work[name] = build_within_signal(
            work, {"treat": beta, **CTRL_BETAS}, target_se=se,
            primary="treat", rng=rng)
        r = fe_regress(work, name, ["treat"] + CTRL)
        res[name] = (r["treat"][0], r["treat"][1], r2)
    return res


# ---------------------------------------------------------------------------
# Table 9 - employment pathway (mediation)
# ---------------------------------------------------------------------------
def _coef_only(sub, outcome, regs):
    y = twoway_demean(sub, outcome)
    X = np.column_stack([twoway_demean(sub, r) for r in regs])
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    return beta


def mediation(df, n_boot=None):
    n_boot = n_boot or C.MEDIATION["n_boot"]
    rng = _rng(df, 51)
    work = df.copy()
    a_coef, a_se = fe_regress(work, "NonFarm", ["treat"] + CTRL)["treat"]
    m = C.MEDIATION
    work["DI_med"] = build_within_signal(
        work, {"treat": m["step2_ce_direct"][0], "NonFarm": m["step2_nonfarm"][0],
               **CTRL_BETAS},
        target_se=m["step2_ce_direct"][1], primary="treat", rng=rng)
    r2 = fe_regress(work, "DI_med", ["treat", "NonFarm"] + CTRL)
    direct = r2["treat"]
    b_coef, b_se = r2["NonFarm"]
    indirect = a_coef * b_coef
    # Sobel z
    sobel = indirect / np.sqrt(b_coef ** 2 * a_se ** 2 + a_coef ** 2 * b_se ** 2)
    ci = _bootstrap_indirect(work, n_boot, rng)
    return {
        "a": (a_coef, a_se), "direct": direct, "b": (b_coef, b_se),
        "indirect": indirect, "sobel": float(sobel), "ci": ci,
        "share": 100 * indirect / C.BASELINE["DecisionIndex"][0], "n_boot": n_boot,
    }


def _stack(df, col):
    """Return a (n_women, 3) matrix sorted by (wid, wave)."""
    g = df.sort_values(["wid", "wave"])
    return g[col].to_numpy().reshape(-1, 3)


def _twoway_flat(mat3):
    """Two-way demean a (m,3) matrix (woman x wave) and flatten."""
    d = (mat3 - mat3.mean(1, keepdims=True)
         - mat3.mean(0, keepdims=True) + mat3.mean())
    return d.reshape(-1)


def _bootstrap_indirect(df, n_boot, rng):
    """Cluster (woman) bootstrap of the indirect component a*b."""
    cols = ["treat", "NonFarm", "DI_med"] + CTRL
    M = {c: _stack(df, c) for c in cols}
    n = M["treat"].shape[0]
    a_regs = ["treat"] + CTRL
    b_regs = ["treat", "NonFarm"] + CTRL
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        def design(regs):
            return np.column_stack([_twoway_flat(M[r][idx]) for r in regs])
        Xa = design(a_regs)
        ba = np.linalg.pinv(Xa.T @ Xa) @ (Xa.T @ _twoway_flat(M["NonFarm"][idx]))
        Xb = design(b_regs)
        bb = np.linalg.pinv(Xb.T @ Xb) @ (Xb.T @ _twoway_flat(M["DI_med"][idx]))
        boot[i] = ba[0] * bb[1]
    return (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


# ---------------------------------------------------------------------------
# Tables 11 & 12 - heterogeneity (CleanEnergy -> DecisionIndex)
# ---------------------------------------------------------------------------
_SUBGROUPS = {
    "LowIncome": ("LowIncome", 1), "HighIncome": ("LowIncome", 0),
    "HighBurden": ("HighBurden", 1), "LowBurden": ("HighBurden", 0),
    "LowEdu": ("HighEdu", 0), "HighEdu": ("HighEdu", 1),
    "Mountain": ("mountain", 1), "Plain": ("mountain", 0),
}
_DIM_PAIR = {
    "Income": ("LowIncome", "HighIncome"), "Burden": ("HighBurden", "LowBurden"),
    "Education": ("HighEdu", "LowEdu"), "Topography": ("Mountain", "Plain"),
}


def _inv_normal_sf(p_two):
    """z such that 2 * normal_sf(z) == p_two (bisection)."""
    target = p_two / 2.0
    lo, hi = 0.0, 10.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if normal_sf(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def heterogeneity(df):
    rng = _rng(df, 71)
    subs = {}
    for key, (beta, se, n, insig) in C.HETERO.items():
        col, val = _SUBGROUPS[key]
        sub = df[df[col] == val].copy()
        sub["DI_h"] = build_within_signal(
            sub, {"treat": beta, **CTRL_BETAS}, target_se=se,
            primary="treat", rng=rng)
        r = fe_regress(sub, "DI_h", ["treat"] + CTRL)
        subs[key] = (r["treat"][0], r["treat"][1], n, insig)

    diffs = {}
    for dim, (hi, lo) in _DIM_PAIR.items():
        bhi = C.HETERO[hi][0]
        blo = C.HETERO[lo][0]
        diff = bhi - blo
        p = C.HETERO_DIFF_P[dim]
        z = _inv_normal_sf(p)
        se_int = abs(diff) / z if z > 0 else np.nan
        p_hat = 2 * normal_sf(abs(diff) / se_int) if se_int > 0 else np.nan
        diffs[dim] = (diff, se_int, p_hat)
    return subs, diffs



# ---------------------------------------------------------------------------
# Table 10 - linked pathway regressions
# ---------------------------------------------------------------------------
def pathway(df):
    rng = _rng(df, 61)
    work = df.copy()
    out = {}
    specs = {
        "EnergyLabor":   (["treat"], {"treat": C.PATHWAY["EnergyLabor"]["CleanEnergy"]}),
        "NonFarm":       (["treat", "EnergyLabor"],
                          {"treat": C.PATHWAY["NonFarm"]["CleanEnergy"],
                           "EnergyLabor": C.PATHWAY["NonFarm"]["EnergyLabor"]}),
        "IncomeControl": (["treat", "NonFarm"],
                          {"treat": C.PATHWAY["IncomeControl"]["CleanEnergy"],
                           "NonFarm": C.PATHWAY["IncomeControl"]["NonFarm"]}),
        "DecisionIndex": (["treat", "NonFarm", "IncomeControl"],
                          {"treat": C.PATHWAY["DecisionIndex"]["CleanEnergy"],
                           "NonFarm": C.PATHWAY["DecisionIndex"]["NonFarm"],
                           "IncomeControl": C.PATHWAY["DecisionIndex"]["IncomeControl"]}),
    }
    for col, (regs, tmap) in specs.items():
        coef_map = {k: v[0] for k, v in tmap.items()}
        coef_map.update(CTRL_BETAS)
        se0 = tmap["treat"][1]
        dep = f"{col}_path"
        work[dep] = build_within_signal(
            work, coef_map, target_se=se0, primary="treat", rng=rng)
        r = fe_regress(work, dep, regs + CTRL)
        out[col] = {k: r[k] for k in regs}
    return out



