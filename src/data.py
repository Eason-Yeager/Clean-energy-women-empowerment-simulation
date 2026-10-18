"""Synthetic CFPS-style balanced panel calibrated to the manuscript.

Real CFPS microdata require registration/application (see the paper's Data
Availability Statement), so this module generates a balanced panel that matches
the published sample design (cohort structure, Table 2 descriptives) and the
cross-variable relationships used by the estimators.  Swapping in a real CFPS
extract with the same column names lets the estimator code run unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .core import build_within_signal


# ---------------------------------------------------------------------------
def _group_draw(rng, n_ctrl, n_treat, m_ctrl, m_treat, sd):
    """Woman-level baseline draws whose group means equal the Table 2 targets."""
    ctrl = rng.normal(m_ctrl, sd, n_ctrl)
    trt = rng.normal(m_treat, sd, n_treat)
    # re-centre each group exactly onto its target mean (kills sampling noise)
    ctrl += m_ctrl - ctrl.mean()
    trt += m_treat - trt.mean()
    return ctrl, trt


def _bernoulli_group(rng, n_ctrl, n_treat, p_ctrl, p_treat):
    """Binary baseline draws with the exact target proportion of ones."""
    def exact(n, p):
        v = np.zeros(n)
        k = int(round(p * n))
        v[rng.permutation(n)[:k]] = 1.0
        return v
    return exact(n_ctrl, p_ctrl), exact(n_treat, p_treat)


def build_panel(seed: int = C.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ----- woman-level design ------------------------------------------------
    n = C.N_WOMEN
    cohort = np.array(
        ["c2020"] * C.COHORT_2020 + ["c2022"] * C.COHORT_2022 + ["never"] * C.N_NEVER
    )
    rng.shuffle(cohort)
    adopt = np.where(cohort == "c2020", 2020,
                     np.where(cohort == "c2022", 2022, 9999))
    treated_woman = cohort != "never"

    wid = np.arange(n)
    province = wid % C.N_PROVINCES
    # topography is a province-level attribute (first provinces are mountainous)
    mount_prov = set(range(0, 11))  # ~ enough to reach ~1807 mountainous women

    # ----- woman-level baseline (2018) levels, calibrated to Table 2 ---------
    n_ctrl, n_treat = C.N_NEVER, C.N_TREATED
    base = {}
    for name in ["HouseWork", "EnergyLabor", "DecisionIndex",
                 "Age", "Education", "HHSize", "DepRatio", "lnPerCapitaInc"]:
        fm, fsd, mc, mt, *_ = C.DESCRIPTIVES[name]
        between = (n_ctrl * (mc - fm) ** 2 + n_treat * (mt - fm) ** 2) / n
        wsd = np.sqrt(max(fsd ** 2 - between, 1e-6))
        c, t = _group_draw(rng, n_ctrl, n_treat, mc, mt, wsd)
        base[name] = (c, t)
    for name in ["NonFarm", "IncomeControl"]:
        _, _, mc, mt, *_ = C.DESCRIPTIVES[name]
        c, t = _bernoulli_group(rng, n_ctrl, n_treat, mc, mt)
        base[name] = (c, t)

    # map group draws back to woman order
    def by_woman(name):
        c, t = base[name]
        out = np.empty(n)
        out[~treated_woman] = c
        out[treated_woman] = t
        return out

    baseline = {k: by_woman(k) for k in base}

    # fuel-collection hours at baseline (initial energy-related burden)
    fuel = np.clip(baseline["EnergyLabor"] * 0.5 + rng.normal(0, 0.4, n), 0, None)

    # ----- long (panel) format ----------------------------------------------
    rows = []
    for w in C.WAVES:
        rows.append(pd.DataFrame({
            "wid": wid, "hid": wid, "wave": w, "province": province,
            "cohort": cohort, "adopt": adopt,
        }))
    df = pd.concat(rows, ignore_index=True)
    df["treat"] = (df["wave"].values >= df["adopt"].values).astype(float)
    df["rel"] = np.where(df["cohort"].values == "never", np.nan,
                         (df["wave"].values - df["adopt"].values) / 2.0)
    df["mountain"] = df["province"].isin(mount_prov).astype(float)
    return _attach_variables(df, rng, baseline, fuel, treated_woman)


# ---------------------------------------------------------------------------
def _wave_idx(df):
    return ((df["wave"].values - 2018) // 2).astype(float)  # 0,1,2


def _attach_variables(df, rng, baseline, fuel, treated_woman):
    n = C.N_WOMEN
    widx = _wave_idx(df)
    w = df["wid"].values

    # time-varying controls = woman baseline + mild wave drift + noise
    def expand(name, drift):
        return baseline[name][w] + drift * widx + rng.normal(0, 0.05, len(df))

    df["Education"] = np.clip(expand("Education", 0.05), 0, 22)
    df["HHSize"] = np.clip(expand("HHSize", -0.06) + rng.normal(0, 0.1, len(df)), 1, None)
    df["DepRatio"] = np.clip(expand("DepRatio", 0.0), 0, 0.95)
    df["lnPerCapitaInc"] = expand("lnPerCapitaInc", 0.12)
    df["Age"] = baseline["Age"][w] + 2.0 * widx

    # ----- baseline-defined subgroup flags (time-invariant) ------------------
    def top_flag(values, k):
        """Flag the k women with the largest baseline value."""
        order = np.argsort(-values)
        flag = np.zeros(n)
        flag[order[:k]] = 1.0
        return flag

    low_income = top_flag(-baseline["lnPerCapitaInc"], C.HETERO["LowIncome"][2] // 3)
    high_burden = top_flag(fuel, C.HETERO["HighBurden"][2] // 3)
    high_edu = top_flag(baseline["Education"], C.HETERO["HighEdu"][2] // 3)

    df["LowIncome"] = low_income[w]
    df["HighBurden"] = high_burden[w]
    df["HighEdu"] = high_edu[w]
    df["fuel_base"] = fuel[w]

    # long-term male migrant households (dropped in Table 6, column 2)
    n_mig = C.MIGRANT_OBS_DROP // 3
    mig = np.zeros(n)
    mig[rng.choice(n, n_mig, replace=False)] = 1.0
    df["migrant"] = mig[w]

    # ----- canonical mediator realisations (used as regressors) --------------
    ctrl_cols = ["Education", "HHSize", "DepRatio", "lnPerCapitaInc"]
    ctrl_betas = {"Education": 0.004, "HHSize": 0.01,
                  "DepRatio": -0.03, "lnPerCapitaInc": 0.02}

    for name in C.OUTCOME_ORDER:
        beta, se, _ = C.BASELINE[name]
        coef_map = {"treat": beta, **ctrl_betas}
        df[name] = build_within_signal(
            df, coef_map, target_se=se, primary="treat", rng=rng)

    # stash the Table 2 baseline draws for the descriptive routine
    df.attrs["baseline"] = baseline
    df.attrs["treated_woman"] = treated_woman
    df.attrs["fuel"] = fuel
    return df

