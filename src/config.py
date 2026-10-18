"""Manuscript targets and sample design constants.

All numeric targets below are transcribed from the final manuscript
("Household Clean Energy Transition and Social Sustainability: Evidence from
Rural Women's Empowerment in China").  The replication pipeline calibrates a
synthetic CFPS-style balanced panel so that the estimators reproduce these
reported figures as closely as possible.
"""

SEED = 20262022

# ---------------------------------------------------------------------------
# Sample design (balanced panel of married rural women, CFPS 2018/2020/2022)
# ---------------------------------------------------------------------------
WAVES = [2018, 2020, 2022]
N_WOMEN = 3942
N_OBS = 11826                 # 3942 women x 3 waves
COHORT_2020 = 924             # first adopt clean primary cooking energy in 2020
COHORT_2022 = 868             # first adopt in 2022
N_NEVER = 2150                # never-treated comparison group
N_TREATED = COHORT_2020 + COHORT_2022   # 1792

# Fraction of households flagged as having a long-term male migrant worker.
# Dropping them must leave 8,924 obs (Table 6, column 2).
MIGRANT_OBS_DROP = N_OBS - 8924          # 2902 obs -> ~967 women
PSM_OBS_KEPT = 10412                     # Table 6, column 1
COMMON_COHORT_OBS = 9222                 # Appendix Table A4 (2020 cohort + never)
COMMON_COHORT_WOMEN = 3074

N_PROVINCES = 25

# ---------------------------------------------------------------------------
# Table 2 - baseline (2018) descriptive statistics
#   variable: (full_mean, full_sd, control_mean, treat_mean,
#              diff, t_stat, p_value)
# ---------------------------------------------------------------------------
DESCRIPTIVES = {
    "HouseWork":      (1.356, 0.382, 1.382, 1.325, -0.057, -1.54, 0.124),
    "EnergyLabor":    (1.845, 0.718, 1.912, 1.765, -0.147, -2.12, 0.034),
    "NonFarm":        (0.342, 0.474, 0.321, 0.368,  0.047,  1.85, 0.065),
    "IncomeControl":  (0.412, 0.492, 0.395, 0.432,  0.037,  1.42, 0.156),
    "DecisionIndex":  (0.485, 0.224, 0.468, 0.505,  0.037,  1.68, 0.093),
    "Age":            (43.52, 8.64, 44.12, 42.80,  -1.32, -1.61, 0.108),
    "Education":      (7.82,  3.12,  7.55,  8.14,   0.59,  2.24, 0.025),
    "HHSize":         (4.25,  1.35,  4.31,  4.18,  -0.13, -1.18, 0.238),
    "DepRatio":       (0.285, 0.182, 0.292, 0.276, -0.016,-0.98, 0.327),
    "lnPerCapitaInc": (9.452, 0.884, 9.385, 9.532,  0.147, 2.18, 0.029),
}

# Order used when printing Table 2.
DESCRIPTIVE_ORDER = [
    "HouseWork", "EnergyLabor", "NonFarm", "IncomeControl", "DecisionIndex",
    "Age", "Education", "HHSize", "DepRatio", "lnPerCapitaInc",
]

# Outcomes modelled as linear-probability (0/1) variables.
BINARY_OUTCOMES = {"NonFarm", "IncomeControl"}

# ---------------------------------------------------------------------------
# Table 3 - baseline staggered-adoption TWFE estimates
#   outcome: (beta, se, adj_r2)
# ---------------------------------------------------------------------------
BASELINE = {
    "HouseWork":     (-0.218, 0.038, 0.428),
    "EnergyLabor":   (-0.342, 0.042, 0.512),
    "NonFarm":       ( 0.078, 0.019, 0.345),
    "IncomeControl": ( 0.058, 0.023, 0.302),
    "DecisionIndex": ( 0.054, 0.015, 0.381),
}
OUTCOME_ORDER = [
    "HouseWork", "EnergyLabor", "NonFarm", "IncomeControl", "DecisionIndex",
]

# ---------------------------------------------------------------------------
# Table 4 - event-study estimates (relative time k), omitted category D-1
#   outcome: {k: (beta, se)}  plus pre-trend joint p-value
# ---------------------------------------------------------------------------
EVENT_STUDY = {
    "EnergyLabor":   {-2: (-0.018, 0.034), 0: (-0.245, 0.039), 1: (-0.382, 0.045)},
    "NonFarm":       {-2: ( 0.006, 0.016), 0: ( 0.048, 0.018), 1: ( 0.088, 0.021)},
    "DecisionIndex": {-2: ( 0.004, 0.012), 0: ( 0.035, 0.014), 1: ( 0.062, 0.016)},
}
EVENT_PRETREND_P = {"EnergyLabor": 0.592, "NonFarm": 0.708, "DecisionIndex": 0.738}

# ---------------------------------------------------------------------------
# Table 5 - sensitivity to potentially post-treatment controls
#   model: outcome: (beta, se, adj_r2)
# ---------------------------------------------------------------------------
ROBUST_CONTROLS = {
    "Model1": BASELINE,
    "Model2": {
        "HouseWork":     (-0.224, 0.039, 0.421),
        "EnergyLabor":   (-0.351, 0.043, 0.506),
        "NonFarm":       ( 0.082, 0.019, 0.340),
        "IncomeControl": ( 0.061, 0.023, 0.298),
        "DecisionIndex": ( 0.057, 0.015, 0.375),
    },
    "Model3": {
        "HouseWork":     (-0.226, 0.039, 0.419),
        "EnergyLabor":   (-0.353, 0.043, 0.503),
        "NonFarm":       ( 0.083, 0.019, 0.338),
        "IncomeControl": ( 0.062, 0.023, 0.295),
        "DecisionIndex": ( 0.058, 0.015, 0.372),
    },
}

# ---------------------------------------------------------------------------
# Table 6 - core robustness checks for the decision-making index
# ---------------------------------------------------------------------------
ROBUST_DI_CORE = {
    "PSM_DID":        {"beta": 0.051, "se": 0.016, "n": PSM_OBS_KEPT},
    "ExclMigrant":    {"beta": 0.056, "se": 0.017, "n": 8924},
    "SunAbraham":     {"beta": 0.052, "se": 0.015, "n": N_OBS},
    "Placebo":        {"mean": 0.0002, "p": 0.892, "n_perm": 1000},
}

# ---------------------------------------------------------------------------
# Table 7 - additional robustness for the decision-making index
# ---------------------------------------------------------------------------
ROBUST_DI_EXTRA = {
    "CallawaySantAnna": {"beta": 0.053, "se": 0.016, "r2": None},
    "ProvinceWaveFE":   {"beta": 0.049, "se": 0.015, "r2": 0.402},
    "Winsorized":       {"beta": 0.052, "se": 0.014, "r2": 0.385},
}

# ---------------------------------------------------------------------------
# Table 8 - supplementary non-target outcomes (outcome specificity)
#   outcome: (beta, se, adj_r2)
# ---------------------------------------------------------------------------
NON_TARGET = {
    "HusbandHousework": (-0.012, 0.028, 0.215),
    "RespondentHeight": ( 0.002, 0.008, 0.892),
    "HouseholdSize":    (-0.015, 0.022, 0.654),
}

# ---------------------------------------------------------------------------
# Table 9 - observable employment pathway (mediation decomposition)
# ---------------------------------------------------------------------------
MEDIATION = {
    "step1_ce_on_nonfarm": (0.078, 0.019),      # NonFarm ~ CleanEnergy
    "step2_ce_direct":     (0.036, 0.015),      # DecisionIndex ~ CE | NonFarm
    "step2_nonfarm":       (0.228, 0.031),      # NonFarm coef in step 2
    "sobel_z":             3.86,
    "indirect":            0.018,
    "indirect_ci":         (0.009, 0.027),
    "share":               33.3,
    "n_boot":              1000,
}

# ---------------------------------------------------------------------------
# Table 10 - linked pathway regressions
#   column: {regressor: (beta, se)}
# ---------------------------------------------------------------------------
PATHWAY = {
    "EnergyLabor":   {"CleanEnergy": (-0.342, 0.042)},
    "NonFarm":       {"CleanEnergy": (0.056, 0.018), "EnergyLabor": (-0.065, 0.014)},
    "IncomeControl": {"CleanEnergy": (0.031, 0.015), "NonFarm": (0.142, 0.021)},
    "DecisionIndex": {"CleanEnergy": (0.025, 0.014), "NonFarm": (0.112, 0.025),
                       "IncomeControl": (0.185, 0.028)},
}
PATHWAY_ORDER = ["EnergyLabor", "NonFarm", "IncomeControl", "DecisionIndex"]

# ---------------------------------------------------------------------------
# Tables 11 & 12 - heterogeneity in the CE -> DecisionIndex association
#   subgroup: (beta, se, n, insignificant_flag)
# ---------------------------------------------------------------------------
HETERO = {
    "LowIncome":    (0.079, 0.021, 5913, False),
    "HighIncome":   (0.028, 0.016, 5913, False),   # p<0.10
    "HighBurden":   (0.088, 0.023, 5682, False),
    "LowBurden":    (0.021, 0.017, 6144, True),
    "LowEdu":       (0.031, 0.020, 5214, True),
    "HighEdu":      (0.072, 0.018, 6612, False),
    "Mountain":     (0.081, 0.022, 5420, False),
    "Plain":        (0.034, 0.016, 6406, False),
}
HETERO_DIFF_P = {
    "Income": 0.012, "Burden": 0.003, "Education": 0.035, "Topography": 0.018,
}

# ---------------------------------------------------------------------------
# Appendix Table A4 - common-cohort event study (2020 cohort + never-treated)
# ---------------------------------------------------------------------------
COMMON_COHORT_EVENT = {
    "EnergyLabor":   {0: (-0.238, 0.043), 1: (-0.365, 0.049)},
    "NonFarm":       {0: ( 0.045, 0.021), 1: ( 0.081, 0.024)},
    "DecisionIndex": {0: ( 0.032, 0.016), 1: ( 0.058, 0.018)},
}

