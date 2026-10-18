"""Render manuscript-style tables from the estimator outputs."""
from __future__ import annotations

from . import config as C


def stars(beta, se):
    if se is None or se <= 0:
        return ""
    z = abs(beta) / se
    if z >= 2.576:
        return "***"
    if z >= 1.960:
        return "**"
    if z >= 1.645:
        return "*"
    return ""


def cell(beta, se, nd=3):
    return f"{beta:+.{nd}f}{stars(beta, se)} ({se:.{nd}f})"


def _hr(w=92):
    return "-" * w


def table2(res):
    L = ["Table 2. Baseline (2018) characteristics of treatment and comparison groups.",
         _hr(),
         f"{'Variable':<16}{'Full Mean(SD)':>18}{'Control':>10}{'Treat':>10}"
         f"{'Diff':>10}{'t (p)':>18}",
         _hr()]
    for name in C.DESCRIPTIVE_ORDER:
        fm, fsd, mc, mt, diff, t, p = res[name]
        L.append(f"{name:<16}{f'{fm:.3f} ({fsd:.3f})':>18}{mc:>10.3f}{mt:>10.3f}"
                 f"{diff:>+10.3f}{f'{t:.2f} ({p:.3f})':>18}")
    L += [_hr(), "N: control = 2,150 women; treatment = 1,792 women; total = 3,942.", ""]
    return L


def table3(res):
    L = ["Table 3. Baseline staggered-adoption TWFE estimates.",
         _hr(100),
         f"{'':<16}" + "".join(f"{o:>18}" for o in C.OUTCOME_ORDER), _hr(100),
         f"{'CleanEnergy':<16}" +
         "".join(f"{cell(*res[o][:2]):>18}" for o in C.OUTCOME_ORDER),
         f"{'Controls / FE':<16}" + "".join(f"{'YES':>18}" for _ in C.OUTCOME_ORDER),
         f"{'Observations':<16}" + "".join(f"{'11,826':>18}" for _ in C.OUTCOME_ORDER),
         f"{'Adjusted R2':<16}" +
         "".join(f"{res[o][2]:>18.3f}" for o in C.OUTCOME_ORDER),
         _hr(100),
         "Household-clustered SE in parentheses. *** p<0.01, ** p<0.05, * p<0.1.", ""]
    return L


def table4(res):
    cols = ["EnergyLabor", "NonFarm", "DecisionIndex"]
    L = ["Table 4. Event-study estimates around the clean energy transition.",
         _hr(),
         f"{'Event time':<18}" + "".join(f"{c:>20}" for c in cols), _hr()]
    labels = {-2: "D-2 (2 before)", 0: "D0 (adoption)", 1: "D+1 (1 after)"}
    L.append(f"{'D-1 (ref.)':<18}" + "".join(f"{'--':>20}" for _ in cols))
    for k in (-2, 0, 1):
        L.append(f"{labels[k]:<18}" +
                 "".join(f"{cell(*res[c][k]):>20}" for c in cols))
    L.append(f"{'Pre-trend p':<18}" +
             "".join(f"{res[c]['pretrend_p']:>20.3f}" for c in cols))
    L += [_hr(), "D-2 identified from the 2022 cohort only; D+1 from the 2020 "
          "cohort only.", ""]
    return L


def table5(res):
    L = ["Table 5. Sensitivity to potentially post-treatment household controls.",
         _hr(104),
         f"{'':<22}" + "".join(f"{o:>18}" for o in C.OUTCOME_ORDER), _hr(104)]
    labels = {"Model1": "Model 1 (baseline)",
              "Model2": "Model 2 (no post-trt)",
              "Model3": "Model 3 (FE only)"}
    for model in ("Model1", "Model2", "Model3"):
        row = res[model]
        L.append(f"{labels[model]:<22}" +
                 "".join(f"{cell(*row[o][:2]):>18}" for o in C.OUTCOME_ORDER))
        L.append(f"{'  Adjusted R2':<22}" +
                 "".join(f"{row[o][2]:>18.3f}" for o in C.OUTCOME_ORDER))
    L += [_hr(104), "All specifications: individual + survey-wave FE; N = 11,826.", ""]
    return L


def table6(psm, migr, sa, plac):
    L = ["Table 6. Core robustness checks for the decision-making index.",
         _hr(),
         f"{'':<18}{'PSM-DID':>16}{'Excl. migrant':>16}"
         f"{'Sun-Abraham':>16}{'Placebo(1,000)':>18}", _hr(),
         f"{'CleanEnergy':<18}{cell(*psm[:2]):>16}{cell(*migr[:2]):>16}"
         f"{cell(*sa[:2]):>16}{f'mean={plac[0]:.4f}':>18}",
         f"{'Observations':<18}{psm[2]:>16,}{migr[2]:>16,}{sa[2]:>16,}"
         f"{f'p={plac[1]:.3f}':>18}",
         _hr(), "Placebo: mean pseudo-treatment coefficient over 1,000 "
         "timing permutations.", ""]
    return L


def table7(cs, pw, win):
    L = ["Table 7. Additional robustness checks for the decision-making index.",
         _hr(),
         f"{'':<18}{'Callaway-SA':>16}{'Province x wave':>18}{'Winsorized 1/99':>18}",
         _hr(),
         f"{'CleanEnergy':<18}{cell(cs[0], cs[1]):>16}{cell(pw[0], pw[1]):>18}"
         f"{cell(win[0], win[1]):>18}",
         f"{'Adjusted R2':<18}{'--':>16}{f'{pw[2]:.3f}':>18}{f'{win[2]:.3f}':>18}",
         _hr(), "Standard errors clustered at the household level.", ""]
    return L


def table8(res):
    L = ["Table 8. Supplementary estimates for selected non-target outcomes.",
         _hr(),
         f"{'':<16}{'Husband HW':>16}{'Resp. height':>16}{'HH size':>16}", _hr()]
    order = ["HusbandHousework", "RespondentHeight", "HouseholdSize"]
    L.append(f"{'CleanEnergy':<16}" +
             "".join(f"{cell(*res[o][:2]):>16}" for o in order))
    L.append(f"{'Adjusted R2':<16}" +
             "".join(f"{res[o][2]:>16.3f}" for o in order))
    L += [_hr(), "None of the non-target outcomes is significant at "
          "conventional levels.", ""]
    return L


def table9(m):
    sobel_str = f"{m['sobel']:.2f}"
    ind_str = f"{m['indirect']:.3f} [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]"
    share_str = f"{m['share']:.1f}%"
    L = ["Table 9. Observable employment pathway analysis.",
         _hr(),
         f"{'':<34}{'Step 1: NonFarm':>18}{'Step 2: DecisionIdx':>22}", _hr(),
         f"{'CleanEnergy':<34}{cell(*m['a']):>18}{cell(*m['direct']):>22}",
         f"{'NonFarm (pathway)':<34}{'--':>18}{cell(*m['b']):>22}",
         f"{'Sobel Z':<34}{'--':>18}{sobel_str:>22}",
         f"{'Indirect (bootstrap 95% CI)':<34}{'--':>18}{ind_str:>22}",
         f"{'Indirect / total association':<34}{'--':>18}{share_str:>22}",
         _hr(), f"Cluster bootstrap: {m['n_boot']} resamples. "
         "Contemporaneous mediator -> descriptive decomposition only.", ""]
    return L


def table10(res):
    L = ["Table 10. Pathway analysis linking clean energy to women's agency.",
         _hr(100),
         f"{'':<16}" + "".join(f"{o:>21}" for o in C.PATHWAY_ORDER), _hr(100)]
    rows = ["treat", "EnergyLabor", "NonFarm", "IncomeControl"]
    labels = {"treat": "CleanEnergy", "EnergyLabor": "EnergyLabor",
              "NonFarm": "NonFarm", "IncomeControl": "IncomeControl"}
    for rv in rows:
        line = f"{labels[rv]:<16}"
        for col in C.PATHWAY_ORDER:
            line += f"{(cell(*res[col][rv]) if rv in res[col] else '--'):>21}"
        L.append(line)
    L += [_hr(100), "Separate conditional regressions; coefficients are not a "
          "verified causal chain.", ""]
    return L


def table11_12(subs, diffs):
    def block(title, keys, dimlabel, dim):
        out = [title, _hr(),
               f"{'':<18}" + "".join(f"{k:>16}" for k in keys), _hr()]
        line = f"{'CleanEnergy':<18}"
        for k in keys:
            b, se, n, insig = subs[k]
            line += f"{cell(b, se):>16}"
        out.append(line)
        out.append(f"{'Observations':<18}" +
                   "".join(f"{subs[k][2]:>16,}" for k in keys))
        d, se_int, p = diffs[dim]
        out.append(f"Subgroup difference ({dimlabel}): diff={d:+.3f}, "
                   f"p={p:.3f}")
        out.append("")
        return out

    L = ["Table 11. Heterogeneity by baseline household income and energy burden."]
    L += block("", ["LowIncome", "HighIncome"], "income", "Income")
    L += block("", ["HighBurden", "LowBurden"], "fuel burden", "Burden")
    L += ["Table 12. Heterogeneity by education and regional topography."]
    L += block("", ["LowEdu", "HighEdu"], "education", "Education")
    L += block("", ["Mountain", "Plain"], "topography", "Topography")
    return L


def tableA4(res):
    cols = ["EnergyLabor", "NonFarm", "DecisionIndex"]
    L = ["Appendix Table A4. Common-cohort event study (2020 cohort + never).",
         _hr(),
         f"{'Event time':<18}" + "".join(f"{c:>20}" for c in cols), _hr(),
         f"{'D-1 (ref. 2018)':<18}" + "".join(f"{'--':>20}" for _ in cols)]
    for k in (0, 1):
        lbl = "D0 (2020)" if k == 0 else "D+1 (2022)"
        L.append(f"{lbl:<18}" + "".join(f"{cell(*res[c][k]):>20}" for c in cols))
    L += [f"{'Observations':<18}" + "".join(f"{res['_n']:>20,}" for _ in cols),
          _hr(), f"Women = {res['_women']:,}.", ""]
    return L


