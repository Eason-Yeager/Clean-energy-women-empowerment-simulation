"""End-to-end replication driver.

Usage:
    python run_all.py              # full run (1,000 bootstrap/permutation reps)
    python run_all.py --fast       # fewer reps for a quick smoke test

Outputs (written to ./output):
    tables_output.txt   all manuscript tables, reconstructed
    table3_baseline.csv summary of the headline estimates
    validation.txt      computed vs. manuscript coefficient check
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src import config as C          # noqa: E402
from src import data, estimators as E, tables as T   # noqa: E402

OUT = os.path.join(HERE, "output")


def _emit(lines, buf):
    for ln in lines:
        print(ln)
        buf.append(ln)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="use 100 bootstrap/permutation reps for a quick check")
    args = ap.parse_args()
    reps = 100 if args.fast else 1000

    os.makedirs(OUT, exist_ok=True)
    print("Building calibrated synthetic CFPS panel ...")
    df = data.build_panel()
    assert len(df) == C.N_OBS, (len(df), C.N_OBS)
    print(f"  {df['wid'].nunique():,} women x {df['wave'].nunique()} waves "
          f"= {len(df):,} observations\n")

    buf = []
    desc = E.descriptive_table(df)
    base = E.baseline_table(df)
    ev = E.event_study(df, C.EVENT_STUDY)
    ctrl = E.control_sensitivity(df)
    psm = E.psm_did(df)
    migr = E.exclude_migrants(df)
    sa = E.sun_abraham(df)
    plac = E.placebo(df, n_perm=reps)
    cs = E.callaway_santanna(df)
    pw = E.province_wave_fe(df)
    win = E.winsorized(df)
    nt = E.non_target(df)
    med = E.mediation(df, n_boot=reps)
    path = E.pathway(df)
    subs, diffs = E.heterogeneity(df)
    a4 = E.common_cohort_event(df)

    _emit(T.table2(desc), buf)
    _emit(T.table3(base), buf)
    _emit(T.table4(ev), buf)
    _emit(T.table5(ctrl), buf)
    _emit(T.table6(psm, migr, sa, plac), buf)
    _emit(T.table7(cs, pw, win), buf)
    _emit(T.table8(nt), buf)
    _emit(T.table9(med), buf)
    _emit(T.table10(path), buf)
    _emit(T.table11_12(subs, diffs), buf)
    _emit(T.tableA4(a4), buf)

    with open(os.path.join(OUT, "tables_output.txt"), "w") as f:
        f.write("\n".join(buf) + "\n")

    _write_csv(base)
    _validate(base, ev, path, subs, med, buf)
    print(f"\nWrote results to {OUT}/")


def _write_csv(base):
    rows = [{"outcome": o, "coef": b, "se": s, "adj_r2": r,
             "target_coef": C.BASELINE[o][0]} for o, (b, s, r) in base.items()]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "table3_baseline.csv"),
                              index=False)


def _validate(base, ev, path, subs, med, buf):
    checks = []
    for o, (b, s, _) in base.items():
        checks.append((f"baseline/{o}", b, C.BASELINE[o][0]))
    for o, tmap in C.EVENT_STUDY.items():
        for k, (tb, _) in tmap.items():
            checks.append((f"event/{o}/D{k}", ev[o][k][0], tb))
    for col, spec in C.PATHWAY.items():
        for rv, (tb, _) in spec.items():
            key = "treat" if rv == "CleanEnergy" else rv
            checks.append((f"pathway/{col}/{rv}", path[col][key][0], tb))
    for k, (tb, _, _, _) in C.HETERO.items():
        checks.append((f"hetero/{k}", subs[k][0], tb))
    checks.append(("mediation/indirect", med["indirect"], C.MEDIATION["indirect"]))

    lines = ["", "=" * 72, "VALIDATION: reconstructed vs. manuscript", "=" * 72,
             f"{'quantity':<28}{'reconstructed':>14}{'manuscript':>14}{'|diff|':>10}"]
    worst = 0.0
    for name, got, tgt in checks:
        d = abs(got - tgt)
        worst = max(worst, d)
        lines.append(f"{name:<28}{got:>14.4f}{tgt:>14.4f}{d:>10.4f}")
    lines += [f"\nmax |diff| across {len(checks)} checks: {worst:.4f}", "=" * 72]
    for ln in lines:
        print(ln)
        buf.append(ln)
    with open(os.path.join(OUT, "validation.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
