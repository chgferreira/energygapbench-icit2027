#!/usr/bin/env python3
"""Summarize practical reconstruction effects from the frozen ICIT results.

This supplementary descriptive analysis does not refit, select a new comparator,
alter the primary estimand, or add inferential tests. Positive percentage changes
in the output denote lower error after correction. The instantaneous hypothesis
is retained as the original directional formulation, not selected by test score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SITES = ["H1", "H2", "I1"]
DURATIONS = [15, 60, 180]
METHODS = ["median", "locf", "linear", "seasonal", "kalman_filter", "kalman_smoother", "SAITS", "MOMENT-1-base"]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_core(core: pd.DataFrame) -> pd.DataFrame:
    """Preserve the equal-duration/equal-fold means from the original report."""
    required = ["site", "method", "variant", "nmae", "mae_kw"]
    if not set(required).issubset(core.columns):
        raise ValueError("Core report lacks required reconstruction fields")
    if core.duplicated(["site", "method", "variant"]).any():
        raise ValueError("Duplicate core report policy keys")
    rows = []
    for site in SITES:
        for method in METHODS:
            base = core.loc[core.site.eq(site) & core.method.eq(method) & core.variant.eq("P")]
            corrected = core.loc[core.site.eq(site) & core.method.eq(method + "+energy_consistent") & core.variant.eq("E:instantaneous")]
            if len(base) != 1 or len(corrected) != 1:
                raise ValueError(f"Incomplete policy coverage for {site}/{method}")
            p, e = base.iloc[0], corrected.iloc[0]
            values = np.asarray([p.nmae, e.nmae, p.mae_kw, e.mae_kw], dtype=float)
            if not np.isfinite(values).all() or p.nmae <= 0:
                raise ValueError(f"Cannot calculate a relative change for {site}/{method}")
            rows.append({
                "site": site, "method": method, "hypothesis": "instantaneous",
                "P_nmae": float(p.nmae), "E_nmae": float(e.nmae),
                "P_mae_kw": float(p.mae_kw), "E_mae_kw": float(e.mae_kw),
                "mae_decrease_kw": float(p.mae_kw - e.mae_kw),
                "nmae_decrease_percent": float(100 * (1 - e.nmae / p.nmae)),
            })
    return pd.DataFrame(rows)


def summarize_durations(duration: pd.DataFrame, methods: pd.DataFrame):
    selected = duration.loc[duration.policy.isin(["linear_P", "linear_E"])].copy()
    if selected.duplicated(["site", "duration", "policy"]).any():
        raise ValueError("Duplicate linear duration policy keys")
    expected = {(site, duration, policy) for site in SITES for duration in DURATIONS for policy in ["linear_P", "linear_E"]}
    observed = set(selected[["site", "duration", "policy"]].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError("All three durations and both linear policies are required at every site")
    paired = selected.pivot(index=["site", "duration"], columns="policy", values="nmae").reset_index()
    paired = paired.rename(columns={"linear_P": "P_nmae", "linear_E": "E_nmae"})
    if not np.isfinite(paired[["P_nmae", "E_nmae"]].to_numpy(float)).all() or not paired.P_nmae.gt(0).all():
        raise ValueError("Invalid duration-level errors")
    paired["nmae_decrease_percent"] = 100 * (1 - paired.E_nmae / paired.P_nmae)
    paired["method"] = "linear"
    paired["hypothesis"] = "instantaneous"
    summary = methods.loc[methods.method.eq("linear")].copy()
    checks = []
    for site in SITES:
        group = paired.loc[paired.site.eq(site)]
        row = summary.loc[summary.site.eq(site)].iloc[0]
        for policy in ["P", "E"]:
            difference = abs(float(group[f"{policy}_nmae"].mean()) - float(row[f"{policy}_nmae"]))
            # The independent input CSV is stored to twelve significant digits.
            if difference > 1e-10:
                raise ValueError(f"Duration means do not reproduce the core {site}/{policy} estimand")
            checks.append({"site": site, "policy": policy, "absolute_difference": difference})
        mask = summary.site.eq(site)
        summary.loc[mask, "duration_min_nmae_decrease_percent"] = group.nmae_decrease_percent.min()
        summary.loc[mask, "duration_max_nmae_decrease_percent"] = group.nmae_decrease_percent.max()
    return summary, paired, checks


def summarize_fold_durations(main: pd.DataFrame, summary: pd.DataFrame):
    """Pair every original linear P/E cell without dropping folds or durations."""
    keys = ["site", "fold", "condition", "duration"]
    fields = keys + ["nmae", "mae_kw", "episodes"]
    if not set(fields + ["method", "variant"]).issubset(main.columns):
        raise ValueError("Main report lacks required reconstruction fields")
    p = main.loc[main.method.eq("linear") & main.variant.eq("P"), fields]
    e = main.loc[main.method.eq("linear+energy_consistent") & main.variant.eq("E:instantaneous"), fields]
    condition_for_duration = {15: "C3", 60: "C4", 180: "C5"}
    expected = {(site, fold, condition_for_duration[duration], duration)
                for site in SITES for fold in [1, 2, 3] for duration in DURATIONS}
    for policy, frame in [("P", p), ("E", e)]:
        if frame.duplicated(keys).any():
            raise ValueError(f"Duplicate linear {policy} main-table cell")
        if set(frame[keys].itertuples(index=False, name=None)) != expected:
            raise ValueError(f"All 27 linear {policy} site/fold/duration cells are required")
    paired = p.merge(e, on=keys, how="outer", suffixes=("_P", "_E"), validate="one_to_one")
    if not paired.episodes_P.eq(paired.episodes_E).all() or not paired.episodes_P.gt(0).all():
        raise ValueError("Linear P/E main-table cells have different or empty episode support")
    values = paired[["nmae_P", "nmae_E", "mae_kw_P", "mae_kw_E"]].to_numpy(float)
    if not np.isfinite(values).all() or not paired.nmae_P.gt(0).all():
        raise ValueError("Invalid main-table reconstruction losses")
    paired["nmae_decrease_percent"] = 100 * (1 - paired.nmae_E / paired.nmae_P)
    paired["mae_decrease_kw"] = paired.mae_kw_P - paired.mae_kw_E
    paired["method"] = "linear"
    paired["hypothesis"] = "instantaneous"
    checks = []
    for site in SITES:
        group = paired.loc[paired.site.eq(site)]
        mask = summary.site.eq(site)
        row = summary.loc[mask].iloc[0]
        for metric in ["nmae", "mae_kw"]:
            for policy in ["P", "E"]:
                difference = abs(float(group[f"{metric}_{policy}"].mean()) - float(row[f"{policy}_{metric}"]))
                if difference > 1e-10:
                    raise ValueError(f"Nine-cell means do not reproduce {site}/{policy}/{metric}")
                checks.append({"site": site, "policy": policy, "metric": metric, "absolute_difference": difference})
        summary.loc[mask, "improved_fold_duration_cells"] = int(group.nmae_decrease_percent.gt(0).sum())
        summary.loc[mask, "total_fold_duration_cells"] = len(group)
        summary.loc[mask, "fold_duration_min_nmae_decrease_percent"] = group.nmae_decrease_percent.min()
        summary.loc[mask, "fold_duration_max_nmae_decrease_percent"] = group.nmae_decrease_percent.max()
    for field in ["improved_fold_duration_cells", "total_fold_duration_cells"]:
        summary[field] = summary[field].astype(int)
    return summary, paired, checks


def latex_table(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
        r"Site & $P$ MAE & $E$ MAE & Cells & Range \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{row.site} & {row.P_mae_kw:.3f} & {row.E_mae_kw:.3f} & "
            f"{row.improved_fold_duration_cells}/{row.total_fold_duration_cells} & "
            f"{row.fold_duration_min_nmae_decrease_percent:.2f}--{row.fold_duration_max_nmae_decrease_percent:.2f}\\%"
            + r" \\"
        )
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis/outputs"), help="Existing outputs from analyze_icit.py")
    parser.add_argument("--output-dir", type=Path, help="Defaults to --analysis-dir")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    analysis = args.analysis_dir.resolve()
    out = (args.output_dir or args.analysis_dir).resolve()
    if out == run or run in out.parents:
        raise ValueError("Write derived outputs outside the immutable run evidence")
    core_path = run / "reports/core_aggregate.csv"
    main_path = run / "reports/main_table.csv"
    duration_path = analysis / "selected_policy_by_duration.csv"
    methods = parse_core(pd.read_csv(core_path))
    summary, paired, checks = summarize_durations(pd.read_csv(duration_path), methods)
    summary, fold_paired, fold_checks = summarize_fold_durations(pd.read_csv(main_path), summary)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    outputs = {
        "engineering_methods.csv": methods,
        "engineering_summary.csv": summary,
        "engineering_by_duration.csv": paired,
        "engineering_by_fold_duration.csv": fold_paired,
    }
    for name, frame in outputs.items():
        frame.to_csv(out / name, index=False, float_format="%.12g")
    table_path = out / "tables/engineering_summary.tex"
    table_path.write_text(latex_table(summary), encoding="utf8")
    validation = {
        "analysis": "descriptive_engineering_summary", "run_id": run.name,
        "hypothesis": "instantaneous", "illustrative_base": "linear",
        "primary_comparator_B_unchanged": True,
        "inference": "No new intervals or tests. The 27 site/fold/duration means are not independent replicates or a binomial sample.",
        "aggregation": "Original equal-duration/equal-fold site means. Per-duration inputs equally average folds. Relative reductions are ratios of these means, not mean per-episode percentages.",
        "units": {"mae": "kW", "relative_changes": "percent; positive means lower corrected error"},
        "range_definition": "Table ranges are the minimum and maximum percentage nMAE decreases over nine fold-by-duration cells within each site, not uncertainty intervals. Separate CSV columns also retain ranges over three duration means after averaging folds.",
        "duration_cells": len(paired),
        "duration_cells_with_lower_corrected_nmae": int(paired.nmae_decrease_percent.gt(0).sum()),
        "site_duration_to_core_checks": checks,
        "fold_duration_cells": len(fold_paired),
        "fold_duration_cells_with_lower_corrected_nmae": int(fold_paired.nmae_decrease_percent.gt(0).sum()),
        "all_fold_duration_cells_improve": bool(fold_paired.nmae_decrease_percent.gt(0).all()),
        "fold_duration_to_core_checks": fold_checks,
        "sources_sha256": {
            "reports/core_aggregate.csv": sha256(core_path),
            "reports/main_table.csv": sha256(main_path),
            "analysis_outputs/selected_policy_by_duration.csv": sha256(duration_path),
        },
        "script_sha256": sha256(Path(__file__)),
        "outputs_sha256": {name: sha256(out / name) for name in [*outputs, "tables/engineering_summary.tex"]},
        "software": {"numpy": np.__version__, "pandas": pd.__version__},
    }
    (out / "engineering_validation.json").write_text(json.dumps(validation, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
