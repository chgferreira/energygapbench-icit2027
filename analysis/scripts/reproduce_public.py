#!/usr/bin/env python3
"""Rebuild figures and validate summaries using published aggregates only.

This does not resample individual episodes or reconstruct restricted telemetry.
Bootstrap arrays contain aggregate contrast estimates, not individual losses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_icit import METHODS, SITES, compare_tables, make_figures, sha256

RUN_ID = "conference-20261101T165101Z-1789282"


def verify_draw_summaries(records, arrays, columns):
    """Verify stored intervals and centered exploratory p-values from mean draws."""
    checks = []
    for row in records:
        matches = [i for i, col in enumerate(columns)
                   if all(row.get(k) == v for k, v in col.items() if k != "difference")]
        if len(matches) != 1:
            raise ValueError("Contrast column is not uniquely identified")
        draws = arrays[f"{row['site']}_block{row['block_days']}"][:, matches[0]]
        valid = draws[np.isfinite(draws)]
        if (len(draws), len(valid), len(draws) - len(valid)) != (
                row["attempted"], row["valid"], row["invalid"]):
            raise ValueError("Bootstrap attempt accounting mismatch")
        ci = np.quantile(valid, [.025, .975]) if len(valid) else None
        if ci is None or not np.allclose(ci, row["ci95"], rtol=0, atol=1e-14):
            raise ValueError("Published percentile interval does not match mean draws")
        supported = row["temporal_support_sufficient"]
        expected_p = ((1 + np.count_nonzero(np.abs(valid - row["estimate"]) >= abs(row["estimate"])))
                      / (len(draws) + 1)) if supported else None
        actual_p = row["p_exploratory"]
        if (expected_p is None) != (actual_p is None) or (
                expected_p is not None and not np.isclose(expected_p, actual_p, rtol=0, atol=1e-15)):
            raise ValueError("Exploratory p-value mismatch")
        if row["confirmatory"] or row["physical_confirmatory"] or row["p_holm"] is not None:
            raise ValueError("Unexpected confirmatory inference")
        checks.append({"site": row["site"], "block_days": row["block_days"],
                       "contrast": row["contrast"], "ci_max_absolute_error": float(np.max(np.abs(ci - row["ci95"]))),
                       "p_value_reproduced": True, "temporal_support_sufficient": supported})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/public_outputs"))
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    inputs = set()

    def source(name):
        path = root / name
        inputs.add(path)
        return path

    def csv(name):
        return pd.read_csv(source(name))

    reports = f"results/{RUN_ID}/reports"
    core = csv("analysis/outputs/reconstruction_all_methods_all_hypotheses.csv")
    cells = csv("analysis/outputs/reconstruction_cells.csv")
    metrics = ["nmae", "mae_kw", "forecast_nmae", "forecast_mae_kw"]
    grouped = cells.groupby(["site", "method", "variant"], sort=True)
    if not grouped.size().eq(9).all():
        raise ValueError("Expected all nine fixed fold/duration cells per policy")
    means = grouped[metrics].mean().where(grouped[metrics].count().eq(9))
    check = compare_tables(means.reset_index(), core, ["site", "method", "variant"], metrics, "equal-cell aggregate")
    duration = csv("analysis/outputs/selected_policy_by_duration.csv")
    contrasts = pd.DataFrame(json.loads(source(f"{reports}/contrasts.json").read_text()))
    make_figures(core, duration, contrasts, figures)

    rows = []
    for method in METHODS:
        row = {"method": method}
        for site in SITES:
            for suffix, variant in [("P", "P"), ("E", "E:instantaneous")]:
                name = method + ("+energy_consistent" if suffix == "E" else "")
                match = core.loc[core.site.eq(site) & core.method.eq(name) & core.variant.eq(variant)]
                if len(match) != 1:
                    raise ValueError("Missing or duplicate method table cell")
                row[f"{site}_{suffix}_nmae"] = float(match.nmae.iloc[0])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "methods_nmae.csv", index=False)

    controls = csv("analysis/outputs/energy_control_summary/control_by_fold_duration.csv")
    control_metrics = ["nmae", "mae_kw", "nmae_original_E", "mae_kw_original_E", "nmae_P", "mae_kw_P"]
    if not controls.groupby(["site", "policy"]).size().eq(9).all():
        raise ValueError("Incomplete control strata")
    controls = controls.groupby(["site", "policy"], as_index=False)[control_metrics].mean()
    control_check = compare_tables(controls, csv("analysis/outputs/energy_control_summary/control_summary.csv"),
                                   ["site", "policy"], control_metrics, "energy controls")
    controls.to_csv(out / "energy_controls.csv", index=False)

    checks = {}
    for kind, directory, stem, schema_name in [
        ("forecast", "analysis/forecast_v2/outputs", "forecast", "bootstrap_draws_schema.json"),
        ("controls", "analysis/energy_controls/statistics", "control", "control_draws_schema.json")]:
        records = json.loads(source(f"{directory}/{stem}_contrasts.json").read_text())
        schema = json.loads(source(f"{directory}/{schema_name}").read_text())
        with np.load(source(f"{directory}/{stem}_bootstrap_draws.npz"), allow_pickle=False) as arrays:
            checks[kind] = verify_draw_summaries(records, arrays, schema["columns"])
        pd.DataFrame(checks[kind]).to_csv(out / f"{kind}_interval_checks.csv", index=False)

    summary = {"status": "passed", "scope": "aggregate_only_reproduction",
               "individual_data_included": False, "episode_bootstrap_resampling_performed": False,
               "support_flags": "read from archived summaries; calendar geometry is not re-audited here",
               "reconstruction": check, "energy_controls": control_check,
               "forecast_intervals_verified": len(checks["forecast"]),
               "control_intervals_verified": len(checks["controls"]),
               "script_sha256": sha256(Path(__file__)),
               "inputs_sha256": {str(p.relative_to(root)): sha256(p) for p in sorted(inputs)}}
    (out / "validation.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Verified {len(core)} policy aggregates, 54 forecast and 18 control intervals. Outputs: {out}")


if __name__ == "__main__":
    main()
