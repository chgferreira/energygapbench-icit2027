#!/usr/bin/env python3
"""Regenerate descriptive control tables from authorized episode-level results.

This command uses no private telemetry, no original experiment state, and no
model fitting. Equal duration and fold weights reproduce the study estimand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["site", "fold", "partition", "condition", "mask_seed", "gap_id"]
POLICIES = ("uniform_shift_tolerant", "uniform_shift_no_deadzone")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, content):
    Path(path).write_text(json.dumps(content, indent=2, sort_keys=True,
                                   allow_nan=False) + "\n")


def summarize(controls_dir, output_dir):
    controls_dir, output_dir = Path(controls_dir), Path(output_dir)
    status = json.loads((controls_dir / "control_validation.json").read_text())
    protocol = json.loads((controls_dir / "control_protocol.json").read_text())
    if status["status"] != "completed_descriptive_posthoc_replay":
        raise ValueError("Full control replay has not completed")
    path = controls_dir / "control_paired_losses.parquet"
    if digest(path) != status["outputs_sha256"][path.name]:
        raise ValueError("Saved paired control loss checksum mismatch")
    frame = pd.read_parquet(path)
    if len(frame) != status["expected_test_rows"] or frame.duplicated(KEYS + ["policy"]).any():
        raise ValueError("Incomplete or duplicate paired control episode rows")
    if set(frame.partition) != {"test"} or set(frame.policy) != set(POLICIES):
        raise ValueError("Unexpected evaluation partition or control policies")
    metrics = ["nmae", "mae_kw", "nmae_original_E", "mae_kw_original_E", "nmae_P", "mae_kw_P", "difference_vs_original_E"]
    if not np.isfinite(frame[metrics].to_numpy()).all():
        raise ValueError("Nonfinite paired reconstruction losses")
    counts = frame.groupby(KEYS, dropna=False).size()
    if not counts.eq(len(POLICIES)).all():
        raise ValueError("Policies do not share identical episode support")
    cells = frame.groupby(["site", "fold", "duration", "policy"], as_index=False)[metrics].mean()
    expected = {(s, f, d, p) for s in protocol["sites"] for f in protocol["folds"]
                for d in (15., 60., 180.) for p in POLICIES}
    if set(cells[["site", "fold", "duration", "policy"]].itertuples(index=False, name=None)) != expected:
        raise ValueError("Incomplete site/fold/duration support")
    summary = cells.groupby(["site", "policy"], as_index=False)[metrics].mean()
    summary["reduction_vs_P_percent"] = 100 * (1 - summary.nmae / summary.nmae_P)
    summary["original_E_reduction_vs_P_percent"] = 100 * (1 - summary.nmae_original_E / summary.nmae_P)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    cells.to_csv(output_dir / "control_by_fold_duration.csv", index=False)
    summary.to_csv(output_dir / "control_summary.csv", index=False)
    statuses = frame.groupby(["site", "policy", "status", "reason"], dropna=False).size().reset_index(name="episodes")
    statuses.to_csv(output_dir / "control_status_counts.csv", index=False)
    rows = []
    for site in protocol["sites"]:
        selected = summary[summary.site == site].set_index("policy")
        nozone = selected.loc["uniform_shift_no_deadzone"]
        zone = selected.loc["uniform_shift_tolerant"]
        if not np.allclose([nozone.nmae_P, nozone.nmae_original_E], [zone.nmae_P, zone.nmae_original_E], rtol=0, atol=1e-15):
            raise ValueError("Archived comparison changed between policies")
        rows.append(f"{site} & {zone.nmae_P:.4f} & {nozone.nmae:.4f} & {zone.nmae:.4f} & {zone.nmae_original_E:.4f} \\\\")
    table = "\n".join([
        "% Descriptive post-hoc nMAE. Equal duration and fold weights; no inferential test.",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Site & Linear P & Shift, $z=0$ & Shift, $z=1$ & Proposed E \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""])
    (output_dir / "tables" / "energy_controls.tex").write_text(table)
    report = {"status": "reproduced_complete_descriptive_control_summary",
        "no_inferential_claims": True, "script_sha256": digest(__file__),
        "input_sha256": {p.name: digest(p) for p in (path, controls_dir / "control_protocol.json", controls_dir / "control_validation.json")},
        "paired_rows": len(frame), "unique_test_episodes": len(counts),
        "policies": list(POLICIES), "site_fold_duration_policy_cells": len(cells),
        "original_P_reproduced_episodes": sum(r["episodes"] for r in status["linear_P_reproduction"]),
        "maximum_original_P_absolute_error": max(r["max_abs_mae_or_nmae_error"] for r in status["linear_P_reproduction"]),
        "explanation": "z is the deadzone width in units of train tolerance. Each policy is separately selected on validation. Both scalar controls use gamma=0. This is not an isolated ablation of the flexible correction.",
        "outputs_sha256": {str(p.relative_to(output_dir)): digest(p) for p in output_dir.rglob("*") if p.is_file() and p.name != "summary_validation.json"}}
    write_json(output_dir / "summary_validation.json", report)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(summarize(args.controls_dir, args.output_dir).to_string(index=False))


if __name__ == "__main__":
    main()
