#!/usr/bin/env python3
"""Exploratory paired reconstruction contrasts for the post hoc scalar controls.

All inference reads immutable episode losses. It never reselects correction
parameters or reconstructs the original E trajectory. The existing verified
calendar-day resampling kernel is shared with the versioned forecast analysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import forecast_common_target_v2 as bootstrap

REVISION = "energy-control-paired-contrasts-v1"
KEYS = ["site", "fold", "partition", "condition", "mask_seed", "gap_id"]
POLICIES = ("uniform_shift_no_deadzone", "uniform_shift_tolerant")


def validate_selection(directory):
    """Audit the saved validation choice without making a new test-set choice."""
    trials = pd.read_csv(directory / "validation_trials.csv")
    selected = pd.read_csv(directory / "selected_parameters.csv")
    keys = ["site", "fold", "policy"]
    expected = {(site, fold, policy) for site in bootstrap.SITES for fold in bootstrap.FOLDS for policy in POLICIES}
    if set(selected[keys].itertuples(index=False, name=None)) != expected or selected.duplicated(keys).any():
        raise ValueError("Incomplete saved validation selections")
    if trials.duplicated(keys + ["trial"]).any() or not np.isfinite(trials.validation_nmae).all():
        raise ValueError("Invalid saved validation candidate scores")
    counts = trials.groupby(keys).size()
    if len(counts) != 18 or not counts.eq(8).all():
        raise ValueError("Expected all eight validation candidates for every control/site/fold")
    ordered = trials.sort_values(keys + ["validation_nmae", "trial"]).groupby(keys, sort=False).head(1)
    checked = selected.merge(ordered, on=keys, suffixes=("_selected", "_ranked"), validate="one_to_one")
    for field in ("trial", "lambda_energy", "tau_multiplier", "validation_nmae"):
        if not np.allclose(checked[field + "_selected"], checked[field + "_ranked"], rtol=0, atol=1e-14):
            raise ValueError("Saved control parameters do not match validation ranking")
    return len(checked)


def pair_control_losses(frame, archived):
    """Return E minus each control, with original E/P and reconnection days checked."""
    if set(frame.partition) != {"test"} or set(frame.policy) != set(POLICIES):
        raise ValueError("Unexpected control partition or policy")
    if frame.duplicated(KEYS + ["policy"]).any():
        raise ValueError("Duplicate control episode-policy rows")
    counts = frame.groupby(KEYS, dropna=False).size()
    if not counts.eq(len(POLICIES)).all():
        raise ValueError("Control policies do not share identical episodes")
    reference = archived.loc[archived.partition.eq("test") & archived.method.eq("linear+energy_consistent") & archived.variant.eq("E:instantaneous"), KEYS + ["duration", "day", "nmae", "mae_kw"]].rename(columns={"duration": "archived_duration", "day": "archived_day", "nmae": "archived_E_nmae", "mae_kw": "archived_E_mae_kw"})
    base = archived.loc[archived.partition.eq("test") & archived.method.eq("linear") & archived.variant.eq("P"), KEYS + ["nmae", "mae_kw"]].rename(columns={"nmae": "archived_P_nmae", "mae_kw": "archived_P_mae_kw"})
    if reference.duplicated(KEYS).any() or base.duplicated(KEYS).any():
        raise ValueError("Duplicate archived linear policy rows")
    reference = reference.merge(base, on=KEYS, how="outer", indicator=True, validate="one_to_one")
    if not reference._merge.eq("both").all():
        raise ValueError("Archived P and E episode identities differ")
    reference = reference.drop(columns="_merge")
    matched = frame.merge(reference, on=KEYS, how="outer", indicator=True, validate="many_to_one")
    if len(matched) != len(reference) * len(POLICIES) or not matched._merge.eq("both").all():
        raise ValueError("Control support differs from all archived linear P/E episodes")
    matched = matched.drop(columns="_merge")
    required = ["nmae", "mae_kw", "nmae_original_E", "mae_kw_original_E", "nmae_P", "mae_kw_P", "difference_vs_original_E"]
    if not np.isfinite(matched[required].to_numpy()).all():
        raise ValueError("Unavailable losses must not be silently discarded")
    for supplied, source in (("duration", "archived_duration"), ("nmae_original_E", "archived_E_nmae"), ("mae_kw_original_E", "archived_E_mae_kw"), ("nmae_P", "archived_P_nmae"), ("mae_kw_P", "archived_P_mae_kw")):
        if not np.allclose(matched[supplied], matched[source], rtol=0, atol=1e-13):
            raise ValueError(f"Archived reference changed in control packet: {supplied}")
    if not np.allclose(matched.difference_vs_original_E, matched.nmae - matched.archived_E_nmae, rtol=0, atol=1e-13):
        raise ValueError("Control-minus-E difference in source does not match the losses")
    matched["day"] = matched.archived_day
    matched["E_minus_control_nmae"] = matched.archived_E_nmae - matched.nmae
    matched["E_minus_control_mae_kw"] = matched.archived_E_mae_kw - matched.mae_kw
    return matched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--controls-dir", type=Path, required=True)
    parser.add_argument("--calendar-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run, controls, out = args.run_dir.resolve(), args.controls_dir.resolve(), args.output_dir.resolve()
    if out == run or run in out.parents or out == controls or controls in out.parents:
        raise ValueError("Statistical outputs require a separate directory from frozen or replay inputs")
    paths = [run / "protocol.lock.json", run / "protocol.lock.sha256", run / "reports/episode_losses.parquet",
             controls / "control_paired_losses.parquet", controls / "control_protocol.json", controls / "control_validation.json",
             controls / "selected_parameters.csv", controls / "validation_trials.csv", args.calendar_manifest.resolve()]
    before = {str(p): bootstrap.sha256(p) for p in paths}
    status = json.loads((controls / "control_validation.json").read_text())
    protocol = json.loads((controls / "control_protocol.json").read_text())
    if status["status"] != "completed_descriptive_posthoc_replay":
        raise ValueError("Wait for the full control replay to finish before inference")
    for path in paths[3:]:
        if path.name in status.get("outputs_sha256", {}) and bootstrap.sha256(path) != status["outputs_sha256"][path.name]:
            raise ValueError(f"Control artifact checksum mismatch: {path.name}")
    if protocol["inputs_sha256"]["episode_losses.parquet"] != bootstrap.sha256(paths[2]):
        raise ValueError("Control replay references a different frozen episode table")
    if protocol.get("selection") != "validation_equal_duration_episode_nmae_then_fixed_grid_index":
        raise ValueError("Unexpected control selection policy")
    selection_count = validate_selection(controls)
    lock = json.loads(paths[0].read_text())
    if bootstrap.canonical_hash(bootstrap.scientific_config(lock)) != paths[1].read_text().strip():
        raise ValueError("Scientific protocol fingerprint mismatch")
    calendars = bootstrap.load_calendars(args.calendar_manifest, paths[0], lock)
    frame = pd.read_parquet(paths[3])
    archived = pd.read_parquet(paths[2], columns=KEYS + ["duration", "day", "method", "variant", "nmae", "mae_kw"], filters=[("partition", "==", "test"), ("method", "in", ["linear", "linear+energy_consistent"])])
    matched = pair_control_losses(frame, archived)
    if len(matched) != 137160 or len(matched) != status["expected_test_rows"]:
        raise ValueError("Expected all 68,580 test episodes under both scalar controls")
    grouping = ["site", "fold", "duration", "policy"]
    metrics = ["nmae", "mae_kw", "archived_E_nmae", "archived_E_mae_kw", "archived_P_nmae", "archived_P_mae_kw", "E_minus_control_nmae", "E_minus_control_mae_kw"]
    cells = matched.groupby(grouping, as_index=False)[metrics].mean()
    counts = matched.groupby(grouping, as_index=False).size().rename(columns={"size": "episode_instances"})
    cells = cells.merge(counts, on=grouping, validate="one_to_one")
    expected_cells = {(s, f, d, p) for s in bootstrap.SITES for f in bootstrap.FOLDS for d in bootstrap.DURATIONS for p in POLICIES}
    if set(cells[grouping].itertuples(index=False, name=None)) != expected_cells:
        raise ValueError("Incomplete fixed fold/duration strata")
    site_scores = cells.groupby(["site", "policy"], as_index=False)[metrics].mean()
    out.mkdir(parents=True, exist_ok=True)
    cells.to_csv(out / "control_contrast_cells.csv", index=False, float_format="%.12g")
    site_scores.to_csv(out / "control_contrast_site_scores.csv", index=False, float_format="%.12g")
    ids = bootstrap.EPISODE
    pivot = matched.pivot(index=ids, columns="policy", values="E_minus_control_nmae").reindex(columns=POLICIES)
    rows = pivot.index.to_frame(index=False)
    contrasts, draws_by_site = [], {}
    for site in bootstrap.SITES:
        keep = rows.site.eq(site).to_numpy()
        arrays = bootstrap.day_arrays(rows.loc[keep], pivot.to_numpy()[keep], calendars[site])
        expected_point = site_scores.loc[site_scores.site.eq(site)].set_index("policy").loc[list(POLICIES), "E_minus_control_nmae"].to_numpy()
        if not np.allclose(bootstrap.fixed_estimate(arrays), expected_point, rtol=0, atol=1e-13):
            raise ValueError("Daily reduction changed the equal-weight paired estimand")
        for block in bootstrap.BLOCK_LENGTHS:
            print(f"Paired energy-control contrasts: {site}, {block}-day blocks", flush=True)
            draws = bootstrap.bootstrap_draws(arrays, block)
            draws_by_site[f"{site}_block{block}"] = draws
            for policy, summary in zip(POLICIES, bootstrap.summarize_bootstrap(arrays, draws, block)):
                summary["interval_kind"] = "pointwise_exploratory_conditional_control_replay"
                summary["reason"] = "Post hoc comparison conditional on fitted and validation-selected policies and realized masks; physical semantics unconfirmed" + ("; insufficient temporal support or invalid bootstrap draws" if not summary["temporal_support_sufficient"] else "")
                contrasts.append({"analysis_revision": REVISION, "site": site, "control": policy,
                                  "contrast": "archived_linear_E_minus_" + policy, "endpoint": "reconstruction_nmae",
                                  "measurement_hypothesis": "instantaneous", "seed": bootstrap.SEED,
                                  "control_parameters_selected_on": "validation", "original_E_source": "frozen_episode_losses",
                                  "independent_confirmation": False, **summary})
    bootstrap.write_json(out / "control_contrasts.json", contrasts)
    flat = pd.DataFrame(contrasts)
    flat["ci_lower"] = [r["ci95"][0] if r["ci95"] else None for r in contrasts]
    flat["ci_upper"] = [r["ci95"][1] if r["ci95"] else None for r in contrasts]
    flat.drop(columns=["ci95", "duration_days", "support_blocks_by_fold"]).to_csv(out / "control_contrasts.csv", index=False, float_format="%.12g")
    np.savez_compressed(out / "control_bootstrap_draws.npz", **draws_by_site)
    bootstrap.write_json(out / "control_draws_schema.json", {"arrays": "{site}_block{days}", "shape": [bootstrap.ATTEMPTS, len(POLICIES)], "columns": [{"control": p, "difference": "archived_linear_E_nmae_minus_control_nmae"} for p in POLICIES], "rows": "replicate IDs 0..9999; invalid draws retained as NaN without replacement", "base_seed": bootstrap.SEED, "same_resample_for_both_controls": True})
    table = [r"\begin{tabular}{lrr}", r"\toprule", r"Site & E minus shift, $z=0$ & E minus shift, $z=1$ \\", r"\midrule"]
    for site in bootstrap.SITES:
        entries = [next(r for r in contrasts if r["site"] == site and r["control"] == p and r["block_days"] == 7) for p in POLICIES]
        label = site + (r"$^{*}$" if not entries[0]["temporal_support_sufficient"] else "")
        cells_tex = [f"{r['estimate']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]" for r in entries]
        table.append(label + " & " + " & ".join(cells_tex) + r" \\")
    table.extend([r"\bottomrule", r"\end{tabular}"])
    (out / "control_contrasts.tex").write_text("\n".join(table) + "\n")
    if before != {str(p): bootstrap.sha256(p) for p in paths}:
        raise RuntimeError("Control inputs changed during inference; rerun on stable completed outputs")
    bootstrap.write_json(out / "control_statistics_summary.json", {"analysis_revision": REVISION,
        "post_hoc": True, "confirmatory": False, "independent_confirmation": False,
        "unique_episode_instances": len(rows), "paired_policy_rows": len(matched), "fold_duration_policy_cells": len(cells),
        "saved_validation_choices_verified": selection_count, "control_parameters_reselected": False,
        "original_P_and_E_losses_verified_against_frozen_archive": True, "inputs_unchanged": True,
        "estimand": "Episode-level E-minus-control loss differences, equally averaged within each duration/fold and equally across the three durations and three folds; sites separate.",
        "attempts_per_contrast": bootstrap.ATTEMPTS, "seed": bootstrap.SEED, "main_block_days": 7, "sensitivity_block_days": [2, 14],
        "multiplicity": "18 exploratory site/control/block-length summaries; pointwise intervals and unadjusted exploratory p-values; no familywise or FDR control claim.",
        "limits": ["The control analysis was designed after the original results were inspected.", "Both controls use constant displacement and gamma=0; comparison against flexible E is not an isolated deadzone or smoothness ablation.", "The two scalar policies have separately validation-selected settings; differences do not isolate a numerical parameter while holding all others fixed.", "Original E trajectories are not reconstructed; their archived losses are preserved.", "Intervals condition on realized masks and fitted/selected policies; H1 fails the main seven-day support rule.", "No meter semantics, external generalization, or independent confirmation is established."]})
    sources = {str(p.relative_to(run)) if p.is_relative_to(run) else "controls/" + p.name if p.is_relative_to(controls) else "calendar_provenance.json": bootstrap.sha256(p) for p in paths}
    outputs = {str(p.relative_to(out)): bootstrap.sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "control_statistics_manifest.json"}
    bootstrap.write_json(out / "control_statistics_manifest.json", {"analysis_revision": REVISION, "input_sha256": sources,
        "script_sha256": bootstrap.sha256(Path(__file__)), "bootstrap_implementation_sha256": bootstrap.sha256(Path(bootstrap.__file__)),
        "output_sha256": outputs, "python_version": sys.version.split()[0], "numpy_version": np.__version__, "pandas_version": pd.__version__})
    print(flat.loc[flat.block_days.eq(7), ["site", "control", "estimate", "ci_lower", "ci_upper", "temporal_support_sufficient", "p_exploratory"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
