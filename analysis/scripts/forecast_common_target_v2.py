#!/usr/bin/env python3
"""Versioned one-hour forecast analysis on common originally observed targets.

The frozen v1 report is never modified. This revision applies a documented
available-target estimand and exploratory paired calendar-day bootstrap. It
first reproduces the original reconstruction intervals using the independently
derived full test calendars, including empty days.
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REVISION = "forecast-common-target-v2"
SITES = ("H1", "H2", "I1")
FOLDS = (1, 2, 3)
DURATIONS = (15, 60, 180)
HYPOTHESES = ("instantaneous", "interval_mean_start", "interval_mean_end")
BLOCK_LENGTHS = (7, 2, 14)
ATTEMPTS = 10000
SEED = 20261031
METHODS = ("median", "locf", "linear", "seasonal", "kalman_filter", "kalman_smoother", "SAITS", "MOMENT-1-base")
EPISODE = ["site", "fold", "condition", "duration", "mask_seed", "gap_id", "day"]
POLICIES = [(m, "P") for m in METHODS] + [(m + "+energy_consistent", "E:" + h) for h in HYPOTHESES for m in METHODS]
CONTRASTS = [(h, c) for h in HYPOTHESES for c in ("contrast_energy_gain", "contrast_complexity_gain")]
PRESENTATION = {"progress", "output_dir", "state_dir", "checkpoint_every_steps", "checkpoint_every_seconds", "checkpoint_optimizer_steps", "checkpoint_active_seconds", "progress_refresh_seconds", "progress_persist_seconds", "status_snapshot_seconds", "heartbeat_seconds", "resume", "budget_total_hours", "budget_seconds", "timezone", "retry_failed"}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def scientific_config(config):
    return {k: scientific_config(v) if isinstance(v, dict) else v for k, v in config.items() if k not in PRESENTATION}


def stable_seed(seed, replicate):
    return int.from_bytes(hashlib.sha256(f"{seed}|{replicate}".encode()).digest()[:8], "big")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_calendars(path, lock_path, lock):
    manifest = json.loads(path.read_text())
    if manifest["protocol_json_sha256"] != sha256(lock_path):
        raise ValueError("Calendar manifest belongs to another frozen protocol")
    if manifest.get("derived_from_episode_dates") is not False:
        raise ValueError("Calendar support must not be truncated to observed episodes")
    calendars = {}
    for site in SITES:
        item = manifest["sites"][site]
        expected = {s["path"]: s["sha256"] for s in lock["datasets"][site]["config"]["files"]}
        actual = {s["path"]: s["sha256"] for s in item["sources"]}
        if actual != expected:
            raise ValueError("Calendar source hashes differ from frozen dataset hashes")
        calendars[site] = {}
        for fold in FOLDS:
            spec = item["folds"][str(fold)]
            days = spec["calendar_days"]
            if not days or days[0] != spec["validation_end_exclusive"][:10] or days[-1] != spec["test_end"][:10]:
                raise ValueError("Calendar endpoints do not match the elapsed-time split")
            if any((date.fromisoformat(b) - date.fromisoformat(a)).days != 1 for a, b in zip(days, days[1:])):
                raise ValueError("Full calendar must include contiguous empty days")
            calendars[site][fold] = days
    return calendars


def choose_baselines(path):
    scores = pd.read_csv(path)
    if scores.duplicated(["site", "fold", "method"]).any() or not set(scores.method).issubset(METHODS[:6]):
        raise ValueError("Invalid validation baseline score table")
    if len(scores) != 54 or not np.isfinite(scores.validation_equal_duration_nmae).all():
        raise ValueError("Incomplete six-baseline validation coverage")
    ordered = scores.sort_values(["site", "fold", "validation_equal_duration_nmae", "latency_p50_seconds", "method"], na_position="last")
    selected = ordered.groupby(["site", "fold"], sort=False).head(1)
    return {(r.site, int(r.fold)): r.method for r in selected.itertuples()}


def load_wide(path):
    columns = EPISODE + ["partition", "method", "variant", "nmae", "forecast_nmae", "forecast_mae_kw", "seed_evaluations"]
    episodes = pd.read_parquet(path, columns=columns)
    episodes = episodes.loc[episodes.partition.eq("test")].copy()
    if episodes.duplicated(EPISODE + ["method", "variant"]).any():
        raise ValueError("Duplicate episode-policy rows")
    expected_seeds = np.where(episodes.method.str.startswith("SAITS"), 3, 1)
    if not np.array_equal(episodes.seed_evaluations, expected_seeds):
        raise ValueError("Input must average losses over the declared training seeds")
    if set(zip(episodes.method, episodes.variant)) != set(POLICIES):
        raise ValueError("Expected the eight methods under all four information variants")
    wide = {metric: episodes.pivot(index=EPISODE, columns=["method", "variant"], values=metric).reindex(columns=pd.MultiIndex.from_tuples(POLICIES, names=["method", "variant"]))
            for metric in ("nmae", "forecast_nmae", "forecast_mae_kw")}
    if not np.isfinite(wide["nmae"].to_numpy()).all():
        raise ValueError("Incomplete reconstruction policy coverage")
    finite = np.isfinite(wide["forecast_nmae"].to_numpy()) & np.isfinite(wide["forecast_mae_kw"].to_numpy())
    observed = finite.all(axis=1)
    if not np.all(observed | ~finite.any(axis=1)):
        raise ValueError("Forecast availability differs across policies; investigate rather than select favorable support")
    return wide, observed


def paired_values(wide, selected):
    """Keep the same row identities for every hypothesis and both contrasts."""
    rows = wide.index.to_frame(index=False)
    result = np.empty((len(wide), len(CONTRASTS)))
    policy_scores = np.empty((len(wide), 1 + 2 * len(HYPOTHESES)))
    for site in SITES:
        for fold in FOLDS:
            mask = rows.site.eq(site).to_numpy() & rows.fold.eq(fold).to_numpy()
            method = selected[(site, fold)]
            base = wide[(method, "P")].to_numpy()[mask]
            policy_scores[mask, 0] = base
            for h_i, hypothesis in enumerate(HYPOTHESES):
                corrected = wide[(method + "+energy_consistent", "E:" + hypothesis)].to_numpy()[mask]
                linear = wide[("linear+energy_consistent", "E:" + hypothesis)].to_numpy()[mask]
                result[mask, h_i * 2] = corrected - base
                result[mask, h_i * 2 + 1] = corrected - linear
                policy_scores[mask, 1 + h_i * 2] = corrected
                policy_scores[mask, 2 + h_i * 2] = linear
    return rows, result, policy_scores


def day_arrays(rows, values, calendars):
    """Reduce episodes only after pairing; retain zero-count calendar dates."""
    if len(rows) != len(values) or not np.isfinite(values).all():
        raise ValueError("Daily reduction requires aligned finite paired losses")
    names = [f"v{i}" for i in range(values.shape[1])]
    frame = rows[["fold", "duration", "day"]].reset_index(drop=True).copy()
    frame[names] = values
    daily = frame.groupby(["fold", "duration", "day"], sort=True).agg(**{n: (n, "sum") for n in names}, count=("day", "size")).reset_index()
    arrays = {}
    for fold in FOLDS:
        days = calendars[fold]
        lookup = {d: i for i, d in enumerate(days)}
        sums = np.zeros((len(days), len(DURATIONS), len(names)))
        counts = np.zeros((len(days), len(DURATIONS)))
        for row in daily.loc[daily.fold.eq(fold)].itertuples(index=False):
            if row.day not in lookup or row.duration not in DURATIONS:
                raise ValueError("Episode outside declared calendar or duration support")
            i, j = lookup[row.day], DURATIONS.index(int(row.duration))
            sums[i, j] = [getattr(row, n) for n in names]
            counts[i, j] = row.count
        arrays[fold] = (sums, counts)
    return arrays


def fixed_estimate(arrays):
    values = []
    for fold in FOLDS:
        sums, counts = arrays[fold]
        denominator = counts.sum(axis=0)
        if np.any(denominator == 0):
            raise ValueError("A required fold-duration stratum is empty")
        values.extend(sums.sum(axis=0) / denominator[:, None])
    return np.mean(values, axis=0)


def bootstrap_draws(arrays, block_days, attempts=ATTEMPTS, seed=SEED):
    """Use one day resample for all paired policies, hypotheses and contrasts.

    Invalid draws retain their replicate ID as NaN. They are never redrawn.
    The RNG ordering matches the frozen day-reduction implementation.
    """
    if block_days < 1 or attempts < 1:
        raise ValueError("Positive block length and attempt count required")
    nmetrics = arrays[FOLDS[0]][0].shape[-1]
    result = np.full((attempts, nmetrics), np.nan)
    for replicate in range(attempts):
        rng = np.random.default_rng(stable_seed(seed, replicate))
        values = []
        for fold in FOLDS:
            sums, counts = arrays[fold]
            n = len(sums)
            if n == 0:
                raise ValueError("Empty test calendar")
            length = min(block_days, n)
            starts = rng.integers(0, n - length + 1, size=(n + length - 1) // length)
            positions = (starts[:, None] + np.arange(length)).ravel()[:n]
            denominator = counts[positions].sum(axis=0)
            if np.any(denominator == 0):
                values = None
                break
            values.extend(sums[positions].sum(axis=0) / denominator[:, None])
        if values is not None:
            result[replicate] = np.mean(values, axis=0)
    return result


def summarize_bootstrap(arrays, draws, block_days):
    point = fixed_estimate(arrays)
    blocks_by_fold = {str(f): len(arrays[f][1]) // block_days for f in FOLDS}
    duration_days = {str(d): sum(int((arrays[f][1][:, j] > 0).sum()) for f in FOLDS) for j, d in enumerate(DURATIONS)}
    geometric_support = sum(blocks_by_fold.values()) >= 20 and all(v >= 4 for v in blocks_by_fold.values()) and all(v >= 20 for v in duration_days.values())
    summaries = []
    for i, estimate in enumerate(point):
        valid = draws[:, i][np.isfinite(draws[:, i])]
        invalid = len(draws) - len(valid)
        supported = bool(geometric_support and invalid == 0)
        p = float((1 + np.count_nonzero(np.abs(valid - estimate) >= abs(estimate))) / (len(draws) + 1)) if supported else None
        summaries.append({"estimate": float(estimate), "ci95": np.quantile(valid, [.025, .975]).tolist() if len(valid) else None,
                          "attempted": len(draws), "valid": len(valid), "invalid": invalid,
                          "block_days": block_days, "support_blocks": sum(blocks_by_fold.values()),
                          "support_blocks_by_fold": blocks_by_fold, "duration_days": duration_days,
                          "temporal_support_sufficient": supported,
                          "interval_kind": "pointwise_exploratory_conditional_available_target",
                          "confirmatory": False, "physical_confirmatory": False,
                          "p_exploratory": p, "p_exploratory_adjusted": False,
                          "p_raw": None, "p_holm": None, "multiplicity_control": "none",
                          "familywise_or_fdr_control_claimed": False,
                          "reason": "Conditional on observed one-hour targets and fitted models; unconfirmed physical measurement semantics" + ("; insufficient temporal support or invalid bootstrap draws" if not supported else "")})
    return summaries


def verify_reconstruction(rows, values, calendars, frozen):
    audit, replay = [], {}
    for site in SITES:
        keep = rows.site.eq(site).to_numpy()
        arrays = day_arrays(rows.loc[keep], values[keep], calendars[site])
        for block in BLOCK_LENGTHS:
            print(f"Verifying original reconstruction intervals: {site}, {block}-day blocks", flush=True)
            draws = bootstrap_draws(arrays, block)
            for (hypothesis, contrast), actual in zip(CONTRASTS, summarize_bootstrap(arrays, draws, block)):
                matches = [r for r in frozen if r["site"] == site and r["hypothesis"] == hypothesis and r["contrast"] == contrast and r["endpoint"] == "nmae" and r["block_days"] == block]
                if len(matches) != 1:
                    raise ValueError("Missing frozen reconstruction comparator for replay")
                expected = matches[0]
                for field in ("estimate", "ci95"):
                    if not np.allclose(actual[field], expected[field], rtol=1e-10, atol=1e-12):
                        raise ValueError(f"Calendar/bootstrap reconstruction replay mismatch: {site}/{hypothesis}/{contrast}/{block}/{field}")
                for field in ("attempted", "invalid", "support_blocks", "duration_days", "temporal_support_sufficient"):
                    if actual[field] != expected[field]:
                        raise ValueError(f"Reconstruction replay support differs: {field}")
                errors = np.abs(np.asarray(actual["ci95"]) - np.asarray(expected["ci95"]))
                audit.append({"site": site, "hypothesis": hypothesis, "contrast": contrast, "block_days": block,
                              "estimate_absolute_error": abs(actual["estimate"] - expected["estimate"]),
                              "ci_max_absolute_error": float(errors.max()), "support_fields_identical": True})
            replay[f"{site}_block{block}"] = draws
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--calendar-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    out = args.output_dir.resolve()
    if out == run or run in out.parents:
        raise ValueError("Versioned outputs must not be written into the frozen run")
    out.mkdir(parents=True, exist_ok=True)
    reports = run / "reports"
    inputs = [run / "protocol.lock.json", run / "protocol.lock.sha256", reports / "episode_losses.parquet", reports / "selected_baseline_validation.csv", reports / "contrasts.json", reports / "contrasts_block_sensitivity.json", args.calendar_manifest.resolve()]
    before = {str(p): sha256(p) for p in inputs}
    lock = json.loads(inputs[0].read_text())
    fingerprint = canonical_hash(scientific_config(lock))
    if fingerprint != inputs[1].read_text().strip():
        raise ValueError("Scientific protocol fingerprint mismatch")
    calendars = load_calendars(args.calendar_manifest, inputs[0], lock)
    selected = choose_baselines(reports / "selected_baseline_validation.csv")
    wide, observed = load_wide(reports / "episode_losses.parquet")
    rows, reconstruction, _ = paired_values(wide["nmae"], selected)
    if len(rows) != 68580 or int(observed.sum()) != 68217:
        raise ValueError("Run-specific episode or common-target count differs from the reviewed revision")
    frozen = json.loads((reports / "contrasts.json").read_text()) + json.loads((reports / "contrasts_block_sensitivity.json").read_text())
    replay_audit = verify_reconstruction(rows, reconstruction, calendars, frozen)
    pd.DataFrame(replay_audit).to_csv(out / "reconstruction_interval_replay.csv", index=False, float_format="%.16g")
    # All methods are evaluated on exactly this common support. It is determined
    # solely by finite supplied target losses, without conditioning on their size.
    support = rows.copy()
    support["common_observed_1h_target"] = observed
    coverage = support.groupby(["site", "fold", "duration"], sort=True).agg(eligible_episode_instances=("gap_id", "size"), observed_target_instances=("common_observed_1h_target", "sum")).reset_index()
    coverage["excluded_target_instances"] = coverage.eligible_episode_instances - coverage.observed_target_instances
    if len(coverage) != 27 or coverage.observed_target_instances.eq(0).any():
        raise ValueError("A required site/fold/duration stratum is empty")
    coverage.to_csv(out / "forecast_target_support.csv", index=False)
    support.to_parquet(out / "forecast_target_membership.parquet", index=False)
    calendar_rows = [{"site": site, "fold": fold, "day": day} for site in SITES for fold in FOLDS for day in calendars[site][fold]]
    pd.DataFrame(calendar_rows).to_csv(out / "full_test_calendars.csv", index=False)
    all_metrics, all_cells = [], []
    for metric in ("forecast_nmae", "forecast_mae_kw"):
        available = wide[metric].loc[observed]
        cells = available.groupby(level=["site", "fold", "duration"]).mean()
        overall = cells.groupby(level="site").mean()
        for (site, fold, duration), row in cells.iterrows():
            for (method, variant), score in row.items():
                all_cells.append({"site": site, "fold": int(fold), "duration": int(duration), "method": method, "variant": variant, "metric": metric, "value": float(score)})
        for site, row in overall.iterrows():
            for (method, variant), score in row.items():
                all_metrics.append({"site": site, "method": method, "variant": variant, "metric": metric, "value": float(score), "analysis_revision": REVISION, "population": "common_originally_observed_1h_targets"})
    pd.DataFrame(all_metrics).to_csv(out / "forecast_all_methods.csv", index=False, float_format="%.12g")
    pd.DataFrame(all_cells).to_csv(out / "forecast_all_method_cells.csv", index=False, float_format="%.12g")
    available_wide = wide["forecast_nmae"].loc[observed]
    available_rows, differences, scores = paired_values(available_wide, selected)
    contrast_output, policy_output, draw_output = [], [], {}
    for site in SITES:
        keep = available_rows.site.eq(site).to_numpy()
        arrays = day_arrays(available_rows.loc[keep], differences[keep], calendars[site])
        score_arrays = day_arrays(available_rows.loc[keep], scores[keep], calendars[site])
        points = fixed_estimate(score_arrays)
        for h_i, hypothesis in enumerate(HYPOTHESES):
            p, e, linear = map(float, (points[0], points[1 + h_i * 2], points[2 + h_i * 2]))
            policy_output.append({"site": site, "hypothesis": hypothesis, "B_P_nmae": p, "B_E_nmae": e, "linear_E_nmae": linear,
                                  "energy_difference": e - p, "complexity_difference": e - linear,
                                  "relative_error_change_percent": 100 * (e / p - 1), "analysis_revision": REVISION})
        for block in BLOCK_LENGTHS:
            print(f"Forecast common-target bootstrap: {site}, {block}-day blocks", flush=True)
            draws = bootstrap_draws(arrays, block)
            draw_output[f"{site}_block{block}"] = draws
            for (hypothesis, contrast), summary in zip(CONTRASTS, summarize_bootstrap(arrays, draws, block)):
                contrast_output.append({"site": site, "hypothesis": hypothesis, "contrast": contrast, "endpoint": "forecast_nmae_1h_common_observed_target", "analysis_revision": REVISION, "seed": SEED, **summary})
    policies = pd.DataFrame(policy_output)
    policies.to_csv(out / "forecast_selected_policies.csv", index=False, float_format="%.12g")
    write_json(out / "forecast_contrasts.json", contrast_output)
    flat = pd.DataFrame(contrast_output)
    flat["ci_lower"] = [r["ci95"][0] if r["ci95"] else None for r in contrast_output]
    flat["ci_upper"] = [r["ci95"][1] if r["ci95"] else None for r in contrast_output]
    flat.drop(columns=["ci95", "support_blocks_by_fold", "duration_days"]).to_csv(out / "forecast_contrasts.csv", index=False, float_format="%.12g")
    np.savez_compressed(out / "forecast_bootstrap_draws.npz", **draw_output)
    write_json(out / "bootstrap_draws_schema.json", {"keys": "{site}_block{days}", "shape": [ATTEMPTS, len(CONTRASTS)], "rows": "replicate IDs 0 through 9999, NaN means invalid draw retained without replacement", "columns": [{"hypothesis": h, "contrast": c} for h, c in CONTRASTS], "seed": SEED, "seed_derivation": "First eight SHA256 bytes of base_seed|replicate_id, interpreted big endian", "pairing": "One shared calendar-day resample across all contrast columns, separately within each fold and site"})
    table = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Site & $B(P)$ & $B(E)$ & Change (\%) & $\Delta$ [95\% interval] \\", r"\midrule"]
    for site in SITES:
        p = policies.loc[policies.site.eq(site) & policies.hypothesis.eq("instantaneous")].iloc[0]
        c = next(r for r in contrast_output if r["site"] == site and r["hypothesis"] == "instantaneous" and r["contrast"] == "contrast_energy_gain" and r["block_days"] == 7)
        lo, hi = c["ci95"]
        label = site + (r"$^{*}$" if not c["temporal_support_sufficient"] else "")
        table.append(f"{label} & {p.B_P_nmae:.4f} & {p.B_E_nmae:.4f} & {p.relative_error_change_percent:+.2f} & {c['estimate']:+.4f} [{lo:+.4f}, {hi:+.4f}]" + r" \\")
    table.extend([r"\bottomrule", r"\end{tabular}"])
    (out / "forecast_table.tex").write_text("\n".join(table) + "\n")
    summary = {"analysis_revision": REVISION, "run_id": run.name, "original_frozen_report_modified": False,
               "chronology": "Available-target reporting revision after inspection of the completed frozen execution; not preregistered or confirmatory.",
               "estimand": "Mean of per-episode seed-averaged forecast losses on common originally observed one-hour targets; equal weights for each of the three durations and each of the three folds; installations analyzed separately.",
               "eligible_episode_instances": len(rows), "common_target_episode_instances": int(observed.sum()),
               "excluded_target_episode_instances": int((~observed).sum()), "evaluated_policies": len(POLICIES),
               "all_policies_share_target_support": True, "required_fold_duration_strata_per_site": 9,
               "reconstruction_interval_replays": len(replay_audit),
               "maximum_replayed_ci_absolute_error": max(r["ci_max_absolute_error"] for r in replay_audit),
               "bootstrap_attempts_per_contrast": ATTEMPTS, "bootstrap_seed": SEED,
               "main_block_days": 7, "sensitivity_block_days": [2, 14], "confirmatory": False,
               "multiplicity": "54 exploratory contrast/measurement-assumption/block-length summaries; pointwise intervals and unadjusted exploratory p-values; no familywise or FDR control claim.",
               "conditioning": ["Observed future targets", "Fitted reconstruction and forecast models", "Realized artificial masks", "Validation-selected baseline and correction policy"],
               "limitations": ["No correction for target-missingness selection bias or full training/model-selection uncertainty.", "H1 fails the declared temporal-support heuristic for the main seven-day analysis.", "Meter semantics remain hypothetical; interval-mean I1 identity is not a metrological validation.", "This revision does not reclassify original unavailable forecast endpoints as originally completed.", "Derived calendar provenance is reconstructed from hash-verified source timestamps, not original server split manifests; exact reconstruction interval replay checks the reconstruction."]}
    write_json(out / "forecast_revision_summary.json", summary)
    after = {str(p): sha256(p) for p in inputs}
    if before != after:
        raise RuntimeError("An immutable input changed during the analysis")
    outputs = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "forecast_revision_manifest.json"}
    source_hashes = {str(p.relative_to(run)) if p.is_relative_to(run) else "calendar_provenance.json": sha256(p) for p in inputs}
    write_json(out / "forecast_revision_manifest.json", {"analysis_revision": REVISION, "scientific_fingerprint": fingerprint,
               "inputs_sha256": source_hashes, "script_sha256": sha256(Path(__file__)), "outputs_sha256": outputs,
               "python_version": sys.version.split()[0], "numpy_version": np.__version__, "pandas_version": pd.__version__})
    print(policies.to_string(index=False), flush=True)
    print(f"Completed {REVISION}. Original inputs unchanged. Outputs: {out}", flush=True)


if __name__ == "__main__":
    main()
