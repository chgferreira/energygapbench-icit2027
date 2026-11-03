#!/usr/bin/env python3
"""Post-hoc same-information scalar energy controls on immutable episode masks.

The original experiment is not edited. Original linear P episode losses must
reproduce before any new test outcome is accepted. Correction parameters are
selected using validation only. Archived linear E losses remain the comparator;
its unavailable original trajectories are not reconstructed or invented.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import multiprocessing
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from energy_replay_common import (KEYS, checked_manifests, digest, fold_scale,
    load_lock, load_losses, load_series, output_directory, source_digest, write_json)
from energygap.data import SeriesData
from energygap.evaluation.energy import integrate_signed
from energygap.methods.baselines import LinearImputer
from energygap.methods.calibration import fit_energy_tolerances, constraint_from_query
from energygap.protocols.availability import build_query
from energygap.protocols.splits import rolling_origin_splits

POLICIES = ("uniform_shift_tolerant", "uniform_shift_no_deadzone")
GRID = tuple({"lambda_energy": weight, "tau_multiplier": multiplier}
             for weight, multiplier in itertools.product((0., .1, 1., 10.), (1., 2.)))


def scalar_control(base, times, constraint, *, scale, lambda_energy, tau_multiplier,
                   deadzone):
    """Constant interior displacement, fixed original boundaries and operator.

    J(c)=(c/s)^2 + lambda sum_d [max(0, |E_d(p+c)-D_d|/(m*tau_d)-z)]^2.
    z=1 for the tolerance control and z=0 for its no-deadzone sensitivity.
    Both use gamma=0. Only directions admitted by the executed calibration enter.
    With two directions both residuals enter separately. The same piecewise-
    linear, zero-crossing-exact instantaneous operator is used throughout.

    The fidelity term bounds any candidate improving the identity by
    |c|<=s*sqrt(J(0)). Bounded searches on each side of zero plus the identity
    provide a deterministic feasible solution; no global optimum is claimed.
    No truth, hidden counters, or test-selected parameter enters this function.
    """
    base, times = np.asarray(base, float), np.asarray(times, float)
    if len(base) < 3 or not np.isfinite(base).all() or not np.isfinite(times).all():
        raise ValueError("Finite boundary-inclusive target segment required")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Strictly increasing physical timestamps required")
    if scale <= 0 or not np.isfinite(scale) or lambda_energy < 0 or tau_multiplier <= 0:
        raise ValueError("Invalid fixed control parameters")
    if constraint.semantics != "instantaneous":
        raise ValueError("This comparison fixes the instantaneous directional operator")
    if not constraint.reliable:
        return base.copy(), "unavailable", constraint.reason
    active = [(d, getattr(constraint, d + "_kwh"), getattr(constraint, "tau_" + d))
              for d in ("import", "export") if getattr(constraint, d + "_kwh") is not None]
    if not active or any(e < 0 or t is None or t <= 0 or not np.isfinite([e, t]).all()
                         for _, e, t in active):
        return base.copy(), "unavailable", "no_positive_calibrated_admissible_direction"
    if lambda_energy == 0:
        return base.copy(), "disabled", "lambda_zero_identity"
    work = base.copy()

    def objective(shift):
        work[1:-1] = base[1:-1] + shift
        energy = integrate_signed(work, times, semantics="instantaneous")
        penalty = sum(max(0., abs(energy[d + "_kwh"] - value) / (tau_multiplier * tau)
                          - float(deadzone)) ** 2 for d, value, tau in active)
        return (shift / scale) ** 2 + lambda_energy * penalty

    before = objective(0.)
    if before == 0:
        return base.copy(), "success", "within_directional_tolerance"
    bound = float(scale * np.sqrt(before))
    if not np.isfinite(bound) or bound <= 0:
        return base.copy(), "failed", "unrepresentable_search_bound"
    candidates = [(before, 0.)]
    for bounds in ((-bound, 0.), (0., bound)):
        fit = minimize_scalar(objective, bounds=bounds, method="bounded",
                              options={"xatol": max(1e-10, scale * 1e-9), "maxiter": 150})
        if fit.success and np.isfinite([fit.x, fit.fun]).all():
            candidates.append((float(fit.fun), float(fit.x)))
    if len(candidates) == 1:
        return base.copy(), "failed", "all_scalar_searches_failed_identity_fallback"
    _, shift = min(candidates, key=lambda pair: (pair[0], abs(pair[1]), pair[1]))
    result = base.copy()
    result[1:-1] += shift
    return result, "success", "identity" if shift == 0 else "bounded_scalar_reconciliation"


def worker_init():
    from threadpoolctl import threadpool_limits
    global _THREAD_LIMIT
    _THREAD_LIMIT = threadpool_limits(limits=1)


def evaluate_case(task):
    case, policy, parameters = task
    corrected, status, reason = scalar_control(case["base"], case["times"], case["constraint"],
        scale=case["scale"], **parameters, deadzone=policy == "uniform_shift_tolerant")
    # Truth is consumed only after the control returned its trajectory.
    mae = float(np.mean(np.abs(corrected[1:-1] - case["truth"])))
    return {**case["key"], "policy": policy, "mae_kw": mae,
            "nmae": mae / case["scale"], "status": status, "reason": reason}


def score_cases(executor, cases, policy, parameters):
    tasks = ((case, policy, parameters) for case in cases)
    rows = list(executor.map(evaluate_case, tasks, chunksize=32)) if executor else list(map(evaluate_case, tasks))
    frame = pd.DataFrame(rows)
    if set(frame.duration) != {15., 60., 180.} or not np.isfinite(frame.nmae).all():
        raise ValueError("Complete finite C3-C5 support is required")
    return frame, float(frame.groupby("duration").nmae.mean().mean())


def make_cases(series, fold, manifests, calibration, scale, archived):
    lookup = archived[(archived.site == series.site_id) & (archived.fold == fold.fold_id)
                      & (archived.method == "linear") & (archived.variant == "P")].set_index(KEYS)
    model = LinearImputer()
    cases, max_error, checked = [], 0., 0
    for manifest in manifests:
        hidden = frozenset(manifest.hidden_indices)
        for episode in manifest.episodes:
            query, truth = build_query(series, episode, information_tier="E", artificial_hidden=hidden)
            mask = np.asarray(query.target_gap_mask, bool)
            positions = np.flatnonzero(mask)
            left, right = positions[0] - 1, positions[-1] + 1
            base_full = model.impute(query.values, query.times)
            base = base_full[left:right + 1]
            expected = np.asarray(truth.values)[mask]
            if not np.isfinite(base).all() or not np.isfinite(expected).all():
                raise ValueError("Invalid original target or linear reconstruction")
            mae = float(np.mean(np.abs(base[1:-1] - expected)))
            key = {"site": series.site_id, "fold": fold.fold_id,
                   "partition": manifest.partition, "condition": manifest.condition,
                   "mask_seed": manifest.seed, "gap_id": episode.gap_id,
                   "duration": float(episode.duration_minutes)}
            original = lookup.loc[tuple(key[k] for k in KEYS)]
            errors = np.abs(np.array([mae, mae / scale]) - np.array([original.mae_kw, original.nmae]))
            max_error = max(max_error, float(errors.max()))
            if not np.allclose([mae, mae / scale], [original.mae_kw, original.nmae], rtol=1e-10, atol=1e-12):
                raise ValueError(f"Linear P replay differs from original episode {episode.gap_id}")
            constraint = constraint_from_query(query, calibration, episode.duration_minutes,
                                               metadata={"power_semantics": "instantaneous"})
            cases.append({"key": key, "base": base, "times": np.asarray(query.times)[left:right + 1],
                          "truth": expected, "constraint": constraint, "scale": scale})
            checked += 1
    return cases, {"episodes": checked, "max_abs_mae_or_nmae_error": max_error,
                   "rtol": 1e-10, "atol": 1e-12}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--profile-only", action="store_true", help="Time metadata-ordered validation cases only; no test")
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        raise ValueError("workers must be between 1 and 32")
    output = output_directory(args.output_dir, args.run_dir)
    lock = load_lock(args.run_dir)
    protocol = {"analysis_kind": "post_hoc_exploratory_same_information_control",
        "run_id": args.run_dir.name, "sites": lock["sites"], "folds": lock["folds"],
        "conditions": lock["conditions"], "policies": list(POLICIES), "parameter_grid": list(GRID),
        "operator": "instantaneous_directional_piecewise_linear_exact_zero_crossings",
        "correction_family": "constant_displacement_inside_gap_fixed_boundaries", "gamma": 0,
        "direction_admission": "same_as_original_positive_train_tau_and_admissible_counter",
        "selection": "validation_equal_duration_episode_nmae_then_fixed_grid_index",
        "test_policy": "all_original_episodes_once_after_validation_selection",
        "comparator": "archived_linear_E_instantaneous_episode_losses",
        "reproduction_gate": "all_linear_P_validation_and_test_episode_losses_match",
        "seed_loss_note": "linear_is_deterministic_one_evaluation_per_episode",
        "inferential_claims": False, "source_sha256": source_digest(),
        "script_sha256": digest(__file__), "helpers_sha256": digest(Path(__file__).with_name("energy_replay_common.py")),
        "inputs_sha256": {"protocol.lock.json": digest(args.run_dir / "protocol.lock.json"),
                           "episode_losses.parquet": digest(args.run_dir / "reports/episode_losses.parquet")},
        "limitations": ["Analysis was designed after the original results were available; it is not preregistered.",
            "Restricted displacement and gamma=0 differ from the original flexible correction; this is not an isolated component ablation.",
            "The two new policies isolate the deadzone within the scalar family, conditional on separately validation-selected parameters.",
            "The original E trajectories and original selected parameters are absent; archived E losses are compared without inventing them.",
            "No neural retraining, forecasting replay, metrological confirmation, new sites, or confirmatory hypothesis test is performed."]}
    protocol_path = output / "control_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Supplemental protocol already exists with different content; choose a new output directory")
    write_json(protocol_path, protocol)  # Frozen before new test evaluation.
    report = {"status": "incomplete", "scientific_complete": False,
              "protocol_sha256": digest(protocol_path), "inputs_sha256": dict(protocol["inputs_sha256"]),
              "linear_P_reproduction": [], "selection_partition": "validation", "test_rows": 0}
    write_json(output / "control_validation.json", report)
    archived = load_losses(args.run_dir)
    trials, selected, tests = [], [], []
    started = perf_counter()
    context = multiprocessing.get_context("spawn")
    executor = ProcessPoolExecutor(args.workers, mp_context=context, initializer=worker_init) if args.workers > 1 else None
    worker_init()
    try:
        for site in lock["sites"]:
            print(f"Loading frozen data: {site}", flush=True)
            series, hashes = load_series(lock, site, args.data_root)
            report["inputs_sha256"].update(hashes)
            for fold in rolling_origin_splits(series):
                indices, _, scale, _ = fold_scale(series, fold)
                conditional = SeriesData(series.records, site, series.nominal_seconds,
                    {**series.metadata, "power_semantics": "instantaneous", "hypothesis_only": True})
                calibration = fit_energy_tolerances(conditional, indices, cutoff_time=fold.train_end)
                manifests, hashes = checked_manifests(args.manifest_dir, site, fold, "validation", lock, archived)
                report["inputs_sha256"].update({"manifests/" + k: v for k, v in hashes.items()})
                validation, check = make_cases(series, fold, manifests, calibration, scale, archived)
                report["linear_P_reproduction"].append({"site": site, "fold": fold.fold_id, "partition": "validation", **check})
                print(f"Validated P replay {site}/fold{fold.fold_id}: {len(validation)} validation episodes", flush=True)
                if args.profile_only:
                    sample = [c for d in (15., 60., 180.)
                              for c in [v for v in validation if v["key"]["duration"] == d][:30]]
                    tic = perf_counter()
                    for policy in POLICIES:
                        for parameters in GRID:
                            score_cases(executor, sample, policy, parameters)
                    seconds = perf_counter() - tic
                    write_json(output / "profile.json", {"validation_cases": len(sample), "policies": len(POLICIES),
                        "candidates": len(GRID), "seconds": seconds, "profile_only": True})
                    print(f"Profile: {seconds:.3f}s / {len(sample) * len(POLICIES) * len(GRID)} case-candidates", flush=True)
                    return
                choices = {}
                for policy in POLICIES:
                    candidate_rows = []
                    for trial, parameters in enumerate(GRID):
                        tic = perf_counter()
                        result, loss = score_cases(executor, validation, policy, parameters)
                        row = {"site": site, "fold": fold.fold_id, "policy": policy, "trial": trial,
                               **parameters, "validation_nmae": loss, "episodes": len(result),
                               "elapsed_seconds": perf_counter() - tic}
                        candidate_rows.append(row)
                        print(f"Validation {site}/fold{fold.fold_id}/{policy} {trial+1}/{len(GRID)}: {row['elapsed_seconds']:.1f}s", flush=True)
                    best = min(candidate_rows, key=lambda r: (r["validation_nmae"], r["trial"]))
                    choices[policy] = {k: best[k] for k in ("lambda_energy", "tau_multiplier")}
                    trials.extend(candidate_rows)
                    selected.append(best)
                # Durable selection is written before loading/evaluating this fold's test queries.
                pd.DataFrame(trials).to_csv(output / "validation_trials.csv", index=False)
                pd.DataFrame(selected).to_csv(output / "selected_parameters.csv", index=False)
                manifests, hashes = checked_manifests(args.manifest_dir, site, fold, "test", lock, archived)
                report["inputs_sha256"].update({"manifests/" + k: v for k, v in hashes.items()})
                test, check = make_cases(series, fold, manifests, calibration, scale, archived)
                report["linear_P_reproduction"].append({"site": site, "fold": fold.fold_id, "partition": "test", **check})
                for policy in POLICIES:
                    result, _ = score_cases(executor, test, policy, choices[policy])
                    tests.append(result)
                pd.concat(tests, ignore_index=True).to_parquet(output / "control_episode_losses.parquet", index=False)
                report["test_rows"] = sum(map(len, tests))
                write_json(output / "control_validation.json", report)
                print(f"Completed {site}/fold{fold.fold_id}; elapsed {(perf_counter()-started)/60:.1f} min", flush=True)
    finally:
        if executor:
            executor.shutdown(cancel_futures=True)
    result = pd.concat(tests, ignore_index=True)
    reference = archived[(archived.partition == "test") & (archived.method == "linear+energy_consistent")
                         & (archived.variant == "E:instantaneous")][KEYS + ["nmae", "mae_kw"]]
    base = archived[(archived.partition == "test") & (archived.method == "linear")
                    & (archived.variant == "P")][KEYS + ["nmae", "mae_kw"]]
    paired = result.merge(reference, on=KEYS, suffixes=("", "_original_E"), validate="many_to_one")
    paired = paired.merge(base.rename(columns={"nmae": "nmae_P", "mae_kw": "mae_kw_P"}),
                          on=KEYS, validate="many_to_one")
    if len(paired) != len(result) or len(result) != len(reference) * len(POLICIES):
        raise ValueError("Incomplete original E/common-support control pairing")
    paired["difference_vs_original_E"] = paired.nmae - paired.nmae_original_E
    paired.to_parquet(output / "control_paired_losses.parquet", index=False)
    metrics = ["nmae", "mae_kw", "nmae_original_E", "mae_kw_original_E", "nmae_P", "mae_kw_P", "difference_vs_original_E"]
    cells = paired.groupby(["site", "fold", "duration", "policy"], as_index=False)[metrics].mean()
    cells.to_csv(output / "control_by_fold_duration.csv", index=False)
    summary = cells.groupby(["site", "policy"], as_index=False)[metrics].mean()
    summary["reduction_vs_P_percent"] = 100 * (1 - summary.nmae / summary.nmae_P)
    summary["original_E_reduction_vs_P_percent"] = 100 * (1 - summary.nmae_original_E / summary.nmae_P)
    summary.to_csv(output / "control_summary.csv", index=False)
    report.update(status="completed_descriptive_posthoc_replay", scientific_complete=False,
                  elapsed_seconds=perf_counter()-started, test_rows=len(result),
                  paired_reference_rows=len(reference), expected_test_rows=len(reference)*len(POLICIES),
                  no_inferential_claims=True, original_E_trajectory_reproduction=False)
    report["outputs_sha256"] = {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != "control_validation.json"}
    write_json(output / "control_validation.json", report)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
