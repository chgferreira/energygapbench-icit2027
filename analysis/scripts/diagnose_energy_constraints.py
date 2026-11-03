#!/usr/bin/env python3
"""Audit recorded energy outcomes and optionally reconstruct train tolerances.

No experiment state is opened for writing. Equal archived losses are an outcome,
not a solver diagnosis. With original inputs, calibration is reconstructed from
the frozen data contract; this is explicitly separate from original status logs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from energy_replay_common import (KEYS, digest, fold_scale, load_lock, load_losses,
    load_series, output_directory, source_digest, write_json)
from energygap.data import SeriesData
from energygap.evaluation.energy import integrate_signed, counter_increment
from energygap.methods.calibration import fit_energy_tolerances
from energygap.methods.correction import calibrate_tolerance, correct_energy, EnergyConstraint
from energygap.protocols.splits import rolling_origin_splits


def residual_audit(series, indices, cutoff, semantics):
    """Independently retain q95 and zero counts omitted by the original export.

    Interval sampling and admission match fit_energy_tolerances. The caller
    verifies all reconstructed tolerance dictionaries against that function.
    """
    rows = [series.records[i] for i in indices]
    result = {}
    for direction in ("import", "export"):
        groups = {}
        for duration in (15, 60, 180):
            width = int(round(duration * 60 / series.nominal_seconds)) + 1
            residuals = []
            for left in range(0, max(0, len(rows) - width), width):
                interval = rows[left:left + width + 1]
                values = np.array([r.value_canonical for r in interval])
                times = np.array([r.grid_time.timestamp() for r in interval])
                if (not all(r.observed_original and not r.natural_missing for r in interval)
                    or not all(r.available_to_method and r.availability_time < cutoff
                               and (r.counter_availability_time or r.availability_time) < cutoff
                               and r.measurement_time < cutoff for r in interval)
                    or not np.isfinite(values).all()
                    or (np.diff(times) != series.nominal_seconds).any()):
                    continue
                counters = [getattr(r, direction + "_counter") for r in interval]
                if any(counter_increment(a, b)[0] is None for a, b in zip(counters[:-1], counters[1:])):
                    continue
                energy = integrate_signed(values, times, semantics=semantics)[direction + "_kwh"]
                residuals.append(abs(energy - (counters[-1] - counters[0])))
            groups[duration] = residuals
        result[direction] = groups
    return result


def recorded_diagnostics(run_dir):
    """Query a complete run's SQLite read-only; never instantiate StateStore."""
    path = Path(run_dir) / "state.sqlite"
    if not path.exists():
        return [], [], "not_in_compact_bundle"
    selections, counts = [], Counter()
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        statement = """SELECT r.job_id, r.payload, j.metadata FROM results r
          JOIN jobs j ON j.job_id=r.job_id
          WHERE json_extract(r.payload,'$.phase') IN ('correction_selection','inference')"""
        for job_id, raw, metadata in connection.execute(statement):
            item, meta = json.loads(raw), json.loads(metadata)
            if item.get("phase") == "correction_selection":
                selections.append({"job_id": job_id, "site": meta.get("site"),
                    "fold": meta.get("fold"), "method": meta.get("method"),
                    **{k: item.get(k) for k in ("hypothesis", "trial", "parameters", "loss", "failures")}})
            elif item.get("variant", "").startswith("E:"):
                diag = item.get("diagnostics", {}) or {}
                key = (item.get("site"), item.get("fold"), item.get("method"),
                       item.get("variant"), item.get("status"), item.get("reason"),
                       diag.get("selected_start"), diag.get("solver_variant"))
                counts[key] += 1
    selected = []
    groups = {}
    for item in selections:
        groups.setdefault((item["job_id"], item["hypothesis"]), []).append(item)
    for rows in groups.values():
        best = min(rows, key=lambda r: (r["loss"] is None, r["loss"] or 0., r["trial"]))
        selected.append({**best, "selection_source": "recorded_validation_grid",
                         "recorded_trials": len(rows)})
    names = ["site", "fold", "method", "variant", "status", "reason", "selected_start", "solver_variant"]
    statuses = [dict(zip(names, key), count=count) for key, count in counts.items()]
    return selected, statuses, "read_only_sqlite_recorded_statuses"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--sites", nargs="+", default=["I1"])
    args = parser.parse_args()
    output = output_directory(args.output_dir, args.run_dir)
    lock = load_lock(args.run_dir)
    evidence = {"protocol.lock.json": digest(args.run_dir / "protocol.lock.json"),
        "reports/episode_losses.parquet": digest(args.run_dir / "reports/episode_losses.parquet")}
    report = {"run_id": args.run_dir.name, "source_sha256": source_digest(),
        "script_sha256": digest(__file__), "inputs_sha256": evidence,
        "physical_semantics_confirmed": False, "original_run_modified": False,
        "equality_does_not_identify_solver_cause": True,
        "calibration_provenance": "not_reconstructed_without_original_data",
        "limitations": ["The compact bundle does not retain original per-query solver diagnostics or selected correction parameters.",
            "Recomputed calibration identifies current source behavior on frozen inputs, not a recovered original solver log."]}
    selected, statuses, ledger = recorded_diagnostics(args.run_dir)
    report["recorded_diagnostic_status"] = ledger
    write_json(output / "recorded_correction_parameters.json", selected)
    write_json(output / "recorded_solver_statuses.json", statuses)
    equality, tolerances = [], []
    for site in args.sites:
        archived = load_losses(args.run_dir, site)
        for method in lock["methods"]:
            base = archived[(archived.method == method) & (archived.variant == "P")
                            & (archived.partition == "test")]
            for hypothesis in ("interval_mean_start", "interval_mean_end"):
                corrected = archived[(archived.method == method + "+energy_consistent")
                                     & (archived.variant == "E:" + hypothesis)
                                     & (archived.partition == "test")]
                paired = base.merge(corrected, on=KEYS, suffixes=("_P", "_E"), validate="one_to_one")
                if len(paired) != len(base) or len(paired) != len(corrected):
                    raise ValueError("Incomplete paired interval-mean episode support")
                values = paired[["nmae_P", "nmae_E", "mae_kw_P", "mae_kw_E"]].to_numpy()
                if not np.isfinite(values).all():
                    raise ValueError("Missing/nonfinite reconstruction outcomes")
                equality.append({"site": site, "method": method, "hypothesis": hypothesis,
                    "episodes": len(paired), "equal_nmae": int((paired.nmae_P == paired.nmae_E).sum()),
                    "equal_mae_kw": int((paired.mae_kw_P == paired.mae_kw_E).sum()),
                    "max_abs_nmae_difference": float((paired.nmae_P - paired.nmae_E).abs().max())})
        if args.data_root is None:
            continue
        print(f"Reconstructing train-only calibration: {site}", flush=True)
        series, hashes = load_series(lock, site, args.data_root)
        evidence.update(hashes)
        for fold in rolling_origin_splits(series):
            indices, _, _, _ = fold_scale(series, fold)
            for hypothesis in lock["physical_hypotheses"]:
                conditional = SeriesData(series.records, site, series.nominal_seconds,
                    {**series.metadata, "power_semantics": hypothesis, "hypothesis_only": True})
                fitted = fit_energy_tolerances(conditional, indices, cutoff_time=fold.train_end)
                residuals = residual_audit(series, indices, fold.train_end, hypothesis)
                for direction, groups in residuals.items():
                    audited = calibrate_tolerance(groups, resolution_kwh=conditional.metadata.get("counter_resolution"))
                    if audited != fitted["directions"][direction]:
                        raise ValueError("Independent residual audit does not reproduce executed calibration")
                    pooled = np.concatenate([np.asarray(x) for x in groups.values()])
                    for duration, values in groups.items():
                        item = audited[str(duration)]
                        sample = np.asarray(values) if len(values) >= 30 else pooled if len(pooled) >= 30 else np.array([])
                        q95 = float(np.quantile(sample, .95)) if len(sample) else None
                        tolerances.append({"site": site, "fold": fold.fold_id,
                            "hypothesis": hypothesis, "direction": direction, "duration": duration,
                            **item, "empirical_q95_kwh": q95, "zero_residuals": int(np.count_nonzero(np.asarray(values) == 0)),
                            "effective_counter_resolution_kwh": conditional.metadata.get("counter_resolution"),
                            "diagnostic": "zero_empirical_tolerance_unknown_resolution" if q95 == 0 and item["tau_kwh"] is None
                            else "positive_tolerance" if item["tau_kwh"] is not None else "insufficient_support"})
    pd.DataFrame(equality).to_csv(output / "interval_mean_loss_equality.csv", index=False)
    pd.DataFrame(tolerances).to_csv(output / "train_tolerance_audit.csv", index=False)
    if tolerances:
        report["calibration_provenance"] = "reconstructed_from_hash_verified_original_data_and_frozen_contract"
        data = pd.DataFrame(tolerances)
        report["zero_export_tau_strata"] = int(((data.direction == "export") & data.tau_kwh.isna()
                                               & (data.empirical_q95_kwh == 0)).sum())
        report["reconstructed_calibration_rows"] = len(data)
        report["mechanism"] = ("calibrate_tolerance retains only positive empirical q95/resolution values; "
            "a zero export q95 with no known resolution becomes tau_export=None even with ample support. "
            "The interval-mean wrapper requires both directional tolerances before forming net tau, "
            "so this blocks its constraint. Instantaneous correction can retain the admissible import direction. "
            "No tolerance policy or original result was changed.")
    report["outputs_sha256"] = {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != "diagnostic_validation.json"}
    write_json(output / "diagnostic_validation.json", report)
    print(json.dumps({"recorded_diagnostics": ledger, "calibration_rows": len(tolerances),
                      "output": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
