"""Pair all available pilot imputers with the same frozen downstream ridge.

This combines already-computed native outputs with fresh train-only references.
It is a one-fold, known-support integrity pilot, not a confirmatory experiment.
No neural/foundation model is trained or called by this script.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import pickle
import sqlite3
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from energygap.data import DatasetConfig, ingest
from energygap.data.provenance import sha256_file, write_immutable_json
from energygap.evaluation.reconstruction import reconstruction_metrics, training_scale
from energygap.methods.baselines import create_baseline
from energygap.pipeline import BASELINES, code_hash
from energygap.protocols.splits import rolling_origin_splits
from energygap.protocols.windows import match_future_target
from energygap.runtime.checkpoints import CheckpointStore
from energygap.runtime.scheduler import run_units
from energygap.runtime.state import StateStore, atomic_bytes, atomic_json, fingerprint


LEARNED = ("BRITS", "SAITS", "CSDI", "ImputeFormer", "T1", "MOMENT-1-base", "TS-ICL")


def read_ledger(path):
    with sqlite3.connect(f"file:{path/'state.sqlite'}?mode=ro", uri=True) as db:
        rows = [json.loads(payload) for payload, in db.execute("SELECT payload FROM results ORDER BY result_key")]
        config = json.loads(db.execute("SELECT value FROM meta WHERE key='config'").fetchone()[0])
    return rows, config


def safe_query_input(point, available_mask):
    """Recover received values only; no unavailable model output becomes input."""
    point = np.asarray(point, float); available = np.asarray(available_mask, bool)
    if point.shape != (512,) or available.shape != point.shape:
        raise ValueError("Expected matching L512 point/available arrays")
    if not np.isfinite(point[available]).all():
        raise ValueError("Originally received support contains nonfinite values")
    return np.where(available, point, np.nan)


def verified_array(path, checksum):
    if sha256_file(path) != checksum:
        raise ValueError("Committed prediction checksum mismatch: "+str(path))
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def load_inputs(root, source_id, ridge_id, tsicl_id, moment_audit_id):
    source = root/"runs"/source_id
    records, config = read_ledger(source)
    source_rows = [row for row in records if row.get("phase") == "inference"]
    moment_rows = sorted([row for row in source_rows if row["method"] == "MOMENT-1-base"],
                         key=lambda row: (row["site"], row["partition"], row["condition"], row["query_id"]))
    expected = {(s, p, c) for s in ("H1", "H2", "I1") for p in ("validation", "test") for c in ("C3", "C4", "C5")}
    if len(moment_rows) != 18 or {(row["site"], row["partition"], row["condition"]) for row in moment_rows} != expected:
        raise ValueError("Source must contain the exact predefined 18 query cells")
    cases = {}; source_files = {}
    for site in ("H1", "H2", "I1"):
        if sha256_file(root/"configs/datasets"/f"{site}.yaml") != config["inputs"][site]["dataset_config_sha256"]:
            raise ValueError("Dataset contract changed after native model pilot")
    for row in moment_rows:
        if row["status"] != "success":
            raise ValueError("Received-value provenance requires successful observed preservation")
        path = source/row["artifact"]
        arrays = verified_array(path, row["sha256"])
        safe = safe_query_input(arrays["point"], arrays["available_mask"])
        target = np.asarray(arrays["target_gap_mask"], bool)
        if np.any(target & arrays["available_mask"]):
            raise ValueError("Hidden target is still marked available")
        cases[row["query_id"]] = {"metadata": row, "values": safe, "times": arrays["times"],
            "available": arrays["available_mask"].astype(bool), "target": target,
            "safe_input_sha256": hashlib.sha256(safe.tobytes()+arrays["times"].tobytes()+arrays["available_mask"].tobytes()).hexdigest()}
        source_files[str(path.relative_to(root))] = row["sha256"]
    learned = {}
    for row in source_rows:
        if row["method"] == "MOMENT-1-base":
            continue  # Shared-environment override output is deliberately superseded.
        if row["method"] not in LEARNED or row["status"] != "success":
            raise ValueError("Unexpected or failed learned source row")
        arrays = verified_array(source/row["artifact"], row["sha256"])
        case = cases[row["query_id"]]
        if not np.array_equal(arrays["times"], case["times"]) or not np.array_equal(arrays["available_mask"], case["available"]):
            raise ValueError("Native source changed query support")
        if not np.array_equal(arrays["point"][case["available"]], case["values"][case["available"]]):
            raise ValueError("Native source did not preserve original observations")
        learned[(row["query_id"], row["method"])] = {"point": arrays["point"], "record": row,
            "prediction_path": str((source/row["artifact"]).relative_to(root)), "sha256": row["sha256"],
            "variant": "official_source_bounded_50steps" if row["method"] == "T1" else "native_PyPOTS_2epochs_16trainwindows",
            "environment_id": fingerprint(config["packages"]), "environment_lock": "environments/neural.lock.txt",
            "point_estimator": "sample_median_10" if row["method"] == "CSDI" else "native_deterministic_point",
            "capability": "joint_samples" if row["method"] == "CSDI" else "deterministic"}
        source_files[str((source/row["artifact"]).relative_to(root))] = row["sha256"]
    audit = source/"audits"/moment_audit_id
    worker = json.loads((audit/"worker-result.json").read_text())
    isolated = verified_array(audit/"predictions.npz", worker["output_sha256"])
    isolated_inputs = verified_array(audit/"input.npz", worker["input_identity"]["input_sha256"])
    moment_manifest = json.loads((audit/"manifest.json").read_text())
    if set(isolated["query_ids"].tolist()) != set(cases):
        raise ValueError("Official-pin MOMENT support differs")
    if worker["environment"]["dependency_override"] or worker["native_parity_max_abs_error"] != 0:
        raise ValueError("MOMENT requires successful official-pin native-parity audit")
    for index, query_id in enumerate(isolated["query_ids"].tolist()):
        point = isolated["point_estimate"][index]; case = cases[query_id]
        if isolated_inputs["query_ids"].tolist()[index] != query_id or not np.array_equal(
                isolated_inputs["values"][index], case["values"], equal_nan=True) or not np.array_equal(
                isolated_inputs["observed_mask"][index], case["available"]) or not np.array_equal(
                isolated_inputs["times"][index], case["times"]):
            raise ValueError("Isolated MOMENT admitted input differs from native pilot")
        if not np.array_equal(point[case["available"]], case["values"][case["available"]]):
            raise ValueError("Isolated MOMENT changed received observations")
        learned[(query_id, "MOMENT-1-base")] = {"point": point,
            "prediction_path": str((audit/"predictions.npz").relative_to(root)), "sha256": worker["output_sha256"],
            "variant": "frozen_native_official_pins_isolated", "environment_id": fingerprint(worker["environment"]),
            "environment_lock": "environments/moment.lock.txt", "point_estimator": "native_reconstruction",
            "capability": "deterministic", "record": {"inference_seconds": worker["inference_seconds"][index]}}
    source_files[str((audit/"predictions.npz").relative_to(root))] = worker["output_sha256"]
    tsicl = root/"runs"/tsicl_id
    tsicl_records, tsicl_config = read_ledger(tsicl)
    tsicl_rows = [row for row in tsicl_records if row.get("phase") == "inference"]
    if len(tsicl_rows) != 18 or {row["query_id"] for row in tsicl_rows} != set(cases):
        raise ValueError("TS-ICL support differs; no silent intersection is allowed")
    for row in tsicl_rows:
        if row["status"] != "success":
            raise ValueError("TS-ICL source inference failed")
        arrays = verified_array(tsicl/row["artifact"], row["sha256"])
        case = cases[row["query_id"]]; point = arrays["point_estimate"]
        safe_sha = hashlib.sha256(np.nan_to_num(case["values"], nan=0).tobytes()+
            case["available"].tobytes()+case["times"].astype(np.float64).tobytes()).hexdigest()
        if safe_sha != row["safe_input_sha256"]:
            raise ValueError("TS-ICL worker admitted input/masks/times differ from v2 pilot")
        if not np.array_equal(point[case["available"]], case["values"][case["available"]]):
            raise ValueError("TS-ICL changed received values")
        level = int(np.flatnonzero(np.isclose(arrays["quantile_levels"], .5))[0])
        if not np.array_equal(point[~case["available"]], arrays["quantiles"][~case["available"], level]):
            raise ValueError("TS-ICL primary point is not its native q0.5")
        learned[(row["query_id"], "TS-ICL")] = {"point": point, "record": row,
            "prediction_path": str((tsicl/row["artifact"]).relative_to(root)), "sha256": row["sha256"],
            "variant": "native_imputation_frozen_noncommercial_isolated", "environment_id": sha256_file(root/"env/tsicl.lock"),
            "environment_lock": "env/tsicl.lock", "point_estimator": "native_quantile_0.5", "capability": "marginal_quantiles"}
        source_files[str((tsicl/row["artifact"]).relative_to(root))] = row["sha256"]
    if len(learned) != 18*7:
        raise ValueError("Incomplete 7-model shared-support collection")
    ridge_rows, ridge_config = read_ledger(root/"runs"/ridge_id)
    ridge = {}
    for site in ("H1", "H2", "I1"):
        row, = [row for row in ridge_rows if row.get("key") == f"{site}:fold1:fit"]
        path = root/"runs"/ridge_id/row["artifact"]
        if sha256_file(path) != row["sha256"]:
            raise ValueError("Frozen forecaster checkpoint differs from committed checksum")
        with path.open("rb") as stream:
            payload = pickle.load(stream)  # Trusted own local checkpoint, checksum verified.
        if payload["alpha"] != row["alpha"] or payload["forecast"].horizon_steps != 12:
            raise ValueError("Frozen forecaster identity/horizon mismatch")
        ridge[site] = {"model": payload["forecast"], "sha256": row["sha256"], "path": str(path.relative_to(root)),
                       "alpha": payload["alpha"], "scale": payload["scale"]}
        source_files[str(path.relative_to(root))] = row["sha256"]
    issuance = {row["query_id"]: row["issuance_time_epoch"] for row in moment_manifest["queries"]}
    for query_id, case in cases.items():
        if issuance[query_id] != case["times"][-1]:
            raise ValueError("Selected query has right-boundary jitter; explicit availability reconstruction required")
        case["issuance"] = issuance[query_id]
    return cases, learned, ridge, source_files


def report(store, root):
    out = store.path/"reports"; out.mkdir(exist_ok=True)
    rows = [row for row in store.results() if row.get("phase") == "evaluation"]
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No paired pilot evaluations committed")
    frame.to_csv(out/"paired_episode_metrics.csv", index=False)
    frame.to_parquet(out/"paired_episode_metrics.parquet", index=False)
    grouping = ["site", "partition", "method", "variant", "environment_id"]
    summary = frame.groupby(grouping, dropna=False).agg(episodes=("query_id", "count"),
        gap_macro_mae_kw=("mae_kw", "mean"), gap_macro_nmae=("nmae", "mean"),
        forecast_mae_kw=("forecast_mae_kw", "mean"), forecast_nmae=("forecast_nmae", "mean"),
        forecast_degradation_vs_original_history_kw=("forecast_degradation_kw", "mean")).reset_index()
    summary.to_csv(out/"paired_summary.csv", index=False)
    reference = frame[frame.method == "locf"][["query_id", "mae_kw", "forecast_mae_kw"]]
    paired = frame.merge(reference, on="query_id", suffixes=("", "_locf"))
    paired["reconstruction_gain_vs_locf_kw"] = paired.mae_kw_locf-paired.mae_kw
    paired["forecast_gain_vs_locf_kw"] = paired.forecast_mae_kw_locf-paired.forecast_mae_kw
    paired.to_csv(out/"paired_differences_vs_locf.csv", index=False)
    details = {"schema_version": 1, "experiment_id": store.path.name, "scope": "one_fold_real_paired_integrity_pilot_exploratory",
        "conference_complete": False, "confirmatory": False, "model_selection_on_test": False,
        "n_methods": int(frame.method.nunique()), "n_queries": int(frame.query_id.nunique()),
        "n_reconstruction_rows": len(frame), "n_forecast_rows": int(frame.forecast_mae_kw.notna().sum()),
        "n_forecast_queries": int(frame[frame.forecast_mae_kw.notna()].query_id.nunique()),
        "same_13_methods_per_query": bool(frame.groupby("query_id").method.nunique().eq(13).all()),
        "same_support_across_methods": True, "learned_models_retrained": False,
        "MOMENT_variant": "isolated official pins; shared override outputs excluded from evaluated rows",
        "TSICL_capability": "marginal quantiles; q0.5 primary; no joint sample claim",
        "forecaster": "existing checksum-verified fold1 direct ridge, identical weights across methods",
        "forecast_horizon_seconds": 3600, "matching_tolerance_seconds": 90,
        "forecast_partition": "test only; validation forecasts NA by design",
        "normalization": "q95-q05 from fold1 training only; checked against frozen forecast source",
        "aggregation": "one episode per C3/C4/C5 -> equal duration weights1/3 within site/partition",
        "exclusions": frame[["query_id", "forecast_exclusion_reason"]].drop_duplicates().astype(object).where(
            pd.notna(frame[["query_id", "forecast_exclusion_reason"]].drop_duplicates()), None).to_dict("records"),
        "limitations": ["Known18querysupport, one fold, one mask seed per partition; no population/general superiority or convergence claim.",
            "PyPOTS2epochs/16windows; T1 official source50steps with documented schedule/mask adaptation.",
            "TS-ICL code/checkpoint/outputs retain noncommercial restrictions; no redistribution or upload.",
            "No energy correction: real power and timestamp interval semantics are unresolved.",
            "LOCF is a fixed descriptive paired reference, not selected using test.",
            "Latency across isolated environments is not combined into a hardware-fair ranking."],
        "runtime": store.status()}
    atomic_json(out/"paired_pilot.json", details)
    atomic_json(root/"artifacts/reports/paired_integrity_pilot.json", details)
    return details


def run(root=Path("."), *, source_id="real-models-pilot-v2", ridge_id="pilot-083764a78be2ffec",
        tsicl_id="tsicl-real-pilot-40e413ace2709b80", moment_audit_id="moment-isolated-official-pins-v1",
        experiment_id=None, resume="auto", progress="none", budget_total_hours=.05):
    root = Path(root).resolve()
    cases, learned, ridge, source_files = load_inputs(root, source_id, ridge_id, tsicl_id, moment_audit_id)
    dataset_files = {site: sha256_file(root/"configs/datasets"/f"{site}.yaml") for site in ("H1", "H2", "I1")}
    config = {"schema_version": 1, "scope": "paired_known_support_integrity_pilot", "script_sha256": sha256_file(__file__),
        "code_sha256": code_hash(),
        "sources": source_files, "datasets": dataset_files, "query_inputs": {qid: case["safe_input_sha256"] for qid, case in cases.items()},
        "methods": [*BASELINES, *LEARNED], "task": "R", "information": "P", "context": 512,
        "fold": 1, "forecast_seconds": 3600, "forecast_tolerance_seconds": 90, "forecast_partition": "test",
        "reconstruction_reference": "locf_fixed", "budget_total_hours": budget_total_hours, "progress": progress}
    identity = fingerprint(config)
    run_path = root/"runs"/(experiment_id or "paired-integrity-"+identity[:16])
    with StateStore(run_path, config, resume=resume, budget_seconds=budget_total_hours*3600) as store:
        if store.get_meta("status") in ("completed", "completed_with_failures"):
            return report(store, root)
        write_immutable_json(run_path/"protocol.lock.json", store.get_meta("config"))
        atomic_bytes(run_path/"protocol.lock.sha256", (identity+"\n").encode())
        write_immutable_json(run_path/"paired_support.json", {"queries": [{"query_id": qid,
            "site": case["metadata"]["site"], "partition": case["metadata"]["partition"],
            "condition": case["metadata"]["condition"], "safe_input_sha256": case["safe_input_sha256"],
            "target_count": int(case["target"].sum()), "issuance_epoch": case["issuance"]}
            for qid, case in cases.items()], "methods": [*BASELINES, *LEARNED],
            "truth_path_policy": "separate evaluator files read only after all baseline inference shards committed"})
        atomic_json(run_path/"plan.json", {"fit_baseline_jobs": 3, "baseline_query_jobs": 18,
            "baseline_output_rows": 108, "evaluate_rows": 234, "learned_model_retraining": False,
            "budget_seconds": budget_total_hours*3600, "cost_estimate": "bounded lightweight baseline evaluation; no neural calls"})
        for site in ("H1", "H2", "I1"):
            store.add_job("fit:"+site, 1, metadata={"phase": "fit_baselines", "site": site})
            store.add_job("baseline-inference:"+site, 6, dependencies=["fit:"+site], metadata={"phase": "baseline_inference", "site": site})
        store.add_job("evaluate", 234, dependencies=["baseline-inference:"+site for site in ("H1", "H2", "I1")], metadata={"phase": "evaluation"})
        store.start_session()
        checkpoints = CheckpointStore(run_path, identity, ledger=store)
        datasets = {}; models = {}
        for site in ("H1", "H2", "I1"):
            series = ingest(DatasetConfig.load(root/"configs/datasets"/f"{site}.yaml"), root=root, purpose="statistical")
            fold = rolling_origin_splits(series)[0]; train = fold.indices(series, "train")
            values = np.asarray(series.values); times = np.asarray(series.times)
            scale, reason = training_scale(values[train])
            if not np.isclose(scale, ridge[site]["scale"], rtol=1e-12, atol=1e-12):
                raise ValueError("Raw training normalizer differs from frozen forecaster source")
            datasets[site] = {"series": series, "fold": fold, "values": values, "times": times, "scale": scale,
                              "by_grid": {row.grid_time.timestamp(): row for row in series.records}}
            for case in cases.values():
                if case["metadata"]["site"] != site:
                    continue
                received = np.array([datasets[site]["by_grid"][float(time)].value_canonical for time in case["times"]])
                if not np.array_equal(received[case["available"]], case["values"][case["available"]]):
                    raise ValueError("Raw original received values differ from saved query provenance")
            job = "fit:"+site
            if store.job(job)["status"] != "succeeded":
                def fit(index, seed, site=site, values=values, times=times, train=train, job=job):
                    tic = perf_counter()
                    trained = {name: create_baseline(name).fit(values[train], times[train], split="train") for name in BASELINES}
                    manifest = checkpoints.save(job, {"models": trained, "scale": scale}, 1)
                    directory = checkpoints._job_dir(job)/manifest["generation"]; artifact = directory/"state.pkl"
                    models[site] = trained
                    return {"key": job, "phase": "fit_baselines", "site": site, "status": "success",
                        "fit_partition": "fold1_train_elapsed40pct", "fit_seconds": perf_counter()-tic,
                        "artifact": str(artifact.relative_to(run_path)), "sha256": manifest["files"]["state.pkl"]["sha256"]}
                run_units(store, job, fit, shard_size=1, progress=progress, signature="six_baselines_train_fit")
            else:
                saved = checkpoints.load(job)
                if saved is None:
                    raise RuntimeError("Confirmed baseline checkpoint is unavailable")
                models[site] = saved[0]["models"]
            site_cases = [(qid, case) for qid, case in cases.items() if case["metadata"]["site"] == site]
            def infer(index, seed, site=site):
                qid, case = site_cases[index]; result = []
                for name, model in models[site].items():
                    tic = perf_counter(); point = model.impute(case["values"], case["times"], task="R")
                    if not np.array_equal(point[case["available"]], case["values"][case["available"]]):
                        raise ValueError("Baseline changed a received observation")
                    buffer = io.BytesIO(); np.savez_compressed(buffer, point=point)
                    artifact = run_path/"predictions"/(hashlib.sha256((name+qid).encode()).hexdigest()+".npz")
                    checksum = atomic_bytes(artifact, buffer.getvalue())
                    result.append({"key": "baseline:"+name+":"+qid, "phase": "baseline_inference", "method": name,
                        "site": site, "query_id": qid, "status": "success", "inference_seconds": perf_counter()-tic,
                        "artifact": str(artifact.relative_to(run_path)), "sha256": checksum})
                return result
            run_units(store, "baseline-inference:"+site, infer, shard_size=1, progress=progress, signature="six_baselines_RP_L512")
        # Every baseline inference result is now committed. Open evaluator truth only below.
        baseline_outputs = [row for row in store.results() if row.get("phase") == "baseline_inference"]
        for row in baseline_outputs:
            arrays = verified_array(run_path/row["artifact"], row["sha256"])
            learned[(row["query_id"], row["method"])] = {"point": arrays["point"], "record": row,
                "prediction_path": str((run_path/row["artifact"]).relative_to(root)), "sha256": row["sha256"],
                "variant": "train_only_reference", "environment_id": "baseline-shared-numpy-"+np.__version__,
                "environment_lock": "uv.lock", "point_estimator": "native_deterministic_point", "capability": "deterministic"}
        work = sorted(learned)
        if len(work) != 234:
            raise ValueError("Shared evaluation support is incomplete")
        evaluator = {}; forecast_targets = {}
        for qid, case in cases.items():
            site = case["metadata"]["site"]; data = datasets[site]
            truth_path = root/"data/evaluation_truth"/source_id/(hashlib.sha256(qid.encode()).hexdigest()+".json")
            truth = json.loads(truth_path.read_text())
            if truth["query_id"] != qid:
                raise ValueError("Truth query identity mismatch")
            y = np.asarray([np.nan if value is None else value for value in truth["values"]])
            mask = np.asarray(truth["evaluation_mask"], bool)
            if not np.array_equal(mask, case["target"]) or np.any(mask & case["available"]):
                raise ValueError("Evaluation support differs across methods")
            evaluator[qid] = {"truth": y, "mask": mask, "sha256": sha256_file(truth_path)}
            zone = ZoneInfo(data["series"].metadata["config"]["timestamp"]["timezone"])
            issuance = datetime.fromtimestamp(case["issuance"], zone)
            if case["metadata"]["partition"] == "test":
                target, reason = match_future_target(data["series"], issuance, 3600, tolerance_seconds=90, fold=data["fold"])
            else:
                target, reason = None, "validation_not_downstream_test_endpoint"
            reference = []
            for time in case["times"]:
                row = data["by_grid"].get(float(time))
                reference.append(row.value_canonical if row is not None and row.observed_original and
                    not row.natural_missing and row.available_to_method and row.availability_time <= issuance else np.nan)
            if not np.array_equal(np.asarray(reference)[mask], y[mask]):
                raise ValueError("Separate evaluation truth differs from original raw target observations")
            reference_forecast = ridge[site]["model"].predict(reference[-288:], case["issuance"])
            reference_error = abs(reference_forecast-target["value"]) if target is not None else None
            forecast_targets[qid] = {"target": target, "reason": reason, "reference_error": reference_error,
                "reference_forecast": reference_forecast, "issuance": issuance.isoformat()}
        def evaluate(index, seed):
            qid, method = work[index]; source = learned[(qid, method)]; case = cases[qid]
            meta = case["metadata"]; site = meta["site"]; scoring = evaluator[qid]
            point = np.asarray(source["point"], float)
            if point.shape != (512,) or not np.isfinite(point[scoring["mask"]]).all():
                raise ValueError("Point source has invalid primary output")
            metrics = reconstruction_metrics(scoring["truth"], point, scoring["mask"], scale_train=datasets[site]["scale"])
            downstream = forecast_targets[qid]
            prediction = ridge[site]["model"].predict(point[-288:], case["issuance"])
            target = downstream["target"]
            forecast_error = abs(prediction-target["value"]) if target is not None else None
            return {"key": "paired:"+method+":"+qid, "phase": "evaluation", "status": "success", "site": site,
                "query_id": qid, "gap_id": meta["gap_id"], "method": method, "variant": source["variant"],
                "environment_id": source["environment_id"], "environment_lock": source["environment_lock"],
                "point_estimator": source["point_estimator"], "probabilistic_capability": source["capability"],
                "partition": meta["partition"], "condition": meta["condition"], "duration": meta["duration"],
                "fold": 1, "task": "R", "information_tier": "P", "context": 512, "mask_seed": meta["mask_seed"],
                "training_seed": 11 if method in ("BRITS", "SAITS", "CSDI", "ImputeFormer", "T1") else None,
                "epochs_requested": 2 if method in ("BRITS", "SAITS", "CSDI", "ImputeFormer") else None,
                "optimizer_steps_requested": 50 if method == "T1" else None,
                "issuance_time": downstream["issuance"], "safe_input_sha256": case["safe_input_sha256"],
                "source_prediction_path": source["prediction_path"], "source_prediction_sha256": source["sha256"],
                "evaluation_truth_sha256": scoring["sha256"], "forecaster_sha256": ridge[site]["sha256"],
                "forecaster_path": ridge[site]["path"], "forecaster_alpha": ridge[site]["alpha"],
                **metrics, "forecast_mae_kw": forecast_error,
                "forecast_nmae": forecast_error/datasets[site]["scale"] if forecast_error is not None else None,
                "forecast_prediction_kw": prediction if target is not None else None,
                "forecast_reference_mae_kw": downstream["reference_error"],
                "forecast_degradation_kw": forecast_error-downstream["reference_error"] if forecast_error is not None else None,
                "forecast_target_measurement_time": target["measurement_time"].isoformat() if target is not None else None,
                "forecast_target_offset_seconds": target["offset_seconds"] if target is not None else None,
                "forecast_exclusion_reason": downstream["reason"], "fallback": None}
        run_units(store, "evaluate", evaluate, shard_size=13, progress=progress, signature="paired_reconstruction_frozen_ridge")
        store.complete()
        return report(store, root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--source-id", default="real-models-pilot-v2")
    parser.add_argument("--ridge-id", default="pilot-083764a78be2ffec")
    parser.add_argument("--tsicl-id", default="tsicl-real-pilot-40e413ace2709b80")
    parser.add_argument("--moment-audit-id", default="moment-isolated-official-pins-v1")
    parser.add_argument("--experiment-id")
    parser.add_argument("--resume", choices=("auto", "require", "never"), default="auto")
    parser.add_argument("--progress", choices=("none", "log", "tqdm"), default="none")
    parser.add_argument("--budget-total-hours", type=float, default=.05)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({key: result[key] for key in ("experiment_id", "n_methods", "n_queries",
        "n_reconstruction_rows", "n_forecast_rows", "confirmatory")}, indent=2))


if __name__ == "__main__":
    main()
