"""Small REAL-data integration pilot, deliberately separate from conference claims.

No physics is inferred from unresolved power/timestamp semantics. Fits retain
native PyPOTS losses. Interrupted native fits restart from their beginning; only
completed fits and completed query shards are resumable in this script.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import PackageNotFoundError, version
import io
import json
from pathlib import Path
import random
from time import perf_counter
import traceback

import numpy as np
import pandas as pd

from energygap.data import DatasetConfig, ingest
from energygap.data.provenance import sha256_file, write_immutable_json
from energygap.evaluation.probabilistic import summarize_samples
from energygap.evaluation.reconstruction import reconstruction_metrics, training_scale
from energygap.masking.generators import generate_mask
from energygap.pipeline import code_hash
from energygap.protocols.availability import build_query
from energygap.protocols.splits import rolling_origin_splits
from energygap.runtime.checkpoints import CheckpointStore
from energygap.runtime.scheduler import run_units
from energygap.runtime.signals import StopRequested
from energygap.runtime.device import resolve_device, torch_device_metadata, synchronize
from energygap.runtime.threads import requested_threads
from energygap.runtime.state import (BudgetExhausted, StateStore, atomic_bytes,
    atomic_json, fingerprint, stable_seed)


PYPOTS_METHODS = ("BRITS", "SAITS", "CSDI", "ImputeFormer")
LOCAL_METHODS = (*PYPOTS_METHODS, "T1")
METHODS = (*LOCAL_METHODS, "MOMENT-1-base")


def package_versions():
    result = {}
    for name in ("numpy", "pypots", "torch", "momentfm", "transformers"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def training_windows(series, fold, count=16, context=512):
    """Evenly spaced chronological train-only windows; natural gaps stay missing."""
    indices = fold.indices(series, "train")
    if len(indices) < context:
        raise ValueError("Insufficient training history for L=512")
    admitted = np.asarray([r.value_canonical if r.observed_original and
        not r.natural_missing and r.available_to_method and r.availability_time < fold.train_end
        else np.nan for r in series.records], dtype=np.float32)
    eligible = [i for i in indices if i >= context-1 and
        fold.partition(series.records[i-context+1].grid_time) == "train" and
        np.isfinite(admitted[i-context+1:i+1]).sum() >= context//2]
    if len(eligible) < count:
        raise ValueError("Too few training windows with at least 50% original support")
    anchors = np.asarray(eligible)[np.linspace(0, len(eligible)-1, count, dtype=int)]
    windows = np.stack([admitted[i-context+1:i+1] for i in anchors])[:, :, None]
    manifest = {"partition": "train", "selection": "evenly_spaced_chronological_eligible_anchors",
        "context": context, "count": count, "minimum_original_fraction": .5,
        "end_record_ids": [series.records[i].source_record_id for i in anchors],
        "end_times": [series.records[i].grid_time.isoformat() for i in anchors],
        "values_sha256": hashlib.sha256(windows.tobytes()).hexdigest(),
        "natural_missing_not_supervision": True}
    return windows, manifest


def prepare(root, sites, windows):
    datasets, cases, inputs, blocked = {}, {}, {}, []
    for site in sites:
        try:
            cfgpath = root/"configs/datasets"/f"{site}.yaml"
            config = DatasetConfig.load(cfgpath)
            config.validate(purpose="statistical")
            series = ingest(config, root=root, purpose="statistical")
            fold = rolling_origin_splits(series)[0]
            train, train_manifest = training_windows(series, fold, windows)
            scale, scale_reason = training_scale(np.asarray(series.values)[fold.indices(series, "train")])
            site_cases = []
            masks = []
            for partition, seed in (("validation", 1001), ("test", 2001)):
                for condition in ("C3", "C4", "C5"):
                    manifest = generate_mask(series, fold, partition=partition,
                        condition=condition, seed=seed, max_episodes=1, physical=False)
                    masks.append(manifest.to_dict())
                    for episode in manifest.episodes:
                        query, truth = build_query(series, episode, information_tier="P",
                            context_length=512, artificial_hidden=manifest.hidden_indices)
                        site_cases.append((query, truth, episode))
            inputs[site] = {"dataset_config_sha256": sha256_file(cfgpath),
                "sources": [{"path": entry["path"], "sha256": entry["sha256"]}
                            for entry in config.data["files"]],
                "split": fold.manifest(), "training_windows": train_manifest, "masks": masks}
            datasets[site] = {"train": train, "scale": scale, "scale_reason": scale_reason,
                "physical_semantics_confirmed": False, "metadata": series.metadata}
            cases[site] = site_cases
        except (ValueError, FileNotFoundError, TypeError) as error:
            blocked.append({"site": site, "status": "blocked", "reason": str(error)})
    return datasets, cases, inputs, blocked


def checkpoint_record(checkpoints, job, payload, step, run):
    manifest = checkpoints.save(job, payload, step=step)
    directory = checkpoints._job_dir(job)/manifest["generation"]
    path = directory/"state.pkl"
    return {"artifact": str(path.relative_to(run)), "sha256": manifest["files"]["state.pkl"]["sha256"],
            "checkpoint_generation": manifest["generation"], "checkpoint_step": step}


def native_fit(name, train, *, epochs, seed, source_root=None, device='cpu'):
    device = resolve_device(device, allow_missing_torch=False)
    if name == "T1":
        from energygap.methods.t1 import T1Adapter
        model = T1Adapter(context=512, n_features=1, seed=seed, source_root=source_root, device=device)
        model.fit_windows(train, split="train", steps=50)
        module, optimizer = model.model, None
        parameters, elapsed = vars(model.config), model.metadata["fit_seconds"]
    else:
        from energygap.methods.neural import PyPOTSAdapter
        model = PyPOTSAdapter(name, context=512, n_features=1, epochs=epochs, seed=seed, device=device)
        model.fit_windows(train, split="train")
        module, optimizer = model.model.model, model.model.optimizer.state_dict()
        parameters, elapsed = model.parameters, model.timings["fit_seconds"]
    import torch
    payload = {"method": name, "mean": model.mean, "scale": model.scale,
        "parameters": parameters, "epochs_requested": epochs if name != "T1" else None,
        "optimizer_steps_requested": 50 if name == "T1" else None, "seed": seed,
        "source_root": str(source_root) if name == "T1" else None,
        "weights": {key: tensor.detach().cpu().clone() for key, tensor in module.state_dict().items()},
        "optimizer_at_native_fit_end": optimizer,
        "rng_at_native_fit_end": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if device.startswith('cuda') else []},
        "device": device, "runtime": torch_device_metadata(device),
        "supports_in_progress_training_resume": False,
        "weights_are_native_selected_best_but_optimizer_is_final_epoch": True,
        "native_fit_seconds": elapsed,
        "training_variant": "official_T1_source_bounded_schedule_adapted" if name == "T1" else "native_PyPOTS_fit_original_losses"}
    return model, payload


def load_native(payload, *, device='cpu'):
    device = resolve_device(device, allow_missing_torch=False)
    if payload.get('device', 'cpu') != device:
        raise ValueError('Completed native fit device differs; use a new experiment for another runtime')
    if payload["method"] == "T1":
        from energygap.methods.t1 import T1Adapter
        model = T1Adapter(context=512, n_features=1, seed=payload["seed"], source_root=payload["source_root"], device=device,
                          **{key: value for key, value in payload["parameters"].items()
                             if key not in ("seq_len", "pred_len", "enc_in")})
        model.mean, model.scale = payload["mean"], payload["scale"]
        model.model.load_state_dict(payload["weights"]); model.model.eval()
        return model
    from energygap.methods.neural import PyPOTSAdapter
    model = PyPOTSAdapter(payload["method"], context=512, n_features=1,
        epochs=payload["epochs_requested"], seed=payload["seed"], device=device)
    model.mean, model.scale = payload["mean"], payload["scale"]
    model.model.model.load_state_dict(payload["weights"])
    model.model.model.eval()
    return model


def write_report(store, root):
    rows = store.results()
    inference = [row for row in rows if row.get("phase") == "inference"]
    fit = [row for row in rows if row.get("phase") in ("fit", "load")]
    path = store.path/"reports"
    path.mkdir(exist_ok=True)
    if inference:
        pd.DataFrame(inference).drop(columns=["probabilistic"], errors="ignore").to_csv(path/"model_pilot_records.csv", index=False)
    report = {"schema_version": 1, "experiment_id": store.path.name,
        "scope": "real_data_integration_pilot_not_converged_scientific_comparison",
        "conference_complete": False, "task": "R", "information_tier": "P", "context": 512,
        "folds": [1], "native_training_epochs_requested": store.get_meta("config")["epochs"],
        "native_training_seed": 11, "T1_optimizer_steps_requested": 50,
        "CSDI_samples": 10, "native_diffusion_steps": 50,
        "physical_results": "not_run_unconfirmed_timestamp_and_power_semantics",
        "forecast": "not_run_in_this_supplementary_integration_pilot",
        "runtime": store.status(), "device": store.get_meta("config")["device"],
        "torch_runtime": store.get_meta("config").get("torch_runtime"), "fit_and_load": fit,
        "n_episode_method_records": len(inference),
        "n_success": sum(row["status"] == "success" for row in inference),
        "n_failed": sum(row["status"] != "success" for row in inference),
        "unavailable_or_failed_jobs": [{"job_id": job["job_id"], "status": job["status"], "reason": job["reason"]}
            for job in store.status()["jobs"] if job["status"] in ("blocked", "unavailable", "unsupported", "failed_terminal")],
        "limitations": ["Two epochs and 16/32 windows validate integration, not convergence or model superiority.",
            "Only fold 1, one mask seed per partition and at most one episode per duration.",
            "Native PyPOTS fit is an indivisible recovery unit; interrupted fit repeats; completed fit and inference shards persist.",
            "No operational fallback used in this integrity supplement; failures remain explicit and no confirmatory ranking is computed.",
            "Frozen MOMENT is loaded once per process and shared across sites; local training is not_applicable.",
            "Indivisible native fit/query can overrun the cumulative budget before the next safe boundary; no hard per-unit timeout."]}
    atomic_json(path/"models_pilot.json", report)
    atomic_json(root/"artifacts/reports/models_pilot.json", report)
    return report


def run(root=Path("."), *, experiment_id=None, resume="auto", progress="log",
        windows=16, epochs=2, budget_total_hours=.1, sites=("H1", "H2", "I1"), methods=METHODS,
        dry_run=False, device='auto'):
    root = Path(root).resolve()
    device = resolve_device(device, allow_missing_torch=False)
    import torch
    torch.use_deterministic_algorithms(True)
    torch_runtime = torch_device_metadata(device)
    datasets, cases, inputs, blocked = prepare(root, sites, windows)
    config = {"schema_version": 1, "scope": "optional_real_models_integrity_pilot",
        "code_sha256": code_hash(), "script_sha256": sha256_file(__file__),
        "inputs": inputs, "blocked": blocked, "sites": list(sites), "methods": list(methods),
        "context": 512, "folds": [1], "conditions": ["C3", "C4", "C5"],
        "training_windows": windows, "epochs": epochs, "training_seed": 11,
        "sampling_samples": 10, "diffusion_steps": 50, "device": device, "torch_runtime": torch_runtime, "threads": requested_threads(),
        "packages": package_versions(), "information": "P", "physics": "disabled",
        "budget_total_hours": budget_total_hours, "progress": progress}
    identity = fingerprint(config)
    experiment_id = experiment_id or "models-pilot-"+identity[:16]
    run_path = root/"runs"/experiment_id
    local = [name for name in methods if name in LOCAL_METHODS]
    has_moment = "MOMENT-1-base" in methods
    plan = {"experiment_id": experiment_id, "scientific_fingerprint": identity,
        "profile": "incomplete_integrity_pilot", "n_fit_jobs": len(datasets)*len(local),
        "n_shared_frozen_load_jobs": int(has_moment),
        "n_inference_queries": sum(len(v) for v in cases.values())*len(methods),
        "n_optimizer_steps_upper_planned": len(datasets)*(
            sum(name in PYPOTS_METHODS for name in local)*epochs*int(np.ceil(windows/8))+
            (50 if "T1" in local else 0)),
        "budget_seconds": budget_total_hours*3600, "cost_estimate_status": "calibrating_L512_"+device+"_pilot",
        "estimated_seconds": None, "estimated_finish": None,
        "reason_no_ETA": "tiny L24 fixture timings are not an honest estimate for L512 diffusion/attention",
        "supports_native_in_progress_fit_resume": False, "blocked": blocked,
        "scope_reduction": "fold1; one episode per C3/C4/C5 and partition; seed11; 2epochs; no HPO; no forecast"}
    with StateStore(run_path, config, resume=resume, budget_seconds=budget_total_hours*3600) as store:
        if store.get_meta("status") in ("completed", "completed_with_failures"):
            return write_report(store, root)
        write_immutable_json(run_path/"plan.json", plan)
        write_immutable_json(run_path/"protocol.lock.json", store.get_meta("config"))
        atomic_bytes(run_path/"protocol.lock.sha256", (identity+"\n").encode())
        for site, details in inputs.items():
            write_immutable_json(run_path/"manifests"/f"{site}-fold1.json", details)
        for item in blocked:
            job = "data:"+item["site"]
            store.add_job(job, 1, metadata={"phase": "data", "site": item["site"]})
            store.mark(job, "blocked", item["reason"])
        if has_moment:
            store.add_job("load:MOMENT-1-base", 1, metadata={"phase": "load", "method": "MOMENT-1-base", "local_training": "not_applicable"})
        for site in datasets:
            for name in local:
                store.add_job(f"fit:{site}:{name}", 1, metadata={"phase": "fit", "site": site, "method": name, "epochs": epochs})
            for name in methods:
                dependency = f"fit:{site}:{name}" if name in local else "load:MOMENT-1-base"
                store.add_job(f"inference:{site}:{name}", len(cases[site]), dependencies=[dependency],
                    metadata={"phase": "inference", "site": site, "method": name, "task": "R", "information_tier": "P"})
        if dry_run:
            atomic_json(run_path/"progress.json", store.status())
            return {"plan": plan, "status": store.status()}
        store.start_session()
        checkpoints = CheckpointStore(run_path, identity, ledger=store)
        models = {}
        try:
            with StopRequested() as stop:
                if has_moment:
                    job = "load:MOMENT-1-base"
                    if store.job(job)["status"] not in ("succeeded", "unavailable", "failed_terminal"):
                        def load_frozen(index, seed):
                            tic = perf_counter()
                            try:
                                import torch
                                from energygap.runtime.threads import configure_torch_threads
                                configure_torch_threads(torch)
                                from energygap.methods.foundation import MOMENTFrozen, MOMENT_REVISION, MOMENT_SHA256
                                model = MOMENTFrozen(cache_dir=root/"artifacts/checkpoints/hf", device=device, allow_network=False)
                                model._load(); synchronize(device)
                                models["MOMENT-1-base"] = model
                                state = {"checkpoint": "AutonLab/MOMENT-1-base", "revision": MOMENT_REVISION,
                                    "weights_sha256": MOMENT_SHA256, "training": "not_applicable", "device": device,
                                    "runtime": torch_runtime}
                                artifact = checkpoint_record(checkpoints, job, state, 1, run_path)
                                return {"key": job, "phase": "load", "method": "MOMENT-1-base", "status": "success",
                                    "load_seconds": perf_counter()-tic, "training": "not_applicable", **artifact, **state}
                            except Exception as error:
                                return {"key": job, "phase": "load", "method": "MOMENT-1-base", "status": "unavailable",
                                    "reason": type(error).__name__+": "+str(error), "load_seconds": perf_counter()-tic}
                        run_units(store, job, load_frozen, shard_size=1, progress=progress, stop=stop, signature="MOMENT_cached_load:"+fingerprint(torch_runtime))
                        record = store.results(job)[0]
                        if record["status"] != "success":
                            store.mark(job, "unavailable", record["reason"])
                for site, data in datasets.items():
                    if stop.requested:
                        break
                    for name in methods:
                        if stop.requested:
                            break
                        fit_job = f"fit:{site}:{name}" if name in local else "load:MOMENT-1-base"
                        if name in local and store.job(fit_job)["status"] not in ("succeeded", "unavailable", "failed_terminal"):
                            def fit_native(index, seed, name=name, site=site, fit_job=fit_job):
                                tic = perf_counter()
                                try:
                                    model, payload = native_fit(name, data["train"], epochs=epochs, seed=11,
                                        source_root=root/"artifacts/upstream/T1", device=device)
                                    synchronize(device)
                                    models[(site, name)] = model
                                    step = 50 if name == "T1" else epochs*int(np.ceil(windows/8))
                                    artifact = checkpoint_record(checkpoints, fit_job, payload, step, run_path)
                                    return {"key": fit_job, "phase": "fit", "site": site, "method": name,
                                        "status": "success", "fit_seconds": perf_counter()-tic,
                                        "native_fit_seconds": payload["native_fit_seconds"],
                                        "epochs_requested": epochs if name != "T1" else None,
                                        "optimizer_steps_requested": 50 if name == "T1" else None,
                                        "training_variant": payload["training_variant"],
                                        "training_seed": 11, "training_windows": windows,
                                        "training_resume": "completed_fit_only", **artifact}
                                except Exception as error:
                                    return {"key": fit_job, "phase": "fit", "site": site, "method": name,
                                        "status": "unavailable" if isinstance(error, (ImportError, ModuleNotFoundError)) else "failed",
                                        "reason": type(error).__name__+": "+str(error), "traceback": traceback.format_exc(),
                                        "fit_seconds": perf_counter()-tic, "training_resume": "not_validated_in_progress"}
                            run_units(store, fit_job, fit_native, shard_size=1, progress=progress, stop=stop,
                                signature=f"native_fit:{name}:L512:windows{windows}:epochs{epochs}:"+fingerprint(torch_runtime))
                            completed = store.results(fit_job)
                            if completed and completed[0]["status"] != "success":
                                status = "unavailable" if completed[0]["status"] == "unavailable" else "failed_terminal"
                                store.mark(fit_job, status, completed[0]["reason"])
                        job = f"inference:{site}:{name}"
                        if store.job(fit_job)["status"] != "succeeded":
                            store.mark(job, "blocked", "Model dependency did not complete successfully: "+fit_job)
                            continue
                        if store.job(job)["status"] == "succeeded":
                            continue
                        def infer(index, unused_seed, name=name, site=site, fit_job=fit_job, job=job):
                            query, truth, episode = cases[site][index]
                            sample_seed = stable_seed(store.fingerprint, name, query.query_id, "joint_samples")
                            record = {"key": f"{job}:{query.query_id}", "phase": "inference", "site": site,
                                "method": name, "query_id": query.query_id, "gap_id": episode.gap_id,
                                "mask_seed": episode.mask_seed, "partition": episode.partition, "condition": episode.condition,
                                "duration": episode.duration_minutes, "task": "R", "information_tier": "P", "context": 512,
                                "training_seed": 11 if name in local else None,
                                "sample_seed": sample_seed if name == "CSDI" else None,
                                "n_samples": 10 if name == "CSDI" else None, "fold": 1,
                                "scope": "integrity_pilot_no_convergence_claim", "fallback": None, "device": device}
                            started = perf_counter()
                            try:
                                loading = perf_counter()
                                if name in local:
                                    if (site, name) not in models:
                                        loaded = checkpoints.load(fit_job)
                                        if loaded is None:
                                            raise RuntimeError("Completed model checkpoint missing or corrupt; cannot infer")
                                        models[(site, name)] = load_native(loaded[0], device=device)
                                    model = models[(site, name)]
                                else:
                                    if name not in models:
                                        from energygap.methods.foundation import MOMENTFrozen
                                        models[name] = MOMENTFrozen(cache_dir=root/"artifacts/checkpoints/hf", device=device, allow_network=False)
                                        models[name]._load()
                                    model = models[name]
                                synchronize(device); reload_seconds = perf_counter()-loading
                                tic = perf_counter()
                                if name in local:
                                    output = model.infer(query.values, task="R") if name == "T1" else model.infer(query.values, task="R", samples=10, sample_seed=sample_seed)
                                    point = output["point_estimate"][:, 0]
                                    draws = output["samples"][:, :, 0] if output["samples"] is not None else None
                                else:
                                    point = model.impute_query(query); draws = None
                                synchronize(device); infer_seconds = perf_counter()-tic
                                observed = np.asarray(query.available_to_method, bool)
                                if not np.array_equal(point[observed], np.asarray(query.values)[observed]):
                                    raise AssertionError("Observed preservation failed")
                                metrics = reconstruction_metrics(truth.values, point, truth.evaluation_mask, scale_train=data["scale"])
                                if metrics["mae_kw"] is None:
                                    raise RuntimeError(metrics["reason"])
                                buffer = io.BytesIO()
                                arrays = {"point": point, "times": np.asarray(query.times),
                                    "available_mask": observed, "target_gap_mask": np.asarray(query.target_gap_mask)}
                                if draws is not None:
                                    arrays["joint_samples"] = draws
                                np.savez_compressed(buffer, **arrays)
                                target = run_path/"predictions"/hashlib.sha256(record["key"].encode()).hexdigest()
                                artifact = target.with_suffix(".npz")
                                checksum = atomic_bytes(artifact, buffer.getvalue())
                                if draws is not None:
                                    scored = summarize_samples(draws, truth.values, truth.evaluation_mask, scale_train=data["scale"])
                                    record["probabilistic"] = {"capability": "joint_samples", "point_estimator": "sample_median",
                                        "crps": scored["crps"], "pinball": scored["pinball"], "intervals": scored["intervals"]}
                                truthpath = root/"data/evaluation_truth"/experiment_id/hashlib.sha256(query.query_id.encode()).hexdigest()
                                atomic_json(truthpath.with_suffix(".json"), {"query_id": query.query_id,
                                    "values": [float(value) if np.isfinite(value) else None for value in truth.values],
                                    "evaluation_mask": list(truth.evaluation_mask)})
                                return {**record, "status": "success", **metrics, "reload_seconds": reload_seconds,
                                    "inference_seconds": infer_seconds, "elapsed_seconds": perf_counter()-started,
                                    "artifact": str(artifact.relative_to(run_path)), "sha256": checksum}
                            except Exception as error:
                                return {**record, "status": "failed", "reason": type(error).__name__+": "+str(error),
                                    "mae_kw": None, "rmse_kw": None, "nmae": None,
                                    "elapsed_seconds": perf_counter()-started, "traceback": traceback.format_exc()}
                        run_units(store, job, infer, shard_size=1, progress=progress, stop=stop,
                            signature=f"infer:{name}:L512:S{10 if name=='CSDI' else 1}:"+fingerprint(torch_runtime))
                if stop.requested:
                    store.end_session("interrupted")
                else:
                    store.complete()
        except BudgetExhausted:
            store.end_session("paused_budget")
        except BaseException:
            store.end_session("interrupted")
            raise
        return write_report(store, root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--experiment-id")
    parser.add_argument("--resume", choices=("auto", "never", "require"), default="auto")
    parser.add_argument("--progress", choices=("tqdm", "log", "none"), default="log")
    parser.add_argument("--windows", type=int, choices=(16, 32), default=16)
    parser.add_argument("--epochs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--budget-total-hours", type=float, default=.1)
    parser.add_argument("--sites", nargs="+", choices=("H1", "H2", "I1"), default=["H1", "H2", "I1"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    args = vars(parser.parse_args())
    report = run(**args)
    print(json.dumps(report.get("plan") or {key: report[key] for key in
        ("experiment_id", "n_episode_method_records", "n_success", "n_failed", "scope")}, indent=2))


if __name__ == "__main__":
    main()
