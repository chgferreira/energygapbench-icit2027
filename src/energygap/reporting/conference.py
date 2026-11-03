"""Durable, explicitly conditional reports for the available-method server study.

This report consumes committed inference records. It never selects methods on
test losses and never upgrades hypothetical meter semantics to confirmed physics.
All products stay under the run directory, separate from manuscript sources.
"""
from __future__ import annotations

from datetime import date, timedelta
from contextlib import closing, contextmanager
from io import BytesIO
import csv
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import uuid

import numpy as np
import pandas as pd

from energygap.runtime.scheduler import run_units
from energygap.runtime.state import StateStore, TERMINAL, BudgetExhausted, atomic_bytes, atomic_json, canonical, fsync_directory
from energygap.runtime.progress import Progress
from energygap.runtime.eta import ETAEstimator
from energygap.statistics.paired import collapse_replicates, fixed_estimand, holm, stable_seed
from energygap.reporting.study_scope import study_design, committed_coverage
from energygap.reporting.bootstrap_execution import BootstrapExecutor

BASELINES = ("median", "locf", "linear", "seasonal", "kalman_filter", "kalman_smoother")
CORE_DURATIONS = (15, 60, 180)
CORE_CONDITIONS = ("C3", "C4", "C5")
METRICS = ("nmae", "mae_kw", "forecast_nmae", "forecast_mae_kw")
EPISODE_IDS = ("site", "fold", "partition", "condition", "duration", "mask_seed",
               "method", "variant", "gap_id", "day")
REPORT_CHUNK_SIZE = 10000
LOG = logging.getLogger(__name__)


def _notify_report(observer, phase, units, seconds, *, job_id="report", kind="report", report_info=None):
    if observer is None:
        return
    # Older integrations may expose the original explicit keyword signature.
    # Inspect it before calling; never swallow a TypeError inside the observer.
    try:
        parameters = inspect.signature(observer).parameters.values()
        accepts_info = any(p.kind == p.VAR_KEYWORD or p.name == "report_info" for p in parameters)
    except (TypeError, ValueError):
        accepts_info = False
    metadata = dict(job_id=job_id, kind=kind)
    if accepts_info and report_info is not None:
        metadata["report_info"] = report_info
    observer(phase, units, seconds, **metadata)


class _ReportMonitor:
    """Parent-only progress for real export stages, with honest unknown totals."""
    def __init__(self, observer, progress, stop, store=None):
        self.observer, self.progress, self.stop, self.store = observer, progress, stop, store
        self.info, self.started, self.last, self.done = {}, time.monotonic(), time.monotonic(), 0

    def check(self):
        _check_report_stop(self.stop)
        if self.store is not None:
            self.store.check_budget()

    def stage(self, stage, label, *, total=None, unit="registros", **extra):
        self.check()
        self.started = self.last = time.monotonic()
        self.done = 0
        self.info = dict(stage=stage, label=label, done=0, total=total, unit=unit,
                         status="running", cost_family=stage, progress_key=stage, plan_complete=False, **extra)
        if self.progress != "none":
            LOG.info("Relatório: %s", label)
        self.advance(0, force=True)

    def advance(self, done=None, *, force=True, status="running"):
        _check_report_stop(self.stop)
        now = time.monotonic()
        if not force and now-self.last < 1.:
            return
        self.check()
        current = self.done if done is None else done
        self.info.update(done=current, status=status, elapsed_seconds=now-self.started)
        _notify_report(self.observer, "report", current-self.done, now-self.last,
                       report_info=dict(self.info))
        self.done, self.last = current, now

    def pulse(self):
        self.advance(force=False)


@contextmanager
def _watch_sql(db, monitor):
    """Make long SQLite scans interruptible without publishing partial exports."""
    if monitor is None:
        yield
        return
    failure = []
    def handler():
        try:
            monitor.pulse()
            return 0
        except BaseException as exc:
            failure.append(exc)
            return 1
    db.set_progress_handler(handler, 100000)
    try:
        yield
    except sqlite3.OperationalError:
        if failure:
            raise failure[0]
        raise
    finally:
        db.set_progress_handler(None, 0)


def _check_report_stop(stop):
    if stop is not None and stop.requested:
        # Share the runner's interruption class without importing the scientific
        # executor during ordinary report-module import.
        from energygap.pipeline import GracefulStop
        raise GracefulStop('Signal at a committed report/bootstrap boundary')


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _mean_complete(values):
    """An unavailable seed loss is not silently discarded from a policy score."""
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return float(numeric.mean()) if len(numeric) and np.isfinite(numeric).all() else np.nan


def _episode_losses(frame):
    if frame.empty:
        return pd.DataFrame(columns=[*EPISODE_IDS, *METRICS, "latency_seconds", "seed_evaluations"])
    frame = frame.copy()
    for column in METRICS:
        if column not in frame:
            frame[column] = np.nan
    if "latency_seconds" not in frame:
        frame["latency_seconds"] = np.nan
    if "timing_mode" in frame:
        frame.loc[frame["timing_mode"].fillna('single_query') != 'single_query', "latency_seconds"] = np.nan
    # Each mask replica keeps its identity. Repeated training seeds are averaged
    # as LOSSES for one episode before any temporal/duration aggregation.
    grouped = frame.groupby(list(EPISODE_IDS), dropna=False, sort=True)
    result = grouped.agg(**{m: (m, _mean_complete) for m in METRICS},
                         latency_seconds=("latency_seconds", lambda values: float(values.median())
                             if values.notna().all() else np.nan),
                         seed_evaluations=("gap_id", "size")).reset_index()
    return result


def _select_baselines(episodes):
    chosen, scores = {}, []
    validation = episodes[(episodes.partition == "validation") &
                          (episodes.variant == "P") & episodes.method.isin(BASELINES) &
                          episodes.condition.isin(CORE_CONDITIONS)]
    for (site, fold), rows in validation.groupby(["site", "fold"]):
        candidates = []
        for method, group in rows.groupby("method"):
            duration = group.groupby("duration").nmae.apply(_mean_complete)
            if set(CORE_DURATIONS) <= set(duration.index) and np.isfinite(
                    duration.reindex(CORE_DURATIONS).to_numpy()).all():
                score = float(duration.reindex(CORE_DURATIONS).mean())
                latency = float(group.latency_seconds.median())
                if not np.isfinite(latency):
                    latency = float("inf")
                candidates.append((score, latency, str(method)))
                scores.append(dict(site=site, fold=int(fold), method=method,
                                   validation_equal_duration_nmae=score,
                                   latency_p50_seconds=latency))
        if candidates:
            chosen[(str(site), int(fold))] = min(candidates)[2]
    return chosen, scores


def _calendars(run_dir, config, site, folds):
    declared = config.get("test_calendars", {}).get(site, {})
    result = {}
    for fold in folds:
        if str(fold) in declared or fold in declared:
            result[fold] = declared.get(str(fold), declared.get(fold))
            continue
        path = run_dir / "manifests" / f"{site}:fold{fold}-split.json"
        if not path.is_file():
            continue
        manifest = json.loads(path.read_text())
        first = date.fromisoformat(manifest["validation_end_exclusive"][:10])
        last = date.fromisoformat(manifest["test_end"][:10])
        result[fold] = [(first + timedelta(days=i)).isoformat()
                        for i in range((last - first).days + 1)]
    return result


def _bootstrap_kernel(rows, calendars, *, block_days, seed, folds, durations=CORE_DURATIONS):
    """The existing moving-block estimator, reduced to day sums/counts for speed.

    Days with no episodes remain in the sampling calendar. Its RNG draws and
    estimand match statistics.paired.bootstrap_replicate; seeds/masks add no days.
    """
    arrays = {}
    for fold in folds:
        days = list(calendars[fold])
        if not days:
            raise ValueError("Empty test calendar")
        if any((date.fromisoformat(str(b)) - date.fromisoformat(str(a))).days != 1
               for a, b in zip(days, days[1:])):
            raise ValueError("Calendars must include contiguous empty days")
        lookup = {str(day): i for i, day in enumerate(days)}
        sums = np.zeros((len(days), len(durations)), dtype=float)
        counts = np.zeros_like(sums)
        for row in rows:
            if int(row["fold"]) != fold:
                continue
            if str(row["day"]) not in lookup:
                raise ValueError("Episode day lies outside its declared test calendar")
            if int(row["duration"]) not in durations:
                continue
            i, j = lookup[str(row["day"])], durations.index(int(row["duration"]))
            sums[i, j] += float(row["difference"])
            counts[i, j] += 1
        arrays[fold] = (sums, counts)

    def replicate(index):
        rng = np.random.default_rng(stable_seed(seed, index))
        cells = []
        for fold in folds:
            sums, counts = arrays[fold]
            n = len(sums)
            length = min(block_days, n)
            starts = rng.integers(0, n - length + 1, size=(n + length - 1) // length)
            selected = (starts[:, None] + np.arange(length)).ravel()[:n]
            totals = counts[selected].sum(axis=0)
            if np.any(totals == 0):
                return None
            cells.extend((sums[selected].sum(axis=0) / totals).tolist())
        return float(np.mean(cells))
    return replicate


def _persisted_bootstrap(store, rows, calendars, contrast_id, attempts, block_days,
                         folds, progress, progress_observer=None, stop=None):
    _check_report_stop(stop)
    rows = collapse_replicates(rows)
    estimate = fixed_estimand(rows, folds, CORE_DURATIONS)
    if estimate is None:
        return dict(estimate=None, ci95=None, p_raw=None, confirmatory=False,
                    attempted=0, invalid=0, reason="Missing required fold-duration stratum")
    identity = dict(rows=rows, calendars={str(k): list(v) for k, v in calendars.items()},
                    contrast_id=contrast_id, attempts=attempts, block_days=block_days,
                    seed=20261031, folds=list(folds), algorithm="paired-moving-block-day-reduction-v1")
    digest = hashlib.sha256(canonical(identity)).hexdigest()[:20]
    job_id = f"bootstrap:conference:{contrast_id}:{digest}"
    store.add_job(job_id, attempts, metadata=dict(phase="bootstrap", contrast_id=contrast_id,
                  input_sha256=digest, attempts=attempts, block_days=block_days, seed=20261031))
    compute_value = _bootstrap_kernel(rows, calendars, block_days=block_days,
                                     seed=20261031, folds=folds)

    def compute(index, unused_seed):
        value = compute_value(index)
        return dict(key=f"{job_id}:replicate:{index:08d}", phase="bootstrap",
                    contrast_id=contrast_id, replicate_id=index, value=value,
                    sample_seed=stable_seed(20261031, index),
                    status="invalid" if value is None else "valid")

    if store.job(job_id)["status"] != "succeeded" and store.get_meta("status") in {
            "completed", "completed_with_failures"}:
        # A stand-alone derived report must acquire a real coordinator session
        # before adding statistics to a previously completed compute ledger.
        store.set_meta("status", "planned")
    run_units(store, job_id, compute, shard_size=100, progress=progress, stop=stop,
              signature=f"conference-bootstrap:days{block_days}:rows{len(rows)}",
              reservation_seconds=1., progress_observer=progress_observer,
              publish_eta=progress_observer is None)
    _check_report_stop(stop)
    draws = sorted(store.results(job_id), key=lambda r: r["replicate_id"])
    if [r["replicate_id"] for r in draws] != list(range(attempts)):
        raise RuntimeError("Bootstrap incomplete; no provisional inferential summary")
    valid = np.asarray([r["value"] for r in draws if r["value"] is not None])
    invalid = attempts - len(valid)
    blocks = sum(len(calendars[f]) // block_days for f in folds)
    duration_days = {d: len({(r["fold"], r["day"]) for r in rows if r["duration"] == d})
                     for d in CORE_DURATIONS}
    supported = (blocks >= 20 and all(len(calendars[f]) // block_days >= 4 for f in folds)
                 and all(n >= 20 for n in duration_days.values()) and invalid == 0)
    p = float((1 + np.sum(np.abs(valid - estimate) >= abs(estimate))) / (attempts + 1)) if supported else None
    return dict(estimate=estimate,
                ci95=list(map(float, np.quantile(valid, [.025, .975]))) if len(valid) else None,
                p_raw=None, p_exploratory=p, confirmatory=False,
                interval_kind="exploratory_conditional_valid", attempted=attempts, invalid=invalid,
                support_blocks=blocks, duration_days=duration_days, temporal_support_sufficient=supported,
                job_id=job_id, reason="Physical measurement semantics remain hypothetical" +
                ("; insufficient temporal support or invalid bootstrap draws" if not supported else ""))


def _contrast_rows(run_dir, config, episodes, chosen, store, attempts, block_days, progress,
                   progress_observer=None, stop=None):
    folds = tuple(map(int, config.get("folds", (1, 2, 3))))
    sites = list(config.get("sites", sorted(episodes.site.unique())))
    hypotheses = list(config.get("physical_hypotheses", ()))
    primary = episodes[(episodes.partition == "test") & episodes.condition.isin(CORE_CONDITIONS)]
    rows = []
    for hypothesis in hypotheses:
        variant = f"E:{hypothesis}"
        family = []
        for site in sites:
            calendars = _calendars(run_dir, config, site, folds)
            for contrast in ("contrast_energy_gain", "contrast_complexity_gain"):
                for metric in ("nmae", "forecast_nmae"):
                    paired, unavailable = [], 0
                    for fold in folds:
                        base = chosen.get((str(site), fold))
                        if base is None:
                            continue
                        subset = primary[(primary.site == site) & (primary.fold == fold)]
                        proposed = subset[(subset.method == base + "+energy_consistent") &
                                          (subset.variant == variant)]
                        ref_method = base if contrast == "contrast_energy_gain" else "linear+energy_consistent"
                        ref_variant = "P" if contrast == "contrast_energy_gain" else variant
                        reference = subset[(subset.method == ref_method) & (subset.variant == ref_variant)]
                        keys = ["gap_id", "condition", "duration", "day", "mask_seed"]
                        merged = proposed.merge(reference, on=keys, suffixes=("_a", "_b"),
                                                how="outer", validate="one_to_one")
                        for _, row in merged.iterrows():
                            if not (np.isfinite(row[metric + "_a"]) and np.isfinite(row[metric + "_b"])):
                                unavailable += 1
                                continue
                            paired.append(dict(fold=fold, day=str(row.day), duration=int(row.duration),
                                               gap_id=f"{row.mask_seed}:{row.gap_id}",
                                               difference=float(row[metric + "_a"] - row[metric + "_b"])))
                    if set(calendars) == set(folds) and paired and not unavailable:
                        result = _persisted_bootstrap(store, paired, calendars,
                            f"{hypothesis}:{site}:{contrast}:{metric}", attempts, block_days, folds, progress,
                            progress_observer, stop)
                    else:
                        result = dict(estimate=None, ci95=None, p_raw=None, confirmatory=False,
                                      attempted=0, invalid=0,
                                      reason="Missing matched eligible outputs, selected baseline, or full test calendars")
                    family.append(dict(site=site, hypothesis=hypothesis, contrast=contrast, endpoint=metric,
                                       physical_confirmatory=False, block_days=block_days,
                                       unmatched_or_unscorable=unavailable, **result))
        # Hypothetical branches each retain their fixed twelve-test family.
        for row, adjusted in zip(family, holm([r["p_raw"] for r in family], family_size=max(12, len(family)))):
            row["p_holm"] = adjusted
        rows.extend(family)
    return rows


def _export_frame(path, frame):
    export = frame.copy()
    for column in (name for name in export.columns if export[name].dtype == object):
        if export[column].map(lambda x: isinstance(x, (dict, list, tuple))).any():
            export[column] = export[column].map(lambda x: json.dumps(_jsonable(x), sort_keys=True)
                                               if isinstance(x, (dict, list, tuple)) else x)
    atomic_bytes(path.with_suffix(".csv"), export.to_csv(index=False).encode())
    buffer = BytesIO()
    export.to_parquet(buffer, index=False)
    atomic_bytes(path.with_suffix(".parquet"), buffer.getvalue())


def _export_forecasts(db, out, chunk_size=10000, monitor=None):
    """Stream every horizon and exclusion; no trajectory arrays enter Python.

    A fixed Arrow schema permits all-null validation chunks followed by measured
    test chunks without inferring an incompatible null-only Parquet column.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    fields = [("key", pa.string()), ("site", pa.string()), ("fold", pa.int64()),
              ("partition", pa.string()), ("condition", pa.string()), ("duration", pa.float64()),
              ("mask_seed", pa.int64()), ("train_seed", pa.float64()), ("method", pa.string()),
              ("variant", pa.string()), ("gap_id", pa.string()), ("day", pa.string()),
              ("issuance_time", pa.string()), ("horizon_seconds", pa.int64()),
              ("mae_kw", pa.float64()), ("nmae", pa.float64()),
              ("target_offset_seconds", pa.float64()), ("exclusion_reason", pa.string())]
    schema = pa.schema(fields)
    identifiers = [name for name, _ in fields[:13]]
    select = [f"json_extract(r.payload,'$.{name}') AS {name}" for name in identifiers]
    select += ["CAST(f.key AS INTEGER) AS horizon_seconds"]
    select += [f"json_extract(f.value,'$.{name}') AS {name}" for name in
               ("mae_kw", "nmae", "target_offset_seconds", "exclusion_reason")]
    query = ("SELECT " + ",".join(select) +
             " FROM results AS r, json_each(r.payload,'$.forecasts') AS f "
             "WHERE json_extract(r.payload,'$.phase')='inference' ORDER BY r.result_key,CAST(f.key AS INTEGER)")
    columns = [name for name, _ in fields]
    token = uuid.uuid4().hex
    csv_path, parquet_path = out / "forecasts_long.csv", out / "forecasts_long.parquet"
    csv_temp = out / f".forecasts-long.{token}.csv.tmp"
    parquet_temp = out / f".forecasts-long.{token}.parquet.tmp"
    count, wrote = 0, False
    writer = pq.ParquetWriter(parquet_temp, schema)
    try:
        with csv_temp.open("w", newline="") as csv_file:
            for chunk in pd.read_sql_query(query, db, chunksize=chunk_size):
                table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
                writer.write_table(table)
                chunk.to_csv(csv_file, index=False, header=not wrote)
                count += len(chunk)
                wrote = True
                if monitor is not None:
                    monitor.advance(count)
            if not wrote:
                pd.DataFrame(columns=columns).to_csv(csv_file, index=False)
            csv_file.flush()
            os.fsync(csv_file.fileno())
        writer.close()
        with parquet_temp.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(csv_temp, csv_path)
        os.replace(parquet_temp, parquet_path)
        fsync_directory(out)
        if monitor is not None:
            monitor.advance(count, status="completed")
    finally:
        writer.close()
        csv_temp.unlink(missing_ok=True)
        parquet_temp.unlink(missing_ok=True)
    return count


def _export_phase(db, out, phase, stem, chunk_size=10000, monitor=None):
    """Stream a separate scientific track with typed scalars and JSON diagnostics."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    types = {}
    for key, kind in db.execute(
            "SELECT DISTINCT j.key,j.type FROM results AS r,json_each(r.payload) AS j "
            "WHERE json_extract(r.payload,'$.phase')=? ORDER BY j.key", (phase,)):
        types.setdefault(key, set()).add(kind)
    schema_fields = []
    for key, kinds in types.items():
        present = kinds - {"null"}
        datatype = (pa.bool_() if present and present <= {"true", "false"} else
                    pa.int64() if present and present <= {"integer"} else
                    pa.float64() if present and present <= {"integer", "real"} else pa.string())
        schema_fields.append((key, datatype))
    schema = pa.schema(schema_fields or [("phase", pa.string())])
    fields = [name for name, _ in schema_fields]
    # json_each is used only to discover this small track's scalar schema. The
    # export reads only that phase and never materializes unrelated predictions.
    select = ",".join(f"json_extract(payload,{json.dumps('$.' + key)}) AS {json.dumps(key)}" for key in fields)
    query = (f"SELECT {select} FROM results WHERE json_extract(payload,'$.phase')=? ORDER BY result_key"
             if fields else "SELECT NULL AS phase WHERE 0")
    token = uuid.uuid4().hex
    csv_temp, parquet_temp = out / f".{stem}.{token}.csv.tmp", out / f".{stem}.{token}.parquet.tmp"
    writer, count, wrote = pq.ParquetWriter(parquet_temp, schema), 0, False
    try:
        with csv_temp.open("w", newline="") as csv_file:
            for chunk in pd.read_sql_query(query, db, params=(phase,) if fields else (), chunksize=chunk_size):
                for key, datatype in schema_fields:
                    if pa.types.is_boolean(datatype):
                        chunk[key] = chunk[key].map(lambda x: None if pd.isna(x) else bool(x))
                    elif pa.types.is_string(datatype):
                        # Arrays/objects arrive as SQLite JSON strings. Null-only
                        # columns retain null rather than the text "None".
                        chunk[key] = chunk[key].map(lambda x: None if pd.isna(x) else str(x))
                writer.write_table(pa.Table.from_pandas(chunk, schema=schema, preserve_index=False))
                chunk.to_csv(csv_file, index=False, header=not wrote)
                count += len(chunk)
                wrote = True
                if monitor is not None:
                    monitor.advance(count)
            if not wrote:
                pd.DataFrame(columns=schema.names).to_csv(csv_file, index=False)
            csv_file.flush()
            os.fsync(csv_file.fileno())
        writer.close()
        with parquet_temp.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(csv_temp, out / f"{stem}.csv")
        os.replace(parquet_temp, out / f"{stem}.parquet")
        fsync_directory(out)
        if monitor is not None:
            monitor.advance(count, status="completed")
    finally:
        writer.close()
        csv_temp.unlink(missing_ok=True)
        parquet_temp.unlink(missing_ok=True)
    return count


_RAW_TEXT = ("key", "phase", "run_id", "site", "partition", "condition", "method", "variant",
             "gap_id", "day", "status", "reason", "metric_reason", "query_id", "issuance_time",
             "information_tier", "task", "probabilistic_capability", "point_estimator",
             "energy_metrics_reason", "energy_metrics_status", "physical_hypothesis",
             "gross_energy_reason", "probabilistic", "correction_diagnostics", "timing_mode", "latency_reason", "performance_reason",
             "oom_retry_timing_policy")
_RAW_INTS = ("fold", "mask_seed", "train_seed", "sample_seed", "context", "n", "n_original_targets",
             "original_target_points", "physical_confirmatory", "inference_batch_size", "batch_retry_scalar",
             "requested_batch_size", "adaptive_batch_size", "oom_retries", "energy_cpu_workers")
_RAW_REAL = (*METRICS, "duration", "latency_seconds", "rmse_kw", "bias_kw", "nrmse", "requested_rate", "realized_rate",
             "batch_wall_seconds", "inference_amortized_seconds", "oom_retry_seconds", "correction_service_seconds",
             "sign_error_rate", "import_peak_error_kw", "export_peak_error_kw",
             "import_peak_time_error_seconds", "export_peak_time_error_seconds", "ramp_mae_kw_per_minute",
             "integral_boundary_span_seconds",
             *(direction + suffix for direction in ("net", "import", "export") for suffix in
               ("_reference_power_integral_kwh", "_estimated_power_integral_kwh",
                "_power_integral_error_kwh", "_input_counter_residual_kwh")))
_RAW_SCHEMA = [(name, "TEXT") for name in _RAW_TEXT] + [(name, "INTEGER") for name in _RAW_INTS] + [(name, "REAL") for name in _RAW_REAL]


def _sql_name(name):
    return '"' + name.replace('"', '""') + '"'


def _complete_average(metric):
    return f"CASE WHEN COUNT({_sql_name(metric)})=COUNT(*) THEN AVG({_sql_name(metric)}) END"


def _export_query(db, query, path, fields, params=(), *, chunk_size=REPORT_CHUNK_SIZE, monitor=None):
    """Bounded row batches to CSV/Parquet; null-only chunks keep a fixed schema."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    arrow = {"TEXT": pa.string(), "INTEGER": pa.int64(), "REAL": pa.float64()}
    schema = pa.schema([(name, arrow[kind]) for name, kind in fields])
    names = [name for name, _ in fields]
    token = uuid.uuid4().hex
    csv_temp = path.with_name(f".{path.name}.{token}.csv.tmp")
    pq_temp = path.with_name(f".{path.name}.{token}.parquet.tmp")
    writer, count = pq.ParquetWriter(pq_temp, schema), 0
    try:
        cursor = db.execute(query, params)
        with csv_temp.open("w", newline="") as stream:
            csv_writer = csv.writer(stream)
            csv_writer.writerow(names)
            while True:
                rows = cursor.fetchmany(chunk_size)
                if not rows:
                    break
                csv_writer.writerows(rows)
                arrays = [pa.array([row[i] for row in rows], type=schema.field(i).type)
                          for i in range(len(names))]
                writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
                count += len(rows)
                if monitor is not None:
                    monitor.advance(count)
            stream.flush()
            os.fsync(stream.fileno())
        writer.close()
        with pq_temp.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(csv_temp, path.with_suffix(".csv"))
        os.replace(pq_temp, path.with_suffix(".parquet"))
        fsync_directory(path.parent)
        if monitor is not None:
            monitor.advance(count, status="completed")
    finally:
        writer.close()
        csv_temp.unlink(missing_ok=True)
        pq_temp.unlink(missing_ok=True)
    return count


def _compact_jsonl(source, out, *, chunk_size=REPORT_CHUNK_SIZE, monitor=None):
    """Retain every compact metric/diagnostic, including future extra fields."""
    destination = out / "episodes_records.jsonl"
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        count = 0
        with temporary.open("w") as stream:
            cursor = source.execute("SELECT json_remove(payload,'$.prediction','$.samples',"
                "'$.filled_missing_indices','$.filled_missing_values','$.forecasts') FROM results "
                "WHERE json_extract(payload,'$.phase')='inference' ORDER BY result_key")
            while True:
                rows = cursor.fetchmany(chunk_size)
                if not rows:
                    break
                for row in rows:
                    stream.write(row[0] + "\n")
                count += len(rows)
                if monitor is not None:
                    monitor.advance(count)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        fsync_directory(out)
        if monitor is not None:
            monitor.advance(count, status="completed")
    finally:
        temporary.unlink(missing_ok=True)


def _compact_report_source(source_path, scratch, monitor=None):
    """Read large inference JSON once; retain every field used by reports.

    Only inference trajectory arrays are removed. Forecasts, future scalar
    fields, and diagnostics remain; fit and separate energy-track payloads are
    copied without modification. The original ledger is attached read-only.
    """
    with closing(sqlite3.connect(scratch, uri=True)) as db:
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA cache_size=-65536")
        db.execute("PRAGMA journal_mode=OFF")  # Disposable derived scratch only.
        db.execute("ATTACH DATABASE ? AS ledger", (f"file:{source_path}?mode=ro",))
        db.execute("CREATE TABLE results(result_key TEXT PRIMARY KEY,payload TEXT NOT NULL)")
        with _watch_sql(db, monitor):
            db.execute("INSERT INTO results SELECT result_key,CASE WHEN "
                "json_extract(payload,'$.phase')='inference' THEN "
                "json_remove(payload,'$.prediction','$.samples','$.filled_missing_indices','$.filled_missing_values') "
                "ELSE payload END FROM ledger.results "
                "WHERE json_extract(payload,'$.phase') IN ('inference','fit','energy_track') ORDER BY result_key")
            db.commit()
            count = db.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        if monitor is not None:
            monitor.info["total"] = count
            monitor.advance(count, status="completed")
    return count


def _build_analysis(source_path, scratch, monitor=None):
    """SQLite performs the large seed/episode grouping with disk-backed sorting."""
    db = sqlite3.connect(scratch, uri=True)
    try:
        with _watch_sql(db, monitor):
            _populate_analysis(db, source_path)
        return db
    except BaseException:
        db.close()
        raise


def _populate_analysis(db, source_path):
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-65536")
    db.execute("PRAGMA journal_mode=OFF")  # Rebuildable scratch; ledger is read-only.
    db.execute("ATTACH DATABASE ? AS ledger", (f"file:{source_path}?mode=ro",))
    declarations = ",".join(f"{_sql_name(name)} {kind}" for name, kind in _RAW_SCHEMA)
    db.execute(f"CREATE TABLE raw ({declarations})")
    # Historical records measured one query per native call.  New batched cost
    # fields describe amortized throughput and must never become query latency.
    mode = "COALESCE(json_extract(payload,'$.timing_mode'),'single_query')"
    def extract(name):
        value = f"json_extract(payload,'$.{name}')"
        if name == 'timing_mode':
            return mode
        if name == 'inference_batch_size':
            return f"CASE WHEN {mode}='single_query' THEN COALESCE({value},1) ELSE {value} END"
        if name == 'latency_seconds':
            return f"CASE WHEN {mode}='single_query' THEN {value} END"
        return value
    extracted = ",".join(extract(name) for name, _ in _RAW_SCHEMA)
    db.execute(f"INSERT INTO raw SELECT {extracted} FROM ledger.results WHERE json_extract(payload,'$.phase')='inference' ORDER BY result_key")
    # A median within one episode has at most the configured training-seed
    # multiplicity (three in the core), rather than all dataset episodes.
    class Median:
        def __init__(self): self.values = []
        def step(self, value):
            if value is not None: self.values.append(value)
        def finalize(self):
            return float(np.median(self.values)) if self.values else None
    db.create_aggregate("seed_median", 1, Median)
    ids = ",".join(map(_sql_name, EPISODE_IDS))
    averages = ",".join(f"{_complete_average(metric)} AS {_sql_name(metric)}" for metric in METRICS)
    db.execute(f"CREATE TABLE episode_losses AS SELECT {ids},{averages},"
               "CASE WHEN COUNT(latency_seconds)=COUNT(*) THEN seed_median(latency_seconds) END AS latency_seconds,"
               "COUNT(*) AS seed_evaluations "
               f"FROM raw GROUP BY {ids}")
    db.execute("CREATE INDEX loss_pair ON episode_losses(partition,site,fold,method,variant,condition,gap_id,mask_seed,duration,day)")
    db.commit()



def _selection_sql(db):
    baseline_literals = ",".join("'" + name + "'" for name in BASELINES)
    where = ("partition='validation' AND variant='P' AND condition IN ('C3','C4','C5') "
             f"AND method IN ({baseline_literals})")
    # Latency tie-breaking is calculated by SQLite window order statistics, so
    # even a large validation stratum is never collected in a Python list.
    sql = f"""WITH cells AS (
       SELECT site,fold,method,duration,{_complete_average('nmae')} AS score
       FROM episode_losses WHERE {where} GROUP BY site,fold,method,duration),
       scores AS (SELECT site,fold,method,AVG(score) AS score FROM cells
         WHERE duration IN (15,60,180) GROUP BY site,fold,method
         HAVING COUNT(*)=3 AND COUNT(score)=3),
       ordered AS (SELECT site,fold,method,latency_seconds,
          ROW_NUMBER() OVER (PARTITION BY site,fold,method ORDER BY latency_seconds) AS position,
          COUNT(*) OVER (PARTITION BY site,fold,method) AS size
          FROM episode_losses WHERE {where} AND latency_seconds IS NOT NULL),
       latencies AS (SELECT site,fold,method,AVG(latency_seconds) AS latency
          FROM ordered WHERE position IN ((size+1)/2,(size+2)/2) GROUP BY site,fold,method)
       SELECT s.site,s.fold,s.method,s.score,l.latency FROM scores AS s
       LEFT JOIN latencies AS l USING(site,fold,method)
       ORDER BY s.site,s.fold,s.score,COALESCE(l.latency,1e300),s.method"""
    chosen, scores = {}, []
    for site, fold, method, score, latency in db.execute(sql):
        chosen.setdefault((site, int(fold)), method)
        scores.append(dict(site=site, fold=int(fold), method=method,
                           validation_equal_duration_nmae=score, latency_p50_seconds=latency))
    db.execute("CREATE TABLE chosen(site TEXT,fold INTEGER,method TEXT,PRIMARY KEY(site,fold))")
    db.executemany("INSERT INTO chosen VALUES(?,?,?)", [(s, f, m) for (s, f), m in chosen.items()])
    db.commit()
    return chosen, scores


def _export_performance(db, out, *, chunk_size=REPORT_CHUNK_SIZE, monitor=None):
    """Keep amortized batch throughput separate from single-query latency.

    A batch's wall time is repeated on its query records for provenance.  Summing
    it would count the same batch repeatedly; sum per-query amortized costs
    instead.  The ratio of measured query count to summed cost gives aggregate
    throughput, rather than averaging per-query reciprocals.
    """
    groups = ('site', 'fold', 'partition', 'method', 'variant', 'timing_mode', 'inference_batch_size', 'energy_cpu_workers')
    names = ','.join(map(_sql_name, groups))
    query = f"""WITH costs AS (
      SELECT *,CASE WHEN timing_mode='batch_amortized' AND inference_amortized_seconds>=0
                         THEN inference_amortized_seconds
                    WHEN timing_mode='single_query' AND latency_seconds>=0 THEN latency_seconds END AS cost
      FROM raw), totals AS (
      SELECT {names},COUNT(*) AS query_count,COUNT(cost) AS timed_query_count,
             SUM(CASE WHEN batch_retry_scalar THEN 1 ELSE 0 END) AS scalar_retry_query_count,
             SUM(CASE WHEN oom_retries>0 THEN 1 ELSE 0 END) AS oom_affected_query_count,
             SUM(cost) AS total_measured_cost_seconds,
             SUM(CASE WHEN timing_mode='parallel_cpu_service_cost' AND correction_service_seconds>=0
                      THEN correction_service_seconds END) AS total_correction_service_seconds
      FROM costs GROUP BY {names})
      SELECT *,CASE WHEN timed_query_count>0 THEN total_measured_cost_seconds/timed_query_count END
                 AS mean_measured_cost_seconds,
               CASE WHEN total_measured_cost_seconds>0 THEN timed_query_count/total_measured_cost_seconds END
                 AS measured_queries_per_second,
               CASE WHEN timing_mode='parallel_cpu_service_cost' THEN
                 'Parallel correction service time only; excludes queue/native inference; no latency or throughput claim; consult ledger wall time'
                    WHEN timed_query_count<query_count THEN
                 'Missing timings; statistics describe measured outputs only'
                    WHEN scalar_retry_query_count>0 THEN
                 'Includes scalar retries; measured costs exclude failed batch work; consult ledger wall accounting'
                    WHEN oom_affected_query_count>0 THEN
                 'Adaptive CUDA OOM retry; measured costs exclude failed attempts; consult ledger wall accounting'
                    WHEN total_measured_cost_seconds=0 THEN
                 'Measured total cost is zero; throughput unavailable'
                    WHEN timing_mode='batch_amortized' THEN
                 'Amortized inference/correction cost; not single-query latency or full pipeline wall time'
                    ELSE 'Single-query inference/correction timings; excludes other pipeline work' END AS performance_reason
      FROM totals ORDER BY {names}"""
    fields = [(name, 'INTEGER' if name in ('fold', 'inference_batch_size', 'energy_cpu_workers') else 'TEXT') for name in groups]
    fields += [('query_count', 'INTEGER'), ('timed_query_count', 'INTEGER'), ('scalar_retry_query_count', 'INTEGER'),
               ('oom_affected_query_count', 'INTEGER'),
               ('total_measured_cost_seconds', 'REAL'), ('total_correction_service_seconds', 'REAL'), ('mean_measured_cost_seconds', 'REAL'),
               ('measured_queries_per_second', 'REAL'), ('performance_reason', 'TEXT')]
    if monitor is not None:
        monitor.stage('export_performance', 'exportando custos computacionais...')
    _export_query(db, query, out / 'performance', fields, chunk_size=chunk_size, monitor=monitor)


def _materialize_tables(db, out, config, selections, *, chunk_size=REPORT_CHUNK_SIZE, monitor=None):
    main_ids = ("site", "fold", "condition", "duration", "method", "variant")
    ids = ",".join(main_ids)
    averages = ",".join(f"{_complete_average(m)} AS {m}" for m in METRICS)
    db.execute(f"CREATE TABLE main AS SELECT {ids},{averages},COUNT(*) AS episodes,"
               "NULL AS latency_p50_seconds,NULL AS latency_p95_seconds,'evaluated' AS status,NULL AS reason,"
               "NULL AS performance_reason "
               f"FROM episode_losses WHERE partition='test' GROUP BY {ids}")
    # Exact interpolated quantiles, with file-backed SQL window sorting.
    for quantile, column in ((.5, "latency_p50_seconds"), (.95, "latency_p95_seconds")):
        db.execute(f"""CREATE TABLE latency_{int(quantile*100)} AS WITH ordered AS (
          SELECT {ids},latency_seconds,ROW_NUMBER() OVER(PARTITION BY {ids} ORDER BY latency_seconds) AS position,
          COUNT(*) OVER(PARTITION BY {ids}) AS size FROM episode_losses
          WHERE partition='test' AND latency_seconds IS NOT NULL), targets AS (
          SELECT *,1+(size-1)*{quantile} AS target FROM ordered)
          SELECT {ids},SUM(CASE WHEN position=CAST(target AS INTEGER)
             THEN latency_seconds*(1-(target-CAST(target AS INTEGER)))
             WHEN position=CAST(target AS INTEGER)+1
             THEN latency_seconds*(target-CAST(target AS INTEGER)) ELSE 0 END) AS value
          FROM targets GROUP BY {ids}""")
        match = " AND ".join(f"l.{name}=main.{name}" for name in main_ids)
        db.execute(f"UPDATE main SET {column}=(SELECT value FROM latency_{int(quantile*100)} AS l WHERE {match})")
    db.execute(f"CREATE TABLE timing_support AS SELECT {ids},"
               "SUM(CASE WHEN timing_mode!='single_query' THEN 1 ELSE 0 END) AS non_single,"
               "SUM(CASE WHEN timing_mode='parallel_cpu_service_cost' THEN 1 ELSE 0 END) AS parallel_cpu,"
               "SUM(CASE WHEN latency_seconds IS NULL THEN 1 ELSE 0 END) AS missing "
               f"FROM raw WHERE partition='test' GROUP BY {ids}")
    timing_match = " AND ".join(f"t.{name}=main.{name}" for name in main_ids)
    db.execute(f"UPDATE main SET performance_reason=(SELECT CASE WHEN parallel_cpu>0 THEN "
               "'Parallel CPU correction: service time excludes queue/native inference; isolated-query latency unmeasured' "
               "WHEN non_single>0 THEN "
               "'Batched inference: single-query latency unmeasured; see performance.csv for amortized cost' "
               "WHEN missing>0 THEN 'Incomplete single-query timings; latency summaries unavailable' "
               f"ELSE 'Single-query latency measured' END FROM timing_support AS t WHERE {timing_match})")
    db.execute(f"UPDATE main SET latency_p50_seconds=NULL,latency_p95_seconds=NULL WHERE EXISTS "
               f"(SELECT 1 FROM timing_support AS t WHERE {timing_match} AND (non_single>0 OR missing>0))")
    for method in config.get("methods", ()):
        info = config.get("availability", {}).get(method, {})
        if info.get("status") in {"unavailable", "unsupported", "blocked"}:
            db.execute("INSERT INTO main(method,variant,status,reason,episodes) VALUES(?,'P',?,?,0)",
                       (method, info["status"], info.get("reason")))
    fields = [(name, "INTEGER" if name == "fold" else "REAL" if name == "duration" else "TEXT") for name in main_ids]
    fields += [(m, "REAL") for m in METRICS] + [("episodes", "INTEGER"), ("latency_p50_seconds", "REAL"),
               ("latency_p95_seconds", "REAL"), ("status", "TEXT"), ("reason", "TEXT"), ("performance_reason", "TEXT")]
    if monitor is not None:
        monitor.stage("export_main_table", "exportando tabela principal...")
    _export_query(db, "SELECT * FROM main ORDER BY site,fold,condition,duration,method,variant", out / "main_table", fields, chunk_size=chunk_size, monitor=monitor)
    _export_performance(db, out, chunk_size=chunk_size, monitor=monitor)
    folds = tuple(map(int, config.get("folds", (1, 2, 3))))
    cells = ("SELECT site,fold,duration,method,variant," + averages + ",COUNT(*) AS episodes FROM episode_losses "
             "WHERE partition='test' AND condition IN ('C3','C4','C5') "
             "GROUP BY site,fold,duration,method,variant")
    overall = ",".join(f"CASE WHEN COUNT(*)={len(folds)*3} AND COUNT({m})=COUNT(*) THEN AVG({m}) END AS {m}" for m in METRICS)
    query = ("SELECT site,method,variant,'equal_duration_then_equal_fold' AS aggregation,SUM(episodes) AS episodes," +
             overall + " FROM (" + cells + ") GROUP BY site,method,variant ORDER BY site,method,variant")
    aggregate_fields = [(n, "TEXT") for n in ("site", "method", "variant", "aggregation")]
    aggregate_fields += [("episodes", "INTEGER")] + [(m, "REAL") for m in METRICS]
    if monitor is not None:
        monitor.stage("export_core_aggregate", "exportando agregados por local...")
    _export_query(db, query, out / "core_aggregate", aggregate_fields, chunk_size=chunk_size, monitor=monitor)
    _export_frame(out / "selected_baseline_validation", pd.DataFrame(selections))


def _daily_pairs(db, site, hypothesis, contrast, metric):
    comparison = "c.method" if contrast == "contrast_energy_gain" else "'linear+energy_consistent'"
    ref_variant = "P" if contrast == "contrast_energy_gain" else "E:" + hypothesis
    scope = "e.partition='test' AND e.condition IN ('C3','C4','C5') AND e.site=?"
    ctes = ("WITH a AS (SELECT e.* FROM episode_losses AS e JOIN chosen AS c USING(site,fold) "
            f"WHERE {scope} AND e.method=c.method||'+energy_consistent' AND e.variant=?), "
            "b AS (SELECT e.* FROM episode_losses AS e JOIN chosen AS c USING(site,fold) "
            f"WHERE {scope} AND e.method={comparison} AND e.variant=?) ")
    params = (site, "E:" + hypothesis, site, ref_variant)
    keys = ("fold", "condition", "gap_id", "duration", "day", "mask_seed")
    on = " AND ".join(f"a.{key}=b.{key}" for key in keys)
    bad_query = (ctes + f"SELECT (SELECT COUNT(*) FROM a LEFT JOIN b ON {on} "
                 f"WHERE b.gap_id IS NULL OR a.{metric} IS NULL OR b.{metric} IS NULL) + "
                 f"(SELECT COUNT(*) FROM b LEFT JOIN a ON {on} WHERE a.gap_id IS NULL)")
    unavailable = int(db.execute(bad_query, params).fetchone()[0])
    daily = [dict(fold=int(fold), day=day, duration=int(duration), total=float(total), count=int(count))
             for fold, day, duration, total, count in db.execute(ctes +
                f"SELECT a.fold,a.day,a.duration,SUM(a.{metric}-b.{metric}),COUNT(*) FROM a JOIN b ON {on} "
                f"WHERE a.{metric} IS NOT NULL AND b.{metric} IS NOT NULL "
                "GROUP BY a.fold,a.day,a.duration ORDER BY a.fold,a.day,a.duration", params)]
    return daily, unavailable


def _prepare_daily_bootstrap(daily, calendars, contrast_id, attempts, block_days, folds):
    """Preserve the predecessor identity and exact daily-array construction."""
    groups = {(fold, duration): [0., 0] for fold in folds for duration in CORE_DURATIONS}
    for row in daily:
        cell = groups[(row["fold"], row["duration"])]
        cell[0] += row["total"]
        cell[1] += row["count"]
    estimate = float(np.mean([total/count for total, count in groups.values()])) if all(c for _, c in groups.values()) else None
    if estimate is None:
        return None
    identity = dict(daily=daily, calendars={str(k): list(v) for k,v in calendars.items()},
                    contrast_id=contrast_id, attempts=attempts, block_days=block_days, seed=20261031,
                    folds=list(folds), algorithm="paired-moving-block-sql-daily-v1")
    digest = hashlib.sha256(canonical(identity)).hexdigest()[:20]
    jid = f"bootstrap:conference:{contrast_id}:{digest}"
    arrays = {}
    for fold in folds:
        days = calendars[fold]
        if not days or any((date.fromisoformat(b)-date.fromisoformat(a)).days != 1 for a,b in zip(days,days[1:])):
            raise ValueError("Calendars must include contiguous empty days")
        lookup = {str(day): i for i,day in enumerate(days)}
        sums, counts = np.zeros((len(days),3)), np.zeros((len(days),3))
        for row in daily:
            if row["fold"] != fold: continue
            if row["day"] not in lookup: raise ValueError("Episode day outside full test calendar")
            i, j = lookup[row["day"]], CORE_DURATIONS.index(row["duration"])
            sums[i,j], counts[i,j] = row["total"], row["count"]
        arrays[fold] = sums, counts

    return dict(estimate=estimate, job_id=jid, input_sha256=digest, arrays=arrays)


def _register_daily_bootstrap(store, prepared, contrast_id, attempts, block_days):
    if prepared is not None:
        store.add_job(prepared["job_id"], attempts, metadata=dict(phase="bootstrap", contrast_id=contrast_id,
                      input_sha256=prepared["input_sha256"], attempts=attempts, block_days=block_days, seed=20261031))


def _run_daily_draws(store, prepared, daily, contrast_id, attempts, block_days, folds, progress,
                     observer, stop, executor, plan_state=None, report_details=None):
    jid = prepared["job_id"]
    job = store.job(jid)
    if job["status"] == "succeeded":
        return
    if store.session_id is None:
        store.start_session()
    initial = done = job["completed_units"]
    signature = f"conference-bootstrap:days{block_days}:daily{len(daily)}"
    estimator = ETAEstimator(store.get_meta("eta_history", {})) if observer is None else None
    job_started = last_heartbeat = time.monotonic()
    cost_family = f"bootstrap:days{block_days}:site{(report_details or {}).get('site', 'unknown')}:daily{len(daily)}"
    def notify(units=0, seconds=0.):
        info = dict(stage="bootstrap", label="bootstrap pareado", done=done, total=attempts,
                    unit="réplicas", status="completed" if store.job(jid)["status"] == "succeeded" else "running", contrast_id=contrast_id, block_days=block_days,
                    progress_key=jid, cost_family=cost_family, elapsed_seconds=time.monotonic()-job_started,
                    plan_complete=False, **(report_details or {}))
        if plan_state is not None:
            placeholders = ",".join("?" for _ in plan_state["jobs"])
            states = list(store.db.execute(
                f"SELECT job_id,status,total_units,completed_units FROM jobs WHERE job_id IN ({placeholders})",
                tuple(plan_state["jobs"]))) if plan_state["jobs"] else []
            info.update(jobs_total=len(states), jobs_done=sum(r["status"] == "succeeded" for r in states),
                        attempts_total=sum(r["total_units"] for r in states),
                        attempts_confirmed=sum(r["completed_units"] for r in states),
                        bootstrap_plan_complete=True,
                        bootstrap_remaining_plan=[dict(family=family, remaining_units=sum(
                            r["total_units"]-r["completed_units"] for r in states if plan_state["jobs"][r["job_id"]] == family))
                            for family in dict.fromkeys(plan_state["jobs"].values())])
        _notify_report(observer, signature, units, seconds, job_id=jid, kind="bootstrap", report_info=info)
    with store.claim(jid) as lease, Progress(attempts, initial, "none" if observer is not None else progress, jid) as display:
        notify()
        reservation = None
        started = time.monotonic()
        def check():
            nonlocal last_heartbeat
            _check_report_stop(stop)
            now = time.monotonic()
            # Include current uncommitted wall work when checking the cumulative
            # ceiling, rather than waiting for the whole bootstrap to finish.
            budget = store.get_meta("budget_seconds")
            live = store.get_meta("active_seconds", 0.) - sum(store._parent_budget_reservations.values())
            if reservation is not None:
                live += now-started-1.
            if budget is not None and live >= budget:
                store.set_meta("status", "paused_budget")
                raise BudgetExhausted("Cumulative budget during report bootstrap; uncommitted chunk will repeat")
            if now-last_heartbeat >= 1.:
                store.heartbeat(jid, lease.token)
                last_heartbeat = now
                notify()
        chunks = executor.chunks(prepared["arrays"], folds, block_days, initial, attempts, check)
        try:
            while done < attempts:
                check()
                store.check_budget()
                started = time.monotonic()
                reservation = store.begin_work(jid, lease.token, 1.)
                first, values = next(chunks)
                records = [dict(key=f"{jid}:replicate:{index:08d}", phase="bootstrap", contrast_id=contrast_id,
                                replicate_id=index, value=value, sample_seed=stable_seed(20261031, index),
                                status="invalid" if value is None else "valid")
                           for index, value in enumerate(values, first)]
                done = first+len(values)
                store.publish(jid, lease.token, records, done)
                store.finish_work(reservation, time.monotonic()-started)
                reservation = None
                store.snapshot()
                elapsed = time.monotonic()-started
                estimate = None
                if estimator is not None:
                    estimator.observe(signature, len(records), elapsed)
                    store.set_meta("eta_history", estimator.history)
                    estimate = estimator.estimate(attempts-done, signature)
                    store.set_meta("eta", estimate)
                notify(len(records), elapsed)
                display.update(done, estimate)
            store.finish(jid, lease.token)
            notify()
        except BaseException:
            if reservation is not None:
                store.finish_work(reservation, time.monotonic()-started)
            raise


def _persisted_daily_bootstrap(store, daily, calendars, contrast_id, attempts, block_days, folds, progress,
                               progress_observer=None, stop=None, *, executor=None, prepared=None,
                               plan_state=None, report_details=None):
    _check_report_stop(stop)
    prepared = prepared or _prepare_daily_bootstrap(daily, calendars, contrast_id, attempts, block_days, folds)
    if prepared is None:
        return dict(estimate=None, ci95=None, p_raw=None, confirmatory=False, attempted=0, invalid=0,
                    reason="Missing required fold-duration stratum")
    _register_daily_bootstrap(store, prepared, contrast_id, attempts, block_days)
    jid, estimate = prepared["job_id"], prepared["estimate"]
    if store.job(jid)["status"] != "succeeded" and store.get_meta("status") in {"completed", "completed_with_failures"}:
        store.set_meta("status", "planned")
    if executor is None:
        # Preserve the existing direct-call interruption granularity. The server
        # reporting entry point supplies its explicit larger runtime chunk size.
        with BootstrapExecutor(1, 100) as own:
            _run_daily_draws(store, prepared, daily, contrast_id, attempts, block_days, folds, progress,
                             progress_observer, stop, own, plan_state, report_details)
    else:
        _run_daily_draws(store, prepared, daily, contrast_id, attempts, block_days, folds, progress,
                         progress_observer, stop, executor, plan_state, report_details)
    _check_report_stop(stop)
    draws = sorted(store.results(jid),key=lambda r:r["replicate_id"])
    if [r["replicate_id"] for r in draws] != list(range(attempts)):
        raise RuntimeError("Bootstrap incomplete; no provisional inferential summary")
    valid = np.asarray([r["value"] for r in draws if r["value"] is not None])
    invalid = attempts-len(valid)
    blocks = sum(len(calendars[f])//block_days for f in folds)
    duration_days = {d:sum(row["duration"]==d for row in daily) for d in CORE_DURATIONS}
    supported = (blocks>=20 and all(len(calendars[f])//block_days>=4 for f in folds)
                 and all(n>=20 for n in duration_days.values()) and not invalid)
    p = float((1+np.sum(np.abs(valid-estimate)>=abs(estimate)))/(attempts+1)) if supported else None
    return dict(estimate=estimate,ci95=list(map(float,np.quantile(valid,[.025,.975]))) if len(valid) else None,
                p_raw=None,p_exploratory=p,confirmatory=False,interval_kind="exploratory_conditional_valid",
                attempted=attempts,invalid=invalid,support_blocks=blocks,duration_days=duration_days,
                temporal_support_sufficient=supported,job_id=jid,
                reason="Physical measurement semantics remain hypothetical" +
                ("; insufficient temporal support or invalid bootstrap draws" if not supported else ""))


def _plan_contrasts_sql(db, path, config, store, attempts, block_lengths, monitor=None):
    """Reduce paired episodes once, then register every eligible draw job up front."""
    folds = tuple(map(int, config.get("folds", (1, 2, 3))))
    sites = config.get("sites", [r[0] for r in db.execute("SELECT DISTINCT site FROM episode_losses")])
    hypotheses = config.get("physical_hypotheses", ())
    pairs = []
    if monitor is not None:
        monitor.stage("plan_bootstrap", "preparando contrastes e plano de bootstrap...",
                      total=len(hypotheses)*len(sites)*4, unit="contrastes")
    for hypothesis in hypotheses:
        for site in sites:
            calendars = _calendars(path, config, site, folds)
            for contrast in ("contrast_energy_gain", "contrast_complexity_gain"):
                for metric in ("nmae", "forecast_nmae"):
                    if monitor is not None:
                        monitor.check()
                    daily, unavailable = _daily_pairs(db, site, hypothesis, contrast, metric)
                    pairs.append(dict(site=site, hypothesis=hypothesis, contrast=contrast, endpoint=metric,
                                      daily=daily, calendars=calendars, unavailable=unavailable))
                    if monitor is not None:
                        monitor.advance(len(pairs))
    requests, jobs = [], {}
    with store.defer_snapshots(), store.transaction():
        for days in block_lengths:
            for pair in pairs:
                contrast_id = f"{pair['hypothesis']}:{pair['site']}:{pair['contrast']}:{pair['endpoint']}"
                prepared = None
                if set(pair["calendars"]) == set(folds) and pair["daily"] and not pair["unavailable"]:
                    prepared = _prepare_daily_bootstrap(pair["daily"], pair["calendars"], contrast_id, attempts, days, folds)
                    _register_daily_bootstrap(store, prepared, contrast_id, attempts, days)
                requests.append(dict(pair, block_days=days, contrast_id=contrast_id, prepared=prepared))
                if prepared is not None:
                    jobs[prepared["job_id"]] = f"bootstrap:days{days}:site{pair['site']}:daily{len(pair['daily'])}"
    return requests, dict(jobs=jobs), folds


def _execute_contrast_plan(requests, plan_state, folds, store, attempts, progress, observer, stop, executor):
    results = {}
    for request in requests:
        _check_report_stop(stop)
        days = request["block_days"]
        details = {key: request[key] for key in ("site", "hypothesis", "contrast", "endpoint")}
        if request["prepared"] is not None:
            result = _persisted_daily_bootstrap(store, request["daily"], request["calendars"], request["contrast_id"],
                attempts, days, folds, progress, observer, stop, executor=executor, prepared=request["prepared"],
                plan_state=plan_state, report_details=details)
        else:
            result = dict(estimate=None, ci95=None, p_raw=None, confirmatory=False, attempted=0, invalid=0,
                          reason="Missing matched eligible outputs, selected baseline, or full test calendars")
            if request["daily"] and not request["unavailable"] and set(request["calendars"]) == set(folds):
                result["reason"] = "Missing required fold-duration stratum"
        results.setdefault(days, []).append(dict(**details, physical_confirmatory=False, block_days=days,
            unmatched_or_unscorable=request["unavailable"], p_holm=None, **result))
    return results

def _contrasts_sql(db, path, config, chosen, store, attempts, block_days, progress,
                    progress_observer=None, stop=None):
    folds = tuple(map(int,config.get("folds",(1,2,3))))
    sites = config.get("sites",[r[0] for r in db.execute("SELECT DISTINCT site FROM episode_losses")])
    rows = []
    for hypothesis in config.get("physical_hypotheses",()):
        for site in sites:
            _check_report_stop(stop)
            calendars = _calendars(path,config,site,folds)
            for contrast in ("contrast_energy_gain","contrast_complexity_gain"):
                for metric in ("nmae","forecast_nmae"):
                    daily, unavailable = _daily_pairs(db,site,hypothesis,contrast,metric)
                    if set(calendars)==set(folds) and daily and not unavailable:
                        result = _persisted_daily_bootstrap(store,daily,calendars,
                            f"{hypothesis}:{site}:{contrast}:{metric}",attempts,block_days,folds,progress,
                            progress_observer,stop)
                    else:
                        result = dict(estimate=None,ci95=None,p_raw=None,confirmatory=False,attempted=0,invalid=0,
                                      reason="Missing matched eligible outputs, selected baseline, or full test calendars")
                    rows.append(dict(site=site,hypothesis=hypothesis,contrast=contrast,endpoint=metric,
                                     physical_confirmatory=False,block_days=block_days,
                                     unmatched_or_unscorable=unavailable,p_holm=None,**result))
    return rows


def generate_conference_report(run_dir, bootstrap_attempts=10000, block_days=7,
                               *, store=None, progress="none", progress_observer=None, stop=None,
                               report_workers=1, report_chunk_size=None, bootstrap_chunk_size=1000):
    """Report committed records and durably resume attempted bootstrap draws.

    Pass the runner's existing ``store`` to share its session and cumulative
    budget. Do not wrap this function in another timed work reservation.
    """
    if bootstrap_attempts <= 0 or block_days <= 0:
        raise ValueError("Positive bootstrap attempts and block days required")
    report_chunk_size = REPORT_CHUNK_SIZE if report_chunk_size is None else report_chunk_size
    for name, value in (("report_workers", report_workers), ("report_chunk_size", report_chunk_size),
                        ("bootstrap_chunk_size", bootstrap_chunk_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    monitor = _ReportMonitor(progress_observer, progress, stop, store)
    path = Path(run_dir)
    out = path / "reports"
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".energygap-report-source-", dir=out) as scratch:
        compact_path = Path(scratch) / "compact.sqlite"
        monitor.stage("compact_report_source", "preparando fonte compacta dos registros confirmados...")
        _compact_report_source(path / "state.sqlite", compact_path, monitor)
        return _generate_conference_report_from_source(path, out, compact_path, bootstrap_attempts, block_days,
            store, progress, progress_observer, stop, report_workers, report_chunk_size, bootstrap_chunk_size, monitor)


def _generate_conference_report_from_source(path, out, report_source, bootstrap_attempts, block_days,
        store, progress, progress_observer, stop, report_workers, report_chunk_size, bootstrap_chunk_size, monitor):
    def report_stage(message):
        monitor.stage("publish_summaries", message)
    with closing(sqlite3.connect(f"file:{path / 'state.sqlite'}?mode=ro", uri=True)) as ledger:
        meta = {k: json.loads(v) for k, v in ledger.execute("SELECT key,value FROM meta")}
    with closing(sqlite3.connect(f"file:{report_source}?mode=ro", uri=True)) as source:
        monitor.stage('read_fit_records', 'conferindo ajustes de modelos confirmados...')
        with _watch_sql(source, monitor):
            fit_records = [json.loads(r[0]) for r in source.execute(
                "SELECT payload FROM results WHERE json_extract(payload,'$.phase')='fit' ORDER BY result_key")]
        monitor.stage('export_forecasts', 'exportando previsões por horizonte...')
        with _watch_sql(source, monitor):
            n_forecast_records = _export_forecasts(source, out, chunk_size=report_chunk_size, monitor=monitor)
        _check_report_stop(stop)
        monitor.stage('export_energy_track', 'exportando a trilha de energia...')
        with _watch_sql(source, monitor):
            n_energy_track_records = _export_phase(source, out, "energy_track", "energy_track", chunk_size=report_chunk_size, monitor=monitor)
        _check_report_stop(stop)
        monitor.stage('export_compact_records', 'exportando os registros de inferência...')
        with _watch_sql(source, monitor):
            _compact_jsonl(source, out, chunk_size=report_chunk_size, monitor=monitor)
        _check_report_stop(stop)
    config = meta.get("config", {})
    sensitivity_days = config.get("experiment_settings", {}).get("block_sensitivity_days", [2, 14])
    if (not isinstance(sensitivity_days, (list, tuple)) or
            any(isinstance(d, bool) or not isinstance(d, int) or d <= 0 for d in sensitivity_days)):
        raise ValueError("Block sensitivities must be positive integer day lengths")
    sensitivity_days = list(dict.fromkeys(d for d in sensitivity_days if d != block_days))
    owns_store = store is None
    active_store = StateStore(path, resume="require") if owns_store else store
    monitor.store = active_store
    try:
        with tempfile.TemporaryDirectory(prefix=".energygap-analysis-", dir=out) as scratch:
            monitor.stage('aggregate_losses', 'agregando perdas em SQLite...')
            analysis = _build_analysis(report_source, Path(scratch) / "aggregation.sqlite", monitor=monitor)
            try:
                n_inference_records = analysis.execute("SELECT COUNT(*) FROM raw").fetchone()[0]
                n_fallback_records = analysis.execute("SELECT COUNT(*) FROM raw WHERE status='fallback'").fetchone()[0]
                n_episode_losses = analysis.execute("SELECT COUNT(*) FROM episode_losses").fetchone()[0]
                n_test_episode_losses = analysis.execute("SELECT COUNT(*) FROM episode_losses WHERE partition='test'").fetchone()[0]
                monitor.stage("export_episodes", "exportando métricas por consulta...", total=n_inference_records)
                _export_query(analysis, "SELECT * FROM raw", out / "episodes", _RAW_SCHEMA, chunk_size=report_chunk_size, monitor=monitor)
                id_fields = [(name, "INTEGER" if name in ('fold','mask_seed') else
                              "REAL" if name == 'duration' else "TEXT") for name in EPISODE_IDS]
                loss_fields = id_fields + [(m,"REAL") for m in METRICS] + [("latency_seconds","REAL"),("seed_evaluations","INTEGER")]
                monitor.stage("export_episode_losses", "exportando perdas por episódio...", total=n_episode_losses)
                _export_query(analysis,"SELECT * FROM episode_losses",out / "episode_losses",loss_fields, chunk_size=report_chunk_size, monitor=monitor)
                monitor.stage('select_baselines', 'selecionando baselines e exportando tabelas...')
                with _watch_sql(analysis, monitor):
                    chosen, selections = _selection_sql(analysis)
                    _materialize_tables(analysis, out, config, selections, chunk_size=report_chunk_size, monitor=monitor)
                    requests, plan_state, folds = _plan_contrasts_sql(analysis, path, config, active_store,
                        bootstrap_attempts, [block_days, *sensitivity_days], monitor)
                with BootstrapExecutor(report_workers, bootstrap_chunk_size) as executor:
                    grouped = _execute_contrast_plan(requests, plan_state, folds, active_store, bootstrap_attempts,
                        progress, progress_observer, stop, executor)
                contrasts = grouped.get(block_days, [])
                sensitivities = [row for days in sensitivity_days for row in grouped.get(days, [])]
            finally:
                analysis.close()
        if owns_store and active_store.session_id is not None:
            active_store.complete()
        jobs = [dict(r) for r in active_store.db.execute("SELECT job_id,status,reason,metadata,total_units,completed_units FROM jobs")]
        scope_design = study_design(path,config,fit_records,jobs,committed_coverage(active_store.db))
        runtime = active_store.get_meta("status")
    except BaseException:
        if owns_store and active_store.session_id is not None:
            active_store.end_session("paused_budget" if active_store.get_meta("status") == "paused_budget" else "interrupted")
        raise
    finally:
        if owns_store:
            active_store.close()
            monitor.store = None
    _export_frame(out / "fits", pd.DataFrame(fit_records))
    report_stage('publicando os resumos finais...')
    atomic_json(out / "contrasts.json", _jsonable(contrasts))
    atomic_json(out / "contrasts_block_sensitivity.json", _jsonable(sensitivities))
    compute_jobs = [j for j in jobs if json.loads(j["metadata"]).get("phase") != "report"]
    unresolved = [j for j in compute_jobs if j["status"] not in TERMINAL]
    availability = config.get("availability", {})
    summary = dict(profile=config.get("profile"),scope=config.get("scope"),
        runtime_execution_history=meta.get('runtime_execution_history', []),
        study_profile=scope_design['study_profile'],study_design=scope_design,
        scope_reduced_by_design=scope_design['scope_reduced_by_design'],full_catalogue_reproduced=False,
        execution_complete=bool(compute_jobs) and not unresolved,
        scientific_complete=False,scientific_experiment_complete=False,runtime_status=runtime,
        n_confirmed_inference_records=n_inference_records,n_forecast_records=n_forecast_records,
        n_energy_track_records=n_energy_track_records,n_fit_records=len(fit_records),
        n_fit_failures=sum(r.get("status") in {"fit_failed","failed","timeout","oom"} for r in fit_records),
        n_fallback_records=n_fallback_records,n_episode_losses=n_episode_losses,
        n_test_episode_losses=n_test_episode_losses,
        selected_B_validation={f"{s}:fold{f}":m for (s,f),m in chosen.items()},
        selected_B_scope="baseline methods only; validation C3-C5 with equal duration weights",
        candidate_availability=availability,
        blocked_methods=scope_design['requested_methods_unavailable'],
        omitted_methods_by_design=scope_design['omitted_methods_by_design'],
        omitted_conditions_by_design=scope_design['omitted_conditions_by_design'],
        unresolved_jobs=[{k:j[k] for k in ("job_id","status","reason")} for j in unresolved],
        n_confirmatory_tests=0,physical_confirmatory=False,physical_hypotheses=config.get("physical_hypotheses",[]),
        bootstrap_attempts=bootstrap_attempts,bootstrap_block_days=block_days,
        bootstrap_block_sensitivity_days=sensitivity_days,analysis_directory=str(out),
        report_backend="disk_backed_SQLite_aggregation_streamed_exports_daily_bootstrap",
        performance_table="performance.csv; grouped by timing mode and actual inference batch size",
        batched_latency_policy="Single-query latency is NA for batched outputs; amortized costs are throughput measurements",
        parallel_cpu_timing_policy="Correction worker service times exclude queue/native inference and do not estimate latency or pipeline throughput; consult ledger wall time",
        report_chunk_size=report_chunk_size, report_workers=report_workers, bootstrap_chunk_size=bootstrap_chunk_size,
        complete_compact_record_export="episodes_records.jsonl; durable prediction arrays remain in SQLite/shards",
        limitations=["Unknown measurement semantics prevent confirmatory physical conclusions.",
            "Available-method execution does not imply all eleven catalogue candidates were reproduced.",
            "Comparator B is restricted to baseline methods; learned models have descriptive comparisons.",
            "Training seeds are averaged within episodes; temporal support uses days, not seed counts.",
            "Batched inference costs describe amortized throughput, not isolated-query latency or total pipeline duration.",
            "Scientific completion requires reviewing coverage, unavailable methods and conditional assumptions."])
    atomic_json(out / "summary.json",_jsonable(summary))
    focused=scope_design['study_profile']=='focused_24h'
    title='Focused 24-hour-target conditional study' if focused else 'Available-method conditional study'
    budgets=scope_design['declared_training_budgets']
    scope_text=(f"Study profile: {scope_design['study_profile']}. Reduced scope by design: {scope_design['scope_reduced_by_design']}. "
        "This run does not reproduce the full method catalogue.\n\n"
        f"Requested methods: {', '.join(scope_design['requested_methods'])}. "
        f"Requested conditions: {', '.join(scope_design['requested_conditions']) or 'not recorded'}.\n\n"
        f"Methods omitted by design: {', '.join(scope_design['omitted_methods_by_design']) or 'none'}. "
        f"Conditions omitted by design: {', '.join(scope_design['omitted_conditions_by_design']) or 'none recorded'}. "
        f"Requested but unavailable methods: {', '.join(scope_design['requested_methods_unavailable']) or 'none'}.\n\n"
        f"Declared HPO trial indices: {budgets['hpo_trial_indices']}; maximum epochs/patience: "
        f"{budgets['hpo_max_epochs']}/{budgets['hpo_patience']}. Final maximum epochs/patience: "
        f"{budgets['final_max_epochs']}/{budgets['final_patience']}. "
        "Actual per-fit budgets and saved panel hashes/counts are in summary.json study_design.\n\n"
        f"Full declared-manifest HPO validation scoring completed: {scope_design['hpo_scoring']['full_declared_validation_scoring_completed']}. "
        f"Full declared-manifest final validation completed: {scope_design['final_validation']['full_declared_manifest_evaluation_completed']}. "
        f"Full declared-manifest test evaluation completed: {scope_design['test_evaluation']['full_declared_manifest_evaluation_completed']}. "
        f"Available-method manifest coverage (validation/test): {scope_design['final_validation']['full_available_manifest_evaluation_completed']}/"
        f"{scope_design['test_evaluation']['full_available_manifest_evaluation_completed']}. "
        "These coverage checks use mask manifests, job totals and committed records within this run's declared scope; "
        "they do not establish full-catalogue reproduction. A 24-hour target is not a completion guarantee.\n\n")
    text=(f"# {title}\n\n"+scope_text+
          f"Execution complete: {summary['execution_complete']}. Scientific study complete: False.\n\n"
          f"Committed inference records: {n_inference_records}. Test episode losses: {n_test_episode_losses}.\n\n"
          "Physical branches are exploratory hypotheses. No confirmatory physical p values are reported. "
          "B is chosen among baselines on validation only. Training-seed losses are averaged before episodes; "
          "core aggregates give equal weight to the three durations and folds.\n\n"
          "Reports use disk-backed aggregation and streamed exports. See summary.json, main_table.csv, "
          "core_aggregate.csv, selected_baseline_validation.csv, forecasts_long.csv, energy_track.csv, "
          "episodes_records.jsonl and contrasts.json. Missing support remains NA.\n\n"
          "Batched models have NA single-query latency in the main table. performance.csv separates "
          "timing modes and actual batch sizes, using measured query count divided by summed per-query "
          "cost for throughput. Batch wall times are retained for provenance and are not summed once "
          "per record. These inference/correction costs exclude other pipeline work. "
          "Parallel CPU correction service time is exported separately; it excludes queue and native "
          "inference time and is never converted into isolated-query latency or pipeline throughput.\n")
    atomic_bytes(out / "report.md",text.encode())
    return _jsonable(summary)


def refresh_conference_status(run_dir, *, store=None):
    """Refresh final operational metadata without rescanning inference records."""
    path = Path(run_dir)
    summary_path = path / "reports/summary.json"
    summary = json.loads(summary_path.read_text())
    db = store.db if store is not None else sqlite3.connect(
        f"file:{path / 'state.sqlite'}?mode=ro", uri=True)
    try:
        jobs = [dict(zip(("job_id", "status", "reason"), row)) for row in
                db.execute("SELECT job_id,status,reason FROM jobs")]
        raw_status = db.execute("SELECT value FROM meta WHERE key='status'").fetchone()
        summary["runtime_status"] = json.loads(raw_status[0]) if raw_status else None
        unresolved = [j for j in jobs if j["status"] not in TERMINAL]
        summary["execution_complete"] = bool(jobs) and not unresolved
        summary["unresolved_jobs"] = unresolved
    finally:
        if store is None:
            db.close()
    atomic_json(summary_path, _jsonable(summary))
    return _jsonable(summary)
