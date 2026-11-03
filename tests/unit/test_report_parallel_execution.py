"""Scientific-ID, draw and interruption parity for CPU report scheduling."""
from datetime import date, timedelta
import hashlib
import json
import time

import pytest

from energygap.pipeline import GracefulStop
from energygap.reporting.bootstrap_execution import BootstrapExecutor, worker_environment
from energygap.reporting.conference import (
    _bootstrap_kernel, _persisted_daily_bootstrap, _prepare_daily_bootstrap,
    _register_daily_bootstrap, generate_conference_report, _compact_report_source,
    _compact_jsonl, _export_forecasts, _export_phase, _build_analysis,
)
from energygap.runtime.signals import StopRequested
from energygap.runtime.state import BudgetExhausted, StateStore, canonical
from energygap.statistics.paired import stable_seed


def population():
    calendars = {fold: [(date(2025, fold, 1)+timedelta(days=i)).isoformat() for i in range(28)]
                 for fold in (1, 2)}
    daily, original = [], []
    for fold, days in calendars.items():
        for i, day in enumerate(days):
            for duration in (15, 60, 180):
                # A rare duration generates invalid draws; empty days remain in
                # the calendar and seeds/masks never acquire independent days.
                if i % 5 == 0 or (duration == 180 and i != 26):
                    continue
                value = (i+fold)/8
                daily.append(dict(fold=fold, day=day, duration=duration, total=2*value, count=2))
                original.extend(dict(fold=fold, day=day, duration=duration, difference=value,
                                     gap_id=f"{fold}:{i}:{duration}:{j}") for j in range(2))
    return daily, calendars, original


def test_spawn_parallel_draws_match_legacy_kernel_and_identity_including_invalids(tmp_path):
    daily, calendars, original = population()
    attempts, days, contrast = 79, 2, "instantaneous:H1:energy:nmae"
    reference = _bootstrap_kernel(original, calendars, block_days=days, seed=20261031, folds=(1, 2))
    identity = dict(daily=daily, calendars={str(k): v for k, v in calendars.items()},
                    contrast_id=contrast, attempts=attempts, block_days=days, seed=20261031,
                    folds=[1, 2], algorithm="paired-moving-block-sql-daily-v1")
    expected_id = f"bootstrap:conference:{contrast}:"+hashlib.sha256(canonical(identity)).hexdigest()[:20]
    with StateStore(tmp_path / "parallel", {"test": "bootstrap parity"}) as store:
        with BootstrapExecutor(2, 7) as executor:
            result = _persisted_daily_bootstrap(store, daily, calendars, contrast, attempts,
                days, (1, 2), "none", executor=executor)
        assert result["job_id"] == expected_id
        records = store.results(expected_id)
        assert [r["replicate_id"] for r in records] == list(range(attempts))
        assert [r["value"] for r in records] == [reference(i) for i in range(attempts)]
        assert [r["sample_seed"] for r in records] == [stable_seed(20261031, i) for i in range(attempts)]
        assert {r["status"] for r in records} == {"valid", "invalid"}
        assert result["invalid"] == sum(reference(i) is None for i in range(attempts))
        assert result["p_raw"] is None and result["confirmatory"] is False


def test_old_partial_shard_resumes_with_new_chunks_without_rewriting_committed_draws(tmp_path):
    daily, calendars, _ = population()
    contrast, attempts, days = "old-partial", 83, 7
    prepared = _prepare_daily_bootstrap(daily, calendars, contrast, attempts, days, (1, 2))
    jid = prepared["job_id"]
    with StateStore(tmp_path / "resumed", {"test": "resume parity"}) as store:
        _register_daily_bootstrap(store, prepared, contrast, attempts, days)
        with store.claim(jid) as lease:
            # Model a predecessor shard with its exact durable record schema.
            reference = _bootstrap_kernel(population()[2], calendars, block_days=days, seed=20261031, folds=(1, 2))
            old = [dict(key=f"{jid}:replicate:{i:08d}", phase="bootstrap", contrast_id=contrast,
                        replicate_id=i, value=reference(i), sample_seed=stable_seed(20261031, i),
                        status="invalid" if reference(i) is None else "valid") for i in range(13)]
            store.publish(jid, lease.token, old, 13)
            store.interrupt(jid, lease.token, "predecessor stopped")
        snapshots = dict(store.db.execute("SELECT result_key,payload FROM results"))
        observed = []
        def observer(phase, units, seconds, **metadata):
            if units:
                assert store.job(jid)["completed_units"] >= 13+sum(observed)+units
                observed.append(units)
        with BootstrapExecutor(2, 11) as executor:
            result = _persisted_daily_bootstrap(store, daily, calendars, contrast, attempts,
                days, (1, 2), "none", observer, executor=executor)
        assert sum(observed) == 70
        assert result["attempted"] == attempts
        for key, payload in snapshots.items():
            assert store.db.execute("SELECT payload FROM results WHERE result_key=?", (key,)).fetchone()[0] == payload
        assert [r["value"] for r in store.results(jid)] == [reference(i) for i in range(attempts)]
        assert len(list((store.path / "shards").glob("*.json"))) == 1+7


def test_spawn_workers_have_no_torch_and_one_blas_thread():
    with BootstrapExecutor(2, 1) as executor:
        pool = executor._ensure_pool()
        environments = [pool.apply_async(worker_environment).get(timeout=30) for _ in range(2)]
    assert all(not item["torch_imported"] for item in environments)
    assert all(p["num_threads"] == 1 for item in environments for p in item["threadpools"])


def test_parallel_signal_leaves_only_ordered_confirmed_chunks_and_resumes(tmp_path):
    daily, calendars, _ = population()
    stop = StopRequested()
    with StateStore(tmp_path / "signal", {"test": "parallel signal"}) as store:
        def observer(phase, units, seconds, **metadata):
            if units:
                stop.requested = True
        with pytest.raises(GracefulStop, match="committed report/bootstrap boundary"):
            with BootstrapExecutor(2, 9) as executor:
                _persisted_daily_bootstrap(store, daily, calendars, "signal", 77, 2,
                    (1, 2), "none", observer, stop, executor=executor)
        job = store.db.execute("SELECT job_id,status,completed_units FROM jobs").fetchone()
        assert job["status"] == "interrupted" and job["completed_units"] == 9
        old = store.results(job["job_id"])
        stop.requested = False
        with BootstrapExecutor(2, 13) as executor:
            _persisted_daily_bootstrap(store, daily, calendars, "signal", 77, 2,
                (1, 2), "none", stop=stop, executor=executor)
        assert store.results(job["job_id"])[:9] == old
        assert len(store.results(job["job_id"])) == 77


def test_live_bootstrap_budget_interrupts_an_uncommitted_chunk(tmp_path):
    daily, calendars, _ = population()
    class SlowExecutor:
        def chunks(self, arrays, folds, block_days, start, end, check):
            time.sleep(.03)
            check()
            yield start, []
    with StateStore(tmp_path / "budget", {"test": "budget"}, budget_seconds=.01) as store:
        with pytest.raises(BudgetExhausted, match="during report bootstrap"):
            _persisted_daily_bootstrap(store, daily, calendars, "budget", 77, 2,
                (1, 2), "none", executor=SlowExecutor())
        job = store.db.execute("SELECT job_id,status,completed_units FROM jobs").fetchone()
        assert job["completed_units"] == 0 and job["status"] == "interrupted"
        assert store.results(job["job_id"]) == []
        assert store.get_meta("active_seconds") >= .03


def test_export_signal_preserves_published_artifacts_and_cleans_temporary_files(tmp_path):
    path = tmp_path / "exports"
    with StateStore(path, {"sites": [], "folds": [], "physical_hypotheses": []}) as store:
        store.add_job("inference", 1, metadata={"phase": "inference"})
        with store.claim("inference") as lease:
            records = [dict(key=f"row:{i}", phase="inference", forecasts={"3600": {"nmae": .1}})
                       for i in range(20)]
            store.publish("inference", lease.token, records, 1)
            store.finish("inference", lease.token)
        store.complete()
    reports = path / "reports"
    reports.mkdir()
    published = reports / "forecasts_long.csv"
    published.write_text("previous complete export\n")
    stop = StopRequested()
    def observer(phase, units, seconds, **metadata):
        info = metadata.get("report_info", {})
        if info.get("stage") == "export_forecasts" and info.get("done", 0) >= 4:
            stop.requested = True
    with pytest.raises(GracefulStop):
        generate_conference_report(path, bootstrap_attempts=3, report_chunk_size=4,
                                   progress_observer=observer, stop=stop)
    assert published.read_text() == "previous complete export\n"
    assert not list(reports.glob(".*.tmp"))


def test_compact_report_source_strips_only_inference_arrays_and_preserves_all_report_values(tmp_path):
    import sqlite3
    import pandas as pd
    run = tmp_path / "large-ledger"
    large = [i/8 for i in range(2048)]
    records = []
    for i, (duration, condition) in enumerate(((15, "C3"), (60, "C4"), (180, "C5"))):
        records.append(dict(key=f"inference:{i}", phase="inference", site="H1", fold=1,
            partition="test", condition=condition, duration=duration, mask_seed=2001,
            method="linear", variant="P", gap_id=f"gap:{i}", day="2025-01-01",
            nmae=(i+1)/8, mae_kw=.3, forecast_nmae=.2, forecast_mae_kw=.1,
            latency_seconds=.01, status="success", prediction=large, samples=[large]*16,
            filled_missing_indices=list(range(len(large))), filled_missing_values=large,
            forecasts={"1800": dict(nmae=.125, mae_kw=.25, target_offset_seconds=0.),
                       "3600": dict(nmae=.25, mae_kw=.5, exclusion_reason=None)},
            correction_diagnostics={"kept": True, "residual": -.125},
            future_scalar=17.25, future_diagnostic={"also": "retained"}))
    records += [dict(key="fit:one", phase="fit", samples=[1, 2], prediction=[3], status="success"),
                dict(key="energy:one", phase="energy_track", samples=[4, 5], prediction=[6],
                     filled_missing_values=[7], track="separate", nmae=.125),
                dict(key="bootstrap:one", phase="bootstrap", replicate_id=0, value=.2)]
    with StateStore(run, {"test": "large report arrays"}) as store:
        store.add_job("data", 1)
        with store.claim("data") as lease:
            store.publish("data", lease.token, records, 1)
            store.finish("data", lease.token)
    source = run / "state.sqlite"
    compact = tmp_path / "compact.sqlite"
    assert _compact_report_source(source, compact) == 5
    with sqlite3.connect(source) as original, sqlite3.connect(compact) as reduced:
        originals = dict(original.execute("SELECT result_key,payload FROM results"))
        minimal = dict(reduced.execute("SELECT result_key,payload FROM results"))
        assert "bootstrap:one" not in minimal
        for key in ("fit:one", "energy:one"):
            assert minimal[key] == originals[key]
        for key in ("inference:0", "inference:1", "inference:2"):
            values = json.loads(minimal[key])
            expected = json.loads(originals[key])
            for field in ("prediction", "samples", "filled_missing_indices", "filled_missing_values"):
                expected.pop(field)
                assert field not in values
            assert values == expected
        # Every exported forecast, extra compact diagnostic, and energy-track
        # field must be identical when read from full or stripped JSON.
        directories = []
        for name, connection in (("full", original), ("compact", reduced)):
            out = tmp_path / name
            out.mkdir()
            _export_forecasts(connection, out, chunk_size=2)
            _export_phase(connection, out, "energy_track", "energy_track", chunk_size=2)
            _compact_jsonl(connection, out, chunk_size=2)
            directories.append(out)
        for filename in ("forecasts_long.csv", "energy_track.csv", "episodes_records.jsonl"):
            assert (directories[0] / filename).read_bytes() == (directories[1] / filename).read_bytes()
        for stem in ("forecasts_long", "energy_track"):
            pd.testing.assert_frame_equal(pd.read_parquet(directories[0] / f"{stem}.parquet"),
                                          pd.read_parquet(directories[1] / f"{stem}.parquet"), check_exact=True)
        assert dict(original.execute("SELECT result_key,payload FROM results")) == originals
    databases = [_build_analysis(path, tmp_path / f"analysis-{i}.sqlite") for i, path in enumerate((source, compact))]
    try:
        for table in ("raw", "episode_losses"):
            assert list(databases[0].execute(f"SELECT * FROM {table}")) == list(databases[1].execute(f"SELECT * FROM {table}"))
    finally:
        for db in databases:
            db.close()
    assert compact.stat().st_size < source.stat().st_size/10
