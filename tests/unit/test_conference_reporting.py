from datetime import date, timedelta
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest
import energygap.reporting.conference as reporting

from energygap.reporting.conference import (
    _bootstrap_kernel, _episode_losses, _select_baselines, _export_forecasts,
    generate_conference_report)
from energygap.runtime.state import StateStore
from energygap.statistics.paired import bootstrap_replicate


def row(**overrides):
    value = dict(phase="inference", site="H1", fold=1, partition="test",
                 condition="C3", duration=15, mask_seed=2001, train_seed=None,
                 method="linear", variant="P", gap_id="g1", day="2025-01-01",
                 status="success", nmae=.1, mae_kw=.2, forecast_nmae=.3,
                 forecast_mae_kw=.4, latency_seconds=.01)
    value.update(overrides)
    return value


def ledger(path, config, rows):
    with StateStore(path, config=config) as store:
        store.add_job("inference", 1, metadata={"phase": "inference"})
        with store.claim("inference") as lease:
            store.publish("inference", lease.token,
                          [dict(r, key=f"result:{i}") for i, r in enumerate(rows)], 1)
            store.finish("inference", lease.token)
        store.complete()


def test_seed_losses_average_before_episodes_and_nonfinite_not_discarded():
    frame = pd.DataFrame([row(train_seed=11, nmae=0), row(train_seed=23, nmae=2),
                          row(gap_id="g2", nmae=10),
                          row(gap_id="g3", train_seed=11, nmae=None),
                          row(gap_id="g3", train_seed=23, nmae=1)])
    collapsed = _episode_losses(frame).set_index("gap_id")
    assert collapsed.loc["g1", "nmae"] == 1
    assert collapsed.loc["g2", "nmae"] == 10
    assert np.isnan(collapsed.loc["g3", "nmae"])
    assert collapsed.loc["g1", "seed_evaluations"] == 2


def test_batched_performance_roundtrip_keeps_latency_na_and_uses_ratio_of_costs(tmp_path):
    config = dict(profile='conference', sites=['H1'], folds=[1],
                  methods=['BRITS', 'SAITS', 'linear', 'median'], physical_hypotheses=[])
    rows = [
        row(method='BRITS', variant='E:instantaneous', gap_id='b1', latency_seconds=0.,
            timing_mode='batch_amortized', inference_batch_size=2, batch_wall_seconds=.8,
            inference_amortized_seconds=1.,requested_batch_size=256,adaptive_batch_size=2,
            oom_retries=7,oom_retry_seconds=.1,oom_retry_timing_policy='cumulative;not_additive_across_rows'),
        row(method='BRITS', variant='E:instantaneous', gap_id='b2', latency_seconds=None,
            timing_mode='batch_amortized', inference_batch_size=2, batch_wall_seconds=.8,
            inference_amortized_seconds=3.,requested_batch_size=256,adaptive_batch_size=2,
            oom_retries=7,oom_retry_seconds=.1,oom_retry_timing_policy='cumulative;not_additive_across_rows'),
        row(method='BRITS', variant='E:instantaneous', gap_id='partial', latency_seconds=None,
            timing_mode='batch_amortized', inference_batch_size=1, batch_wall_seconds=.5,
            inference_amortized_seconds=.7),
        row(method='SAITS', gap_id='missing', latency_seconds=None, timing_mode='batch_amortized',
            inference_batch_size=2, inference_amortized_seconds=None),
        row(method='linear', gap_id='legacy-zero', latency_seconds=0.),
        row(method='median', gap_id='legacy-missing', latency_seconds=None),
        row(method='BRITS', gap_id='scalar-retry', latency_seconds=.2, timing_mode='single_query',
            inference_batch_size=1, batch_retry_scalar=True,
            performance_reason='individual query retry after failed batch; ledger wall includes failed attempt'),
    ]
    path = tmp_path / 'timing-study'
    ledger(path, config, rows)
    result = generate_conference_report(path, bootstrap_attempts=2)
    exported = pd.read_csv(path / 'reports/episodes.csv').set_index('gap_id')
    parquet = pd.read_parquet(path / 'reports/episodes.parquet').set_index('gap_id')
    assert pd.isna(exported.loc['b1', 'latency_seconds'])  # a batch is never zero-latency
    for field in ('timing_mode', 'inference_batch_size', 'batch_wall_seconds', 'inference_amortized_seconds'):
        assert parquet.loc['b2', field] == exported.loc['b2', field]
    assert exported.loc['legacy-zero', 'timing_mode'] == 'single_query'
    assert exported.loc['legacy-zero', 'latency_seconds'] == 0.
    assert exported.loc['scalar-retry', 'batch_retry_scalar'] == 1
    assert 'retry after failed batch' in exported.loc['scalar-retry', 'performance_reason']
    assert pd.isna(exported.loc['legacy-missing', 'latency_seconds'])
    compact = {r['gap_id']: r for r in map(json.loads, (path / 'reports/episodes_records.jsonl').read_text().splitlines())}
    assert compact['b2']['inference_batch_size'] == 2
    assert compact['b2']['batch_wall_seconds'] == .8
    assert compact['b2']['oom_retry_timing_policy']=='cumulative;not_additive_across_rows'
    assert parquet.loc['b2','requested_batch_size']==256
    assert parquet.loc['b2','adaptive_batch_size']==2
    assert parquet.loc['b2','oom_retry_seconds']==.1
    main = pd.read_csv(path / 'reports/main_table.csv').set_index(['method', 'variant'])
    assert pd.isna(main.loc[('BRITS', 'E:instantaneous'), 'latency_p50_seconds'])
    assert 'Batched inference' in main.loc[('BRITS', 'E:instantaneous'), 'performance_reason']
    assert main.loc[('BRITS', 'P'), 'latency_p50_seconds'] == .2
    assert main.loc[('linear', 'P'), 'latency_p50_seconds'] == 0.
    assert pd.isna(main.loc[('median', 'P'), 'latency_p50_seconds'])
    performance = pd.read_csv(path / 'reports/performance.csv')
    batch = performance[(performance.method == 'BRITS') & (performance.inference_batch_size == 2)].iloc[0]
    assert batch.query_count == batch.timed_query_count == 2
    assert batch.total_measured_cost_seconds == 4.
    assert batch.mean_measured_cost_seconds == 2.
    assert batch.measured_queries_per_second == .5  # not mean(1/1,1/3), not repeated batch wall
    assert batch.oom_affected_query_count==2
    assert 'exclude failed attempts' in batch.performance_reason
    assert len(performance[performance.method == 'BRITS']) == 3  # partial batches and retries remain separate
    retry = performance[(performance.method == 'BRITS') & (performance.timing_mode == 'single_query')].iloc[0]
    assert retry.scalar_retry_query_count == 1
    assert 'exclude failed batch work' in retry.performance_reason
    zero = performance[performance.method == 'linear'].iloc[0]
    assert zero.total_measured_cost_seconds == 0.
    assert pd.isna(zero.measured_queries_per_second)
    missing = performance[performance.method == 'SAITS'].iloc[0]
    assert missing.timed_query_count == 0
    assert pd.isna(missing.total_measured_cost_seconds)
    assert pd.isna(missing.measured_queries_per_second)
    assert 'amortized' in result['batched_latency_policy']


def test_parallel_cpu_service_times_are_exported_without_latency_or_throughput_claim(tmp_path):
    config = dict(profile='conference', sites=['H1'], folds=[1],
                  methods=['linear'], physical_hypotheses=[])
    rows = [row(method='linear+energy_consistent', variant='E:instantaneous',
                gap_id=f'cpu-{i}', timing_mode='parallel_cpu_service_cost',
                inference_batch_size=1, energy_cpu_workers=8,
                correction_service_seconds=seconds, latency_seconds=None,
                inference_amortized_seconds=None) for i, seconds in enumerate((2., 4.))]
    path = tmp_path / 'parallel-study'
    ledger(path, config, rows)
    result = generate_conference_report(path, bootstrap_attempts=2)
    exported = pd.read_parquet(path / 'reports/episodes.parquet')
    assert sorted(exported.correction_service_seconds) == [2., 4.]
    assert exported.energy_cpu_workers.tolist() == [8, 8]
    assert exported.latency_seconds.isna().all()
    main = pd.read_csv(path / 'reports/main_table.csv')
    assert main.latency_p50_seconds.isna().all()
    assert main.performance_reason.str.contains('Parallel CPU correction').all()
    performance = pd.read_csv(path / 'reports/performance.csv').iloc[0]
    assert performance.total_correction_service_seconds == 6.
    assert pd.isna(performance.total_measured_cost_seconds)
    assert pd.isna(performance.measured_queries_per_second)
    assert 'no latency or throughput claim' in performance.performance_reason
    assert 'queue' in result['parallel_cpu_timing_policy']


def test_selection_uses_only_validation_baselines_and_equal_durations():
    rows = []
    for duration, condition in zip((15, 60, 180), ("C3", "C4", "C5")):
        for method, value in (("linear", .1), ("seasonal", .2), ("SAITS", 0)):
            rows.extend([row(duration=duration, condition=condition, method=method,
                             partition="validation", nmae=value),
                         row(duration=duration, condition=condition, method=method,
                             nmae=0 if method == "seasonal" else 100)])
    selected, scores = _select_baselines(_episode_losses(pd.DataFrame(rows)))
    assert selected == {("H1", 1): "linear"}
    assert {r["method"] for r in scores} == {"linear", "seasonal"}


def test_fast_bootstrap_matches_existing_estimator_including_empty_days():
    calendars = {fold: [(date(2025, 1, 1) + timedelta(days=i)).isoformat()
                        for i in range(15)] for fold in (1, 2, 3)}
    rows = [dict(fold=fold, day=day, duration=duration,
                 gap_id=f"{fold}:{i}:{duration}", difference=fold + i / 10 + duration / 1000)
            for fold, days in calendars.items() for i, day in enumerate(days)
            for duration in (15, 60, 180) if i % 4 != 0]
    fast = _bootstrap_kernel(rows, calendars, block_days=7, seed=20261031, folds=(1, 2, 3))
    for index in range(30):
        expected = bootstrap_replicate(rows, calendars, index)
        assert fast(index) == pytest.approx(expected, abs=1e-14)


def test_report_is_conditional_and_resume_does_not_repeat_draws(tmp_path):
    calendars = {str(fold): [(date(2025, 1, 1) + timedelta(days=i)).isoformat()
                             for i in range(56)] for fold in (1, 2, 3)}
    config = dict(profile="conference", scope="full_available_methods_conditional_physics",
                  sites=["H1"], folds=[1, 2, 3], methods=["linear", "FIM"],
                  availability={"linear": {"status": "available", "reason": None},
                                "FIM": {"status": "unavailable", "reason": "weights unavailable"}},
                  physical_hypotheses=["instantaneous"], test_calendars={"H1": calendars})
    rows = []
    for fold in (1, 2, 3):
        for duration, condition in zip((15, 60, 180), ("C3", "C4", "C5")):
            rows.append(row(fold=fold, duration=duration, condition=condition, partition="validation"))
            for i, day in enumerate(calendars[str(fold)]):
                base = row(fold=fold, duration=duration, condition=condition, day=day,
                           gap_id=f"g:{i}:{duration}", nmae=.1+i*.001+fold*.01)
                rows.extend([base, dict(base, method="linear+energy_consistent",
                                       variant="E:instantaneous", nmae=.05+i*.0003)])
    path = tmp_path / "isolated-analysis" / "study"
    ledger(path, config, rows)
    first = generate_conference_report(path, bootstrap_attempts=11)
    assert first["execution_complete"] is True
    assert first["scientific_complete"] is False
    assert first["n_confirmatory_tests"] == 0
    assert first["blocked_methods"]["FIM"]["reason"] == "weights unavailable"
    contrasts = json.loads((path / "reports/contrasts.json").read_text())
    assert len(contrasts) == 4
    assert all(r["confirmatory"] is False and r["p_raw"] is None and r["p_holm"] is None
               for r in contrasts)
    assert all(r["temporal_support_sufficient"] for r in contrasts)
    sensitivity = json.loads((path / "reports/contrasts_block_sensitivity.json").read_text())
    assert {r["block_days"] for r in sensitivity} == {2, 14}
    assert all(r["confirmatory"] is False and r["p_raw"] is None for r in sensitivity)
    table = pd.read_csv(path / "reports/main_table.csv")
    assert table.loc[table.method == "FIM", "status"].tolist() == ["unavailable"]
    with StateStore(path, resume="require") as store:
        bootstrap = {j[0]: j[1] for j in store.db.execute(
            "SELECT job_id,attempts FROM jobs WHERE job_id LIKE 'bootstrap:%'")}
        assert len(bootstrap) == 12  # four contrasts × three block lengths
        assert len([r for r in store.results() if r.get("phase") == "bootstrap"]) == 132
        job_id = next(j[0] for j in store.db.execute(
            "SELECT job_id FROM jobs WHERE json_extract(metadata,'$.contrast_id')=? "
            "AND json_extract(metadata,'$.block_days')=7", ("instantaneous:H1:contrast_energy_gain:nmae",)))
        actual = sorted(store.results(job_id), key=lambda r: r["replicate_id"])
        reference = [dict(fold=fold, day=day, duration=duration, gap_id=f"{i}:{duration}",
                          difference=(.05+i*.0003)-(.1+i*.001+fold*.01))
                     for fold in (1,2,3) for i,day in enumerate(calendars[str(fold)]) for duration in (15,60,180)]
        expected = [bootstrap_replicate(reference,{int(f):days for f,days in calendars.items()},i) for i in range(11)]
        np.testing.assert_allclose([r["value"] for r in actual],expected,rtol=1e-12,atol=1e-12)
    generate_conference_report(path, bootstrap_attempts=11)
    with StateStore(path, resume="require") as store:
        assert bootstrap == {j[0]: j[1] for j in store.db.execute(
            "SELECT job_id,attempts FROM jobs WHERE job_id LIKE 'bootstrap:%'")}


def test_missing_fold_and_unmatched_outputs_remain_na(tmp_path):
    config = dict(profile="conference", sites=["H1"], folds=[1, 2, 3], methods=["linear"],
                  physical_hypotheses=["interval_mean_start"])
    records = [row(partition="validation", duration=duration, condition=condition)
               for duration, condition in zip((15, 60, 180), ("C3", "C4", "C5"))]
    records += [row(), row(method="linear+energy_consistent", variant="E:interval_mean_start")]
    ledger(tmp_path / "study", config, records)
    generate_conference_report(tmp_path / "study", bootstrap_attempts=3)
    contrasts = json.loads((tmp_path / "study/reports/contrasts.json").read_text())
    assert all(r["estimate"] is None and r["attempted"] == 0 for r in contrasts)
    aggregate = pd.read_csv(tmp_path / "study/reports/core_aggregate.csv")
    assert aggregate.nmae.isna().all()


def test_all_forecast_horizons_stream_with_null_validation_and_explicit_exclusions(tmp_path):
    horizons = (1800, 3600, 10800, 21600)
    validation = {str(h): dict(mae_kw=None, nmae=None, target_offset_seconds=None,
                              exclusion_reason="validation_not_downstream_test") for h in horizons}
    forecasts = {str(h): dict(mae_kw=.2, nmae=.1, target_offset_seconds=0., exclusion_reason=None)
                 for h in horizons}
    forecasts["21600"] = dict(mae_kw=None, nmae=None, target_offset_seconds=None,
                              exclusion_reason="no_original_target_within_90s_in_test_fold")
    records = [row(partition="validation", forecasts=validation, prediction=[99.] * 512,
                   filled_missing_indices=[4, 5], filled_missing_values=[2., 3.]),
               row(forecasts=forecasts, prediction=[99.] * 512),
               dict(phase="fit", status="fit_failed", method="SAITS", reason="native failure")]
    config = dict(profile="conference", sites=["H1"], folds=[1], methods=["linear"],
                  physical_hypotheses=[])
    ledger(tmp_path / "study", config, records)
    summary = generate_conference_report(tmp_path / "study", bootstrap_attempts=3)
    assert summary["n_confirmed_inference_records"] == 2
    assert summary["n_fit_failures"] == 1
    assert summary["n_forecast_records"] == 8
    out = tmp_path / "study/reports"
    # Force all-null first chunks before measured test chunks.
    with sqlite3.connect(tmp_path / "study/state.sqlite") as db:
        assert _export_forecasts(db, out, chunk_size=1) == 8
    flat = pd.read_parquet(out / "forecasts_long.parquet")
    assert set(flat.horizon_seconds) == set(horizons)
    assert len(flat) == 8
    assert flat[flat.partition == "validation"].nmae.isna().all()
    missing = flat[(flat.partition == "test") & (flat.horizon_seconds == 21600)].iloc[0]
    assert pd.isna(missing.nmae)
    assert missing.exclusion_reason == "no_original_target_within_90s_in_test_fold"
    episodes = pd.read_csv(out / "episodes.csv")
    assert len(episodes) == 2
    assert not {"prediction", "filled_missing_indices", "filled_missing_values", "forecasts"} & set(episodes.columns)


def test_energy_donor_track_export_preserves_status_nulls_and_nested_diagnostics(tmp_path):
    config = dict(profile="conference", sites=["H1"], folds=[1], methods=["linear"], physical_hypotheses=[])
    records = [row(), dict(phase="energy_track", method="cpi_energy_donor_adapted", site="H1",
                          physical_hypothesis="interval_mean_start", status="unavailable", reason="missing counter",
                          conditional_power_mae_kw=None, confirmatory=False, donor_diagnostics={}),
               dict(phase="energy_track", method="cpi_energy_donor_adapted", site="H1",
                    physical_hypothesis="interval_mean_end", status="success", reason=None,
                    conditional_power_mae_kw=.2, confirmatory=False, donor_diagnostics={"import": {"donor_id": 7}})]
    ledger(tmp_path / "study", config, records)
    summary = generate_conference_report(tmp_path / "study", bootstrap_attempts=3)
    assert summary["n_energy_track_records"] == 2
    energy = pd.read_parquet(tmp_path / "study/reports/energy_track.parquet")
    assert len(energy) == 2
    assert set(energy.status) == {"success", "unavailable"}
    assert not energy.confirmatory.any()
    assert pd.isna(energy[energy.status == "unavailable"].iloc[0].conditional_power_mae_kw)
    assert json.loads(energy[energy.status == "success"].iloc[0].donor_diagnostics) == {"import": {"donor_id": 7}}


def test_disk_backed_report_uses_bounded_batches_and_sql_seed_means(tmp_path, monkeypatch):
    records = []
    for i in range(45):
        duration, condition = ((15,"C3"),(60,"C4"),(180,"C5"))[i % 3]
        for seed, loss, latency in ((11,float(i),.2*i+.1),(23,float(i+2),.8*i+.1)):
            records.append(row(gap_id=f"g{i}", duration=duration, condition=condition, train_seed=seed,
                               nmae=None if i==7 and seed==23 else loss, latency_seconds=latency,
                               custom_future_metric={"retained":i}, prediction=[99.]*512))
    records.extend(row(partition="validation",duration=d,condition=c)
                   for d,c in ((15,"C3"),(60,"C4"),(180,"C5")))
    config = dict(profile="conference",sites=["H1"],folds=[1],methods=["linear"],physical_hypotheses=[])
    ledger(tmp_path / "study",config,records)
    monkeypatch.setattr(reporting,"REPORT_CHUNK_SIZE",10)
    def forbidden(*args,**kwargs):
        raise AssertionError("Full inference DataFrame aggregation is forbidden")
    monkeypatch.setattr(reporting,"_episode_losses",forbidden)
    monkeypatch.setattr(reporting,"_contrast_rows",forbidden)
    real_export = reporting._export_query
    batches = []
    class Cursor:
        def __init__(self,cursor):self.cursor=cursor
        def fetchmany(self,size):
            assert size <= 10
            rows=self.cursor.fetchmany(size)
            batches.append(len(rows))
            return rows
        def fetchall(self):raise AssertionError("Unbounded fetchall forbidden")
    class Connection:
        def __init__(self,db):self.db=db
        def execute(self,*args):return Cursor(self.db.execute(*args))
    def guarded_export(db,*args,**kwargs):return real_export(Connection(db),*args,**kwargs)
    monkeypatch.setattr(reporting,"_export_query",guarded_export)
    summary=reporting.generate_conference_report(tmp_path / "study",bootstrap_attempts=3)
    assert summary["report_backend"] == "disk_backed_SQLite_aggregation_streamed_exports_daily_bootstrap"
    assert summary["n_confirmed_inference_records"] == 93
    assert summary["n_test_episode_losses"] == 45
    assert max(batches) == 10
    out=tmp_path / "study/reports"
    losses=pd.read_parquet(out / "episode_losses.parquet")
    assert losses[(losses.partition=="test") & (losses.gap_id=="g6")].iloc[0].nmae == 7
    assert pd.isna(losses[(losses.partition=="test") & (losses.gap_id=="g7")].iloc[0].nmae)
    main=pd.read_parquet(out / "main_table.parquet")
    for duration, offset in ((15,0),(60,1),(180,2)):
        value=main[main.duration==duration].iloc[0]
        if duration==60:assert pd.isna(value.nmae)
        else:assert value.nmae == pytest.approx(np.mean([i+1 for i in range(offset,45,3)]))
        expected=np.quantile([.5*i+.1 for i in range(offset,45,3)],[.5,.95])
        np.testing.assert_allclose([value.latency_p50_seconds,value.latency_p95_seconds],expected)
    compact=[json.loads(line) for line in (out / "episodes_records.jsonl").read_text().splitlines()]
    assert len(compact)==93
    assert compact[0]["custom_future_metric"] == {"retained":0}
    assert all("prediction" not in record for record in compact)
    assert not list(out.glob(".energygap-analysis-*"))
