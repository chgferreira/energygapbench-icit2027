"""Reporting counters are presentation, not additional scientific queries."""
import pytest

from energygap.runtime import conference_progress as progress
from energygap.runtime.state import StateStore


def _plan(store):
    store.add_job('inference', 10, metadata={'phase': 'inference', 'signature': 'infer:A:test:C3'})
    store.db.execute("UPDATE jobs SET completed_units=10,status='succeeded' WHERE job_id='inference'")
    store.add_job('report', 1, metadata={'phase': 'report'})


def _info(done, **extra):
    return dict(stage='export_forecasts', label='exportando previsões',
                progress_key='forecast-export', cost_family='export_forecasts',
                done=done, total=1000, unit='registros', status='running', **extra)


def _tick(display, clock, at, done, **extra):
    clock[0] = at
    return display.update('report', job_id='report', report_info=_info(done, **extra))


def test_report_wall_eta_is_calibrated_without_counting_exports_as_queries(tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(progress, 'perf_counter', lambda: clock[0])
    with StateStore(tmp_path / 'run', {'test': 'export progress'}) as store:
        _plan(store)
        display = progress.ConferenceProgress(store, 'none')
        display.training(dict(job_id='old-fit', stage='training', epoch=9))
        first = _tick(display, clock, 0., 0)
        assert first['current_training'] is None
        assert store.get_meta('current_training') is None
        one = _tick(display, clock, 2., 100)
        assert one['current_report']['current_stage_remaining_seconds'] is None
        two = _tick(display, clock, 4., 200)
        info = two['current_report']
        assert info['current_stage_remaining_seconds'] == pytest.approx(16.)
        assert info['current_stage_heuristic_range_seconds'] == pytest.approx([16., 16.])
        assert info['report_remaining_seconds'] is None
        assert (two['confirmed_queries'], two['total_queries']) == (10, 10)
        assert display._activity_counts() == (200, 1000, 'registros')
        assert 'exportando previsões' in two['activity']
        assert 'ETA etapa~16s' in display._activity_postfix()
        assert 'ETA relatório total não estimado' in display._activity_postfix()
        display.close()
        assert store.get_meta('report_progress')['done'] == 200
        assert store.get_meta('report_progress')['eta_calibration'] == 'empirical_wall_rate'


def test_report_refresh_reuses_counts_and_bounds_metadata_writes(tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(progress, 'perf_counter', lambda: clock[0])
    with StateStore(tmp_path / 'run', {'test': 'cheap report ticks'}) as store:
        _plan(store)
        display = progress.ConferenceProgress(store, 'none', persist_seconds=10)
        statements = []
        store.db.set_trace_callback(statements.append)
        _tick(display, clock, 0., 0)
        for index in range(1, 10):
            eta = _tick(display, clock, float(index), index * 10)
            assert eta['current_report']['done'] == index * 10
        assert sum('GROUP BY status' in sql for sql in statements) == 1
        assert sum(sql == 'COMMIT' for sql in statements) == 1
        assert not any('FROM results' in sql for sql in statements)
        assert store.get_meta('report_progress')['done'] == 0
        _tick(display, clock, 10., 100)
        assert store.get_meta('report_progress')['done'] == 100
        assert sum(sql == 'COMMIT' for sql in statements) == 2


def test_restart_uses_saved_rates_without_charging_offline_time(tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(progress, 'perf_counter', lambda: clock[0])
    with StateStore(tmp_path / 'run', {'test': 'restart report ETA'}) as store:
        _plan(store)
        display = progress.ConferenceProgress(store, 'none')
        _tick(display, clock, 0., 0)
        _tick(display, clock, 2., 100)
        _tick(display, clock, 4., 200)
        display.close()
        clock[0] = 10000.
        resumed = progress.ConferenceProgress(store, 'none')
        eta = _tick(resumed, clock, 10000., 200)
        assert eta['current_report']['current_stage_remaining_seconds'] == pytest.approx(16.)
        _tick(resumed, clock, 10002., 300)
        resumed.close()
        row = store.get_meta('conference_cost_history')['report_stage:export_forecasts']
        assert row['seconds'] == pytest.approx(6.)
        assert row['units'] == 300


def test_heartbeat_without_new_counter_does_not_erase_work_wall_time(tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(progress, 'perf_counter', lambda: clock[0])
    with StateStore(tmp_path / 'run', {'test': 'include waiting and commit overhead'}) as store:
        _plan(store)
        display = progress.ConferenceProgress(store, 'none')
        _tick(display, clock, 0., 0)
        _tick(display, clock, 8., 0)
        _tick(display, clock, 10., 100)
        _tick(display, clock, 20., 200)
        row = display.timings['report_stage:export_forecasts']
        assert row['seconds'] == pytest.approx(20.)
        assert row['units'] == 200


def test_bootstrap_has_distinct_family_and_persisted_human_context(tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(progress, 'perf_counter', lambda: clock[0])
    with StateStore(tmp_path / 'run', {'test': 'bootstrap status context'}) as store:
        _plan(store)
        display = progress.ConferenceProgress(store, 'none')
        _tick(display, clock, 0., 0)
        _tick(display, clock, 2., 100)
        _tick(display, clock, 4., 200)
        info = dict(stage='bootstrap', label='bootstrap pareado', done=100, total=10000,
                    unit='réplicas confirmadas', status='running', site='H1', hypothesis='instantaneous',
                    contrast='contrast_energy_gain', endpoint='nmae', block_days=7,
                    contrast_id='instantaneous:H1:contrast_energy_gain:nmae',
                    jobs_done=1, jobs_total=108, attempts_confirmed=10100, attempts_total=1080000)
        eta = display.update('conference-bootstrap:days7', 100, .001, job_id='bootstrap:one',
                             kind='bootstrap', report_info=info)
        current = eta['current_report']
        assert current['current_stage_remaining_seconds'] is None
        assert 'report_stage:bootstrap:days7' not in display.timings
        assert 'bootstrap' not in display.timings
        assert 'dataset H1' in eta['activity']
        assert 'bloco=7d' in eta['activity']
        assert 'tarefas bootstrap=1/108' in display._activity_postfix()
        assert 'réplicas confirmadas=10100/1080000' in display._activity_postfix()
        assert store.get_meta('report_progress')['site'] == 'H1'
        assert eta['confirmed_queries'] == 10


def test_unknown_stage_total_and_uncalibrated_remaining_plan_stay_unknown(tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(progress, 'perf_counter', lambda: clock[0])
    with StateStore(tmp_path / 'run', {'test': 'unknown report work'}) as store:
        _plan(store)
        display = progress.ConferenceProgress(store, 'none')
        _tick(display, clock, 0., 0)
        _tick(display, clock, 2., 100)
        _tick(display, clock, 4., 200)
        info = _info(200)
        info.update(total=None, plan_complete=True,
                    remaining_plan=[dict(family='export_forecasts', remaining_units=800),
                                    dict(family='figures', remaining_units=1)])
        eta = display.update(report_info=info)
        current = eta['current_report']
        assert current['current_stage_remaining_seconds'] is None
        assert current['report_remaining_seconds'] is None
        assert current['known_report_remaining_seconds'] == pytest.approx(16.)
        assert current['uncalibrated_report_families'] == ['figures']
        assert 'total da etapa desconhecido' in display._activity_postfix()
        info.update(total=1000, remaining_plan=[dict(family='export_forecasts', remaining_units=800)])
        assert display.update(report_info=info)['current_report']['report_remaining_seconds'] == pytest.approx(16.)


def test_completed_report_and_return_to_training_clear_stale_presentation(tmp_path):
    with StateStore(tmp_path / 'run', {'test': 'report lifecycle'}) as store:
        _plan(store)
        display = progress.ConferenceProgress(store, 'none')
        display.update(report_info=_info(1000))
        store.set_meta('status', 'completed')
        eta = display.update(force=True)
        assert eta['current_report']['report_remaining_seconds'] == 0.
        assert eta['activity'] == 'Execução concluída'
        store.set_meta('status', 'running')
        display.training(dict(job_id='fit:new', stage='training', epoch=1))
        assert store.get_meta('report_progress') is None
        assert display.current_report == {}


def test_bootstrap_plan_cannot_borrow_an_export_or_other_block_rate(tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(progress, 'perf_counter', lambda: clock[0])
    with StateStore(tmp_path / 'run', {'test': 'remaining bootstrap families'}) as store:
        _plan(store)
        display = progress.ConferenceProgress(store, 'none')
        plan = [dict(family='bootstrap:H1:days7', remaining_units=800),
                dict(family='bootstrap:H1:days14', remaining_units=1000)]
        def tick(at, done, family, key):
            clock[0] = at
            return display.update('conference-bootstrap', 100, .001, kind='bootstrap',
                report_info=dict(stage='bootstrap', label='bootstrap', progress_key=key,
                    cost_family=family, done=done, total=1000, unit='réplicas',
                    bootstrap_plan_complete=True, bootstrap_remaining_plan=plan, plan_complete=False))
        tick(0., 0, 'bootstrap:H1:days7', 'first')
        tick(2., 100, 'bootstrap:H1:days7', 'first')
        current = tick(4., 200, 'bootstrap:H1:days7', 'first')['current_report']
        assert current['bootstrap_remaining_seconds'] is None
        assert current['uncalibrated_bootstrap_families'] == ['bootstrap:H1:days14']
        tick(4., 0, 'bootstrap:H1:days14', 'second')
        tick(8., 100, 'bootstrap:H1:days14', 'second')
        current = tick(12., 200, 'bootstrap:H1:days14', 'second')['current_report']
        assert current['bootstrap_remaining_seconds'] == pytest.approx(800 * .02 + 1000 * .04)
        assert current['report_remaining_seconds'] is None
        assert 'bootstrap' not in display.timings
        assert display.timings['report_stage:bootstrap:H1:days14']['seconds'] == 8.


def test_real_report_callbacks_preserve_query_counts_and_publish_bootstrap_context(tmp_path):
    from energygap.reporting.conference import generate_conference_report

    days = ['2025-01-01', '2025-01-02']
    config = dict(profile='conference', sites=['H1'], folds=[1], methods=['linear'],
                  physical_hypotheses=['instantaneous'], test_calendars={'H1': {'1': days}})
    records = []
    for duration, condition in [(15, 'C3'), (60, 'C4'), (180, 'C5')]:
        base = dict(phase='inference', site='H1', fold=1, partition='validation', condition=condition,
                    duration=duration, mask_seed=2001, train_seed=None, method='linear', variant='P',
                    gap_id=f'validation:{duration}', day=days[0], status='success', nmae=.2,
                    mae_kw=.2, forecast_nmae=.3, forecast_mae_kw=.3, latency_seconds=.01)
        records.append(base)
        for day in days:
            test = dict(base, partition='test', gap_id=f'{day}:{duration}', day=day)
            records.extend([test, dict(test, method='linear+energy_consistent', variant='E:instantaneous', nmae=.1)])
    with StateStore(tmp_path / 'run', config) as store:
        store.add_job('inference', 1, metadata={'phase': 'inference'})
        with store.claim('inference') as lease:
            store.publish('inference', lease.token, [dict(row, key=f'row:{i}') for i, row in enumerate(records)], 1)
            store.finish('inference', lease.token)
        store.add_job('report', 1, metadata={'phase': 'report'})
        store.start_session()
        display = progress.ConferenceProgress(store, 'none')
        seen = []
        def observer(*args, **kwargs):
            seen.append(display.update(*args, **kwargs))
        result = generate_conference_report(store.path, bootstrap_attempts=3, store=store,
                                            progress_observer=observer, report_chunk_size=2,
                                            bootstrap_chunk_size=1)
        display.close()
        assert result['n_confirmed_inference_records'] == len(records)
        assert all(eta['confirmed_queries'] == eta['total_queries'] == 1 for eta in seen)
        bootstrap = [eta['current_report'] for eta in seen if eta['current_report']['stage'] == 'bootstrap']
        assert bootstrap
        assert all(info['site'] == 'H1' and info['hypothesis'] == 'instantaneous' for info in bootstrap)
        assert bootstrap[-1]['attempts_confirmed'] == bootstrap[-1]['attempts_total'] == 36
        assert bootstrap[-1]['jobs_done'] == bootstrap[-1]['jobs_total'] == 12
        assert all(info['report_remaining_seconds'] is None for info in bootstrap)
        assert store.get_meta('report_progress') is not None
