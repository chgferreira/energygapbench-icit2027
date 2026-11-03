import pytest

from energygap.runtime import conference_progress as progress
from energygap.runtime.state import StateStore


def add(store, name, total, phase, *, done=0, status='pending', **metadata):
    store.add_job(name, total, metadata=dict(phase=phase, **metadata))
    store.db.execute('UPDATE jobs SET completed_units=?,status=? WHERE job_id=?', (done, status, name))


def test_fit_hpo_scores_and_inference_have_separate_counts_and_rates(tmp_path):
    with StateStore(tmp_path / 'run', {'test': 'different workload units'}) as store:
        with store.transaction(), store.defer_snapshots():
            add(store, 'hpo:a', 101, 'hpo', method='A')
            add(store, 'hpo:b', 11, 'hpo', method='A', done=4)
            add(store, 'fit', 1, 'fit', method='A')
            add(store, 'inference', 200, 'inference', method='A', partition='test', condition='C3',
                signature='infer:A:test:C3', done=20)
            add(store, 'missing', 0, 'catalogue', status='unavailable')
            add(store, 'report', 1, 'report')
        display = progress.ConferenceProgress(store, 'none')
        display.update('fit:A', 1, 100)
        display.update('hpo_score:A', 10, 20)
        eta = display.update('infer:A:test:C3', 20, 10, force=True)
        # Two fits (one HPO and one final) do not consume the score-query rate.
        assert eta['known_remaining_seconds'] == pytest.approx(200 + 107 * 2 + 180 * .5)
        assert eta['confirmed_queries'] == 23
        assert eta['total_queries'] == 310
        assert (eta['resolved_jobs'], eta['total_jobs']) == (1, 6)
        assert eta['remaining_seconds'] is None
        assert eta['uncalibrated_signatures'] == ['report']
        assert 'ETA parcial=' in display._postfix(eta)
        assert 'indisponíveis=1' in display._postfix(eta)
        assert store.get_meta('conference_eta') == eta


def test_exact_architecture_keys_do_not_borrow_another_fit_rate(tmp_path):
    with StateStore(tmp_path / 'run', {'test': 'architecture timing'}) as store:
        add(store, 'fit64', 1, 'fit', method='A', timing_signature='fit:A:width64')
        add(store, 'fit128', 1, 'fit', method='A', timing_signature='fit:A:width128')
        display = progress.ConferenceProgress(store, 'none')
        eta = display.update('fit:A:width64', 1, 40)
        assert eta['known_remaining_seconds'] == 40
        assert eta['uncalibrated_fit_count'] == 1
        assert eta['uncalibrated_signatures'] == ['fit:A:width128']
        eta = display.update('fit:A:width128', 1, 80, force=True)
        assert eta['remaining_seconds'] == 120


def test_refresh_is_bounded_and_cost_history_is_bounded_without_losing_totals(tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(progress, 'perf_counter', lambda: clock[0])
    with StateStore(tmp_path / 'run', {'test': 'bounded progress'}) as store:
        add(store, 'queries', 10000, 'inference', signature='infer:A:test:C1')
        display = progress.ConferenceProgress(store, 'none')
        trace = []
        store.db.set_trace_callback(trace.append)
        display.update('infer:A:test:C1', 1, .1)
        first_refresh_queries = len(trace)
        for _ in range(9999):
            display.update('infer:A:test:C1', 1, .1)
        assert len(trace) == first_refresh_queries
        store.db.execute("UPDATE jobs SET completed_units=10000,status='succeeded'")
        display.close()
        history = store.get_meta('conference_cost_history')['infer:A:test:C1']
        assert len(history['observations']) == progress.MAX_OBSERVATIONS
        assert history['units'] == 10000
        assert history['seconds'] == pytest.approx(1000.)
        assert store.get_meta('conference_eta')['remaining_seconds'] == 0


def test_training_validation_is_visible_but_never_added_to_confirmed_queries(tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(progress, 'perf_counter', lambda: clock[0])
    with StateStore(tmp_path / 'run', {'test': 'epoch visibility'}) as store:
        add(store, 'hpo:a', 101, 'hpo', method='A')
        display = progress.ConferenceProgress(store, 'none')
        eta = display.training(dict(job_id='hpo:a', epoch=1, max_epochs=100, stage='validation',
                                    validation_queries_done=300, validation_queries_total=1000))
        assert eta['confirmed_queries'] == 0
        assert display._activity_postfix().startswith('val=300/1000 epoch=1/100')
        clock[0] = 2.
        eta = display.training(dict(job_id='hpo:a', epoch=2, stage='training', batches_done=0, batches_total=5))
        assert 'validation_queries_total' not in eta['current_training']
        assert 'batch=0/5' in display._activity_postfix()
        assert eta['confirmed_queries'] == 0
        clock[0] = 4.
        display.training(None)
        assert store.get_meta('current_training') is None


def test_empirical_ranges_and_resume_preserve_bounded_observations(tmp_path):
    with StateStore(tmp_path / 'run', {'test': 'resume costs'}) as store:
        add(store, 'queries', 20, 'inference', signature='infer:A:test:C1')
        display = progress.ConferenceProgress(store, 'none')
        display.update('infer:A:test:C1', 2, 4)
        eta = display.update('infer:A:test:C1', 4, 12, force=True)
        assert eta['remaining_seconds'] == pytest.approx(20 * 16 / 6)
        assert eta['heuristic_range_seconds'] == [40., 60.]
        assert 'not a confidence interval' in eta['range_rule']
        resumed = progress.ConferenceProgress(store, 'none')
        assert resumed.update(force=True)['remaining_seconds'] == eta['remaining_seconds']


def test_new_report_children_update_job_total_without_overwriting_global_eta(tmp_path):
    with StateStore(tmp_path / 'run', {'test': 'report children'}) as store:
        add(store, 'inference', 10, 'inference', done=10, status='succeeded', signature='infer:A:test:C1')
        add(store, 'report', 1, 'report', status='running')
        display = progress.ConferenceProgress(store, 'none')
        first = display.update()
        add(store, 'bootstrap', 100, 'bootstrap', done=20)
        eta = display.update('conference-bootstrap:days7:daily30', 20, 2, kind='bootstrap', force=True)
        assert (first['total_jobs'], eta['total_jobs']) == (2, 3)
        assert eta['known_remaining_seconds'] == pytest.approx(8.)
        assert eta['remaining_seconds'] is None
        assert eta['uncalibrated_signatures'] == ['report']
        assert eta['confirmed_queries'] == 10
        assert eta['total_queries'] == 10
        assert store.get_meta('eta')['total_jobs'] == 3


def test_first_completed_epoch_exposes_fit_time_scenarios_without_charging_a_fit(tmp_path):
    with StateStore(tmp_path / 'run', {'test': 'epoch scenarios'}) as store:
        add(store, 'fit', 1, 'fit', method='A')
        display = progress.ConferenceProgress(store, 'none')
        eta = display.training(dict(job_id='fit', stage='epoch_complete', epoch=1, epoch_seconds=12.,
                                    bad_epochs=0, patience=10, max_epochs=100))
        assert eta['current_fit_heuristic_range_seconds'] == [120., 1188.]
        assert eta['uncalibrated_fit_count'] == 1
        assert not display.timings
        assert 'treino~2.0min–19.8min' in display._activity_postfix()
        display.training(dict(job_id='fit', stage='epoch_complete', epoch=1, epoch_seconds=12.,
                              bad_epochs=0, patience=10, max_epochs=100))
        assert len(display.current_training['epoch_timing_observations']) == 1
        display.training(dict(job_id='fit', stage='epoch_complete', epoch=2, epoch_seconds=10.,
                              bad_epochs=1, patience=10, max_epochs=100))
        assert display.current_training['fit_remaining_heuristic_range_seconds'] == [90., 1176.]
        display.close()
        resumed = progress.ConferenceProgress(store, 'none')
        assert len(resumed.current_training['epoch_timing_observations']) == 2
        resumed.training(dict(job_id='another-fit', stage='epoch_start', epoch=1))
        assert 'fit_remaining_heuristic_range_seconds' not in resumed.current_training
        resumed.training(dict(job_id='another-fit', stage='inference', shard_queries_done=3, shard_queries_total=16))
        assert 'epoch' not in resumed.current_training
        resumed.update(force=True)
        assert 'shard=3/16' in resumed._activity_postfix()
        resumed.stage('report', job_id='report')
        assert resumed.current_training == {}
        resumed.update(force=True)
        assert 'shard=' not in resumed._activity_postfix()


@pytest.mark.parametrize('catalogue', [False, True])
def test_empty_or_unavailable_only_plan_has_zero_remaining_work(tmp_path, catalogue):
    with StateStore(tmp_path / 'run', {'test': 'no executable work'}) as store:
        if catalogue:
            add(store, 'missing', 0, 'catalogue', status='unavailable')
        display = progress.ConferenceProgress(store, 'none')
        eta = display.update()
        assert eta['remaining_seconds'] == 0
        assert eta['heuristic_range_seconds'] == [0., 0.]
        assert eta['uncalibrated_signatures'] == []
        assert eta['resolved_jobs'] == eta['total_jobs'] == int(catalogue)
        assert eta['confirmed_queries'] == eta['total_queries'] == 0


def test_long_labels_do_not_precede_or_truncate_validation_denominator(tmp_path):
    with StateStore(tmp_path / 'run', {'test': 'narrow terminal progress'}) as store:
        add(store, 'fit', 1, 'fit', method='A')
        display = progress.ConferenceProgress(store, 'none')
        job = 'hpo:H1:very_long_architecture_name:' + 'configuration-abc' * 20
        eta = display.training(dict(job_id=job, stage='validation', epoch=1, max_epochs=100,
                                    validation_queries_done=27, validation_queries_total=1625,
                                    inference_batch_index=1, inference_batches_total=51, inference_batch_size=32))
        line = display._activity_postfix()
        assert line.startswith('val=27/1625 epoch=1/100 lote=1/51 n=32')
        assert len(progress._short_label(job)) == 48
        assert display.current_job == job
        assert eta['total_queries'] == 0


def test_validation_refresh_reuses_workload_counts_and_bounds_durable_writes(tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(progress, 'perf_counter', lambda: clock[0])
    with StateStore(tmp_path / 'run', {'test': 'batch presentation is bounded'}) as store:
        add(store, 'fit', 1, 'fit', method='A')
        display = progress.ConferenceProgress(store, 'none', persist_seconds=10)
        statements = []
        store.db.set_trace_callback(statements.append)
        display.training(dict(job_id='fit', stage='validation', epoch=1,
                              validation_queries_done=0, validation_queries_total=1000))
        first_revision = store.progress_revision
        for done in range(1, 10):
            clock[0] = float(done)
            eta = display.training(dict(job_id='fit', stage='validation',
                                        validation_queries_done=done * 32))
            assert eta['current_training']['validation_queries_done'] == done * 32
        assert store.progress_revision == first_revision
        assert sum('GROUP BY status' in sql for sql in statements) == 1
        assert sum(sql == 'COMMIT' for sql in statements) == 1
        assert store.get_meta('current_training')['validation_queries_done'] == 0
        clock[0] = 10.
        display.training(dict(job_id='fit', stage='validation', validation_queries_done=320))
        assert store.get_meta('current_training')['validation_queries_done'] == 320
        assert sum('GROUP BY status' in sql for sql in statements) == 1
        assert sum(sql == 'COMMIT' for sql in statements) == 2


def test_work_commit_refreshes_exact_counts_inside_display_interval(tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(progress, 'perf_counter', lambda: clock[0])
    with StateStore(tmp_path / 'run', {'test': 'work bypasses presentation throttle'}) as store:
        add(store, 'queries', 10, 'inference', signature='infer:A:test:C1')
        display = progress.ConferenceProgress(store, 'none', refresh_seconds=60, persist_seconds=60)
        assert display.update()['confirmed_queries'] == 0
        with store.claim('queries') as lease:
            store.publish('queries', lease.token, [dict(key='first', value=1)], 1)
            eta = display.update('infer:A:test:C1', 1, .5)
            assert eta['confirmed_queries'] == 1
            assert store.get_meta('conference_eta')['confirmed_queries'] == 1
            revision = store.progress_revision
            store.heartbeat('queries', lease.token)
            assert store.progress_revision == revision
            statements = []
            store.db.set_trace_callback(statements.append)
            display.training(dict(job_id='queries', stage='inference', shard_queries_done=1))
            assert not any('GROUP BY status' in sql for sql in statements)


def test_live_counters_do_not_wait_for_presentation_or_persistence_interval(tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(progress, 'perf_counter', lambda: clock[0])
    with StateStore(tmp_path / 'run', {'test': 'current counters remain exact'}) as store:
        add(store, 'fit', 1, 'fit', method='A')
        display = progress.ConferenceProgress(store, 'none', refresh_seconds=5, persist_seconds=60)
        first = display.training(dict(job_id='fit', stage='validation', validation_queries_done=0))
        clock[0] = .01
        latest = display.training(dict(job_id='fit', stage='validation', validation_queries_done=32))
        assert latest['current_training']['validation_queries_done'] == 32
        assert first['current_training']['validation_queries_done'] == 0
        display.close()
        assert store.get_meta('current_training')['validation_queries_done'] == 32


def test_full_status_uses_one_ordered_job_query_and_preserves_decoded_api(tmp_path):
    with StateStore(tmp_path / 'run', {'test': 'status query scales with plan'}) as store:
        with store.defer_snapshots(), store.transaction():
            for index in range(100):
                store.add_job(f'job:{index:03}', index + 1,
                              dependencies=['job:000'] if index else [],
                              metadata=dict(phase='inference', index=index))
        statements = []
        store.db.set_trace_callback(statements.append)
        status = store.status()
        job_reads = [sql for sql in statements if sql.startswith('SELECT') and 'FROM jobs' in sql]
        assert job_reads == ['SELECT * FROM jobs ORDER BY job_id']
        assert status['total_jobs'] == 100
        assert status['total_units'] == 5050
        assert status['jobs'][9]['metadata'] == dict(phase='inference', index=9)
        assert status['jobs'][9]['dependencies'] == ['job:000']
        assert status['jobs'][9] == store.job('job:009')


def test_cached_workload_rolls_back_with_its_transaction(tmp_path):
    with StateStore(tmp_path / 'run', {'test': 'cached counts after rollback'}) as store:
        add(store, 'fit', 1, 'fit', method='A')
        display = progress.ConferenceProgress(store, 'none')
        assert display.update()['resolved_jobs'] == 0
        with pytest.raises(RuntimeError, match='rollback'):
            with store.transaction():
                store.db.execute("UPDATE jobs SET status='succeeded',completed_units=1")
                assert display.update()['resolved_jobs'] == 1
                raise RuntimeError('rollback')
        assert display.update()['resolved_jobs'] == 0
        assert store.get_meta('conference_eta')['resolved_jobs'] == 0


def test_activity_names_model_dataset_fold_seed_and_training_stage(tmp_path):
    config = {'experiment_settings': {'hpo_trials': 6, 'folds': [1, 2, 3], 'train_seeds': [11, 23, 47]}}
    with StateStore(tmp_path / 'run', config) as store:
        job = 'hpo:H1:BRITS:0'
        add(store, job, 101, 'hpo', method='BRITS', site='H1', trial=0)
        display = progress.ConferenceProgress(store, 'none')
        eta = display.training(dict(job_id=job, stage='training', epoch=2, max_epochs=100,
                                    batches_done=3, batches_total=20))
        assert eta['activity'] == 'Treinando BRITS no dataset H1 | fold 1 | seed 11 | HPO 1/6'
        assert display._activity_counts() == (3, 20, 'batch de treino')
        eta = display.training(dict(job_id=job, stage='validation', epoch=2,
                                    validation_queries_done=256, validation_queries_total=1625))
        assert eta['activity'].startswith('Validando BRITS no dataset H1')
        assert display._activity_counts() == (256, 1625, 'consulta de validação')
        assert eta['confirmed_queries'] == 0
        assert store.get_meta('conference_eta')['activity'] == eta['activity']
        display._settings.update(hpo_trials=2, hpo_trial_indices=[3, 5])
        eta = display.training(dict(job_id='hpo:H1:BRITS:5', stage='validation', epoch=1,
            max_epochs=20, validation_queries_done=256, validation_queries_total=300,
            validation_scope='fixed_panel', validation_full_queries_total=1625))
        assert 'HPO 2/2' in eta['activity']
        assert 'painel fixo de 1625 consultas disponíveis' in eta['activity']
        assert display._activity_counts() == (256, 300, 'consulta de validação')


def test_activity_metadata_read_is_once_per_job_not_per_validation_batch(tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(progress, 'perf_counter', lambda: clock[0])
    with StateStore(tmp_path / 'run', {'test': 'labels are cheap'}) as store:
        job = 'H2:fold2:SAITS:seed23:fit'
        add(store, job, 1, 'fit', method='SAITS', site='H2', fold=2, train_seed=23)
        display = progress.ConferenceProgress(store, 'none')
        statements = []
        store.db.set_trace_callback(statements.append)
        for index in range(20):
            clock[0] += 1
            eta = display.training(dict(job_id=job, stage='validation', epoch=1,
                validation_queries_done=index * 32, validation_queries_total=1000))
        job_reads = [sql for sql in statements if sql.startswith('SELECT * FROM jobs WHERE job_id=')]
        assert len(job_reads) == 1
        assert eta['activity'] == 'Validando SAITS no dataset H2 | fold 2 | seed 23'


def test_completed_run_clears_active_training_and_reports_its_final_state(tmp_path):
    with StateStore(tmp_path / 'run', {'test': 'final activity'}) as store:
        add(store, 'fit', 1, 'fit', method='BRITS', site='H1')
        display = progress.ConferenceProgress(store, 'none')
        display.training(dict(job_id='fit', stage='training', epoch=1, batches_done=1, batches_total=5))
        store.set_meta('status', 'completed')
        eta = display.update(force=True)
        assert eta['activity'] == 'Execução concluída'
        assert eta['current_training'] is None
        assert display._activity_counts() == (0, 0, 'etapa')
        assert 'BRITS' not in display._activity_label()
