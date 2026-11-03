from datetime import date, timedelta

import pytest

from energygap.pipeline import GracefulStop
from energygap.reporting.conference import _persisted_daily_bootstrap
from energygap.runtime import scheduler
from energygap.runtime.signals import StopRequested
from energygap.runtime.state import StateStore


def test_observer_receives_committed_shards_without_local_bar_or_eta_overwrite(tmp_path, monkeypatch):
    modes = []
    original = scheduler.Progress
    def capture(total, initial, mode, desc):
        modes.append(mode)
        return original(total, initial, mode, desc)
    monkeypatch.setattr(scheduler, 'Progress', capture)
    with StateStore(tmp_path / 'run', {'test': 'observed scheduler'}) as store:
        store.add_job('boot', 5, metadata={'phase': 'bootstrap'})
        sentinel = {'remaining_seconds': 999, 'scope': 'global'}
        store.set_meta('eta', sentinel)
        observed = []
        def observer(phase, units, seconds, **metadata):
            observed.append((phase, units, seconds, metadata, store.job('boot')['completed_units']))
        result = scheduler.run_units(store, 'boot', lambda i, seed: dict(key=f'k:{i}', seed=seed),
                                     shard_size=2, progress='tqdm', signature='bootstrap:days7',
                                     progress_observer=observer, publish_eta=False)
        assert result['completed'] is True
        assert modes == ['none']
        assert [item[1] for item in observed] == [0, 2, 2, 1, 0]
        assert [item[4] for item in observed] == [0, 2, 4, 5, 5]
        assert all(item[2] >= 0 for item in observed)
        assert all(item[3] == dict(job_id='boot', kind='bootstrap') for item in observed)
        assert store.get_meta('eta') == sentinel
        assert store.get_meta('eta_history') is None
        repeated = scheduler.run_units(store, 'boot', lambda *_: pytest.fail('repeated committed work'),
                                       progress_observer=observer, publish_eta=False)
        assert repeated['executed_units'] == 0


def test_scheduler_default_still_publishes_local_eta(tmp_path):
    with StateStore(tmp_path / 'run', {'test': 'default scheduler'}) as store:
        store.add_job('a', 2)
        scheduler.run_units(store, 'a', lambda i, seed: dict(key=f'k:{i}'), signature='test')
        assert store.get_meta('eta')['remaining_seconds'] == 0
        assert store.get_meta('eta_history')['test'][0]['units'] == 2


def test_stopped_report_bootstrap_commits_once_and_resumes_without_failed_draws(tmp_path):
    stop = StopRequested()
    calendars = {1: [(date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(14)]}
    daily = [dict(fold=1, day=day, duration=duration, total=.1 + duration / 1000, count=1)
             for day in calendars[1] for duration in (15, 60, 180)]
    with StateStore(tmp_path / 'run', {'test': 'report interruption'}) as store:
        observations = []
        def observer(phase, units, seconds, **metadata):
            observations.append(units)
            if units:
                stop.requested = True
        with pytest.raises(GracefulStop, match='committed report/bootstrap boundary'):
            _persisted_daily_bootstrap(store, daily, calendars, 'test', 205, 7, (1,), 'none', observer, stop)
        job = store.db.execute('SELECT job_id,status,completed_units FROM jobs').fetchone()
        assert job['status'] == 'interrupted'
        assert job['completed_units'] == 100
        assert len(store.results(job['job_id'])) == 100
        stop.requested = False
        calls = []
        result = _persisted_daily_bootstrap(store, daily, calendars, 'test', 205, 7, (1,), 'none',
                                            lambda phase, units, seconds, **kw: calls.append(units), stop)
        assert result['attempted'] == 205
        assert sum(calls) == 105
        assert len(store.results(job['job_id'])) == 205
        assert {row['status'] for row in store.results(job['job_id'])} == {'valid'}
