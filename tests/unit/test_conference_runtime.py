"""Behavioral checks for the high-volume runner's bounded-work primitives."""
from dataclasses import replace
from datetime import timedelta
import json
import math
import sqlite3

import numpy as np
import pytest

from energygap.data import SeriesData, make_fixture
from energygap.forecasting.ridge import DirectRidge, select_alpha
from energygap.masking import Episode
from energygap.protocols import build_query
from energygap.reporting.conference import _episode_losses, _select_baselines
from energygap.runtime.state import StateStore
import pandas as pd


def test_query_age_matches_full_history_with_hidden_and_out_of_order_arrivals():
    series = make_fixture(days=5, natural_missing=False)
    rows = series.records.copy()
    episode = Episode("delayed", (1005, 1006, 1007), 1004, 1008, 15, 20, "C3", 2001, 1, "test")
    issuance = rows[1008].availability_time
    # The closest earlier grid rows cannot establish an observed age: they are
    # late, replica-hidden, not yet measured, or naturally missing respectively.
    rows[1004] = replace(rows[1004], availability_time=issuance + timedelta(minutes=20))
    rows[1002] = replace(rows[1002], measurement_time=issuance + timedelta(minutes=1))
    rows[1001] = replace(rows[1001], natural_missing=True, observed_original=False)
    series = SeriesData(rows, series.site_id)
    hidden = frozenset((*episode.indices, 1003))
    query, truth = build_query(series, episode, context_length=4,
                               artificial_hidden=hidden, issuance_time=issuance)
    last, expected = None, []
    for i, record in enumerate(rows[:episode.right_index + 1]):
        admitted = (i not in hidden and record.observed_original and not record.natural_missing and
                    record.available_to_method and math.isfinite(record.value_canonical) and
                    record.availability_time <= issuance and record.measurement_time <= issuance)
        if admitted:
            last = record.grid_time.timestamp()
        if i >= 1005:
            expected.append(math.inf if last is None else record.grid_time.timestamp() - last)
    assert query.elapsed_since_observed == tuple(expected) == (1500., 1800., 2100., 0.)
    assert query.available_to_method == (False, False, False, True)
    assert all(math.isnan(v) for v in query.values[:3])
    assert truth.values[:3] == tuple(rows[i].value_canonical for i in episode.indices)


def test_regular_query_reads_window_and_nearest_preceding_observation_only():
    series = make_fixture(days=10, natural_missing=False)

    class CountedRows:
        def __init__(self, records):
            self.records, self.reads = records, []

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            self.reads.append(index)
            return self.records[index]

        def __iter__(self):
            raise AssertionError("P query must not iterate the entire data series")

    counted = CountedRows(series.records)
    series.records = counted
    episode = Episode("late-window", (2000, 2001, 2002), 1999, 2003, 15, 20, "C3", 2001, 1, "test")
    query, _ = build_query(series, episode, context_length=16,
                            artificial_hidden=frozenset(episode.indices))
    assert len(query.values) == 16
    assert len(counted.reads) < 60
    assert min(counted.reads) == 1987  # one observation immediately before the window


def test_validation_baseline_selection_cannot_be_changed_by_replay_condition():
    records = []
    for method, score in (("linear", .1), ("seasonal", .2)):
        for duration, condition in zip((15, 60, 180), ("C3", "C4", "C5")):
            records.append(dict(site="H1", fold=1, partition="validation", condition=condition,
                                duration=duration, mask_seed=1001, method=method, variant="P",
                                gap_id=condition, day="2025-01-01", nmae=score, latency_seconds=.1))
        records.append(dict(site="H1", fold=1, partition="validation", condition="C6",
                            duration=15, mask_seed=1001, method=method, variant="P",
                            gap_id="replay", day="2025-01-01", nmae=100 if method == "linear" else 0,
                            latency_seconds=.1))
    selected, _ = _select_baselines(_episode_losses(pd.DataFrame(records)))
    assert selected == {("H1", 1): "linear"}


@pytest.mark.parametrize("backend", ["scipy", "torch"])
def test_ridge_predict_many_matches_scalar_with_padding_and_natural_gaps(backend):
    if backend == "torch":
        pytest.importorskip("torch")
    times = np.arange(110) * 300.
    values = np.sin(np.arange(110) / 11) + np.arange(110) / 100
    values[44] = np.nan
    model = DirectRidge(history=20, horizon_steps=6, device="cpu", backend=backend).fit(values, times)
    histories = [values[8:12], values[25:45], np.array([np.nan, 0., np.nan]), values[30:80]]
    issuances = [times[i] for i in (11, 44, 60, 79)]
    expected = np.array([model.predict(h, t) for h, t in zip(histories, issuances)])
    np.testing.assert_allclose(model.predict_many(histories, issuances), expected, rtol=1e-12, atol=1e-12)
    assert model.predict_many([], []).shape == (0,)


@pytest.mark.parametrize("horizon_steps", [6, 12, 36, 72])
def test_alpha_selection_batch_matches_scalar_at_each_physical_horizon(horizon_steps):
    train_times = np.arange(420) * 300.
    val_times = np.arange(420, 455) * 300.
    train = np.sin(train_times / 86400 * 2 * np.pi) + np.cos(train_times / 8600) / 10
    validation = np.sin(val_times / 86400 * 2 * np.pi) + np.cos(val_times / 8600) / 10
    train[[0, 52, 390]], validation[[2, 15]] = np.nan, np.nan
    val_times[10] += 180  # this original target must fail the fixed 90 s clock match
    candidates = (.1, 1.)
    combined, times = np.r_[train, validation], np.r_[train_times, val_times]
    expected = []
    for alpha in candidates:
        model = DirectRidge(alpha=alpha, horizon_steps=horizon_steps).fit(train, train_times)
        errors = []
        for j in range(len(train), len(combined)):
            i = j - horizon_steps
            if i < 0 or not np.isfinite(combined[j]) or abs(times[j] - times[i] - horizon_steps * 300) > 90:
                continue
            history = combined[max(0, i - 287):i + 1]
            errors.append(abs(model.predict(history, times[i]) - combined[j]))
        expected.append((float(np.mean(errors)), alpha))
    selected, scores = select_alpha(train, train_times, validation, val_times,
                                    candidates=candidates, horizon_steps=horizon_steps, device="cpu")
    assert selected == min(expected)[1]
    np.testing.assert_allclose(scores, expected, rtol=1e-10, atol=1e-12)


def test_deferred_snapshots_keep_sqlite_committed_and_flush_after_nested_exception(tmp_path):
    with StateStore(tmp_path / "run", config={"profile": "test"}) as store:
        store.snapshot(force=True)
        baseline = (store.path / "progress.json").read_bytes()
        with pytest.raises(RuntimeError, match="stop planning"):
            with store.defer_snapshots():
                store.add_job("first", 3)
                with store.defer_snapshots():
                    store.add_job("second", 5, dependencies=["first"])
                    assert (store.path / "progress.json").read_bytes() == baseline
                    with sqlite3.connect(store.path / "state.sqlite") as reader:
                        assert reader.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2
                assert (store.path / "progress.json").read_bytes() == baseline
                raise RuntimeError("stop planning")
        snapshot = json.loads((store.path / "progress.json").read_text())
        assert snapshot["total_jobs"] == 2
        assert snapshot["total_units"] == 8
        assert store._defer_snapshots == 0


def test_snapshot_throttle_does_not_delay_sqlite_or_forced_export(tmp_path, monkeypatch):
    now = [10.]
    monkeypatch.setattr("energygap.runtime.state.time.monotonic", lambda: now[0])
    with StateStore(tmp_path / "run", config={"profile": "test"}) as store:
        store.snapshot_interval_seconds = 60
        store.snapshot(force=True)
        store.add_job("first", 1)
        assert json.loads((store.path / "progress.json").read_text())["total_jobs"] == 0
        with sqlite3.connect(store.path / "state.sqlite") as reader:
            assert reader.execute("SELECT total_units FROM jobs WHERE job_id='first'").fetchone()[0] == 1
        now[0] = 71.
        store.snapshot()
        assert json.loads((store.path / "progress.json").read_text())["total_jobs"] == 1
        store.add_job("second", 2)
        assert json.loads((store.path / "progress.json").read_text())["total_jobs"] == 1
        store.snapshot(force=True)
        assert json.loads((store.path / "progress.json").read_text())["total_jobs"] == 2


def test_interrupted_fit_never_calibrates_complete_fit_eta(tmp_path):
    from energygap.conference import _finish_unit, GlobalProgress
    from energygap.pipeline import GracefulStop
    from energygap.runtime.signals import StopRequested
    with StateStore(tmp_path / 'run', config={'profile': 'test'}) as store:
        store.add_job('fit', 1, metadata={'phase': 'fit', 'method': 'SAITS'})
        store.start_session()
        display = GlobalProgress(store, 'none')
        def interrupted():
            raise GracefulStop('unfinished epoch')
        with store.claim('fit') as lease:
            with pytest.raises(GracefulStop):
                _finish_unit(store, 'fit', lease, 1, [], interrupted, display, 'fit:SAITS', StopRequested())
        display.close()
        assert store.job('fit')['completed_units'] == 0
        assert 'fit:SAITS' not in display.timings
        assert store.get_meta('eta')['remaining_seconds'] is None
        assert store.get_meta('active_seconds') >= 0


def test_report_children_use_remaining_actual_budget_without_discarding_crash_reserve(tmp_path):
    from energygap.runtime.state import BudgetExhausted
    with StateStore(tmp_path / 'run', config={'profile':'test'}, budget_seconds=150) as store:
        store.add_job('report', 1)
        store.add_job('bootstrap', 1)
        store.start_session();store.record_active(100)
        with store.claim('report') as parent:
            reservation=store.begin_work('report', parent.token, 60)
            with store.child_work_budget(reservation):
                store.check_budget()
                with store.claim('bootstrap') as child:
                    child_reservation=store.begin_work('bootstrap', child.token, 60)
                    store.finish_work(child_reservation,5)
                    store.check_budget()
                assert store.get_meta('active_seconds') == 165
            # Exclusion is only live/in-memory; a crash still retains its reserve.
            with pytest.raises(BudgetExhausted):store.check_budget()
            store.finish_work(reservation,2)
            store.check_budget()
            assert store.get_meta('active_seconds') == 107
