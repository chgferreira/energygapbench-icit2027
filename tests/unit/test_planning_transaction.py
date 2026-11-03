"""Synthetic plans commit once and never expose partially registered jobs."""
import json
import logging
import sqlite3

import pytest

import energygap.conference as runner
import energygap.methods.conference_models as models
from energygap.data import make_fixture
from energygap.runtime.state import StateStore


@pytest.fixture
def planning_run(tmp_path, monkeypatch):
    statements, exported_transactions, stores = [], [], []

    class TracedStore(StateStore):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.db.set_trace_callback(statements.append)
            stores.append(self)

        def add_job(self, *args, **kwargs):
            super().add_job(*args, **kwargs)
            assert self.db.in_transaction, "planning job unexpectedly autocommitted"
            # Other processes cannot observe any part of the new plan yet.
            with sqlite3.connect(self.path / "state.sqlite") as reader:
                assert reader.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0

        def snapshot(self, force=False):
            if force and not self._defer_snapshots:
                exported_transactions.append(self.db.in_transaction)
            return super().snapshot(force=force)

    monkeypatch.setattr(runner, "StateStore", TracedStore)
    monkeypatch.setattr(runner, "code_hash", lambda: "synthetic-transaction-test")
    monkeypatch.setattr(models, "model_availability", lambda *args, **kwargs: {
        "FIM": {"status": "unavailable", "available": False, "reason": "synthetic unavailable entry"}})
    config = dict(sites=["SYNTH"], folds=[1], conditions=["C3", "C4", "C5"],
                  methods=["linear", "FIM"], train_seeds=[11],
                  validation_mask_seeds=[1001, 1002], test_mask_seeds=[2001, 2002],
                  max_episodes=1, physical_hypotheses=[])
    kwargs = dict(root=tmp_path, settings=config,
                  series_override=[make_fixture(days=10, natural_missing=False)],
                  device="cpu", threads=1, experiment_id="planning", dry_run=True)
    return kwargs, statements, exported_transactions


@pytest.mark.parametrize("progress", ["none", "tqdm"])
def test_all_planned_jobs_share_one_commit_and_snapshot_follows_commit(planning_run, caplog, progress):
    kwargs, statements, exported_transactions = planning_run
    caplog.set_level(logging.INFO, logger="energygap.conference")
    result = runner.run_conference(**kwargs, progress=progress)
    assert result["status"] == "planned"
    assert result["total_jobs"] == 15
    assert statements.count("BEGIN IMMEDIATE") == 1
    assert statements.count("COMMIT") == 1
    assert "ROLLBACK" not in statements
    assert len([sql for sql in statements if sql.startswith("INSERT INTO jobs(")]) == 15
    assert exported_transactions == [False]
    directory = kwargs["root"] / "runs/planning"
    progress_snapshot = json.loads((directory / "progress.json").read_text())
    assert progress_snapshot["total_jobs"] == 15
    assert progress_snapshot["counts"]["unavailable"] == 1
    planning_logs = [record.message for record in caplog.records if record.message.startswith("Planning")]
    if progress == "none":
        assert planning_logs == []
    else:
        assert any(message.startswith("Planning masks: SYNTH:fold1;") for message in planning_logs)
        assert any(message == "Planning committed: 15 jobs and 12 mask manifests" for message in planning_logs)


def test_mask_failure_rolls_back_jobs_before_exporting_snapshot(planning_run, monkeypatch):
    kwargs, statements, exported_transactions = planning_run
    original = runner.generate_mask
    attempts = []

    def fail_during_masks(*args, **mask_kwargs):
        attempts.append(mask_kwargs)
        if len(attempts) == 2:
            raise RuntimeError("synthetic mask generation failure")
        return original(*args, **mask_kwargs)

    monkeypatch.setattr(runner, "generate_mask", fail_during_masks)
    with pytest.raises(RuntimeError, match="synthetic mask generation failure"):
        runner.run_conference(**kwargs, progress="none")
    assert statements.count("BEGIN IMMEDIATE") == 1
    assert statements.count("ROLLBACK") == 1
    assert "COMMIT" not in statements
    assert any(sql.startswith("INSERT INTO jobs(") for sql in statements)
    assert exported_transactions == [False]
    directory = kwargs["root"] / "runs/planning"
    with sqlite3.connect(directory / "state.sqlite") as reader:
        assert reader.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    progress_snapshot = json.loads((directory / "progress.json").read_text())
    assert progress_snapshot["total_jobs"] == progress_snapshot["total_units"] == 0
