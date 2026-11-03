"""Synthetic executor acceptance; these runs make no claims about real sites."""
import json
import sqlite3

import pandas as pd
import pytest

from energygap.data import make_fixture
from energygap.conference import run_conference
import energygap.conference as runner


def settings(physics):
    return dict(sites=["SYNTH"], folds=[1], conditions=["C3", "C4", "C5"],
                methods=["linear", "seasonal"], train_seeds=[11],
                validation_mask_seeds=[1001], test_mask_seeds=[2001],
                context=512, max_episodes=1, max_training_windows=2,
                max_epochs=1, patience=1, hpo_trials=1, samples=10,
                forecast_horizons_seconds=[3600], bootstrap_attempts=10,
                physical_hypotheses=physics, shard_size=1)


@pytest.mark.parametrize("physics", [[], ["instantaneous", "interval_mean_start", "interval_mean_end"]])
def test_available_baseline_executor_plan_run_report_and_completed_resume(tmp_path, monkeypatch, physics):
    # Keep concurrent repository work outside the synthetic protocol identity.
    monkeypatch.setattr(runner, "code_hash", lambda: "synthetic-test-code")
    fixture = make_fixture(days=10, natural_missing=False)
    config = settings(physics)
    kwargs = dict(root=tmp_path, settings=config, series_override=[fixture], device="cpu",
                  threads=1, experiment_id="fixture-conference", progress="none", budget_total_hours=.1)
    plan = run_conference(**kwargs, dry_run=True)
    assert plan["status"] == "planned"
    first = run_conference(**kwargs)
    assert first["status"] == "completed"
    assert first["counts"]["succeeded"] == first["total_jobs"]
    path = tmp_path / "runs/fixture-conference"
    frame = pd.read_csv(path / "reports/episodes.csv")
    assert len(frame[frame.variant == "P"]) == 12
    assert not frame.key.duplicated().any()
    assert frame.loc[frame.partition == "test", "forecast_nmae"].notna().all()
    if physics:
        assert set(frame.variant) == {"P", *("E:" + h for h in physics)}
        assert len(frame[frame.variant != "P"]) == 18
        with sqlite3.connect(path / "state.sqlite") as db:
            selections = [json.loads(r[0]) for r in db.execute(
                "SELECT payload FROM results WHERE json_extract(payload,'$.phase')='correction_selection'")]
        assert len(selections) == 2 * 3 * 16
        assert all(r["selection_partition"] == "validation" for r in selections)
        assert all(r["physical_confirmatory"] is False for r in selections)
    else:
        assert set(frame.variant) == {"P"}
    summary = json.loads((path / "reports/summary.json").read_text())
    assert summary["execution_complete"] is True
    assert summary["scientific_complete"] is False
    assert summary["physical_confirmatory"] is False
    assert summary["n_confirmatory_tests"] == 0
    assert summary["scope"] == "synthetic_executor_validation_not_real_science"
    assert (path / "reports/main_table.parquet").is_file()
    before_artifacts = {str(p.relative_to(path)): p.stat().st_mtime_ns
                        for p in (path / "models").glob("*")}
    with sqlite3.connect(path / "state.sqlite") as db:
        before_results = db.execute("SELECT COUNT(*) FROM results").fetchone()[0]

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("A completed resume cannot re-run preparation/fits")

    monkeypatch.setattr(runner, "_prepare_fold", unexpected_fit)
    second = run_conference(**kwargs, resume="require")
    assert second["status"] == "completed"
    assert before_artifacts == {str(p.relative_to(path)): p.stat().st_mtime_ns
                                for p in (path / "models").glob("*")}
    with sqlite3.connect(path / "state.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM results").fetchone()[0] == before_results


def test_budget_pause_retains_committed_queries_and_resumes_without_refit(tmp_path,monkeypatch):
    from energygap.runtime.state import StateStore
    monkeypatch.setattr(runner,'code_hash',lambda:'budget-test-code')
    fixture=make_fixture(days=10,natural_missing=False)
    config=settings([])
    config['methods']=['linear']
    original=StateStore.publish
    injected=[]
    def publish(self,jid,token,records,completed_units,*args,**kwargs):
        output=original(self,jid,token,records,completed_units,*args,**kwargs)
        if records and records[0].get('phase')=='inference' and not injected:
            injected.append(records[0]['key']);self.record_active(400.,jid)
        return output
    monkeypatch.setattr(StateStore,'publish',publish)
    kwargs=dict(root=tmp_path,settings=config,series_override=[fixture],device='cpu',threads=1,
                experiment_id='paused',progress='none',budget_total_hours=.1)
    first=run_conference(**kwargs)
    assert first['status']=='paused_budget'
    assert first['counts']['pending']>0
    directory=tmp_path/'runs/paused'
    cache=next((directory/'models').glob('*common.pickle'))
    before=cache.stat().st_mtime_ns
    with StateStore(directory,resume='require') as store:
        keys={row['key'] for row in store.results()}
        assert injected[0] in keys
        store.extend_budget(3600.)
    def forbidden(*a,**k):raise AssertionError('committed preparation repeated')
    monkeypatch.setattr(runner,'_prepare_fold',forbidden)
    second=run_conference(**kwargs,resume='require')
    assert second['status']=='completed'
    assert cache.stat().st_mtime_ns==before
    with StateStore(directory,resume='require') as store:
        final=[row['key'] for row in store.results()]
        assert len(final)==len(set(final)) and keys<=set(final)


def test_corrupt_preparation_rebuilds_dependent_report_after_completed_run(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,"code_hash",lambda:"artifact-recovery-test")
    fixture=make_fixture(days=10,natural_missing=False)
    config=settings([])
    config["methods"]=["linear"]
    kwargs=dict(root=tmp_path,settings=config,series_override=[fixture],device="cpu",threads=1,
                experiment_id="recover-artifact",progress="none",budget_total_hours=.1)
    assert run_conference(**kwargs)["status"] == "completed"
    directory=tmp_path / "runs/recover-artifact"
    table=directory / "reports/main_table.parquet"
    before=table.stat().st_mtime_ns
    cache=next((directory / "models").glob("*common.pickle"))
    cache.write_bytes(b"deliberately corrupted test artifact")
    second=run_conference(**kwargs,resume="require")
    assert second["status"] == "completed"
    assert table.stat().st_mtime_ns != before
    with sqlite3.connect(directory / "state.sqlite") as db:
        assert db.execute("SELECT attempts FROM jobs WHERE job_id='report'").fetchone()[0] == 2
        keys=[r[0] for r in db.execute("SELECT result_key FROM results")]
        assert len(keys)==len(set(keys))
    assert pd.read_parquet(table).nmae.notna().all()
