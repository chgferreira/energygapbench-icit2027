"""Actual synthetic end-to-end smoke; no scientific claim about real sites."""
import json
import os
import subprocess
from unittest.mock import patch
import sqlite3
import pandas as pd


def test_fixture_to_masks_correction_forecast_stats_report_and_completed_resume(tmp_path,monkeypatch):
    # macOS's system_profiler can take minutes in a sandbox; bundled fonts suffice
    # for testing exported scientific figures. Numerical code is unaffected.
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/energygap-mpl-tests')
    real_check_output=subprocess.check_output
    def check_output(command,*args,**kwargs):
        if command and command[0]=='system_profiler':
            raise OSError('Use bundled matplotlib fonts in isolated integration test')
        return real_check_output(command,*args,**kwargs)
    with patch('subprocess.check_output',check_output):
        import matplotlib.pyplot
    from energygap.pipeline import run_pipeline,BASELINES
    import energygap.pipeline as pipeline
    first=run_pipeline(root=tmp_path,profile='smoke',max_episodes=1,progress='none',budget_total_hours=.05,experiment_id='full-fixture')
    run=tmp_path/'runs/full-fixture'
    summary=json.loads((run/'reports/summary.json').read_text())
    assert summary['synthetic'] is True
    assert summary['scientific_experiment_complete'] is False
    assert summary['n_confirmatory_tests']==0
    episodes=pd.read_csv(run/'reports/episodes.csv')
    assert len(episodes)==3*2*3*len(BASELINES)*2
    assert not episodes[['key']].duplicated().any()
    assert set(BASELINES)<=set(episodes.method)
    assert {name+'+energy_consistent' for name in BASELINES}<=set(episodes.method)
    assert episodes[episodes.partition=='test'].forecast_mae_kw.notna().any()
    assert episodes.import_power_integral_error_kwh.notna().all()
    for fold in (1,2,3):
        calibrated=json.loads((run/'models'/f'SYNTH:fold{fold}-calibration-selection.json').read_text())
        assert calibrated['calibration']['split']=='train'
        assert all(entry['status']=='selected_validation_only' for entry in calibrated['correction_selection'].values())
        assert all(len(entry['trials'])==16 for entry in calibrated['correction_selection'].values())
    assert (run/'protocol.lock.sha256').is_file()
    assert (run/'reports/episodes.parquet').is_file()
    assert (run/'reports/contrasts.json').is_file()
    assert (run/'reports/error_duration.svg').is_file()
    before={p.name:p.stat().st_mtime_ns for p in (run/'models').glob('*.pickle')}
    db=sqlite3.connect(run/'state.sqlite')
    before_count=db.execute('SELECT count(*) FROM results').fetchone()[0]
    db.close()
    def fail_if_instantiated(*args,**kwargs):
        raise AssertionError('Completed resume must not fit or infer')
    monkeypatch.setattr(pipeline,'create_baseline',fail_if_instantiated)
    second=run_pipeline(root=tmp_path,profile='smoke',max_episodes=1,progress='log',budget_total_hours=.05,experiment_id='full-fixture')
    after={p.name:p.stat().st_mtime_ns for p in (run/'models').glob('*.pickle')}
    assert before==after
    db=sqlite3.connect(run/'state.sqlite')
    assert db.execute('SELECT count(*) FROM results').fetchone()[0]==before_count
    assert db.execute("SELECT count(*) FROM results WHERE json_extract(payload,'$.method') IS NOT NULL").fetchone()[0]==len(episodes)
    db.close()
