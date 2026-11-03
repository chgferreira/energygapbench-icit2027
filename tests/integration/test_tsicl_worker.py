"""Optional isolated TS-ICL execution; never downloads a model during tests."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
PYTHON=ROOT/'.venv-tsicl/bin/python'
CHECKPOINT=ROOT/'artifacts/checkpoints/tsicl/tsicl-v1.ckpt'
pytestmark=pytest.mark.skipif(not PYTHON.exists() or not CHECKPOINT.exists(),reason='optional isolated TS-ICL environment/checkpoint unavailable')


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_isolated_worker_npz_contract_and_marginal_quantiles(tmp_path,device):
    if device.startswith('cuda'):
        probe=subprocess.run([str(PYTHON),'-c','import torch; print(torch.cuda.is_available())'],capture_output=True,text=True,timeout=20)
        assert probe.returncode==0,probe.stderr
        assert probe.stdout.strip() in ('True','False'),probe.stdout
        if probe.stdout.strip()=='False':pytest.skip('actual CUDA unavailable in the isolated TS-ICL interpreter')
    x=np.sin(np.arange(512)/17);x[499:511]=np.nan
    times=np.arange(512,dtype=np.float64)*300
    path=tmp_path/'query.npz';out=tmp_path/'result.npz'
    np.savez(path,values=x,measurement_times=times,availability_times=times,issuance_time=times[-1])
    result=subprocess.run([str(PYTHON),str(ROOT/'scripts/tsicl_worker.py'),'--input',str(path),'--output',str(out),'--device',device],
                           cwd=ROOT,text=True,capture_output=True,timeout=45)
    assert result.returncode==0,result.stderr+result.stdout
    metadata=json.loads(result.stdout)
    assert metadata['device']==device
    assert metadata['capability']=='marginal_quantiles' and metadata['samples'] is None
    assert metadata['primary_estimator']=='native_quantile_0.5'
    with np.load(out,allow_pickle=False) as output:
        assert output['point_estimate'].shape==(512,)
        assert output['quantiles'].shape==(512,7)
        assert np.array_equal(output['point_estimate'][np.isfinite(x)],x[np.isfinite(x)])
        assert np.array_equal(output['point_estimate'][499:511],output['quantiles'][499:511,3])
        assert np.isfinite(output['quantiles']).all()


def test_worker_refuses_truth_tensor_in_input_object(tmp_path):
    times=np.arange(512,dtype=np.float64)*300
    path=tmp_path/'forbidden.npz';out=tmp_path/'must_not_exist.npz'
    np.savez(path,values=np.zeros(512),measurement_times=times,availability_times=times,
             issuance_time=times[-1],truth=np.ones(512))
    result=subprocess.run([str(PYTHON),str(ROOT/'scripts/tsicl_worker.py'),'--input',str(path),'--output',str(out)],
                           cwd=ROOT,text=True,capture_output=True,timeout=45)
    assert result.returncode==2,result.stderr+result.stdout
    assert 'truth/extra tensors refused' in result.stdout
    assert not out.exists()


def test_native_audit_records_actual_pretrained_component_and_isolation():
    report=json.loads((ROOT/'artifacts/reports/tsicl_audit.json').read_text())
    assert report['passed'] and report['release_pipeline_exact_match']
    assert report['checkpoint_bytes']==219150987
    assert report['checkpoint_sha256']=='a67ae9f694c2a83cfc8e7ec41745ff4f41a4a76ee2b17172ec3430d8d29da431'
    assert report['checkpoint_component_used'].startswith('imputer')
    assert report['environment']['torch'].startswith('2.9.1')
    assert report['context']==512 and not report['joint_samples']
    assert report['future_and_hidden_truth_poisoning_max_abs_error']==0
    assert report['availability_delay_poisoning_max_abs_error']==0
    assert report['native_parity_max_abs_error_hidden']==0
    assert not report['scientific_benchmark_executed']


def test_real_pilot_truth_evaluation_is_separate_and_completed_resume_noops():
    report_path=ROOT/'artifacts/reports/tsicl_real_pilot.json'
    if not report_path.exists():pytest.skip('real restricted pilot not present in clean installation')
    report=json.loads(report_path.read_text())
    assert report['n_queries']==report['n_success']==report['n_evaluations']==18
    assert report['context']==512 and report['local_training']=='not_applicable'
    assert report['truth_isolation'].startswith('separate evaluator phase')
    assert len(report['episode_metrics'])==18
    assert all(r['truth_sha256'] and r['prediction_sha256'] for r in report['episode_metrics'])
    import sqlite3
    database=ROOT/'runs'/report['experiment_id']/'state.sqlite'
    if not database.is_file():
        pytest.skip('historical private pilot database is not included in this checkout')
    db=sqlite3.connect(f'file:{database}?mode=ro',uri=True)
    before=db.execute('SELECT COUNT(*) FROM results').fetchone()[0]
    active_before=json.loads(db.execute("SELECT value FROM meta WHERE key='active_seconds'").fetchone()[0]);db.close()
    resumed=subprocess.run([sys.executable,str(ROOT/'scripts/tsicl_real_pilot.py'),
                           '--experiment-id',report['experiment_id'],'--resume','require'],cwd=ROOT,
                           capture_output=True,text=True,timeout=20)
    current_worker_sha=hashlib.sha256((ROOT/'scripts/tsicl_worker.py').read_bytes()).hexdigest()
    if report['worker_sha256']!=current_worker_sha:
        # A historical CPU run cannot be relabeled or resumed with revised CUDA
        # worker semantics; inspect it read-only or create a new experiment.
        assert resumed.returncode!=0
        assert 'scientific fingerprint differs' in resumed.stderr
    else:
        assert resumed.returncode==0,resumed.stdout+resumed.stderr
    db=sqlite3.connect(f'file:{database}?mode=ro',uri=True)
    assert db.execute('SELECT COUNT(*) FROM results').fetchone()[0]==before==36
    assert json.loads(db.execute("SELECT value FROM meta WHERE key='active_seconds'").fetchone()[0])==active_before
    db.close()
