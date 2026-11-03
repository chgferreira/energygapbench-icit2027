"""Legacy-fingerprint fixture retains committed results under CPU-only continuation."""
import json
import shutil

import energygap.conference as runner
from energygap.data import make_fixture
from energygap.pipeline import code_hash
from energygap.runtime.execution_patch import MANIFEST
from energygap.runtime.state import StateStore


def test_runtime_pool_resume_keeps_commits_rng_and_final_scientific_results(tmp_path,monkeypatch):
    cfg=dict(sites=['SYNTH'],folds=[1],conditions=['C3','C4','C5'],methods=['linear'],
        train_seeds=[11],validation_mask_seeds=[1001],test_mask_seeds=[2001],max_episodes=1,
        max_training_windows=2,max_epochs=1,patience=1,hpo_trials=1,samples=2,
        forecast_horizons_seconds=[3600],bootstrap_attempts=2,shard_size=1,
        energy_cpu_workers=1,energy_cpu_max_pending=2,
        physical_hypotheses=['instantaneous','interval_mean_start','interval_mean_end'])
    source_hash=json.loads(MANIFEST.read_text())['source_code_hash']
    monkeypatch.setattr(runner,'code_hash',lambda:source_hash)
    original_publish=StateStore.publish;injected=[]
    def publish(self,jid,token,records,completed_units,*args,**kwargs):
        result=original_publish(self,jid,token,records,completed_units,*args,**kwargs)
        if not injected and any(r.get('partition')=='test' for r in records):
            injected.append(True);self.record_active(400.,jid)
        return result
    monkeypatch.setattr(StateStore,'publish',publish)
    fixture=make_fixture(days=10,natural_missing=False)
    common=dict(settings=cfg,series_override=[fixture],device='cpu',threads=1,
                experiment_id='kept',progress='none',budget_total_hours=.1)
    parent=tmp_path/'parent'
    assert runner.run_conference(root=parent,**common)['status']=='paused_budget'
    monkeypatch.setattr(StateStore,'publish',original_publish)
    with StateStore(parent/'runs/kept',resume='require') as store:
        original={r['key']:r for r in store.results()};fingerprint=store.fingerprint
    outputs={}
    for name in ('reference','optimized'):
        root=tmp_path/name;shutil.copytree(parent,root)
        with StateStore(root/'runs/kept',resume='require') as store:store.extend_budget(3600.)
        overrides={}
        if name=='optimized':
            monkeypatch.setattr(runner,'code_hash',code_hash)
            overrides=dict(energy_cpu_workers=2,energy_cpu_max_pending=4,allow_runtime_patch=True)
        assert runner.run_conference(root=root,**common,resume='require',**overrides)['status']=='completed'
        with StateStore(root/'runs/kept',resume='require') as store:
            assert store.fingerprint==fingerprint
            records={r['key']:r for r in store.results()}
            assert all(records[key]==value for key,value in original.items())
            outputs[name]=records
    assert set(outputs['reference'])==set(outputs['optimized'])
    for key,row in outputs['reference'].items():
        other=outputs['optimized'][key]
        if row.get('phase')=='inference':
            for field in ('prediction','nmae','forecasts','sample_seed'):
                assert row.get(field)==other.get(field)
    report=json.loads((tmp_path/'optimized/runs/kept/reports/summary.json').read_text())
    assert report['runtime_execution_history'][-1]['effective_energy_pool']['workers']==2
