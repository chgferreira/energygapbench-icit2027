"""Serial/spawn correction grids and committed episode metrics use identical data."""
import energygap.conference as runner
from energygap.data import make_fixture
from energygap.runtime.energy_workers import EnergyCorrectionPool
from energygap.runtime.state import StateStore


def test_parallel_grid_and_test_rows_match_serial_and_completed_resume_never_spawns(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'code_hash',lambda:'parallel-correction-test')
    fixture=make_fixture(days=10,natural_missing=False)
    config=dict(sites=['SYNTH'],folds=[1],conditions=['C3','C4','C5'],methods=['linear'],
        train_seeds=[11],validation_mask_seeds=[1001],test_mask_seeds=[2001],max_episodes=1,
        max_training_windows=2,max_epochs=1,patience=1,hpo_trials=1,samples=100,
        forecast_horizons_seconds=[3600],bootstrap_attempts=10,shard_size=3,
        physical_hypotheses=['instantaneous','interval_mean_start','interval_mean_end'])
    outputs={}
    for workers in (1,2):
        current={**config,'energy_cpu_workers':workers,'energy_cpu_max_pending':2}
        status=runner.run_conference(root=tmp_path,settings=current,series_override=[fixture],
            device='cpu',threads=1,experiment_id=f'workers{workers}',progress='none',budget_total_hours=.1)
        assert status['status']=='completed'
        with StateStore(tmp_path/f'runs/workers{workers}',resume='require') as store:
            outputs[workers]=[row for row in store.results() if row.get('phase') in ('inference','correction_selection')]
    assert [row['key'].replace('workers1','workers2') for row in outputs[1]]==[row['key'] for row in outputs[2]]
    for serial,parallel in zip(outputs[1],outputs[2]):
        assert serial['phase']==parallel['phase']
        if serial['phase']=='correction_selection':
            for field in ('hypothesis','trial','parameters','loss','failures'):
                assert serial[field]==parallel[field]
        else:
            for field in ('prediction','mae_kw','nmae','rmse_kw','bias_kw','forecast_nmae','forecasts'):
                assert serial[field]==parallel[field]
            if parallel['variant'].startswith('E:'):
                assert parallel['timing_mode']=='parallel_cpu_service_cost'
                assert parallel['latency_seconds'] is None
                assert parallel['inference_amortized_seconds'] is None
                assert parallel['correction_service_seconds']>=0
                assert parallel['energy_cpu_workers']==2
    def unexpected(*args,**kwargs):
        raise AssertionError('completed resume cannot spawn CPU workers')
    monkeypatch.setattr(EnergyCorrectionPool,'_start',unexpected)
    resumed=runner.run_conference(root=tmp_path,settings={**config,'energy_cpu_workers':2,'energy_cpu_max_pending':2},
        series_override=[fixture],device='cpu',threads=1,experiment_id='workers2',progress='none',resume='require',budget_total_hours=.1)
    assert resumed['status']=='completed'
