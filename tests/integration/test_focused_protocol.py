"""Independent focused-protocol acceptance using real, small SAITS fits."""
import json
import sqlite3

import numpy as np
import pytest

import energygap.conference as runner
from energygap.data import make_fixture
from energygap.runtime.state import IncompatibleResume


def test_epoch_panel_keeps_hpo_and_final_validation_full_and_budgets_distinct(tmp_path,monkeypatch):
    pytest.importorskip('torch');pytest.importorskip('pypots')
    from energygap.methods.conference_models import ConferenceModel
    from energygap.methods import conference_models
    monkeypatch.setattr(runner,'code_hash',lambda:'focused-protocol-independent-test')
    architecture=dict(n_layers=1,d_model=8,n_heads=2,d_k=4,d_v=4,d_ffn=8)
    catalogue=[dict(learning_rate=rate,architecture=architecture) for rate in (.001,.002)]
    monkeypatch.setattr(conference_models,'hpo_configurations',lambda name:catalogue)
    active=[None];callbacks=[]
    original_fit=ConferenceModel.fit_windows
    def fit(self,*args,**kwargs):
        callback=kwargs['selection_callback']
        def select(model,epoch):
            observation=dict(role=model.metadata['training_role'],epoch=epoch,keys=[])
            active[0]=observation
            try:
                return callback(model,epoch)
            finally:
                callbacks.append(observation);active[0]=None
        kwargs['selection_callback']=select
        return original_fit(self,*args,**kwargs)
    monkeypatch.setattr(ConferenceModel,'fit_windows',fit)
    original_infer=ConferenceModel.infer_queries
    def infer(self,queries,**kwargs):
        for query in queries:
            assert not hasattr(query,'original_record_ids')
            admitted=np.asarray(query.available_to_method)&~np.asarray(query.padding_mask)
            assert np.isnan(np.asarray(query.values)[~admitted]).all()
            if active[0] is not None:
                active[0]['keys'].append(query.target_gap_id)
        return original_infer(self,queries,**kwargs)
    monkeypatch.setattr(ConferenceModel,'infer_queries',infer)
    config=dict(study_profile='focused_24h',sites=['SYNTH'],folds=[1],conditions=['C3','C4','C5'],methods=['SAITS'],
        train_seeds=[11],validation_mask_seeds=[1001,1002],test_mask_seeds=[2001],context=512,
        max_episodes=2,max_training_windows=4,max_epochs=2,patience=2,hpo_trials=1,hpo_trial_indices=[1],
        hpo_max_epochs=1,hpo_patience=1,epoch_validation_max_per_stratum=1,epoch_validation_seed=1729,
        batch_size=2,inference_batch_size=4,samples=2,forecast_horizons_seconds=[3600],
        bootstrap_attempts=2,physical_hypotheses=[],shard_size=2)
    kwargs=dict(root=tmp_path,settings=config,series_override=[make_fixture(days=20,natural_missing=False)],
        device='cpu',threads=1,experiment_id='focused-saits',progress='none',budget_total_hours=.1)
    result=runner.run_conference(**kwargs)
    assert result['status']=='completed'
    directory=tmp_path/'runs/focused-saits'
    panel=json.loads((directory/'manifests/SYNTH:fold1-epoch-validation-panel.json').read_text())
    assert panel['candidate_count']==12 and panel['selected_count']==6
    assert not panel['covers_all_candidates'] and panel['selection_seed']==1729
    assert set(panel['selected'][0])=={'duration','mask_seed','gap_id'}
    expected={row['gap_id'] for row in panel['selected']}
    assert callbacks and all(len(row['keys'])==6 and set(row['keys'])==expected for row in callbacks)
    assert [row['role'] for row in callbacks].count('hpo')==1
    assert [row['role'] for row in callbacks].count('final')==2
    with sqlite3.connect(directory/'state.sqlite') as db:
        records=[json.loads(raw[0]) for raw in db.execute('SELECT payload FROM results')]
    fits=[row for row in records if row.get('phase')=='fit']
    assert {(row['training_role'],row['max_epochs'],row['patience']) for row in fits}=={('hpo',1,1),('final',2,2)}
    assert all(row['epoch_validation_panel_sha256']==panel['panel_sha256'] for row in fits)
    assert all(row['epoch_validation_candidates']==12 and row['epoch_validation_selected']==6 for row in fits)
    assert all(not row.get('selection_scores_artifact') for row in fits)
    scores=[row for row in records if row.get('phase')=='hpo_score']
    assert len(scores)==12 and {row['trial'] for row in scores}=={1}
    assert all(row['score_source']=='native_best_checkpoint_inference' for row in scores)
    final_validation=[row for row in records if row.get('phase')=='inference' and row['partition']=='validation']
    assert len(final_validation)==12
    summary=json.loads((directory/'reports/summary.json').read_text())
    design=summary['study_design']
    assert summary['study_profile']=='focused_24h' and not summary['scientific_complete']
    assert design['hpo_scoring']['full_declared_validation_scoring_completed'] is True
    assert design['hpo_scoring']['actual_trial_indices']==[1]
    assert design['final_validation']['full_declared_manifest_evaluation_completed'] is True
    assert design['test_evaluation']['full_declared_manifest_evaluation_completed'] is True
    assert {row['training_role'] for row in design['actual_fit_budgets']}=={'hpo','final'}
    figure_manifest=json.loads((directory/'reports/figures/manifest.json').read_text())
    assert figure_manifest['study_profile']=='focused_24h'
    assert figure_manifest['scope_reduced_by_design'] and not figure_manifest['full_catalogue_reproduced']
    assert 'Focused 24h profile' in next((directory/'reports/figures').glob('*.svg')).read_text()
    assert runner.run_conference(**kwargs,resume='require')['status']=='completed'
    for change in (dict(epoch_validation_seed=1730),dict(epoch_validation_max_per_stratum=2),
                   dict(hpo_max_epochs=2),dict(hpo_trial_indices=[0])):
        with pytest.raises(IncompatibleResume):
            runner.run_conference(**{**kwargs,'settings':{**config,**change}},resume='require')
