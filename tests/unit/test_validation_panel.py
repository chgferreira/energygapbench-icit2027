"""Epoch panels depend only on frozen metadata and never narrow final scoring."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from energygap.conference import _load_selection_scores, load_settings, run_conference
from energygap.methods.validation_panel import select_epoch_validation_panel
from energygap.runtime.state import StateStore, fingerprint


def _candidates(count=50):
    return [dict(duration=duration,mask_seed=seed,gap_id=f'{duration}:{seed}:{index}')
            for duration in (15.,60.,180.) for seed in range(1001,1006) for index in range(count)]


def test_metadata_hash_panel_is_fixed_order_independent_and_balanced_without_truth():
    candidates=_candidates()
    kwargs=dict(site='H1',fold=1,max_per_stratum=20,seed=1729,mask_seeds=list(range(1001,1006)))
    first=select_epoch_validation_panel(candidates,**kwargs)
    reordered=select_epoch_validation_panel(list(reversed(candidates)),**kwargs)
    assert first==reordered
    assert first['candidate_count']==750 and first['selected_count']==300
    assert len(first['strata'])==15
    assert all(row['candidate_count']==50 and row['selected_count']==20 for row in first['strata'])
    assert not first['covers_all_candidates']
    assert first['coverage_status']=='complete' and first['missing_required_strata']==[]
    assert first['selection_values_used'] is False and first['selection_metrics_used'] is False
    assert first['policy']=='fixed_hash_stratified_epoch_validation_panel'
    expected=dict(first);expected.pop('panel_sha256')
    assert first['panel_sha256']==hashlib.sha256(json.dumps(expected,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    for changed in ({'site':'H2'},{'fold':2},{'seed':1730}):
        other=select_epoch_validation_panel(candidates,**{**kwargs,**changed})
        assert first['selected']!=other['selected']
    # The boundary rejects evaluator data rather than silently reading or
    # retaining it alongside the metadata that defines the panel.
    for forbidden in ('truth','values','score','nmae','method','train_seed'):
        with pytest.raises(ValueError,match='only duration, mask_seed and gap_id'):
            select_epoch_validation_panel([{**candidates[0],forbidden:object()}],**kwargs)


def test_uncapped_selection_covers_every_candidate_and_missing_strata_stay_explicit():
    candidates=_candidates(2)
    panel=select_epoch_validation_panel(candidates,site='H1',fold=1,mask_seeds=list(range(1001,1006)))
    assert panel['covers_all_candidates'] and panel['selected_count']==panel['candidate_count']==30
    assert panel['policy']=='all_prespecified_validation_episodes'
    assert panel['selected']==sorted(candidates,key=lambda row:(row['duration'],row['mask_seed'],row['gap_id']))
    sparse=[row for row in candidates if (row['duration'],row['mask_seed'])!=(180.,1005)]
    reduced=select_epoch_validation_panel(sparse,site='H1',fold=1,max_per_stratum=1,
        mask_seeds=list(range(1001,1006)))
    assert reduced['missing_required_strata']==[dict(duration=180.,mask_seed=1005)]
    assert reduced['coverage_status']=='missing_required_strata'
    assert next(row for row in reduced['strata'] if (row['duration'],row['mask_seed'])==(180.,1005))==dict(
        duration=180.,mask_seed=1005,candidate_count=0,selected_count=0)
    with pytest.raises(ValueError,match='Duplicate'):
        select_epoch_validation_panel(candidates+candidates[:1],site='H1',fold=1)


def _settings():
    return dict(sites=['SYNTH'],folds=[1],conditions=['C3','C4','C5'],methods=['SAITS'],
        train_seeds=[11],validation_mask_seeds=[1001,1002],test_mask_seeds=[2001],
        max_epochs=100,patience=10,hpo_trials=2,hpo_trial_indices=[3,5],
        hpo_max_epochs=20,hpo_patience=5,epoch_validation_max_per_stratum=1,
        max_episodes=2,max_training_windows=2,physical_hypotheses=[],forecast_horizons_seconds=[3600])


def test_new_protocol_fields_have_strict_bounds_and_full_defaults_are_unchanged():
    full=load_settings('.',settings={**_settings(),'hpo_trials':1,'hpo_trial_indices':[0],
                                    'epoch_validation_max_per_stratum':None})
    assert full['hpo_trial_indices']==[0]
    defaults=dict(_settings())
    for key in ('hpo_trial_indices','hpo_max_epochs','hpo_patience','epoch_validation_max_per_stratum','epoch_validation_seed'):
        defaults.pop(key,None)
    expanded=load_settings('.',settings=defaults)
    assert expanded['hpo_trial_indices']==[0,1]
    assert expanded['hpo_max_epochs']==100 and expanded['hpo_patience']==10
    assert expanded['epoch_validation_max_per_stratum'] is None and expanded['epoch_validation_seed']==1729
    invalids=[{'hpo_trial_indices':[0]}, {'hpo_trial_indices':[0,0]}, {'hpo_trial_indices':[0,6]},
              {'hpo_trial_indices':[0,True]}, {'hpo_trial_indices':[0,'1']}, {'hpo_trial_indices':None},
              {'hpo_max_epochs':0}, {'hpo_max_epochs':True}, {'hpo_patience':21},
              {'epoch_validation_max_per_stratum':0}, {'epoch_validation_max_per_stratum':True},
              {'epoch_validation_max_per_stratum':1.5}, {'epoch_validation_seed':True}, {'epoch_validation_seed':1.5}]
    for changed in invalids:
        with pytest.raises(ValueError):
            load_settings('.',settings={**_settings(),**changed})
    with pytest.raises(ValueError,match='require C3, C4 and C5'):
        load_settings('.',settings={**_settings(),'conditions':['C3']})
    assert load_settings('.',settings={**_settings(),'conditions':['C3'],'methods':['linear']})['conditions']==['C3']
    for changed in ({'hpo_trial_indices':[1,5]}, {'hpo_max_epochs':21}, {'hpo_patience':4},
                    {'epoch_validation_seed':1730},{'epoch_validation_max_per_stratum':2}):
        assert fingerprint(load_settings('.',settings=_settings()))!=fingerprint(load_settings('.',settings={**_settings(),**changed}))


def test_panel_scores_cannot_masquerade_as_full_best_checkpoint_scores(tmp_path):
    cases=[(SimpleNamespace(seed=1001),SimpleNamespace(gap_id='a')),
           (SimpleNamespace(seed=1001),SimpleNamespace(gap_id='b'))]
    path=tmp_path/'selection.json'
    saved=dict(model_sha256='model',best_epoch=1,scope='fixed_panel',
               scores=[dict(mask_seed=1001,gap_id=gap,nmae=.1) for gap in ('a','b')])
    record=dict(selection_scores_artifact='selection.json',sha256='model',training_metadata={'best_epoch':1})
    def write():
        path.write_text(json.dumps(saved))
        record['selection_scores_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    write()
    assert _load_selection_scores(tmp_path,record,cases)=={}
    saved['scope']='full_prespecified_validation_episodes';write()
    assert len(_load_selection_scores(tmp_path,record,cases))==2
    record['epoch_validation_covers_all_candidates']=False
    assert _load_selection_scores(tmp_path,record,cases)=={}
    record['epoch_validation_covers_all_candidates']=True
    saved['scores']=saved['scores'][:1];write()
    assert _load_selection_scores(tmp_path,record,cases)=={}


def test_nonprefix_hpo_indices_and_epoch_panel_are_frozen_in_the_dry_plan(tmp_path,monkeypatch):
    pytest.importorskip('torch');pytest.importorskip('pypots')
    from energygap.data import make_fixture
    import energygap.conference as runner
    monkeypatch.setattr(runner,'code_hash',lambda:'nonprefix-metadata-panel-plan')
    cfg=_settings()
    result=run_conference(root=tmp_path,settings=cfg,series_override=[make_fixture(days=20,natural_missing=False)],
        device='cpu',threads=1,experiment_id='panel-plan',progress='none',dry_run=True)
    assert result['status']=='planned'
    directory=tmp_path/'runs/panel-plan'
    plan=json.loads((directory/'plan.json').read_text())
    jobs=[row for row in plan['jobs'] if row['metadata'].get('phase')=='hpo']
    assert {row['job_id'] for row in jobs}=={'hpo:SYNTH:SAITS:3','hpo:SYNTH:SAITS:5'}
    assert all(row['metadata']['max_epochs']==20 and row['metadata']['patience']==5 for row in jobs)
    final=[row for row in plan['jobs'] if row['metadata'].get('phase')=='fit']
    assert final and all(row['metadata']['max_epochs']==100 and row['metadata']['patience']==10 for row in final)
    descriptor=plan['epoch_validation_panels']['SYNTH:fold1']
    panel=json.loads((directory/descriptor['artifact']).read_text())
    assert descriptor['sha256']==hashlib.sha256((directory/descriptor['artifact']).read_bytes()).hexdigest()
    assert panel['selected_count']==6 and panel['candidate_count']>6
    assert not panel['covers_all_candidates']
    with StateStore(directory,resume='require') as store:
        assert store.get_meta('config')['experiment_settings']['hpo_trial_indices']==[3,5]


def test_epoch_checkpoint_identity_includes_panel_hash_and_named_scope(tmp_path):
    torch=pytest.importorskip('torch');pytest.importorskip('pypots')
    import numpy as np
    from energygap.methods.conference_models import create_adapter
    kwargs=dict(name='SAITS',root=tmp_path,device='cpu',threads=1,seed=11,context=16,
        params={'architecture':dict(n_layers=1,d_model=8,n_heads=2,d_k=4,d_v=4,d_ffn=8)},
        epochs=1,patience=1,batch_size=2)
    t=np.arange(6*16,dtype=np.float32).reshape(6,16,1)
    train,val=np.sin(t/5),np.cos(t[:3]/4)
    epoch=tmp_path/'epoch.pt'
    model=create_adapter(**kwargs)
    model.metadata.update(epoch_validation_policy='fixed_hash_stratified_epoch_validation_panel',
                          epoch_validation_panel_sha256='panel-1')
    model.fit_windows(train,val,checkpoint_path=epoch,selection_callback=lambda model,epoch:.5)
    assert model.metadata['epoch_selection']=='fixed_panel_validation_equal_duration_gap_nmae'
    changed=create_adapter(**kwargs)
    changed.metadata.update(epoch_validation_policy='fixed_hash_stratified_epoch_validation_panel',
                            epoch_validation_panel_sha256='panel-2')
    with pytest.raises(ValueError,match='differ'):
        changed.fit_windows(train,val,checkpoint_path=epoch,selection_callback=lambda model,epoch:.5)
