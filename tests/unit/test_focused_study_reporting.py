"""Focused scope and full-within-scope evaluation require explicit evidence."""
import json

import pandas as pd

from energygap.reporting.conference import generate_conference_report
from energygap.runtime.state import StateStore,atomic_json


def test_focused_report_separates_omissions_failures_and_saved_panel_evidence(tmp_path):
    path=tmp_path/'focused'
    settings=dict(study_profile='focused_24h',methods=['linear','SAITS'],conditions=['C3','C4','C5'],
        sites=['H1'],folds=[1],train_seeds=[11],validation_mask_seeds=[1001],test_mask_seeds=[2001],
        max_episodes=None,hpo_trial_indices=[0,1],hpo_trials=2,hpo_max_epochs=20,hpo_patience=5,
        max_epochs=100,patience=10,epoch_validation_max_per_stratum=1,epoch_validation_seed=1729,
        omission_reasons={'BRITS':'excluded by the declared resource design'},
        condition_omission_reasons={'C1':'supplementary point-missingness condition'})
    config=dict(profile='conference',scope='focused_available_core_conditional_energy',experiment_settings=settings,
        methods=settings['methods'],conditions=settings['conditions'],sites=['H1'],folds=[1],train_seeds=[11],
        validation_mask_seeds=[1001],test_mask_seeds=[2001],physical_hypotheses=[],availability={
            'linear':dict(status='available'),'SAITS':dict(status='unavailable',reason='optional dependency'),
            'BRITS':dict(status='available'),'FIM':dict(status='unavailable',reason='licence unresolved')})
    with StateStore(path,config=config) as store:
        for partition,seed in [('validation',1001),('test',2001)]:
            for condition,duration in [('C3',15),('C4',60),('C5',180)]:
                jid=f'{partition}:{condition}'
                store.add_job(jid,1,metadata=dict(phase='inference',site='H1',fold=1,method='linear',
                    partition=partition,condition=condition,mask_seed=seed,train_seed=None))
                atomic_json(path/'manifests'/f'H1:fold1-{partition}-{condition}-{seed}.json',
                            dict(episodes=[dict(gap_id=condition)]))
                record=dict(key=jid,phase='inference',site='H1',fold=1,partition=partition,condition=condition,
                    duration=duration,mask_seed=seed,train_seed=None,method='linear',variant='P',gap_id=condition,
                    day='2025-01-01',status='success',nmae=.1,mae_kw=.2,forecast_nmae=.3,forecast_mae_kw=.4,latency_seconds=.01)
                with store.claim(jid) as lease:
                    store.publish(jid,lease.token,[record],1);store.finish(jid,lease.token)
        store.complete()
    panel=dict(policy='fixture_validation_panel',site='H1',fold=1,partition='validation',selection_seed=1729,
        max_per_stratum=1,candidate_count=6,selected_count=3,covers_all_candidates=False,
        candidate_identity_sha256='candidate-hash',selected_identity_sha256='selected-hash',panel_sha256='panel-hash',
        strata=[dict(duration=d,mask_seed=1001,candidate_count=2,selected_count=1) for d in (15,60,180)],
        missing_required_strata=[],selected=[dict(duration=d,mask_seed=1001,gap_id=str(d)) for d in (15,60,180)])
    atomic_json(path/'manifests/H1:fold1-epoch-validation-panel.json',panel)
    result=generate_conference_report(path,bootstrap_attempts=2)
    assert result['study_profile']=='focused_24h'
    assert result['scope_reduced_by_design'] and not result['scientific_complete']
    assert not result['full_catalogue_reproduced']
    assert result['blocked_methods']=={'SAITS':config['availability']['SAITS']}
    assert {'BRITS','FIM'}<=set(result['omitted_methods_by_design'])
    assert result['omitted_conditions_by_design']==['C1','C2','C6']
    design=result['study_design']
    assert design['declared_training_budgets']==dict(hpo_trial_indices=[0,1],hpo_max_epochs=20,hpo_patience=5,
                                                   final_max_epochs=100,final_patience=10)
    assert design['epoch_validation_panels'][0]['selected_count']==3
    assert design['epoch_validation_panels'][0]['candidate_count']==6
    assert design['epoch_validation_panels'][0]['panel_sha256']=='panel-hash'
    assert design['final_validation']['full_available_manifest_evaluation_completed'] is True
    assert design['test_evaluation']['full_available_manifest_evaluation_completed'] is True
    assert design['final_validation']['requested_methods_all_available'] is False
    assert design['final_validation']['full_declared_manifest_evaluation_completed'] is False
    assert design['test_evaluation']['full_declared_manifest_evaluation_completed'] is False
    assert design['omission_reasons']==settings['omission_reasons']
    assert design['condition_omission_reasons']==settings['condition_omission_reasons']
    table=pd.read_csv(path/'reports/main_table.csv')
    assert 'SAITS' in set(table.method)
    assert not {'BRITS','FIM'}&set(table.method)
    text=(path/'reports/report.md').read_text()
    assert 'Focused 24-hour-target' in text and 'not a completion guarantee' in text
    assert 'Methods omitted by design' in text and 'Requested but unavailable methods: SAITS' in text
    assert json.loads((path/'reports/summary.json').read_text())['study_profile']=='focused_24h'


def test_report_does_not_claim_full_validation_from_configuration_alone(tmp_path):
    path=tmp_path/'missing-evidence'
    with StateStore(path,config=dict(profile='conference',methods=['linear'],physical_hypotheses=[],
                                    experiment_settings=dict(study_profile='focused_24h'))) as store:
        store.add_job('empty',0,metadata=dict(phase='inference',partition='validation'))
        with store.claim('empty') as lease:
            store.publish('empty',lease.token,[],0);store.finish('empty',lease.token)
        store.complete()
    result=generate_conference_report(path,bootstrap_attempts=2)
    assert result['study_design']['final_validation']['full_manifest_counts_verified'] is False
    assert result['study_design']['final_validation']['full_declared_manifest_evaluation_completed'] is False
