"""Plan validation uses only local configs; no model imports, downloads, or training."""
import json
from pathlib import Path
import shutil
import pytest
from energygap.runtime.planner import plan,validate_dag,BASELINES
from energygap.methods.registry import REQUIRED,load_registry

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def conference():return plan(ROOT,'conference',budget_total_hours=.1)


def test_full_catalogue_and_exact_scientific_counts(conference):
    p=conference
    assert set(p['catalogue'])==set(BASELINES)|REQUIRED
    assert len(p['catalogue'])==17
    registered=load_registry(ROOT)['methods']
    local=[m for m in registered if m['local_training']=='required']
    frozen=[m for m in registered if m['local_training']=='not_applicable']
    unknown=[m for m in registered if m['local_training'] not in ('required','not_applicable')]
    assert p['training_counts']['M_required_local']==len(local)==5
    assert p['training_counts']['final_trainings']==len(local)*3*3*3==135
    assert p['training_counts']['HPO_trials']==len(local)*3*6==90
    assert p['training_counts']['baseline_fits']==len(BASELINES)*3*3==54
    assert p['training_counts']['frozen_checkpoint_loads']==len(frozen)==5
    assert p['training_counts']['frozen_training_replicates']==0
    assert p['training_counts']['unresolved_training_methods']==[m['name'] for m in unknown]==['CBiT']
    assert p['training_counts']['unresolved_additional_trainings'] is None
    assert p['counts']['train']==135 and p['counts']['hpo']==90
    assert p['counts']['hpo_selection']==15
    assert p['counts']['mask_manifest']==3*3*15*6
    assert p['counts']['inference_manifest']==3*3*15*6*(len(BASELINES)+len(frozen)+len(unknown)+len(local)*3)
    assert p['counts']['bootstrap']==12
    assert p['counts']['energy_reference']==9
    assert p['method_coverage']['T1']['planned_inference_jobs']==3*3*15*6*3
    assert p['method_coverage']['MOMENT-1-base']['train_seeds']==[]


def test_no_missing_dependencies_no_cycles_no_duplicates(conference):
    order=validate_dag(conference['jobs'])
    assert len(order)==len(conference['jobs'])==len(set(order))
    index={key:i for i,key in enumerate(order)}
    for job in conference['jobs']:
        assert all(index[parent]<index[job['job_id']] for parent in job['dependencies'])
    assert conference['dag_validation']['valid']


def test_input_truth_never_an_inference_dependency(conference):
    index={j['job_id']:j for j in conference['jobs']}
    jobs=[j for j in conference['jobs'] if j['phase']=='inference_manifest']
    assert jobs and all(j['n_queries'] is None for j in jobs)
    for job in jobs:
        assert not any(index[d]['phase']=='evaluation_truth' for d in job['dependencies'])
        assert any(index[d]['phase']=='query_build' for d in job['dependencies'])
        assert any(index[d]['phase'] in ('train','baseline_fit','frozen_prepare','implementation_availability') for d in job['dependencies'])
        if job['partition']=='test':assert 'protocol_lock:P' in job['dependencies']
    for job in conference['jobs']:
        if job['phase']=='hpo':
            assert job['fold']==1 and job['selection_partition']=='validation'
            assert not any(':test:' in d for d in job['dependencies'])


def test_shared_masks_forecast_weights_and_no_fake_foundation_seeds(conference):
    index={j['job_id']:j for j in conference['jobs']}
    for job in conference['jobs']:
        if job['phase']=='mask_manifest':assert job['shared_by_all_methods'] and job['immutable']
        if job['phase']=='forecast_fit':assert job['weights_shared_across_imputations']
        if job['phase'] in ('frozen_prepare','model_load'):assert job['train_seed'] is None
    example=[j for j in conference['jobs'] if j['phase']=='inference_manifest' and j['site']=='H1' and j['fold']==1 and j['partition']=='test' and j['condition']=='C3' and j['mask_seed']==2001]
    queries=[d for j in example for d in j['dependencies'] if index[d]['phase']=='query_build']
    expected_variants=sum(len(m['train_seeds']) if m['local_training']=='required' else 1 for m in conference['method_coverage'].values())
    assert len(example)==expected_variants and len(set(queries))==1


def test_statistical_and_physical_contracts_are_separate(conference):
    assert conference['dataset_unresolved']=={'H1':[],'H2':[],'I1':[]}
    assert all(v for v in conference['physical_dataset_unresolved'].values())
    baseline_inferences=[j for j in conference['jobs'] if j['phase']=='inference_manifest' and j['method'] in BASELINES]
    assert all(j['status']=='pending' for j in baseline_inferences)
    corrections=[j for j in conference['jobs'] if j['phase']=='correction']
    assert corrections and all(j['status']=='blocked' and 'physical' in j['reason'] for j in corrections)
    assert all(j['same_wrapper_all_compatible_methods'] for j in corrections)


def test_honest_unknown_cost_blocks_start_not_silent_reduction(conference):
    assert conference['estimated_seconds'] is None and conference['estimated_finish'] is None
    assert not conference['can_start_full_matrix']
    assert conference['budget_decision']=='blocked_pending_cost_calibration'
    assert conference['budget_seconds']==360
    assert conference['unknown_cost_jobs']>0
    assert not conference['conference_complete'] and not conference['execution_supported']
    assert conference['masks']['test_seeds']==list(range(2001,2011))
    assert conference['reduced_plan']['not_conference_substitute']
    # Measured query timings remain costs per query, not fabricated matrix totals.
    for job in conference['jobs']:
        if job['phase']=='inference_manifest':assert job['estimated_seconds'] is None


def test_plan_runs_without_pilot_calibrations_or_dataset_metadata(tmp_path):
    shutil.copy(ROOT/'model_registry.yaml',tmp_path/'model_registry.yaml')
    p=plan(tmp_path,'conference',.1)
    assert p['calibration_sources']==[]
    assert p['cost_calibration']=={}
    assert all(p['dataset_unresolved'].values())
    assert not p['can_start_full_matrix']
    expected_variants=sum(len(m['train_seeds']) if m['local_training']=='required' else 1 for m in p['method_coverage'].values())
    assert p['counts']['inference_manifest']==3*3*15*6*expected_variants
    assert validate_dag(p['jobs'])


def test_validator_rejects_invalid_dags():
    with pytest.raises(ValueError,match='missing'):validate_dag([{'job_id':'a','dependencies':['absent']}])
    with pytest.raises(ValueError,match='cycle'):validate_dag([{'job_id':'a','dependencies':['b']},{'job_id':'b','dependencies':['a']}])
    with pytest.raises(ValueError,match='duplicate'):validate_dag([{'job_id':'a','dependencies':[]},{'job_id':'a','dependencies':[]}])


def test_unreproduced_cbit_never_invents_a_frozen_checkpoint_or_training_semantics(conference):
    jobs=[j for j in conference['jobs'] if j.get('method')=='CBiT']
    assert not any(j['phase'] in ('model_load','frozen_prepare','train','hpo') for j in jobs)
    audit=[j for j in jobs if j['phase']=='implementation_availability']
    assert len(audit)==1 and audit[0]['status']=='blocked'
    assert audit[0]['checkpoint'] is None and audit[0]['planned_training_jobs'] is None
    assert audit[0]['planned_training_seeds'] is None
    inference=[j for j in jobs if j['phase']=='inference_manifest']
    assert len(inference)==3*3*15*6
    assert all(j['status']=='blocked' and j['train_seed'] is None and j['train_seed_semantics']=='unknown_not_reproduced' for j in inference)
    assert all('implementation_availability:CBiT' in j['dependencies'] for j in inference)
    coverage=conference['method_coverage']['CBiT']
    assert coverage['local_training']=='unknown_not_reproduced'
    assert coverage['training_job_count'] is None and not coverage['verified_frozen_role']
    assert coverage['train_seed_semantics']=='unknown_not_reproduced'
