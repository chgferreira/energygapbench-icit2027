"""Continuation accepts a reviewed release pair, not arbitrary scientific edits."""
import copy
import json
import sqlite3

import pytest

from energygap.pipeline import code_hash
from energygap.runtime.execution_patch import MANIFEST, REPORT_MANIFEST, authorize_resume, pool_options, record_execution
from energygap.runtime.state import IncompatibleResume, JobBusy, StateStore, fingerprint


def _parent(tmp_path,source_hash=None):
    manifest = json.loads(MANIFEST.read_text())
    config = dict(profile='conference',code_hash=source_hash or manifest['source_code_hash'],
        experiment_settings=dict(energy_cpu_workers=2,energy_cpu_max_pending=4,
            train_seeds=[11],batch_size=32,inference_batch_size=256,hpo_trials=2),
        execution=dict(device='cpu',threads=1),datasets={'SYNTH':{'sha256':'original'}})
    path=tmp_path/'run'
    with StateStore(path,config) as store:
        store.add_job('kept',1);store.set_meta('status','interrupted')
    return path,config,dict(config,code_hash=code_hash())


def _current_manifest():
    for path in (REPORT_MANIFEST,MANIFEST):
        if path.is_file():
            manifest=json.loads(path.read_text())
            if manifest['target_code_hash']==code_hash():return manifest
    raise AssertionError('No shipped audited registry matches the current source release')


def test_audited_patch_keeps_original_identity_and_backups_database(tmp_path):
    path,frozen,proposed=_parent(tmp_path)
    with pytest.raises(IncompatibleResume):
        StateStore(path,proposed,resume='require')
    restored,guard=authorize_resume(path,proposed)
    assert restored==frozen
    with guard,StateStore(path,restored,resume='require') as store:
        guard.attach(store);store.start_session()
        record_execution(store,frozen['experiment_settings'],dict(workers=3,max_pending=6),code_hash=code_hash(),patch=guard.audit)
        assert store.get_meta('config')==frozen
        assert store.fingerprint==fingerprint(frozen)
        assert store.job('kept')['total_units']==1
        history=store.get_meta('runtime_execution_history')
        assert history[-1]['effective_energy_pool']['workers']==3
        backup=path/guard.audit['database_backup']
        with sqlite3.connect(backup) as db:
            config=json.loads(db.execute("SELECT value FROM meta WHERE key='config'").fetchone()[0])
            assert config==frozen
        assert (path/'runtime_history'/f'{store.session_id}.json').exists()


@pytest.mark.parametrize('field,value',[
    ('batch_size',64),('inference_batch_size',512),('train_seeds',[23]),('hpo_trials',1)])
def test_patch_rejects_numerical_design_changes(tmp_path,field,value):
    path,frozen,proposed=_parent(tmp_path)
    proposed=copy.deepcopy(proposed);proposed['experiment_settings'][field]=value
    with pytest.raises(IncompatibleResume,match='cannot change'):
        authorize_resume(path,proposed)
    assert not (path/'runtime_history').exists()


def test_patch_rejects_unknown_source_and_active_coordinator(tmp_path):
    path,frozen,proposed=_parent(tmp_path)
    with StateStore(path,resume='require') as store:
        store.set_meta('config',{**frozen,'code_hash':'unreviewed-release'})
        with pytest.raises(IncompatibleResume,match='not authorized'):
            authorize_resume(path,proposed)
        store.set_meta('config',frozen);store.start_session()
        _,guard=authorize_resume(path,proposed)
        with pytest.raises(JobBusy,match='Stop the active coordinator'):
            with guard:pass
        assert not (path/'runtime_history').exists()


def test_target_file_hashes_and_nonbatch_fields_are_not_bypassed(tmp_path):
    path,frozen,proposed=_parent(tmp_path)
    for modified in ({**proposed,'execution':dict(device='cuda:0',threads=1)},
                     {**proposed,'datasets':{'SYNTH':{'sha256':'changed'}}}):
        with pytest.raises(IncompatibleResume):authorize_resume(path,modified)
    manifest=_current_manifest()
    manifest['target_source_files']['src/energygap/conference.py']='modified'
    custom=tmp_path/'manifest.json';custom.write_text(json.dumps(manifest))
    with pytest.raises(IncompatibleResume,match='source files differ'):
        authorize_resume(path,proposed,manifest_path=custom)


def test_worker_change_recalibrates_only_affected_phase_rates(tmp_path):
    path,frozen,_=_parent(tmp_path)
    with StateStore(path,resume='require') as store:
        store.start_session()
        costs={key:dict(units=1,seconds=1.) for key in
            ('fit:SAITS','hpo_score:SAITS','infer:SAITS:validation:C3','infer:SAITS:test:C3','correction_selection')}
        store.set_meta('conference_cost_history',costs)
        store.set_meta('conference_timings',costs)
        record_execution(store,frozen['experiment_settings'],dict(workers=3,max_pending=6),code_hash=code_hash())
        assert set(store.get_meta('conference_cost_history'))=={'fit:SAITS','hpo_score:SAITS','infer:SAITS:validation:C3'}


def test_pool_controls_are_bounded_and_do_not_modify_declared_settings():
    settings=dict(energy_cpu_workers=1,energy_cpu_max_pending=2)
    assert pool_options(settings)==dict(workers=1,max_pending=2)
    assert pool_options(settings,2)==dict(workers=2,max_pending=4)
    assert settings==dict(energy_cpu_workers=1,energy_cpu_max_pending=2)
    for workers,pending in [(True,None),(0,None),(2,1),(2,4097),(10**6,None)]:
        with pytest.raises(ValueError):pool_options(settings,workers,pending)


@pytest.mark.parametrize('source_hash',[
    'd62678487fdd69ee05fa444675099bb5f7ad823fa476994662f168da5d17554b',
    'adf62b1ae5a5c61e0d28144ea81fc0733035e53671f0fd53440fde942b5c4141',
])
def test_reporting_patch_accepts_both_audited_predecessors(tmp_path,source_hash):
    path,frozen,proposed=_parent(tmp_path,source_hash)
    restored,guard=authorize_resume(path,proposed)
    assert guard.audit['patch_id']=='reporting-performance-v1'
    assert restored==frozen
    assert guard.audit['scientific_fingerprint']==fingerprint(frozen)
    with guard,StateStore(path,restored,resume='require') as store:
        guard.attach(store)
        assert store.results()==[]
        assert (path/guard.audit['database_backup']).is_file()


def test_current_audited_release_can_continue_its_own_frozen_run(tmp_path):
    path,frozen,proposed=_parent(tmp_path,code_hash())
    restored,guard=authorize_resume(path,proposed)
    assert restored==frozen
    assert guard.audit['source_code_hash']==guard.audit['target_code_hash']==code_hash()


@pytest.mark.parametrize('status',['completed','completed_with_failures'])
def test_terminal_run_is_backed_up_before_reporting_continuation(tmp_path,status):
    path,frozen,proposed=_parent(tmp_path)
    with StateStore(path,resume='require',reconcile_on_open=False) as store:
        store.set_meta('status',status)
    restored,guard=authorize_resume(path,proposed)
    with guard:
        backup=path/guard.audit['database_backup']
        with sqlite3.connect(backup) as db:
            saved=json.loads(db.execute("SELECT value FROM meta WHERE key='config'").fetchone()[0])
            saved_status=json.loads(db.execute("SELECT value FROM meta WHERE key='status'").fetchone()[0])
        assert saved==frozen==restored
        assert saved_status==status


def test_reporting_options_recorded_separately_and_only_report_rates_reset(tmp_path):
    path,frozen,_=_parent(tmp_path)
    reporting=dict(workers=8,chunk_size=20000,bootstrap_chunk_size=1000)
    with StateStore(path,resume='require') as store:
        store.start_session()
        costs={name:dict(units=1,seconds=2.) for name in (
            'fit:SAITS','hpo_score:SAITS','infer:SAITS:test:C3',
            'report','bootstrap','bootstrap:H1','report_stage:episode_export')}
        store.set_meta('conference_cost_history',costs)
        store.set_meta('conference_timings',costs)
        store.set_meta('report_progress',{'stage':'old serial stage'})
        original_config=copy.deepcopy(store.get_meta('config'))
        record_execution(store,frozen['experiment_settings'],dict(workers=2,max_pending=4),
                         code_hash=code_hash(),reporting=reporting)
        assert store.get_meta('config')==original_config
        assert store.get_meta('report_progress') is None
        assert store.get_meta('runtime_execution_history')[-1]['effective_report_execution']==reporting
        assert set(store.get_meta('conference_cost_history'))=={
            'fit:SAITS','hpo_score:SAITS','infer:SAITS:test:C3'}
        assert store.get_meta('conference_timings')==store.get_meta('conference_cost_history')
        store.set_meta('conference_cost_history',costs)
        record_execution(store,frozen['experiment_settings'],dict(workers=2,max_pending=4),
                         code_hash=code_hash(),reporting=reporting)
        assert store.get_meta('conference_cost_history')==costs


def test_historical_cpu_manifest_is_not_rewritten_to_authorize_new_source(tmp_path):
    manifest=json.loads(MANIFEST.read_text())
    assert manifest['patch_id']=='cpu-worker-scaling-v1'
    assert manifest['target_code_hash']=='adf62b1ae5a5c61e0d28144ea81fc0733035e53671f0fd53440fde942b5c4141'
    path,_,proposed=_parent(tmp_path)
    with pytest.raises(IncompatibleResume,match='not this exact audited release'):
        authorize_resume(path,proposed,manifest_path=MANIFEST)
