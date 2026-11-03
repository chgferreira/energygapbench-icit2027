import io
import json
import os
import signal
import subprocess
import sys
from datetime import datetime,timezone,timedelta
from pathlib import Path
import pytest
from energygap.runtime import StateStore,CheckpointStore,IncompatibleResume,LeaseLost,JobBusy,BudgetExhausted,ETAEstimator,Progress,fingerprint
from energygap.runtime.scheduler import run_units
from energygap.runtime.demo import demonstrate,train_fixture,DEMO_CONFIG


def test_identity_excludes_presentation_only():
    config={'data':'sha1','mask':'a','model':{'lr':.1},'progress':'log','output_dir':'a'}
    assert fingerprint(config)==fingerprint({**config,'progress':'none','output_dir':'b'})
    operational={**config,'experiment_settings':{'progress_persist_seconds':10,'heartbeat_seconds':15}}
    assert fingerprint(operational)==fingerprint({**operational,'experiment_settings':{
        'progress_persist_seconds':30,'heartbeat_seconds':60}})
    for changed in ({'data':'sha2'},{'mask':'b'},{'model':{'lr':.2}},{'base_checkpoint':'new'}):
        assert fingerprint(config)!=fingerprint({**config,**changed})


def test_never_require_and_incompatible(tmp_path):
    with pytest.raises(FileNotFoundError):StateStore(tmp_path/'missing',{},resume='require')
    with StateStore(tmp_path/'r',{'data':'a'}) as store:store.add_job('a',1)
    with pytest.raises(FileExistsError):StateStore(tmp_path/'r',{'data':'a'},resume='never')
    with pytest.raises(IncompatibleResume):StateStore(tmp_path/'r',{'data':'b'})
    with StateStore(tmp_path/'r',{'data':'a','progress':'tqdm'},resume='require') as store:
        assert store.job('a')['completed_units']==0


def test_progress_commit_idempotence_completed_resume(tmp_path):
    store=StateStore(tmp_path,{'seed':1});store.add_job('infer',7)
    computed=[]
    def compute(i,s):computed.append(i);return {'key':str(i),'seed':s,'value':i*i}
    result=run_units(store,'infer',compute,shard_size=2)
    assert result['executed_units']==7
    assert store.job('infer')['completed_units']==7
    assert len(store.results())==7
    assert run_units(store,'infer',compute)['executed_units']==0
    assert computed==list(range(7))
    assert store.complete(); store.close()
    with StateStore(tmp_path,{'seed':1}) as resumed:
        assert resumed.status()['status']=='completed'
        assert run_units(resumed,'infer',compute)['executed_units']==0
        assert len(resumed.results())==7


@pytest.mark.parametrize('phase',['during_write','after_rename','before_commit','after_commit'])
def test_crash_consistency_shards(tmp_path,phase):
    store=StateStore(tmp_path,{'x':1});store.add_job('infer',2)
    class Crash(Exception):pass
    def fault(point):
        if point==phase:raise Crash(point)
    with store.claim('infer') as lease:
        with pytest.raises(Crash):store.publish('infer',lease.token,[{'key':'q0','value':0}],1,fault)
    store.close()
    with StateStore(tmp_path,{'x':1}) as resumed:
        expected=0 if phase=='during_write' else 1
        assert resumed.job('infer')['completed_units']==expected
        assert len(resumed.results())==expected
        run_units(resumed,'infer',lambda i,s:{'key':f'q{i}','value':i},shard_size=1)
        assert resumed.results()==[{'key':'q0','value':0},{'key':'q1','value':1}]


def test_committed_missing_artifact_local_recovery(tmp_path):
    store=StateStore(tmp_path,{})
    run_units(store,'a',lambda i,s:{'key':f'a{i}','value':i},total_units=2,shard_size=1)
    run_units(store,'b',lambda i,s:{'key':f'b{i}','value':i},total_units=1,shard_size=1)
    row=store.db.execute("SELECT path FROM shards WHERE job_id='a' ORDER BY end_unit LIMIT 1").fetchone()
    (tmp_path/row[0]).write_text('corrupt')
    store.close()
    with StateStore(tmp_path,{}) as resumed:
        assert resumed.job('a')['status']=='interrupted'
        assert resumed.job('a')['completed_units']==0
        assert resumed.job('b')['status']=='succeeded'
        run_units(resumed,'a',lambda i,s:{'key':f'a{i}','value':i},shard_size=1)
        assert len(resumed.results())==3


def test_two_writers_and_fencing(tmp_path):
    first=StateStore(tmp_path,{});first.add_job('a',1)
    second=StateStore(tmp_path,{})
    lease=first.claim('a')
    with pytest.raises(JobBusy):second.claim('a')
    # Simulate a lease coordinator revoking ownership, retaining stale worker token.
    second.db.execute("UPDATE jobs SET generation=generation+1,owner='replacement' WHERE job_id='a'")
    with pytest.raises(LeaseLost):first.publish('a',lease.token,[{'key':'bad'}],1)
    assert first.results()==[]
    lease.release();first.close();second.close()


def test_corrupt_checkpoint_falls_back_preserves_other_job(tmp_path):
    store=StateStore(tmp_path,{'precision':'float64'});store.add_job('train',20)
    run_units(store,'other',lambda i,s:{'key':'other','v':2},total_units=1)
    cp=CheckpointStore(tmp_path,store.fingerprint,ledger=store)
    cp.save('train',{'step':5,'weight':[1,2]},5)
    latest=cp.save('train',{'step':10,'weight':[3,4]},10)
    path=cp._job_dir('train')/latest['generation']/'state.pkl';path.write_bytes(b'bad')
    state,manifest=cp.load('train')
    assert state=={'step':5,'weight':[1,2]}
    assert cp.last_recovery['fallback']
    assert store.results()==[{'key':'other','v':2}]
    assert len(list(cp._job_dir('train').glob('*/manifest.json')))==2


def test_partial_checkpoint_not_promoted(tmp_path):
    store=StateStore(tmp_path,{})
    cp=CheckpointStore(tmp_path,store.fingerprint)
    cp.save('train',{'step':1},1)
    class Crash(Exception):pass
    def fault(where):
        if where=='checkpoint_before_rename':raise Crash()
    with pytest.raises(Crash):cp.save('train',{'step':2},2,fault)
    assert cp.load('train')[0]['step']==1


def test_budget_cumulative_and_extension(tmp_path):
    with StateStore(tmp_path,{},budget_seconds=10) as store:
        store.record_active(7);store.check_budget()
    with StateStore(tmp_path,{}) as store:
        assert store.get_meta('active_seconds')==7
        store.record_active(4)
        with pytest.raises(BudgetExhausted):store.check_budget()
        assert store.get_meta('status')=='paused_budget'
        store.extend_budget(20);store.check_budget()
        assert store.get_meta('active_seconds')==11
        with pytest.raises(ValueError):store.extend_budget(5)


def test_progress_initial_and_eta_fake_clock():
    stream=io.StringIO()
    p=Progress(40,initial=17,mode='tqdm',desc='partial',stream=stream)
    assert p.bar.n==17
    p.update(20);p.close()
    assert '17/40' in stream.getvalue()
    now=datetime(2026,10,31,19,tzinfo=timezone.utc)
    eta=ETAEstimator(now=lambda:now)
    eta.observe('cpu:median:L512',10,2)
    first=eta.estimate(20,'cpu:median:L512')
    assert first['remaining_seconds']==4
    eta2=ETAEstimator(eta.history,now=lambda:now+timedelta(days=30))
    after=eta2.estimate(20,'cpu:median:L512')
    assert after['remaining_seconds']==4
    assert datetime.fromisoformat(after['estimated_finish'])==now+timedelta(days=30,seconds=4)
    assert eta2.estimate(20,'gpu:csdi:S100')['remaining_seconds'] is None


def test_dag_eta_resources():
    eta=ETAEstimator()
    estimate=eta.schedule([
        {'job_id':'a','total_units':1,'estimated_seconds':3,'resource':'gpu'},
        {'job_id':'b','total_units':1,'estimated_seconds':5,'resource':'gpu'},
        {'job_id':'c','total_units':1,'estimated_seconds':4,'dependencies':['a'],'resource':'cpu'}],{'gpu':1,'cpu':1})
    assert estimate['remaining_seconds']==8
    assert estimate['schedule']['c']['start_seconds_from_now']==3


@pytest.mark.parametrize('mode',['none','log','tqdm'])
def test_progress_backend_science_parity(tmp_path,mode):
    store=StateStore(tmp_path/mode,{'progress':mode})
    run_units(store,'x',lambda i,s:{'key':str(i),'seed':s},total_units=4,progress=mode)
    assert [r['seed'] for r in store.results()]==[__import__('energygap.runtime',fromlist=['stable_seed']).stable_seed(store.fingerprint,'x',i) for i in range(4)]


def test_bootstrap_invalid_ids_not_redrawn_and_hpo_trial_same(tmp_path):
    store=StateStore(tmp_path,{'hpo_configurations':[{'lr':.1},{'lr':.01}]})
    store.add_job('trial:0',4,metadata={'lr':.1,'trial_id':0})
    with store.claim('trial:0') as lease:
        store.publish('trial:0',lease.token,[{'key':'trial0:0','params':{'lr':.1},'value':2}],1)
        store.interrupt('trial:0',lease.token)
    run_units(store,'trial:0',lambda i,s:{'key':f'trial0:{i}','params':{'lr':.1},'value':i})
    run_units(store,'bootstrap',lambda i,s:{'key':f'rep:{i}','replicate_id':i,'seed':s,'status':'invalid' if i==2 else 'valid'},total_units=5)
    rows=store.results('bootstrap')
    assert len(rows)==5 and rows[2]['status']=='invalid'
    assert run_units(store,'bootstrap',lambda *_:pytest.fail('must not redraw'))['executed_units']==0
    assert store.job('trial:0')['attempts']==2
    assert store.job('trial:0')['metadata']['lr']==.1


def test_real_sigterm_and_sigkill_full_training_state(tmp_path):
    report=demonstrate(tmp_path/'demo')
    assert report['passed']
    for case in report['cases']:
        assert case['full_training_state_equal']
        assert case['results_equal']
        assert case['result_keys_unique']
        assert case['tqdm_total']==40
    assert report['cases'][0]['saved_progress']==5
    assert report['cases'][0]['repeated_unconfirmed_steps']==2
    assert report['cases'][1]['saved_progress']==7
    assert report['cases'][1]['repeated_unconfirmed_steps']==0


def test_crash_budget_reservation_is_retained_not_reset(tmp_path):
    with StateStore(tmp_path,{},budget_seconds=100) as store:
        store.start_session();store.add_job('infer',2)
        with store.claim('infer') as lease:
            store.begin_work('infer',lease.token,60)
            # Crash is represented by closing without finish_work/end_session.
        assert store.get_meta('active_seconds')==60
    with StateStore(tmp_path,{}) as store:
        store.start_session()
        assert store.get_meta('active_seconds')==60
        assert store.status()['crash_time_reservations'][0]['status']=='crash_uncertain_charged'
        with store.claim('infer') as lease:
            reservation=store.begin_work('infer',lease.token,60)
            store.finish_work(reservation,5)
        assert store.get_meta('active_seconds')==65


def test_referenced_model_artifact_invalidates_dependents_only(tmp_path):
    import hashlib
    store=StateStore(tmp_path,{})
    model=tmp_path/'model.pickle';model.write_bytes(b'valid')
    digest=hashlib.sha256(model.read_bytes()).hexdigest()
    run_units(store,'fit',lambda i,s:{'key':'fit','artifact':'model.pickle','sha256':digest},total_units=1)
    store.add_job('infer',1,dependencies=['fit'])
    run_units(store,'infer',lambda i,s:{'key':'infer','value':3})
    run_units(store,'independent',lambda i,s:{'key':'independent','value':8},total_units=1)
    model.write_bytes(b'corrupt');store.close()
    with StateStore(tmp_path,{}) as resumed:
        assert resumed.job('fit')['status']=='interrupted'
        assert resumed.job('infer')['status']=='interrupted'
        assert resumed.job('independent')['status']=='succeeded'
        assert resumed.results()==[{'key':'independent','value':8}]


def test_stale_orphan_cannot_publish_after_generation_revoked(tmp_path):
    store=StateStore(tmp_path,{})
    store.add_job('q',1)
    class Crash(Exception):pass
    def fault(point):
        if point=='before_commit':raise Crash()
    with store.claim('q') as old:
        with pytest.raises(Crash):store.publish('q',old.token,[{'key':'old','value':1}],1,fault)
    # A new owner acquired before old output reconciliation is attempted.
    with store.claim('q') as new:
        assert new.token==2
        assert store.reconcile()['recovered']==[]
        store.publish('q',new.token,[{'key':'new','value':2}],1);store.finish('q',new.token)
    assert store.results()==[{'key':'new','value':2}]


def test_experiment_coordinator_lock(tmp_path):
    first=StateStore(tmp_path,{});second=StateStore(tmp_path,{})
    first.start_session()
    with pytest.raises(JobBusy):second.start_session()
    first.close();second.start_session();second.close()


def test_persisted_paired_bootstrap_matches_continuous_and_invalid_ids(tmp_path):
    from energygap.runtime.statistics_job import persisted_paired_bootstrap
    from energygap.statistics.paired import paired_bootstrap
    rows=[{'fold':1,'day':'2026-01-01','duration':15,'gap_id':'a','difference':.1},
          {'fold':1,'day':'2026-01-03','duration':15,'gap_id':'b','difference':.2}]
    calendars={1:['2026-01-01','2026-01-02','2026-01-03']}
    store=StateStore(tmp_path,{'bootstrap_seed':20261031})
    actual=persisted_paired_bootstrap(store,rows,calendars,contrast_id='example',replicates=13,
                                     block_days=1,folds=(1,),durations=(15,),shard_size=3)
    expected=paired_bootstrap(rows,calendars,replicates=13,block_days=1,folds=(1,),durations=(15,))
    for key in expected:assert actual[key]==expected[key]
    count=len(store.results());assert count==13
    # A second invocation never samples, commits, or redraws invalid replicates.
    again=persisted_paired_bootstrap(store,rows,calendars,contrast_id='example',replicates=13,
                                    block_days=1,folds=(1,),durations=(15,),shard_size=3)
    assert again==actual and len(store.results())==count
    assert [r['replicate_id'] for r in sorted(store.results(),key=lambda r:r['replicate_id'])]==list(range(13))
