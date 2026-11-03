"""Tiny real-native neural executor test on explicitly synthetic support."""
import json
from contextlib import contextmanager
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from energygap.data import make_fixture
from energygap.conference import run_conference
import energygap.conference as runner


@pytest.mark.parametrize('cache_write_failure',[False,True])
def test_native_saits_full_runner_selects_epochs_on_all_validation_replicas(tmp_path, monkeypatch, cache_write_failure):
    pytest.importorskip('torch')
    pytest.importorskip('pypots')
    from energygap.methods.conference_models import load_adapter, ConferenceModel
    from energygap.runtime.state import StateStore
    monkeypatch.setattr(runner,'code_hash',lambda:'synthetic-native-selection-test')
    if cache_write_failure:
        original_atomic_json=runner.atomic_json
        def publish_optional_cache(path,value):
            if str(path).endswith('.selection.json'):raise OSError('simulated optional cache write failure')
            return original_atomic_json(path,value)
        monkeypatch.setattr(runner,'atomic_json',publish_optional_cache)
    original_infer = ConferenceModel.infer_queries
    batch_calls = []
    def record_batch(self, queries, **kwargs):
        batch_calls.append((self.name, len(queries), kwargs.get('batch_size')))
        assert all(not hasattr(query, 'original_record_ids') for query in queries)
        for query in queries:
            admitted=np.asarray(query.available_to_method)&~np.asarray(query.padding_mask)
            assert np.isnan(np.asarray(query.values)[~admitted]).all()
        return original_infer(self, queries, **kwargs)
    monkeypatch.setattr(ConferenceModel, 'infer_queries', record_batch)
    original_claim = StateStore.claim
    injected = []
    @contextmanager
    def claim_near_budget(self, job_id, *args, **kwargs):
        with original_claim(self, job_id, *args, **kwargs) as lease:
            if job_id.startswith('hpo:') and not injected:
                # Fifty seconds of actual budget remain. An accounting reserve
                # must not become an extra sixty seconds of measured fit time.
                self.record_active(100. - self.get_meta('active_seconds', 0.))
                injected.append(job_id)
            yield lease
    monkeypatch.setattr(StateStore, 'claim', claim_near_budget)
    settings=dict(sites=['SYNTH'],folds=[1],conditions=['C3','C4','C5'],
        methods=['linear','SAITS'],train_seeds=[11],validation_mask_seeds=[1001,1002],
        test_mask_seeds=[2001],context=512,max_episodes=1,max_training_windows=2,
        max_epochs=1,patience=1,hpo_trials=1,samples=2,batch_size=2,inference_batch_size=4,
        forecast_horizons_seconds=[3600],bootstrap_attempts=10,
        physical_hypotheses=[],shard_size=1)
    fixture=make_fixture(days=20,natural_missing=False)
    kwargs=dict(root=tmp_path,settings=settings,series_override=[fixture],device='cpu',threads=1,
        experiment_id='synthetic-native-selection',progress='none',budget_total_hours=150./3600)
    result=run_conference(**kwargs)
    assert result['status']=='completed'
    assert injected and result['active_seconds'] < 150.
    assert any(name=='SAITS' and count==4 and configured==4 for name,count,configured in batch_calls)
    directory=tmp_path/'runs/synthetic-native-selection'
    with sqlite3.connect(directory/'state.sqlite') as db:
        rows=[json.loads(r[0]) for r in db.execute('SELECT payload FROM results')]
    fits=[row for row in rows if row.get('phase')=='fit']
    assert len(fits)==2
    assert all(row['status']=='success' for row in fits)
    scores=[row for row in rows if row.get('phase')=='hpo_score']
    assert len(scores)==6
    expected_source='native_best_checkpoint_inference' if cache_write_failure else 'best_checkpoint_validation_cache'
    assert all(row['score_source']==expected_source for row in scores)
    if not cache_write_failure:
        assert all(row['source_model_sha256']==fits[0]['sha256'] for row in scores)
    assert {row['mask_seed'] for row in scores}=={1001,1002}
    assert {row['duration'] for row in scores}=={15.,60.,180.}
    inferred=[row for row in rows if row.get('phase')=='inference' and row['method']=='SAITS']
    assert inferred and all(row['latency_seconds'] is None for row in inferred)
    assert all(row['timing_mode']=='batch_amortized' and row['inference_amortized_seconds']>=0 for row in inferred)
    selected_score=float(np.mean([np.mean([row['nmae'] for row in scores if row['duration']==duration]) for duration in (15.,60.,180.)]))
    for fitted in fits:
        model=load_adapter(directory/fitted['artifact'],root=tmp_path,device='cpu',threads=1)
        assert model.metadata['epoch_selection']=='validation_equal_duration_gap_nmae'
        assert model.best_epoch==1
        assert model.history[0]['validation_metric']==pytest.approx(selected_score)
        assert model.batch_size==2 and model.inference_batch_size==4
        assert 'native_validation_metric' in model.history[0]
    before=len(rows)
    again=run_conference(**kwargs,resume='require')
    assert again['status']=='completed'
    with sqlite3.connect(directory/'state.sqlite') as db:
        assert db.execute('SELECT COUNT(*) FROM results').fetchone()[0]==before


def test_csdi_sample_batches_remain_one_query_latency_and_keep_every_joint_draw(tmp_path,monkeypatch):
    pytest.importorskip('torch');pytest.importorskip('pypots')
    monkeypatch.setattr(runner,'code_hash',lambda:'synthetic-csdi-sampling-batch')
    from energygap.methods import conference_models
    tiny={'learning_rate':.001,'architecture':dict(n_layers=1,n_heads=2,n_channels=8,
          d_time_embedding=8,d_feature_embedding=4,d_diffusion_embedding=8,n_diffusion_steps=2)}
    monkeypatch.setattr(conference_models,'hpo_configurations',lambda name:[tiny])
    settings=dict(sites=['SYNTH'],folds=[1],conditions=['C3','C4','C5'],methods=['CSDI'],
        train_seeds=[11],validation_mask_seeds=[1001],test_mask_seeds=[2001],context=512,
        max_episodes=1,max_training_windows=2,max_epochs=1,patience=1,hpo_trials=1,samples=4,
        batch_size=2,inference_batch_size=4,csdi_sample_batch_size=3,
        forecast_horizons_seconds=[3600],bootstrap_attempts=10,physical_hypotheses=[],shard_size=2)
    result=run_conference(root=tmp_path,settings=settings,series_override=[make_fixture(days=20,natural_missing=False)],
        device='cpu',threads=1,experiment_id='csdi-batches',progress='none',budget_total_hours=.1)
    assert result['status']=='completed'
    directory=tmp_path/'runs/csdi-batches'
    with sqlite3.connect(directory/'state.sqlite') as db:
        rows=[json.loads(r[0]) for r in db.execute('SELECT payload FROM results')]
    fits=[r for r in rows if r.get('phase')=='fit']
    assert len(fits)==2 and all(r['status']=='success' for r in fits)
    assert all(r.get('score_source')=='best_checkpoint_validation_cache' for r in rows if r.get('phase')=='hpo_score')
    inferred=[r for r in rows if r.get('phase')=='inference']
    assert len(inferred)==6 and all(r['status']=='success' for r in inferred)
    assert all(r['timing_mode']=='single_query' and r['latency_seconds']>=0 for r in inferred)
    assert all(r['inference_batch_size']==1 and r['csdi_sample_batch_size']==3 and r['n_joint_samples']==4 for r in inferred)
    assert all(r['probabilistic']['sample_count']==4 for r in inferred)
