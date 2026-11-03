from dataclasses import replace, fields
from datetime import timedelta
import math
import numpy as np
import pytest
from energygap.data import make_fixture, SeriesData
from energygap.masking import generate_mask, Episode
from energygap.protocols import rolling_origin_splits, build_query


def setup_case():
    s=make_fixture(days=8,natural_missing=False)
    m=generate_mask(s,rolling_origin_splits(s)[0],condition='C3',max_episodes=4)
    assert m.episodes
    return s,m,m.episodes[0]


def assert_same_inputs(a,b):
    np.testing.assert_allclose(a.values,b.values,equal_nan=True)
    np.testing.assert_equal(a.available_to_method,b.available_to_method)
    assert a.times==b.times
    assert a.input_missing_mask==b.input_missing_mask
    assert a.target_gap_mask==b.target_gap_mask


@pytest.mark.parametrize('task',['R','C'])
def test_future_poisoning(task):
    s,m,e=setup_case()
    t=s.records[e.right_index if task=='R' else e.end_index].availability_time
    q,_=build_query(s,e,issuance_time=t,task=task,artificial_hidden=m.hidden_indices)
    changed=SeriesData([replace(r,value_canonical=1e12,auxiliaries={'temperature':1e12},import_counter=1e12) if r.availability_time>t else r for r in s.records],s.site_id)
    poisoned,_=build_query(changed,e,issuance_time=t,task=task,artificial_hidden=m.hidden_indices)
    assert_same_inputs(q,poisoned)


def test_hidden_truth_isolation():
    s,m,e=setup_case()
    q,truth=build_query(s,e,artificial_hidden=m.hidden_indices)
    poisoned=SeriesData([replace(r,value_canonical=-1e10) if i in m.hidden_indices else r for i,r in enumerate(s.records)],s.site_id)
    b,changed_truth=build_query(poisoned,e,artificial_hidden=m.hidden_indices)
    assert_same_inputs(q,b)
    assert any(a!=b for a,b,mask in zip(truth.values,changed_truth.values,truth.evaluation_mask) if mask)
    assert not any(f.name in ('truth','value_original','records','series') for f in fields(q))


def test_counter_delay_removes_boundary_energy_until_arrival():
    s,m,e=setup_case();t=s.records[e.right_index].availability_time
    q,_=build_query(s,e,information_tier='E',artificial_hidden=m.hidden_indices)
    assert q.energy_intervals[0]['import_kwh'] is not None
    rows=s.records.copy();rows[e.right_index]=replace(rows[e.right_index],counter_availability_time=t+timedelta(seconds=1))
    changed=SeriesData(rows,s.site_id)
    delayed,_=build_query(changed,e,issuance_time=t,information_tier='E',artificial_hidden=m.hidden_indices)
    assert delayed.energy_intervals[0]['import_kwh'] is None
    assert delayed.energy_intervals[0]['import_status']=='endpoint_unavailable'


def test_blackout_hides_aux_and_counter_interior():
    s,m,e=setup_case()
    q,_=build_query(s,e,information_tier='AE',artificial_hidden=m.hidden_indices)
    for j,is_target in enumerate(q.target_gap_mask):
        if is_target:
            assert math.isnan(q.values[j])
            assert all(math.isnan(channel[j]) for channel in q.auxiliaries.values())
    assert len(q.energy_intervals)==1
    assert set(q.energy_intervals[0]) >= {'left_endpoint_id','right_endpoint_id'}
    assert not hasattr(q,'counters')


def test_mask_applied_before_overlapping_windows_and_age():
    s,m,e=setup_case()
    hidden=set(m.hidden_indices)
    for episode in m.episodes:
        q,_=build_query(s,episode,context_length=1024,artificial_hidden=m.hidden_indices)
        for j,rid in enumerate(q.source_record_ids):
            if rid!='padding' and int(rid) in hidden:
                assert math.isnan(q.values[j]) and not q.available_to_method[j]
    q,_=build_query(s,e,artificial_hidden=m.hidden_indices)
    ages=[a for a,mask in zip(q.elapsed_since_observed,q.target_gap_mask) if mask]
    assert ages==[300.,600.,900.]


def test_natural_missing_never_evaluation_truth():
    s,m,e=setup_case();rows=s.records.copy();i=e.indices[0]
    rows[i]=replace(rows[i],observed_original=False,natural_missing=True,value_canonical=math.nan)
    q,truth=build_query(SeriesData(rows,s.site_id),e,artificial_hidden=m.hidden_indices)
    for rid,mask,v in zip(q.source_record_ids,truth.evaluation_mask,truth.values):
        if rid==str(i):assert not mask and math.isnan(v)


def test_padding_is_missing_and_not_ground_truth():
    s,m,e=setup_case()
    e=Episode('early',(1,2,3),0,4,15,20,'C3',2001,1,'train')
    q,truth=build_query(s,e,context_length=512)
    assert sum(q.padding_mask)==507
    assert all(not ev for ev,pad in zip(truth.evaluation_mask,q.padding_mask) if pad)
    assert all(math.isnan(v) for v,pad in zip(q.values,q.padding_mask) if pad)


def test_no_right_boundary_in_c_and_offline_requires_lookahead():
    s,m,e=setup_case()
    q,_=build_query(s,e,task='C',information_tier='E')
    assert q.times[-1]==s.times[e.end_index]
    assert not q.energy_intervals
    with pytest.raises(ValueError):build_query(s,e,task='O')
    q,_=build_query(s,e,task='O',lookahead_seconds=600)
    assert q.times[-1]<=s.times[e.right_index]+600


def test_late_arrival_cannot_enter_earlier_aligned_slot():
    s,m,e=setup_case();t=s.records[e.right_index].availability_time
    rows=s.records.copy();i=e.left_index
    rows[i]=replace(rows[i],availability_time=t+timedelta(seconds=1))
    q,_=build_query(SeriesData(rows,s.site_id),e,issuance_time=t)
    j=q.source_record_ids.index(str(i))
    assert math.isnan(q.values[j]) and not q.available_to_method[j]


def test_context_cannot_silently_drop_episode_support():
    s,m,e=setup_case()
    with pytest.raises(ValueError,match='entirely'):
        build_query(s,e,context_length=2)


def test_ongoing_default_issuance_is_last_target_not_reconnection():
    s,m,e=setup_case()
    q,_=build_query(s,e,task='C')
    assert q.issuance_time==s.records[e.end_index].availability_time
