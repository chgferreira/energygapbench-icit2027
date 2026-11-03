from datetime import datetime, timedelta, timezone
from dataclasses import replace
import math
import pytest
from energygap.data import make_fixture, DatasetConfig, AmbiguousDataError, SeriesData
from energygap.data.units import power_to_kw, energy_to_kwh
from energygap.data.counters import counter_increment
from energygap.data.ingest import parse_number
from energygap.data.timebase import align_records, regularize, nearest_slot
from energygap.protocols import rolling_origin_splits, require_train_indices
from energygap.protocols.windows import match_future_target
from energygap.masking import generate_mask, fit_replay


def test_units_require_evidence_and_equivalence():
    assert power_to_kw(-3500,'W','negative')==power_to_kw(-3.5,'kW','negative')==3.5
    assert energy_to_kwh(3500,'Wh')==energy_to_kwh(3.5,'kWh')
    with pytest.raises(AmbiguousDataError): power_to_kw(1000,None,'positive')
    with pytest.raises(AmbiguousDataError): DatasetConfig({'site_id':'H1'}).validate()


def test_locale_parser_keeps_zeros_negatives_extremes():
    assert parse_number('2.059,40')==2059.4
    assert parse_number('0,00')==0
    assert parse_number('-41839,0')==-41839
    assert parse_number(2.5)==2.5
    assert math.isnan(parse_number(''))


def test_counters_constant_reset_and_known_rollover():
    assert counter_increment(4,4)==(0,'ok')
    assert counter_increment(4,1)==(None,'reset_or_unknown_rollover')
    assert counter_increment(99,2,known_rollover=100)==(3,'known_rollover')
    assert counter_increment(4,5,right_available=False)==(None,'endpoint_unavailable')


def test_alignment_does_not_retime_arrival_or_resolve_conflicts():
    r=make_fixture(days=1).records[0]
    late=replace(r,measurement_time=r.measurement_time+timedelta(seconds=70),availability_time=r.measurement_time+timedelta(seconds=120))
    aligned,info=align_records([late])
    assert aligned[0].grid_time==r.grid_time
    assert aligned[0].measurement_time==late.measurement_time
    assert aligned[0].availability_time==late.availability_time
    conflict=replace(r,value_canonical=9,source_record_id='other')
    rows,audit=align_records([r,conflict])
    assert not rows[0].observed_original and math.isnan(rows[0].value_canonical)
    assert audit['collisions'][0]['kind']=='conflicting'
    assert nearest_slot(r.measurement_time+timedelta(seconds=91)) is None


def test_regularization_creates_missing_slots_without_truth():
    s=make_fixture(days=1)
    rows=regularize([s.records[0],s.records[2]],'SYNTH')
    assert len(rows)==3 and rows[1].natural_missing
    assert not rows[1].observed_original and math.isnan(rows[1].value_canonical)


def test_split_elapsed_time_and_identical_timestamps():
    s=make_fixture(days=10)
    sparse=SeriesData(s.records[:20]+s.records[-30:],s.site_id)
    a=rolling_origin_splits(s);b=rolling_origin_splits(sparse)
    assert [f.manifest() for f in a]==[f.manifest() for f in b]
    seen=set()
    for f in a:
        test=set(f.indices(s,'test'))
        assert not seen & test
        seen|=test
        assert all(f.partition(s.records[i].grid_time)=='train' for i in f.indices(s,'train'))
        with pytest.raises(ValueError): require_train_indices(s,f.indices(s,'test')[:1],f)


def test_mask_immutable_deterministic_and_target_original(tmp_path):
    s=make_fixture(days=7)
    f=rolling_origin_splits(s)[0]
    a=generate_mask(s,f,condition='C4',max_episodes=3)
    b=generate_mask(s,f,condition='C4',max_episodes=3)
    assert a==b and a.episodes
    for episode in a.episodes:
        assert len(episode.indices)==12
        assert episode.duration_minutes==60
        assert episode.boundary_distance_minutes==65
        assert all(s.records[i].observed_original for i in episode.indices)
        assert all(f.partition(s.records[i].grid_time)=='test' for i in episode.indices)
    out=tmp_path/'mask.json';a.write(out);a.write(out)
    c=generate_mask(s,f,seed=2002,max_episodes=3)
    with pytest.raises(FileExistsError):c.write(out)


def test_replay_estimates_only_train():
    s=make_fixture(days=10)
    f=rolling_origin_splits(s)[0]
    model=fit_replay(s,f)
    changed=SeriesData([replace(r,natural_missing=True,observed_original=False,value_canonical=math.nan) if f.partition(r.grid_time)=='test' else r for r in s.records],s.site_id)
    assert fit_replay(changed,f)==model
    assert model['estimation_partition']=='train'


def test_forecast_physical_time_tolerance_and_no_interpolated_targets():
    s=make_fixture(days=7,natural_missing=False)
    t=s.records[100].grid_time
    match,reason=match_future_target(s,t,3600)
    assert match['index']==112 and match['offset_seconds']==0 and reason is None
    rows=s.records.copy();rows[112]=replace(rows[112],observed_original=False,natural_missing=True,value_canonical=math.nan)
    missing=SeriesData(rows,s.site_id)
    assert match_future_target(missing,t,3600)[0] is None
    with pytest.raises(ValueError):match_future_target(s,t,0)


def test_zero_industrial_values_are_valid_mask_targets():
    s=make_fixture(days=4,natural_missing=False)
    f=rolling_origin_splits(s)[0]
    s.records=[replace(r,value_canonical=0.,value_original=0.) for r in s.records]
    m=generate_mask(s,f,max_episodes=2)
    assert m.episodes and m.eligible_count>0


def test_statistical_confirmation_does_not_fabricate_physical_semantics():
    c=DatasetConfig({'timestamp':{'column':'time','timezone':'America/Sao_Paulo','semantics':None},'power':{'column':'p','unit':'W','semantics':None,'import_sign':'negative'},'confirmed':{'timestamp.column':True,'timestamp.timezone':True,'timestamp.semantics':False,'power.column':True,'power.unit':True,'power.semantics':False,'power.import_sign':True}})
    c.validate(purpose='statistical')
    with pytest.raises(AmbiguousDataError,match='semantics'):c.validate()
    assert c.data['power']['semantics'] is None


def test_collision_auxiliary_conflicts_are_not_selected_as_truth():
    r=make_fixture(days=1).records[0]
    other=replace(r,source_record_id='other',auxiliaries={'temperature':500})
    rows,audit=align_records([r,other])
    assert not rows[0].observed_original and audit['collisions'][0]['kind']=='conflicting'
