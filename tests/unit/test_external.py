from datetime import datetime,timedelta
from pathlib import Path
import csv
import io
import json
import math
import zipfile
import pytest
from energygap.data.external import freeze_uci_selection,load_uci_consumers,download_uci,UCI_URL


def fixture(path,start=datetime(2011,1,1),days=4,poison_future=False):
    with path.open('w',newline='',encoding='utf8') as f:
        out=csv.writer(f,delimiter=';');out.writerow(['','MT_001','MT_002','MT_003','MT_004'])
        for i in range(days*96):
            t=start+timedelta(minutes=15*i)
            values=[0 if i<4 else 0 if i==20 else 1,2,0 if i<3*96 else 7,0]
            if poison_future and i>=2*96:values=[9999,8888,7777,6666]
            out.writerow([t.isoformat(sep=' ')]+[str(v).replace('.',',') for v in values])
    return path


def test_selection_fit_only_train_deterministic_and_frozen(tmp_path):
    a=fixture(tmp_path/'a.txt');b=fixture(tmp_path/'b.txt',poison_future=True)
    kwargs={'train_end':'2011-01-03','count':2,'min_train_observations':100}
    ma=freeze_uci_selection(a,manifest_path=tmp_path/'selection.json',**kwargs)
    mb=freeze_uci_selection(b,**kwargs)
    assert ma['selected_consumer_ids']==mb['selected_consumer_ids']
    assert ma['train_fingerprint']==mb['train_fingerprint']
    assert not ma['consumer_statistics']['MT_003']['eligible']
    assert ma['consumer_statistics']['MT_003']['reason']=='no_positive_activation_observed_in_train'
    assert ma['consumer_statistics']['MT_001']['first_active']=='2011-01-01T01:00:00'
    assert ma['eligibility']['fit_partition']=='train'
    freeze_uci_selection(a,manifest_path=tmp_path/'selection.json',**kwargs)
    with pytest.raises(FileExistsError):freeze_uci_selection(a,manifest_path=tmp_path/'selection.json',count=1,min_train_observations=100,train_end='2011-01-03')
    with pytest.raises(ValueError,match='Only 2'):freeze_uci_selection(a,count=30,min_train_observations=100,train_end='2011-01-03')


def test_independent_consumers_preactivation_and_postactivation_zeros(tmp_path):
    source=fixture(tmp_path/'a.txt')
    selection=freeze_uci_selection(source,'2011-01-03',count=2,min_train_observations=100)
    series=load_uci_consumers(source,selection)
    assert {s.site_id for s in series}=={'UCI:MT_001','UCI:MT_002'}
    s=next(s for s in series if s.site_id=='UCI:MT_001')
    assert s.nominal_seconds==900 and len(s.records)==384
    assert not s.records[0].observed_original and s.records[0].value_original==0
    assert s.records[0].quality_flag=='preactivation_placeholder'
    assert math.isnan(s.records[0].value_canonical)
    assert s.records[20].observed_original and s.records[20].value_canonical==0
    assert all(r.import_counter is None and r.export_counter is None for r in s.records)
    assert not s.metadata['physical_correction_allowed']
    assert all(not r.auxiliaries for r in s.records)


@pytest.mark.parametrize('start,transition',[('2011-03-26','2011-03-27'),('2011-10-29','2011-10-30')])
def test_dst_transition_days_excluded_without_forcing_ambiguous_times(tmp_path,start,transition):
    source=fixture(tmp_path/'dst.txt',start=datetime.fromisoformat(start),days=4)
    cutoff=(datetime.fromisoformat(start)+timedelta(days=3)).isoformat()
    selection=freeze_uci_selection(source,cutoff,count=2,min_train_observations=100)
    series=load_uci_consumers(source,selection)
    for s in series:
        assert transition in s.metadata['dst_excluded_dates']
        assert any(r.natural_missing and r.source_file_id=='missing' for r in s.records)
        assert all((b-a)==900 for a,b in zip(s.times,s.times[1:]))
        assert all(not (r.measurement_time.date().isoformat()==transition and r.observed_original) for r in s.records)


def test_manifest_or_source_change_invalidates_loading(tmp_path):
    source=fixture(tmp_path/'a.txt');selection=freeze_uci_selection(source,'2011-01-03',count=2,min_train_observations=100)
    bad=dict(selection);bad['selected_consumer_ids']=bad['selected_consumer_ids'][:1]
    with pytest.raises(ValueError,match='fingerprint'):load_uci_consumers(source,bad)
    source.write_text(source.read_text()+'\n')
    with pytest.raises(ValueError,match='source changed'):load_uci_consumers(source,selection)


def test_official_download_is_explicit_actual_hash_and_idempotent(tmp_path,monkeypatch):
    from energygap.data import external
    calls=[];payload=b'official archive fixture bytes'
    def fake_urlopen(url,timeout):
        calls.append(url);return io.BytesIO(payload)
    monkeypatch.setattr(external,'urlopen',fake_urlopen)
    destination=tmp_path/'official.zip'
    manifest=download_uci(destination)
    assert calls==[UCI_URL] and manifest['bytes']==len(payload)
    import hashlib
    assert manifest['sha256']==hashlib.sha256(payload).hexdigest()
    assert manifest['license']=='CC-BY-4.0'
    assert download_uci(destination)==manifest and calls==[UCI_URL]
    with pytest.raises(ValueError,match='checksum'):download_uci(destination,expected_sha256='0'*64)


def test_archive_read_does_not_extract_untrusted_paths(tmp_path):
    source=fixture(tmp_path/'a.txt');archive=tmp_path/'data.zip'
    with zipfile.ZipFile(archive,'w') as out:
        out.writestr('LD2011_2014.txt',source.read_text());out.writestr('../../escape.txt','bad')
    selection=freeze_uci_selection(archive,'2011-01-03',count=2,min_train_observations=100)
    series=load_uci_consumers(archive,selection)
    assert len(series)==2 and not (tmp_path/'escape.txt').exists()
