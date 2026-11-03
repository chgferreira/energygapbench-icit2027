"""Cached HPO losses belong to one exact best checkpoint and complete mask set."""
import hashlib
import json
from types import SimpleNamespace

from energygap.conference import _load_selection_scores, load_settings


def test_only_complete_checksum_verified_best_checkpoint_scores_are_reused(tmp_path):
    cases=[(SimpleNamespace(seed=seed),SimpleNamespace(gap_id=gap))
           for seed,gap in [(1001,'a'),(1002,'b')]]
    cached=dict(model_sha256='fitted-model-digest',best_epoch=3,
                scores=[dict(mask_seed=1001,gap_id='a',nmae=.12),
                        dict(mask_seed=1002,gap_id='b',nmae=.34)])
    path=tmp_path/'selection.json'
    def write():
        path.write_text(json.dumps(cached))
        record['selection_scores_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    record=dict(selection_scores_artifact='selection.json',sha256='fitted-model-digest',
                training_metadata={'best_epoch':3})
    write()
    assert set(_load_selection_scores(tmp_path,record,cases))=={(1001,'a'),(1002,'b')}
    assert _load_selection_scores(tmp_path,record,list(reversed(cases)))[(1002,'b')]['nmae']==.34
    assert _load_selection_scores(tmp_path,record,cases[:1])=={}
    path.write_text(path.read_text()+' ')
    assert _load_selection_scores(tmp_path,record,cases)=={}
    write()
    cached['best_epoch']=2;write()
    assert _load_selection_scores(tmp_path,record,cases)=={}
    cached['best_epoch']=3;cached['model_sha256']='different-model';write()
    assert _load_selection_scores(tmp_path,record,cases)=={}
    cached['model_sha256']=record['sha256'];cached['scores'][0]['nmae']=float('nan');write()
    assert _load_selection_scores(tmp_path,record,cases)=={}
    path.unlink()
    assert _load_selection_scores(tmp_path,record,cases)=={}
    assert _load_selection_scores(tmp_path,{},cases)=={}


def test_operational_intervals_are_validated_without_reducing_science():
    from pathlib import Path
    import pytest
    settings=load_settings(Path(__file__).resolve().parents[2])
    assert settings['max_episodes'] is None and settings['max_training_windows'] is None
    for key in ('progress_refresh_seconds','progress_persist_seconds','status_snapshot_seconds','heartbeat_seconds'):
        for invalid in (0,-1,True,float('inf')):
            with pytest.raises(ValueError,match='Positive finite interval'):
                load_settings('.',settings={**settings,key:invalid})
