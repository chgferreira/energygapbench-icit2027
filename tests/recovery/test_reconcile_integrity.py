"""Recovery reduces duplicate I/O while retaining byte-level integrity checks."""
from collections import Counter
import hashlib
import json
import logging
import os
from pathlib import Path

import pytest

from energygap.runtime import state
from energygap.runtime.state import StateStore, atomic_bytes


def _commit(store, job_id, records, *, dependencies=()):
    store.add_job(job_id,1,dependencies)
    with store.claim(job_id) as lease:
        path=store.publish(job_id,lease.token,records,1)
        store.finish(job_id,lease.token)
    return path


def _artifact_run(tmp_path):
    path=tmp_path/'run'
    config={'profile':'smoke','seed':11}
    with StateStore(path,config) as store:
        artifact=path/'models/shared.bin'
        checksum=atomic_bytes(artifact,b'original model bytes')
        shards=[_commit(store,'fit:'+str(index),[
            {'key':'fit:'+str(index),'artifact':'models/shared.bin','sha256':checksum}
        ]) for index in range(2)]
    return path,config,artifact,shards


def test_each_committed_shard_read_once_and_shared_artifact_hashed_once(tmp_path,monkeypatch,caplog):
    path,config,artifact,shards=_artifact_run(tmp_path)
    reads=Counter(); hashes=Counter()
    read_bytes=Path.read_bytes
    file_sha256=state.file_sha256
    def read(file):
        reads[file]+=1
        return read_bytes(file)
    def digest(file):
        hashes[Path(file)]+=1
        return file_sha256(file)
    monkeypatch.setattr(Path,'read_bytes',read)
    monkeypatch.setattr(state,'file_sha256',digest)
    caplog.set_level(logging.INFO,logger=state.__name__)
    with StateStore(path,config,resume='require') as store:
        assert len(store.results())==2
        assert all(store.job('fit:'+str(i))['status']=='succeeded' for i in range(2))
    assert all(reads[file]==1 for file in shards)
    assert reads[artifact]==0  # Binary references use bounded-memory streaming.
    assert hashes[artifact]==1
    assert 'committed shards checked 2/2' in caplog.text
    assert 'Recovery complete: 0 invalidated shards, 0 recovered shards' in caplog.text


def test_corrupt_shared_artifact_invalidates_all_users_and_dependents_only(tmp_path,monkeypatch):
    path,config,artifact,shards=_artifact_run(tmp_path)
    with StateStore(path,resume='require',reconcile_on_open=False) as store:
        _commit(store,'dependent',[{'key':'dependent'}],dependencies=('fit:0',))
        _commit(store,'independent',[{'key':'independent'}])
        frozen=store.get_meta('config'); identity=store.fingerprint
    artifact.write_bytes(b'corrupt model bytes!')
    hashes=Counter(); original=state.file_sha256
    def digest(file):
        hashes[Path(file)]+=1
        return original(file)
    monkeypatch.setattr(state,'file_sha256',digest)
    with StateStore(path,config,resume='require') as store:
        assert [record['key'] for record in store.results()]==['independent']
        assert store.job('independent')['status']=='succeeded'
        assert all(store.job(job)['status']=='interrupted' for job in ('fit:0','fit:1','dependent'))
        assert store.get_meta('config')==frozen
        assert store.fingerprint==identity
    assert hashes[artifact]==1


def test_cache_does_not_survive_reconciliation_passes(tmp_path,monkeypatch):
    path,config,artifact,_=_artifact_run(tmp_path)
    hashes=Counter(); original=state.file_sha256
    def digest(file):
        hashes[Path(file)]+=1
        return original(file)
    monkeypatch.setattr(state,'file_sha256',digest)
    with StateStore(path,config,resume='require') as store:
        assert hashes[artifact]==1
        artifact.write_bytes(b'changed after recovery')
        result=store.reconcile()
        assert result['invalid']
        assert not store.results()
    assert hashes[artifact]==2


def test_changed_reference_metadata_forces_new_byte_hash_within_pass(tmp_path,monkeypatch):
    artifact=tmp_path/'model.bin'
    artifact.write_bytes(b'original')
    before=artifact.stat()
    cache={}; hashes=[]; original=state.file_sha256
    def digest(file):
        hashes.append(Path(file))
        return original(file)
    monkeypatch.setattr(state,'file_sha256',digest)
    first=state._checked_file_digest(artifact,cache)
    artifact.write_bytes(b'modified')  # Same length; restore mtime as well.
    os.utime(artifact,ns=(before.st_atime_ns,before.st_mtime_ns))
    second=state._checked_file_digest(artifact,cache)
    assert first!=second
    assert len(hashes)==2


def test_reference_changed_during_hash_is_not_cached_as_valid(tmp_path,monkeypatch):
    artifact=tmp_path/'model.bin'; artifact.write_bytes(b'original')
    cache={}; original=state.file_sha256
    def digest(file):
        result=original(file)
        Path(file).write_bytes(b'changed during validation')
        return result
    monkeypatch.setattr(state,'file_sha256',digest)
    assert state._checked_file_digest(artifact,cache) is None
    assert not cache


def test_orphan_is_read_once_and_recovers_with_verified_reference(tmp_path,monkeypatch):
    path=tmp_path/'run'; config={'seed':11}
    with StateStore(path,config) as store:
        artifact=path/'models/fitted.bin'
        checksum=atomic_bytes(artifact,b'fitted state')
        store.add_job('fit',1)
        def fault(stage):
            if stage=='before_commit':raise RuntimeError('crash before ledger commit')
        with pytest.raises(RuntimeError,match='crash before ledger commit'):
            with store.claim('fit') as lease:
                store.publish('fit',lease.token,[{
                    'key':'fit','artifact':'models/fitted.bin','sha256':checksum
                }],1,fault=fault)
    orphan=next((path/'shards').glob('*.json'))
    reads=Counter(); original=Path.read_bytes
    def read(file):
        reads[file]+=1
        return original(file)
    monkeypatch.setattr(Path,'read_bytes',read)
    with StateStore(path,config,resume='require') as store:
        assert store.job('fit')['completed_units']==1
        assert [record['key'] for record in store.results()]==['fit']
        event=store.db.execute("SELECT payload FROM events WHERE kind='reconciliation'").fetchone()
        assert json.loads(event[0])['recovered']==[orphan.name]
    assert reads[orphan]==1


def test_orphan_with_corrupt_reference_is_not_incorporated(tmp_path):
    path=tmp_path/'run'
    with StateStore(path,{'seed':11}) as store:
        store.add_job('fit',1)
        artifact=path/'models/fitted.bin'
        checksum=atomic_bytes(artifact,b'fitted state')
        def fault(stage):
            if stage=='before_commit':raise RuntimeError('crash')
        with pytest.raises(RuntimeError):
            with store.claim('fit') as lease:
                store.publish('fit',lease.token,[{
                    'key':'fit','artifact':'models/fitted.bin','sha256':checksum
                }],1,fault=fault)
    artifact.write_bytes(b'bad')
    with StateStore(path,resume='require') as store:
        assert store.results()==[]
        assert store.job('fit')['completed_units']==0


def test_corrupted_known_filename_is_verified_before_orphan_skip(tmp_path):
    path,config,_,shards=_artifact_run(tmp_path)
    shards[0].write_bytes(b'{"truncated":')
    with StateStore(path,config,resume='require') as store:
        assert store.job('fit:0')['status']=='interrupted'
        assert store.job('fit:1')['status']=='succeeded'
        assert [record['key'] for record in store.results()]==['fit:1']


@pytest.mark.parametrize('terminal',['completed','completed_with_failures'])
def test_valid_legacy_terminal_run_keeps_results_and_status(tmp_path,terminal):
    path,config,_,_=_artifact_run(tmp_path)
    with StateStore(path,resume='require',reconcile_on_open=False) as store:
        store.set_meta('status',terminal)
        before=store.results(); identity=store.fingerprint
    with StateStore(path,config,resume='require') as store:
        assert store.get_meta('status')==terminal
        assert store.results()==before
        assert store.fingerprint==identity


def test_missing_reference_and_path_escape_fail_closed(tmp_path):
    with StateStore(tmp_path/'run',{}) as store:
        assert not store._referenced_artifacts_valid({'records':[{
            'artifact':'missing.bin','sha256':hashlib.sha256(b'missing').hexdigest()
        }]})
        assert not store._referenced_artifacts_valid({'records':[{
            'artifact':'../outside.bin','sha256':hashlib.sha256(b'outside').hexdigest()
        }]})
        assert not store._referenced_artifacts_valid({'records':[{
            'artifact':'models','sha256':None
        }]})
