"""Checksummed, two-generation checkpoints for trusted local execution state."""
from __future__ import annotations
import hashlib
import json
import os
import pickle
import shutil
import uuid
from pathlib import Path
from typing import Any
from .state import atomic_json, atomic_bytes, fsync_directory, utcnow, IncompatibleResume

class CheckpointStore:
    def __init__(self, run_dir: str | Path, fingerprint: str, keep: int = 2, ledger=None):
        if keep < 2: raise ValueError('at least two generations are required')
        self.root = Path(run_dir) / 'checkpoints'; self.root.mkdir(parents=True, exist_ok=True)
        self.fingerprint, self.keep, self.ledger = fingerprint, keep, ledger
        self.last_recovery: dict = {}

    def _job_dir(self, job_id: str) -> Path:
        path = self.root / hashlib.sha256(job_id.encode()).hexdigest()
        path.mkdir(exist_ok=True); return path

    def save(self, job_id: str, state: Any, step: int, fault=None) -> dict:
        parent = self._job_dir(job_id)
        generation = f'{step:012d}-{uuid.uuid4().hex}'
        staging = parent / ('.' + generation + '.tmp'); staging.mkdir()
        data = pickle.dumps(state, protocol=5)
        checksum = atomic_bytes(staging / 'state.pkl', data)
        manifest = {'schema_version': 1, 'job_id': job_id, 'fingerprint': self.fingerprint,
                    'step': step, 'created_at': utcnow(), 'generation': generation,
                    'files': {'state.pkl': {'sha256': checksum, 'bytes': len(data)}}}
        atomic_json(staging / 'manifest.json', manifest)
        fsync_directory(staging)
        # Validate serialization before publishing the generation.
        pickle.loads(data)
        if fault: fault('checkpoint_before_rename')
        final = parent / generation; os.replace(staging, final); fsync_directory(parent)
        if fault: fault('checkpoint_after_rename')
        atomic_json(parent / 'latest.json', {'generation': generation})
        if self.ledger:
            path = str((final / 'manifest.json').relative_to(self.ledger.path))
            self.ledger.db.execute('INSERT OR REPLACE INTO checkpoints VALUES(?,?,?,?,?)',
                                   (path, job_id, step, hashlib.sha256((final/'manifest.json').read_bytes()).hexdigest(), manifest['created_at']))
            self.ledger.set_meta('last_valid_checkpoint',{'path':path,'job_id':job_id,'step':step,'created_at':manifest['created_at'],'generation':generation})
            self.ledger.snapshot()
        # Only remove older VALID generations after two newer valid ones exist.
        valid = []
        for directory in parent.iterdir():
            if directory.is_dir() and not directory.name.startswith('.'):
                try: valid.append((self._validate(directory)['step'], directory))
                except (ValueError, OSError, KeyError, IncompatibleResume): pass
        for _, directory in sorted(valid, key=lambda item: (item[0], item[1].name), reverse=True)[self.keep:]:
            shutil.rmtree(directory)
        return manifest

    def _validate(self, directory: Path) -> dict:
        manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest['schema_version'] != 1: raise ValueError('checkpoint schema unsupported')
        if manifest['fingerprint'] != self.fingerprint: raise IncompatibleResume('checkpoint scientific fingerprint differs')
        if manifest['generation'] != directory.name: raise ValueError('checkpoint generation differs')
        for name, info in manifest['files'].items():
            if Path(name).name != name: raise ValueError('unsafe checkpoint path')
            data = (directory / name).read_bytes()
            if len(data) != info['bytes'] or hashlib.sha256(data).hexdigest() != info['sha256']:
                raise ValueError('checkpoint checksum mismatch')
        return manifest

    def load(self, job_id: str) -> tuple[Any, dict] | None:
        parent = self._job_dir(job_id)
        candidates = sorted([p for p in parent.iterdir() if p.is_dir() and not p.name.startswith('.')], reverse=True)
        try: requested = json.loads((parent / 'latest.json').read_text())['generation']
        except (ValueError, OSError, KeyError): requested = None
        if requested:
            candidates.sort(key=lambda p: p.name == requested, reverse=True)
        failures = []
        for directory in candidates:
            try:
                manifest = self._validate(directory)
                if manifest['job_id'] != job_id: raise ValueError('checkpoint job mismatch')
                state = pickle.loads((directory / 'state.pkl').read_bytes())
                self.last_recovery = {'requested': requested, 'loaded': directory.name,
                                      'fallback': bool(failures), 'rejected': failures,
                                      'step': manifest['step']}
                if self.ledger:
                    self.ledger.set_meta('last_valid_checkpoint',{'path':str((directory/'manifest.json').relative_to(self.ledger.path)),'job_id':job_id,'step':manifest['step'],'created_at':manifest['created_at'],'generation':directory.name})
                    if failures: self.ledger.event('checkpoint_fallback', job_id=job_id, **self.last_recovery)
                atomic_json(parent / 'latest.json', {'generation': directory.name})
                return state, manifest
            except IncompatibleResume: raise
            except (ValueError, OSError, KeyError, pickle.UnpicklingError, EOFError) as error:
                failures.append({'generation': directory.name, 'reason': str(error)})
        self.last_recovery = {'requested': requested, 'loaded': None, 'fallback': bool(failures), 'rejected': failures, 'step': 0}
        return None
