"""Explicit, exact-release continuation for audited operational patches.

The immutable scientific configuration/fingerprint is retained, including its RNG
identity. This is not a general code-hash bypass: only the shipped source/target
lineage is accepted and every other scientific field must still match.
"""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import uuid

from .state import IncompatibleResume, JobBusy, atomic_json, file_sha256, fingerprint


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / 'configs/runtime_patches/cpu_worker_scaling_v1.json'
REPORT_MANIFEST = ROOT / 'configs/runtime_patches/reporting_performance_v1.json'
PATCH_POLICIES = {
    'cpu-worker-scaling-v1': 'CPU correction worker/queue scheduling only; original scientific and RNG identity retained',
    'reporting-performance-v1': 'Reporting, deterministic bootstrap scheduling, and integrity-read performance only; original scientific and RNG identity retained',
}


def pool_options(settings, workers=None, max_pending=None):
    effective_workers = settings['energy_cpu_workers'] if workers is None else workers
    effective_pending = (settings['energy_cpu_max_pending'] if workers is None else 2 * workers) if max_pending is None else max_pending
    for key, value in [('workers', effective_workers), ('max_pending', effective_pending)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'Runtime energy {key} must be a positive integer')
    if workers is not None or max_pending is not None:
        available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
        if effective_workers > (available or 1):
            raise ValueError('Runtime correction workers exceed the available CPU affinity')
        if effective_pending < effective_workers or effective_pending > 4096:
            raise ValueError('Runtime pending tasks must cover workers and remain at most 4096')
    return dict(workers=effective_workers, max_pending=effective_pending)


def _read_state(path):
    database = Path(path) / 'state.sqlite'
    if not database.is_file():
        raise IncompatibleResume('Runtime patch requires an existing conference run')
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as source:
        return {key: json.loads(value) for key, value in source.execute('SELECT key,value FROM meta')}


def authorize_resume(path, proposed, *, manifest_path=None, source_root=ROOT):
    """Require this exact target release and one explicitly audited predecessor.

    With no explicit manifest, select the registry entry for the actual current
    release. Historical entries retain their historical hashes; merely shipping
    a newer manifest never authorizes arbitrary changed source files.
    """
    from energygap.pipeline import code_hash
    actual_code_hash = code_hash()
    candidates = [Path(manifest_path)] if manifest_path is not None else [REPORT_MANIFEST,MANIFEST]
    selected = None
    for candidate in candidates:
        if manifest_path is None and not candidate.is_file():continue
        raw = candidate.read_bytes()
        manifest = json.loads(raw)
        if manifest.get('target_code_hash') == actual_code_hash:
            selected = candidate,raw,manifest
            break
    if selected is None:
        raise IncompatibleResume('Runtime patch target is not this exact audited release')
    manifest_path,raw,manifest = selected
    if (manifest.get('patch_id') not in PATCH_POLICIES
            or proposed.get('code_hash') != manifest['target_code_hash']):
        raise IncompatibleResume('Runtime patch target is not this exact audited release')
    actual_files = {str(p.relative_to(source_root)): file_sha256(p)
                    for p in (Path(source_root) / 'src/energygap').rglob('*.py')}
    if actual_files != manifest['target_source_files']:
        raise IncompatibleResume('Runtime patch target source files differ from the audited manifest')
    saved = _read_state(path)
    frozen = saved.get('config', {})
    accepted = manifest.get('accepted_source_code_hashes',[manifest.get('source_code_hash')])
    if (not isinstance(accepted,list) or not accepted
            or any(not isinstance(item,str) or len(item)!=64 for item in accepted)):
        raise IncompatibleResume('Runtime patch has no valid audited source lineage')
    if frozen.get('code_hash') not in (*accepted,manifest['target_code_hash']):
        raise IncompatibleResume('Saved source release is not authorized by this runtime patch')
    if frozen.get('profile') != 'conference' or not frozen.get('experiment_settings'):
        raise IncompatibleResume('Runtime patch only supports conference executor state')
    comparable = dict(proposed, code_hash=frozen['code_hash'])
    if fingerprint(comparable) != saved.get('fingerprint') or fingerprint(frozen) != saved.get('fingerprint'):
        raise IncompatibleResume('Runtime patch cannot change data, models, settings, device, batches, threads, or seeds')
    audit = dict(patch_id=manifest['patch_id'], source_code_hash=frozen['code_hash'],
                 target_code_hash=manifest['target_code_hash'], manifest_sha256=hashlib.sha256(raw).hexdigest(),
                 scientific_fingerprint=saved['fingerprint'],
                 policy=PATCH_POLICIES[manifest['patch_id']])
    return frozen, ResumeGuard(Path(path), frozen, audit)


class ResumeGuard:
    def __init__(self, path, config, audit):
        self.path, self.config, self.audit = path, config, audit
        self.lock = None

    def __enter__(self):
        self.lock = (self.path / 'locks/experiment.lock').open('a+')
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            saved = _read_state(self.path)
            if saved['config'] != self.config or saved['fingerprint'] != self.audit['scientific_fingerprint']:
                raise IncompatibleResume('Run identity changed before runtime patch acquired its lock')
            previous = saved.get('runtime_execution_history', [])
            if not any((row.get('patch') or {}).get('target_code_hash') == self.audit['target_code_hash']
                       and (row.get('patch') or {}).get('patch_id') == self.audit['patch_id'] for row in previous):
                directory = self.path / 'runtime_history'
                directory.mkdir(exist_ok=True)
                backup = directory / ('before-' + self.audit['patch_id'] + '-' + uuid.uuid4().hex + '.sqlite')
                with sqlite3.connect((self.path / 'state.sqlite').resolve().as_uri() + '?mode=ro', uri=True) as source:
                    with sqlite3.connect(backup) as destination:
                        source.backup(destination)
                with backup.open('rb') as stream:
                    os.fsync(stream.fileno())
                self.audit.update(database_backup=str(backup.relative_to(self.path)), database_backup_sha256=file_sha256(backup))
                atomic_json(Path(str(backup) + '.json'), self.audit)
            return self
        except BlockingIOError as exc:
            self.__exit__(None, None, None)
            raise JobBusy('Stop the active coordinator safely before applying the runtime patch') from exc
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def attach(self, store):
        # Transfer ownership; start_session must not acquire a second flock.
        store.experiment_lock = self.lock

    def __exit__(self, *_):
        if self.lock is not None and not self.lock.closed:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_UN)
            self.lock.close()


def record_execution(store, settings, options, *, code_hash, patch=None, reporting=None):
    previous = store.get_meta('runtime_execution_history', [])
    old = previous[-1]['effective_energy_pool'] if previous else dict(
        workers=settings['energy_cpu_workers'], max_pending=settings['energy_cpu_max_pending'])
    if old != options:
        # Costs calibrated under a different correction pool are not mixed.
        for key in ('conference_cost_history', 'conference_timings'):
            history = store.get_meta(key, {})
            store.set_meta(key, {name: value for name, value in history.items()
                                if name != 'correction_selection' and not (name.startswith('infer:') and ':test:' in name)})
    old_reporting = previous[-1].get('effective_report_execution') if previous else None
    if reporting is not None and old_reporting != reporting:
        # Serial and parallel report/bootstrap wall rates cannot be pooled.
        # Model fitting and inference histories remain valid and untouched.
        store.set_meta('report_progress',None)
        for key in ('conference_cost_history','conference_timings'):
            history=store.get_meta(key,{})
            store.set_meta(key,{name:value for name,value in history.items()
                                if not (name in ('report','bootstrap') or name.startswith(('report:','report_stage:','bootstrap:')))})
    record = dict(session_id=store.session_id, at=datetime.now(timezone.utc).isoformat(),
                  code_hash=code_hash, scientific_fingerprint=store.fingerprint,
                  starting_result_count=store.db.execute('SELECT COUNT(*) FROM results').fetchone()[0],
                  starting_confirmed_units=store.db.execute('SELECT COALESCE(SUM(completed_units),0) FROM jobs').fetchone()[0],
                  declared_energy_pool=dict(workers=settings['energy_cpu_workers'], max_pending=settings['energy_cpu_max_pending']),
                  effective_energy_pool=options, patch=patch,
                  timing_policy='per-session scheduling recorded; changed CPU correction rates recalibrated')
    if reporting is not None:
        record['effective_report_execution']=dict(reporting)
        record['timing_policy']+='; changed reporting/bootstrap rates recalibrated separately'
    atomic_json(store.path / 'runtime_history' / (store.session_id + '.json'), record)
    store.set_meta('runtime_execution_history', previous + [record])
    store.event('runtime_execution', **record)
