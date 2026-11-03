"""Local, durable experiment ledger. SQLite WAL + POSIX locks; not a distributed backend."""
from __future__ import annotations
import contextlib
import fcntl
import hashlib
import json
import logging
import os
import sqlite3
import stat
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

LOG = logging.getLogger(__name__)

TERMINAL = {'succeeded', 'failed_terminal', 'unavailable', 'unsupported', 'blocked'}
PRESENTATION = {'progress', 'output_dir', 'state_dir', 'checkpoint_every_steps',
                'checkpoint_every_seconds', 'checkpoint_optimizer_steps',
                'checkpoint_active_seconds', 'progress_refresh_seconds',
                'progress_persist_seconds', 'status_snapshot_seconds', 'heartbeat_seconds',
                'resume', 'budget_total_hours',
                'budget_seconds', 'timezone', 'retry_failed'}

def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()

def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def scientific_config(config: dict) -> dict:
    """Remove operational presentation fields, preserving all numeric algorithm choices."""
    return {k: scientific_config(v) if isinstance(v, dict) else v
            for k, v in config.items() if k not in PRESENTATION}

def fingerprint(config: dict) -> str:
    return hashlib.sha256(canonical(scientific_config(config))).hexdigest()

def file_sha256(path: Path) -> str:
    """Bounded-memory validation also for multi-GB report artifacts."""
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _file_identity(info: os.stat_result) -> tuple:
    """Identify the version whose bytes were checked during this one pass."""
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns, info.st_mode)


def _read_stable_bytes(path: Path) -> tuple[bytes, tuple]:
    """Read a shard once, refusing changes observed while reading it."""
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise OSError('artifact is not a regular file')
    data = path.read_bytes()
    identity = _file_identity(before)
    if _file_identity(path.stat()) != identity:
        raise OSError('artifact changed while being read')
    return data, identity


def _checked_file_digest(path: Path, cache: dict | None = None) -> str | None:
    """Hash bytes before reusing a digest for the same unchanged file version.

    The caller owns a new cache for each reconciliation. Metadata is only a
    guard against reuse after modification; it never replaces the first SHA-256
    calculation, and no metadata/digest cache survives the pass.
    """
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        return None
    identity = _file_identity(before)
    key = (str(path), identity)
    if cache is not None and key in cache:
        return cache[key]
    digest = file_sha256(path)
    if _file_identity(path.stat()) != identity:
        return None
    if cache is not None:
        cache[key] = digest
    return digest

def stable_seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256(canonical(parts)).digest()[:8], 'big') % (2**32)

def fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    except OSError:
        # Some filesystems do not expose directory fsync. Local APFS/ext4 are supported.
        pass

def atomic_bytes(path: Path, data: bytes, fault: Callable | None = None) -> str:
    """Publish same-filesystem durable bytes. A crash never makes a partial destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as stream:
        stream.write(data)
        if fault: fault('during_write')
        stream.flush(); os.fsync(stream.fileno())
    if temporary.read_bytes() != data:
        raise OSError('atomic write validation failed')
    os.replace(temporary, path)
    fsync_directory(path.parent)
    if fault: fault('after_rename')
    return hashlib.sha256(data).hexdigest()

def atomic_json(path: Path, value: Any, fault: Callable | None = None) -> str:
    return atomic_bytes(path, canonical(value), fault)

class RuntimeErrorBase(RuntimeError): pass
class IncompatibleResume(RuntimeErrorBase): pass
class LeaseLost(RuntimeErrorBase): pass
class JobBusy(RuntimeErrorBase): pass
class BudgetExhausted(RuntimeErrorBase): pass

@dataclass
class JobLease:
    store: 'StateStore'
    job_id: str
    token: int
    owner: str
    lock_file: Any
    released: bool = False

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb):
        if exc_type:
            try: self.store.interrupt(self.job_id, self.token, str(exc))
            except LeaseLost: pass
        self.release()

    def release(self):
        if not self.released:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
            self.lock_file.close(); self.released = True

class StateStore:
    """Single-machine ledger. Each writer transaction is serialized by SQLite.

    ``claim`` holds a POSIX per-job lock until its JobLease is released. Recovery
    never infers process death from a PID or heartbeat. A generation fences commits.
    """
    def __init__(self, run_dir: str | Path, config: dict | None = None,
                 resume: str = 'auto', budget_seconds: float | None = None, *,reconcile_on_open=True):
        self._defer_snapshots=0
        self._parent_budget_reservations={}
        self._progress_ignored_changes=0
        self._progress_invalidations=0
        self.snapshot_interval_seconds=0.0
        self._last_snapshot_monotonic=0.0
        if resume not in {'auto', 'never', 'require'}: raise ValueError('invalid resume mode')
        self.path = Path(run_dir)
        exists = (self.path / 'state.sqlite').exists()
        if resume == 'never' and self.path.exists() and any(self.path.iterdir()):
            raise FileExistsError('--resume never cannot overwrite existing output')
        if resume == 'require' and not exists: raise FileNotFoundError('resume requires existing state')
        self.path.mkdir(parents=True, exist_ok=True)
        for folder in ('shards', 'checkpoints', 'locks'): (self.path / folder).mkdir(exist_ok=True)
        self.db = sqlite3.connect(self.path / 'state.sqlite', timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs(
          job_id TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'pending',
          total_units INTEGER NOT NULL, completed_units INTEGER NOT NULL DEFAULT 0,
          generation INTEGER NOT NULL DEFAULT 0, owner TEXT, heartbeat TEXT,
          attempts INTEGER NOT NULL DEFAULT 0, dependencies TEXT NOT NULL,
          metadata TEXT NOT NULL, reason TEXT, active_seconds REAL NOT NULL DEFAULT 0,
          created_at TEXT NOT NULL, started_at TEXT, ended_at TEXT);
        CREATE TABLE IF NOT EXISTS shards(
          shard_id TEXT PRIMARY KEY, job_id TEXT NOT NULL, path TEXT NOT NULL,
          checksum TEXT NOT NULL, start_unit INTEGER NOT NULL, end_unit INTEGER NOT NULL,
          generation INTEGER NOT NULL, committed_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS invalidated_artifacts(checksum TEXT PRIMARY KEY, reason TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS results(
          result_key TEXT PRIMARY KEY, job_id TEXT NOT NULL, payload TEXT NOT NULL,
          shard_id TEXT NOT NULL REFERENCES shards(shard_id));
        CREATE TABLE IF NOT EXISTS sessions(
          session_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT,
          active_seconds REAL NOT NULL DEFAULT 0, recovery_uncertainty TEXT);
        CREATE TABLE IF NOT EXISTS work_reservations(
          reservation_id TEXT PRIMARY KEY, session_id TEXT, job_id TEXT NOT NULL,
          reserved_seconds REAL NOT NULL, actual_seconds REAL, created_at TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'open');
        CREATE TABLE IF NOT EXISTS checkpoints(
          path TEXT PRIMARY KEY, job_id TEXT NOT NULL, step INTEGER NOT NULL,
          checksum TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events(
          event_id INTEGER PRIMARY KEY, at TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS results_job_id ON results(job_id);
        CREATE INDEX IF NOT EXISTS shards_job_id ON shards(job_id);
        ''')
        if not exists:
            config = config or {}
            self.set_meta('fingerprint', fingerprint(config))
            self.set_meta('config', config)
            self.set_meta('created_at', utcnow())
            self.set_meta('first_started_at', None)
            self.set_meta('status', 'planned')
            self.set_meta('active_seconds', 0.0)
            self.set_meta('budget_seconds', budget_seconds)
            self.set_meta('resume_count', 0)
        elif config is not None and fingerprint(config) != self.get_meta('fingerprint'):
            self.db.close()
            raise IncompatibleResume('scientific fingerprint differs: data/code/split/mask/model/config/seed must match')
        self.fingerprint = self.get_meta('fingerprint')
        self.session_id: str | None = None
        self.experiment_lock = None
        if reconcile_on_open:self.reconcile()

    def close(self):
        if self.experiment_lock:
            fcntl.flock(self.experiment_lock.fileno(), fcntl.LOCK_UN)
            self.experiment_lock.close(); self.experiment_lock = None
        self.db.close()
    def __enter__(self): return self
    def __exit__(self, *_): self.close()
    @contextlib.contextmanager
    def transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            self._progress_invalidations += 1
            raise

    def set_meta(self, key: str, value: Any):
        self.db.execute('INSERT INTO meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                        (key, json.dumps(value, allow_nan=False)))
        # Metadata and heartbeats do not change job counts or remaining work.
        # Keep their writes out of the progress cache revision without adding a
        # database trigger or altering the durable experiment schema.
        self._progress_ignored_changes += 1

    @property
    def progress_revision(self) -> int:
        """Local writer revision for workload/count caches, including direct SQL.

        SQLite total_changes also advances on rolled-back writes. A rollback
        invalidates estimates that may have read uncommitted work. The supported
        coordinator is the sole writer; status readers use other connections.
        """
        return self.db.total_changes - self._progress_ignored_changes + self._progress_invalidations
    def get_meta(self, key: str, default=None):
        row = self.db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def event(self, kind: str, **payload):
        now = utcnow()
        self.db.execute('INSERT INTO events(at,kind,payload) VALUES(?,?,?)', (now, kind, json.dumps(payload)))
        # JSONL is a convenience export; SQLite remains authoritative after a torn tail.
        with (self.path / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps({'at': now, 'kind': kind, **payload}) + '\n'); stream.flush()

    def start_session(self):
        if self.get_meta('status') in {'completed', 'completed_with_failures'}: return None
        if self.experiment_lock is None:
            self.experiment_lock = (self.path/'locks'/'experiment.lock').open('a+')
            try: fcntl.flock(self.experiment_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self.experiment_lock.close(); self.experiment_lock = None
                raise JobBusy('another coordinator owns the experiment')
        self.session_id = uuid.uuid4().hex
        now = utcnow()
        with self.transaction():
            old = self.db.execute('SELECT session_id FROM sessions WHERE ended_at IS NULL').fetchall()
            for row in old:
                self.db.execute('UPDATE sessions SET ended_at=?, recovery_uncertainty=? WHERE session_id=?',
                                (now, 'active time after last durable accounting unknown; offline time excluded', row[0]))
            self.db.execute("UPDATE work_reservations SET status='crash_uncertain_charged' WHERE status='open'")
            count = self.db.execute('SELECT COUNT(*) FROM sessions').fetchone()[0]
            self.set_meta('resume_count', count)
            self.set_meta('first_started_at', self.get_meta('first_started_at') or now)
            self.set_meta('status', 'running')
            self.db.execute('INSERT INTO sessions(session_id,started_at) VALUES(?,?)', (self.session_id, now))
        self.event('session_started', session_id=self.session_id, resume_count=count)
        self.snapshot()
        return self.session_id

    def end_session(self, status: str | None = None):
        if self.session_id:
            self.db.execute('UPDATE sessions SET ended_at=? WHERE session_id=?', (utcnow(), self.session_id))
        if status: self.set_meta('status', status)
        self.snapshot()

    def begin_work(self, job_id: str, token: int, reserve_seconds: float = 60.0) -> str:
        """Charge a conservative work quantum before computing an indivisible unit.

        A crash retains the reservation; successful completion replaces it by measured
        active duration. Choose reserve >= enforced unit timeout for a true upper bound.
        Without such timeout, crash-time uncertainty remains explicitly unbounded.
        """
        self._verify(job_id, token); self.check_budget()
        if reserve_seconds <= 0: raise ValueError('reservation must be positive')
        identity = uuid.uuid4().hex
        with self.transaction():
            self.db.execute('INSERT INTO work_reservations(reservation_id,session_id,job_id,reserved_seconds,created_at) VALUES(?,?,?,?,?)',
                            (identity,self.session_id,job_id,reserve_seconds,utcnow()))
            self.set_meta('active_seconds',self.get_meta('active_seconds',0.0)+reserve_seconds)
            self.db.execute('UPDATE jobs SET active_seconds=active_seconds+? WHERE job_id=?',(reserve_seconds,job_id))
            if self.session_id:
                self.db.execute('UPDATE sessions SET active_seconds=active_seconds+? WHERE session_id=?',(reserve_seconds,self.session_id))
        return identity

    def finish_work(self, reservation_id: str, actual_seconds: float):
        if actual_seconds < 0: raise ValueError('negative active duration')
        with self.transaction():
            row=self.db.execute('SELECT * FROM work_reservations WHERE reservation_id=?',(reservation_id,)).fetchone()
            if not row or row['status']!='open': raise RuntimeErrorBase('work reservation not open')
            delta=actual_seconds-row['reserved_seconds']
            self.db.execute("UPDATE work_reservations SET status='measured',actual_seconds=? WHERE reservation_id=?",(actual_seconds,reservation_id))
            self.set_meta('active_seconds',self.get_meta('active_seconds',0.0)+delta)
            self.db.execute('UPDATE jobs SET active_seconds=active_seconds+? WHERE job_id=?',(delta,row['job_id']))
            if row['session_id']:
                self.db.execute('UPDATE sessions SET active_seconds=active_seconds+? WHERE session_id=?',(delta,row['session_id']))

    def extend_budget(self, seconds: float):
        old = self.get_meta('budget_seconds')
        if old is not None and seconds < old: raise ValueError('budget extension cannot reduce ceiling')
        self.set_meta('budget_seconds', seconds)
        self.event('budget_extended', previous_seconds=old, new_seconds=seconds)

    def record_active(self, seconds: float, job_id: str | None = None):
        if seconds < 0: raise ValueError('active duration cannot be negative')
        with self.transaction():
            self.set_meta('active_seconds', self.get_meta('active_seconds', 0.0) + seconds)
            if job_id: self.db.execute('UPDATE jobs SET active_seconds=active_seconds+? WHERE job_id=?', (seconds, job_id))
            if self.session_id: self.db.execute('UPDATE sessions SET active_seconds=active_seconds+? WHERE session_id=?', (seconds, self.session_id))

    def check_budget(self):
        total = self.get_meta('budget_seconds')
        charged = self.get_meta('active_seconds', 0) - sum(self._parent_budget_reservations.values())
        if total is not None and charged >= total:
            self.set_meta('status', 'paused_budget'); self.snapshot()
            raise BudgetExhausted('cumulative budget exhausted; extend operational ceiling explicitly')

    @contextlib.contextmanager
    def child_work_budget(self, parent_reservation: str):
        """Do not charge a live parent's overhead reserve again to its children.

        The reservation remains durable for crash accounting. Child work keeps
        its own measured charges; the caller measures the parent's exclusive
        overhead when finishing the reservation. This process-local exclusion
        never survives recovery or discounts a prior uncertain crash charge.
        """
        row=self.db.execute('SELECT status,session_id,reserved_seconds FROM work_reservations WHERE reservation_id=?',
                            (parent_reservation,)).fetchone()
        if not row or row['status']!='open' or row['session_id']!=self.session_id:
            raise RuntimeErrorBase('child budget requires an open parent reservation in this session')
        previous=self._parent_budget_reservations.get(parent_reservation)
        self._parent_budget_reservations[parent_reservation]=row['reserved_seconds']
        try:
            yield
        finally:
            if previous is None:self._parent_budget_reservations.pop(parent_reservation,None)
            else:self._parent_budget_reservations[parent_reservation]=previous

    def add_job(self, job_id: str, total_units: int, dependencies: Iterable[str] = (), metadata: dict | None = None):
        if total_units < 0: raise ValueError('negative work')
        deps, meta = json.dumps(list(dependencies)), json.dumps(metadata or {}, sort_keys=True)
        existing = self.db.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone()
        if existing:
            if (existing['total_units'], existing['dependencies'], existing['metadata']) != (total_units, deps, meta):
                raise IncompatibleResume('logical job changed without new ID')
            return
        self.db.execute('INSERT INTO jobs(job_id,total_units,dependencies,metadata,created_at) VALUES(?,?,?,?,?)',
                        (job_id, total_units, deps, meta, utcnow()))
        self.snapshot()

    def job(self, job_id: str) -> dict:
        row = self.db.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone()
        if not row: raise KeyError(job_id)
        return self._decode_job(row)

    @staticmethod
    def _decode_job(row) -> dict:
        result = dict(row)
        result['metadata'] = json.loads(result['metadata']); result['dependencies'] = json.loads(result['dependencies'])
        return result

    def claim(self, job_id: str, owner: str | None = None) -> JobLease:
        self.check_budget()
        lock_path = self.path / 'locks' / (hashlib.sha256(job_id.encode()).hexdigest() + '.lock')
        lock_file = lock_path.open('a+')
        try: fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_file.close(); raise JobBusy(job_id)
        owner = owner or uuid.uuid4().hex
        try:
            with self.transaction():
                job = self.job(job_id)
                if job['status'] in TERMINAL: raise RuntimeErrorBase('job already resolved: ' + job['status'])
                for dep in job['dependencies']:
                    parent = self.job(dep)
                    if parent['status'] != 'succeeded': raise RuntimeErrorBase('dependency not successful: ' + dep)
                token = job['generation'] + 1
                self.db.execute("UPDATE jobs SET status='running',generation=?,owner=?,heartbeat=?,attempts=attempts+1,started_at=COALESCE(started_at,?) WHERE job_id=?",
                                (token, owner, utcnow(), utcnow(), job_id))
            self.event('job_claimed', job_id=job_id, generation=token, owner=owner)
            return JobLease(self, job_id, token, owner, lock_file)
        except BaseException:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN); lock_file.close(); raise

    def _verify(self, job_id: str, token: int):
        job = self.job(job_id)
        if job['generation'] != token or job['status'] != 'running': raise LeaseLost('stale job generation')
        return job

    def heartbeat(self, job_id: str, token: int):
        self._verify(job_id, token)
        self.db.execute('UPDATE jobs SET heartbeat=? WHERE job_id=? AND generation=?', (utcnow(), job_id, token))
        self._progress_ignored_changes += 1

    def publish(self, job_id: str, token: int, records: list[dict], completed_units: int,
                fault: Callable | None = None) -> Path:
        job = self._verify(job_id, token)
        if completed_units < job['completed_units'] or completed_units > job['total_units']:
            raise ValueError('nonmonotonic/out-of-range confirmed units')
        if len({str(r['key']) for r in records}) != len(records): raise ValueError('duplicate result keys in shard')
        payload = {'schema_version': 1, 'fingerprint': self.fingerprint, 'job_id': job_id,
                   'generation': token, 'start_unit': job['completed_units'], 'end_unit': completed_units,
                   'records': records}
        data = canonical(payload); checksum = hashlib.sha256(data).hexdigest()
        path = self.path / 'shards' / (checksum + '.json')
        atomic_bytes(path, data, fault)
        if fault: fault('before_commit')
        with self.transaction():
            self._verify(job_id, token)
            self._incorporate(path, payload, checksum)
        if fault: fault('after_commit')
        self.snapshot()
        return path

    def _incorporate(self, path: Path, payload: dict, checksum: str):
        job = self.job(payload['job_id'])
        if payload['start_unit'] > job['completed_units']: raise ValueError('shard has gap in work cursor')
        shard_id = checksum
        self.db.execute('INSERT OR IGNORE INTO shards VALUES(?,?,?,?,?,?,?,?)',
                        (shard_id, payload['job_id'], str(path.relative_to(self.path)), checksum,
                         payload['start_unit'], payload['end_unit'], payload['generation'], utcnow()))
        for record in payload['records']:
            key, encoded = str(record['key']), canonical(record).decode()
            old = self.db.execute('SELECT payload,job_id FROM results WHERE result_key=?', (key,)).fetchone()
            if old and (old[0] != encoded or old[1] != payload['job_id']):
                raise IncompatibleResume('result key collision with different scientific value: ' + key)
            self.db.execute('INSERT OR IGNORE INTO results VALUES(?,?,?,?)', (key, payload['job_id'], encoded, shard_id))
        self.db.execute('UPDATE jobs SET completed_units=MAX(completed_units,?),heartbeat=? WHERE job_id=?',
                        (payload['end_unit'], utcnow(), payload['job_id']))

    def finish(self, job_id: str, token: int):
        with self.transaction():
            job = self._verify(job_id, token)
            if job['completed_units'] != job['total_units']: raise RuntimeErrorBase('cannot succeed with incomplete work')
            self.db.execute("UPDATE jobs SET status='succeeded',ended_at=? WHERE job_id=?", (utcnow(), job_id))
        self.event('job_succeeded', job_id=job_id)
        self.snapshot()

    def interrupt(self, job_id: str, token: int, reason: str = 'stop requested'):
        with self.transaction():
            self._verify(job_id, token)
            self.db.execute("UPDATE jobs SET status='interrupted',reason=? WHERE job_id=?", (reason, job_id))
        self.snapshot()

    def mark(self, job_id: str, status: str, reason: str):
        if status not in TERMINAL | {'failed_retryable', 'interrupted', 'pending'}: raise ValueError(status)
        self.db.execute('UPDATE jobs SET status=?,reason=?,ended_at=? WHERE job_id=?', (status, reason, utcnow(), job_id))
        self.snapshot()

    def retry_failed(self, max_retries: int = 2) -> int:
        cursor = self.db.execute("UPDATE jobs SET status='pending' WHERE status='failed_retryable' AND attempts<=?", (max_retries,))
        self.snapshot(); return cursor.rowcount

    def results(self, job_id: str | None = None) -> list[dict]:
        rows = self.db.execute('SELECT payload FROM results' + (' WHERE job_id=?' if job_id else '') + ' ORDER BY result_key',
                               (job_id,) if job_id else ()).fetchall()
        return [json.loads(row[0]) for row in rows]

    def _referenced_artifacts_valid(self, payload: dict, *, checksum_cache=None) -> bool:
        if not isinstance(payload,dict) or not isinstance(payload.get('records',[]),list):
            return False
        for record in payload.get('records',[]):
            if not isinstance(record,dict):return False
            for path_key,digest_key in (('artifact','sha256'),('artifact_sidecar','artifact_sidecar_sha256')):
                if path_key not in record or digest_key not in record:continue
                try:
                    relative=Path(record[path_key])
                    if relative.is_absolute() or '..' in relative.parts:return False
                    expected=record[digest_key]
                    if not isinstance(expected,str) or _checked_file_digest(self.path/relative,checksum_cache)!=expected:return False
                except (OSError,TypeError,ValueError):return False
        return True

    def _invalidate_job(self, job_id: str, reason: str, start_unit: int = 0):
        """Invalidate affected outputs and downstream jobs, retaining unrelated results."""
        rows=self.db.execute('SELECT shard_id,checksum FROM shards WHERE job_id=? AND end_unit>?',(job_id,start_unit)).fetchall()
        for row in rows:
            self.db.execute('INSERT OR REPLACE INTO invalidated_artifacts VALUES(?,?)',(row['checksum'],reason))
            self.db.execute('DELETE FROM results WHERE shard_id=?',(row['shard_id'],))
            self.db.execute('DELETE FROM shards WHERE shard_id=?',(row['shard_id'],))
        self.db.execute("UPDATE jobs SET status='interrupted',completed_units=?,reason=? WHERE job_id=?",(start_unit,reason,job_id))
        for row in self.db.execute('SELECT job_id,dependencies FROM jobs').fetchall():
            if job_id in json.loads(row['dependencies']): self._invalidate_job(row['job_id'],'dependency artifact invalid: '+job_id,0)
        self.set_meta('status','interrupted')

    def reconcile(self) -> dict:
        """Verify durable outputs and recover orphans without redundant reads.

        Every committed shard is read and SHA-256 checked. Shared referenced
        artifacts are streamed and hashed once per unchanged file version in
        this pass. Already checked committed files need not be read again while
        searching for orphans. No persistent integrity shortcut is introduced.
        """
        invalid, recovered = [], []
        checksum_cache = {}
        checked_commits = {}
        started = last_log = time.monotonic()
        rows = self.db.execute('SELECT * FROM shards ORDER BY start_unit').fetchall()
        def progress(stage, completed, total, *, force=False):
            nonlocal last_log
            now=time.monotonic()
            if force or now-last_log>=5.:
                LOG.info('Recovery: %s %d/%d; %d unique artifact versions hashed; elapsed %.1fs',
                         stage,completed,total,len(checksum_cache),now-started)
                last_log=now
        progress('committed shards checked',0,len(rows),force=True)
        for index,row in enumerate(rows,1):
            if not self.db.execute('SELECT 1 FROM shards WHERE shard_id=?',(row['shard_id'],)).fetchone(): continue
            path = self.path / row['path']
            try:
                data,identity=_read_stable_bytes(path)
                valid=hashlib.sha256(data).hexdigest()==row['checksum']
                if valid:valid=self._referenced_artifacts_valid(json.loads(data),checksum_cache=checksum_cache)
                if valid:checked_commits[path]=(row['checksum'],identity)
            except (ValueError,OSError):valid=False
            if not valid:
                with self.transaction():
                    self._invalidate_job(row['job_id'],'committed artifact invalid; localized recovery',row['start_unit'])
                invalid.append(row['path'])
            progress('committed shards checked',index,len(rows))
        progress('committed shards checked',len(rows),len(rows),force=True)
        known = {r[0] for r in self.db.execute('SELECT checksum FROM shards')}
        rejected={r[0] for r in self.db.execute('SELECT checksum FROM invalidated_artifacts')}
        candidates = []
        paths=list((self.path / 'shards').glob('*.json'))
        progress('shard files checked for orphan recovery',0,len(paths),force=True)
        for index,path in enumerate(paths,1):
            try:
                checked=checked_commits.get(path)
                if checked and checked[0] in known and _file_identity(path.stat())==checked[1]:continue
                # An admissible orphan filename must equal its actual digest;
                # a known rejected digest therefore cannot be recovered here.
                if path.stem in rejected:continue
                data,_ = _read_stable_bytes(path); checksum = hashlib.sha256(data).hexdigest()
                payload = json.loads(data)
                if checksum in known or checksum in rejected or path.stem != checksum: continue
                if payload['schema_version'] != 1 or payload['fingerprint'] != self.fingerprint: continue
                job=self.job(payload['job_id'])
                if payload['generation'] != job['generation']: continue
                if not self._referenced_artifacts_valid(payload,checksum_cache=checksum_cache): continue
                if any(self.job(dep)['status']!='succeeded' for dep in job['dependencies']): continue
                candidates.append((payload['start_unit'], path, payload, checksum))
            except (ValueError, KeyError, OSError, TypeError): continue
            finally:progress('shard files checked for orphan recovery',index,len(paths))
        for _, path, payload, checksum in sorted(candidates, key=lambda item: item[0]):
            try:
                with self.transaction(): self._incorporate(path, payload, checksum)
                recovered.append(path.name)
            except (ValueError, IncompatibleResume): continue
        if invalid or recovered: self.event('reconciliation', invalid=invalid, recovered=recovered)
        progress('shard files checked for orphan recovery',len(paths),len(paths),force=True)
        LOG.info('Recovery complete: %d invalidated shards, %d recovered shards; elapsed %.1fs',
                 len(invalid),len(recovered),time.monotonic()-started)
        return {'invalid': invalid, 'recovered': recovered}

    def complete(self):
        jobs = [dict(row) for row in self.db.execute('SELECT * FROM jobs')]
        if any(j['status'] not in TERMINAL for j in jobs): return False
        status = 'completed' if all(j['status'] == 'succeeded' for j in jobs) else 'completed_with_failures'
        self.set_meta('status', status); self.set_meta('ended_at', utcnow()); self.end_session(status)
        return True

    def status(self) -> dict:
        jobs = [self._decode_job(row) for row in self.db.execute('SELECT * FROM jobs ORDER BY job_id')]
        counts = {s: sum(j['status'] == s for j in jobs) for s in sorted(TERMINAL | {'pending','running','interrupted','failed_retryable'})}
        return {'schema_version': 1, 'experiment_id': self.path.name, 'fingerprint': self.get_meta('fingerprint'),
                'status': self.get_meta('status'), 'created_at': self.get_meta('created_at'),
                'first_started_at': self.get_meta('first_started_at'), 'ended_at': self.get_meta('ended_at'),
                'resume_count': self.get_meta('resume_count', 0),
                'active_seconds': self.get_meta('active_seconds', 0), 'budget_seconds': self.get_meta('budget_seconds'),
                'confirmed_units': sum(j['completed_units'] for j in jobs),
                'total_units': sum(j['total_units'] for j in jobs),
                'resolved_jobs': sum(j['status'] in TERMINAL for j in jobs), 'total_jobs': len(jobs),
                'counts': counts, 'jobs': jobs,
                'result_count': self.db.execute('SELECT COUNT(*) FROM results').fetchone()[0],
                'last_checkpoint': self.get_meta('last_valid_checkpoint') or dict(self.db.execute('SELECT * FROM checkpoints ORDER BY created_at DESC LIMIT 1').fetchone() or {}),
                'sessions': [dict(r) for r in self.db.execute('SELECT * FROM sessions ORDER BY started_at')],
                'crash_time_reservations': [dict(r) for r in self.db.execute("SELECT * FROM work_reservations WHERE status!='measured'")],
                'active_accounting': 'measured work plus conservatively retained unfinished reservations; unbounded uncertainty for units without enforced timeout',
                'current_training':self.get_meta('current_training'),
                'report_progress':self.get_meta('report_progress'),
                'eta': self.get_meta('eta', {'remaining_seconds': None, 'estimated_finish': None, 'calibration': 'unavailable'})}

    @contextlib.contextmanager
    def defer_snapshots(self):
        """Batch planning writes; SQLite remains authoritative throughout."""
        self._defer_snapshots+=1
        try:yield
        finally:
            self._defer_snapshots-=1
            if not self._defer_snapshots:self.snapshot(force=True)

    def snapshot(self, force=False):
        now=time.monotonic()
        if self._defer_snapshots:return
        if not force and now-self._last_snapshot_monotonic<self.snapshot_interval_seconds:return
        atomic_json(self.path / 'progress.json', self.status())
        self._last_snapshot_monotonic=now
