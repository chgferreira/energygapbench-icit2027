# Resume and recovery

The supported execution backend is local, POSIX, and single-coordinator, with
SQLite WAL, `synchronous=FULL`, and `BEGIN IMMEDIATE` transactions. Tests cover local
APFS/ext4-style semantics; NFS, multiple machines, changed world sizes, and distributed
coordination are unsupported. Recovery does not protect against loss of the disk.
Both [experiment profiles](experiment_profiles.md) use the same durable protocol.

## State and compatibility

`runs/ID/state.sqlite` is authoritative. Protocol locks, `plan.json`, manifests,
`progress.json`, events, checkpoints, and immutable shards support auditing. Persisted
timestamps are UTC; presentation can use the configured timezone. Sessions/attempts
are separate from scientific observations. A truncated last JSONL line does not
replace database truth. Status uses a read path without loading model weights.

Generic `--resume auto` creates or resumes compatible state; `require` needs existing
compatible state; `never` refuses a nonempty destination. The supported shell uses
`run` for a new ID and `resume ID` for continuation. It never deletes prior runs.
Scientific identity includes supplied data/source/configuration/environment/model
hashes and numerical policies. Precision, batching, HPO, and panel choices are
scientific settings. Presentation controls are excluded. The runner supplies these
identities; the ledger cannot infer which unreported files an adapter consumed.

The saved profile is retained on resume. `--preset` and `--config` overrides are
rejected. A changed method matrix, mask policy, HPO index/budget, epoch-panel seed/
cap/hash, or numerical implementation needs a new ID. A new ID does not erase prior
test exposure. Documentation-only edits do not change the Python code hash.

A narrowly audited [CPU worker patch](runtime_worker_tuning.md) is an explicit
exception for one exact source/target release pair. It retains the original
scientific/RNG identity and committed work, checks all other fields, holds the
coordinator lock, backs up SQLite, and records executable/session lineage. It does
not authorize arbitrary changed-code continuation.

## Ownership and publication

An experiment lock and per-job process lock/generation establish ownership.
Heartbeats are diagnostic; age/PID alone does not prove death. OS locks release
when a process exits. A stale generation cannot commit work after replacement.
The shell/native checks also use a project lock to avoid competing coordinators.

Publication writes on the same filesystem, flushes/fsyncs, verifies bytes/checksum,
renames atomically, and syncs directories when supported. Immutable JSON shards
identify their scientific keys, generation, and cursor before a database commit.
Duplicate keys with different results are compatibility errors, not blind appends.
A valid orphan can be reconciled only after identity/generation/dependency checks.
Missing/corrupt committed artifacts invalidate affected work and dependents while
preserving independent work. Model artifacts and checksum sidecars are checked.

`CheckpointStore` uses generation manifests/checksums and a `latest.json` pointer,
retaining earlier valid generations. A corrupt recent generation falls back with
an explicit recomputation reason. Generic pickle artifacts are trusted local state,
not a format for loading arbitrary third-party data. Native training serialization
stores model/optimizer, scheduler/scaler when applicable, Python/NumPy/torch CPU/CUDA
RNG, sampler, step/epoch, early stopping, and best state as required by each loop.
A utility's existence alone does not make an external library's `fit()` resumable.

## Signals, budgets, and conference training

SIGINT/SIGTERM request a safe stop; they do not write scientific state in the signal
handler. A current indivisible operation reaches a supported boundary before exit.
SIGKILL can require replay after the last valid commit/checkpoint. Stable IDs/seeds
preserve inference and bootstrap replicas; invalid bootstrap attempts stay invalid.

The budget is cumulative. Work reservations are replaced by measured duration on
normal completion; a crash can leave a conservative charged reservation labeled
uncertain. Without an enforced unit timeout it is not a proven upper bound on lost
compute. Offline time does not contribute to throughput. An explicit larger
cumulative ceiling can continue the same settings; it does not add trials/epochs
or change scope. Setup and initial planning are outside the timed ledger.

Conference BRITS/SAITS/CSDI/ImputeFormer/T1 loops persist completed-epoch weights,
optimizer, RNG, history, early stopping, and best validation-selected state. Full
selects on all core validation; focused selects on the frozen panel. Its identity
and hash are checked. A partial epoch repeats from a completed checkpoint. Frozen
models resume through committed inference shards rather than local training.
CPU interrupted/resume tests exist for all five local conference architectures;
CUDA parity/recovery remains target-server evidence, not a local certification.

## Diagnostic recovery paths

Historical standalone diagnostic adapters differ from conference loops. The named
`saits_protocol_training_loop` uses the upstream SAITS architecture/native ORT+MIT
loss in a controlled deterministic CPU loop. Real SIGKILL/SIGTERM tests compare
model, Adam, RNG, sampler/next batch, dynamic masks, trial, early stopping, and best
state. Its constant learning rate has no scheduler, and it is not the scientific
validation/long-run convergence experiment. The historical `PyPOTSAdapter.fit_windows`
remains indivisible within a native fit.

```bash
PYTHONPATH=src python -m pytest tests/recovery -q
PYTHONPATH=src python -m energygap.runtime.demo --output /tmp/energygap-recovery-demo
PYTHONPATH=src python -m energygap.runtime.saits_demo --output /tmp/energygap-saits-demo
```

Use new output directories and the dependencies required by each demo. Small fixtures
validate their loops, not arbitrary recovery of every catalogue method or context-512
scientific convergence. Reports retain repeated work, measured overhead, reservations,
and limitations. Follow the [server guide](../SERVER_GUIDE.md) and
[TODO roadmap](development_status.md) for current-profile execution and missing tests.
