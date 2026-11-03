# CPU worker tuning with audited continuation

The current [reporting performance update](reporting_performance.md) also accepts
the explicitly audited predecessors of this CPU-worker release. The historical
CPU-worker manifest remains unchanged; use the reporting patch for the current
target source and its additional report controls.

**Work in progress.** Spare RAM/VRAM is capacity, not a measure of useful throughput.
The native model can finish a short GPU batch while CPU correction, scoring,
forecasting, and durable publication continue. Increasing memory allocation alone
does not accelerate those phases. The current V100 path stays native float32.

## Supported change

`--energy-cpu-workers N` and `--energy-cpu-max-pending N` control CPU correction
scheduling for a conference session. They do not change training/inference batches,
models, loss, optimizer, precision, masks, seeds, panel, HPO, equations, correction
grid, sample count, or evaluation support. Results remain ordered and the coordinator
alone scores/commits them. Workers retain one BLAS thread each and never run models.

The YAML's declared worker values remain immutable. Effective session values are
recorded separately in `runtime_history/`, the ledger's `runtime_execution_history`,
and `reports/summary.json`. Existing and new corrected rows retain their actual
worker metadata. Correction/test phase rates are recalibrated when pool settings
change; model-fitting/validation rates are retained. Omit overrides to use the YAML;
repeat chosen runtime flags on later resumes.

The [reference host](reference_hardware.md) has two Xeon Gold 6130 processors,
32 physical cores, 64 logical processors and 386 GB RAM. With the existing
main-process request of 32 threads, the recorded continuation used
16 correction workers and 32 pending tasks, compared with the original 8/16. This
can accelerate CPU-limited correction; it is not a measured twofold pipeline speedup.
A larger pool needs evidence from actual CPU/queue throughput and affinity. Memory
headroom does not justify allocating all cores or VRAM indiscriminately.

## Preserve the currently running release

The previous implementation rejects arbitrary changed-code resume. The explicit
`--allow-runtime-patch` exception accepts only the exact predecessor source hash
and exact audited target source files in
`configs/runtime_patches/cpu_worker_scaling_v1.json`. All other scientific/environment
fields must still match. The original configuration, fingerprint, RNG identity,
committed results, fitted models, masks/panels, and budget accounting are retained.
The new executable hash and manifest digest are recorded separately; this is not
an unrestricted hash override or a rewrite of historical provenance.

A POSIX lock refuses an active coordinator. Before the first continued computation,
a consistent SQLite backup with checksum is created under `runs/ID/runtime_history/`.
Shards/models stay in place; normal reconciliation verifies them. A partial unit
may repeat from its last durable boundary. Unknown source releases, changed model
batches/threads/settings, or mismatched target files are rejected rather than
silently starting another study. Other scientific/code changes still need a new
identity and appropriate evidence.

## Update and resume the same ID

1. Record the printed experiment ID and request a safe stop once (Ctrl+C). Wait
   until the Python coordinator exits. Do not replace source files in an active run.
2. Preserve the full `runs/ID/`, raw data, environment, and downloaded artifacts.
   The automatic database backup is additional protection, not a backup of the disk.
3. Extract the targeted runtime patch in the existing project root and verify it:

```bash
conda activate cuda_env
unzip -o /path/to/energygapbench-cpu-worker-patch.zip
sha256sum -c RUNTIME_WORKER_PATCH_MANIFEST.sha256
bash scripts/server.sh resume ID --threads 32 \
  --energy-cpu-workers 16 --energy-cpu-max-pending 32 --allow-runtime-patch
```

Replace `ID`; use `resume`, not `run`. No setup/reinstallation is needed for this
patch. The saved cumulative budget is retained; extend it only through an explicit
`--budget-hours TOTAL_HOURS` if desired. The full updated server ZIP also includes
this support; a targeted patch avoids overwriting unrelated experiment YAMLs.
An old `SERVER_FULL_MANIFEST.sha256` describes the earlier whole package; after a
targeted patch use its own manifest and the recorded source/target lineage.

A refusal to recognize the source/version is not a reason to edit protocol locks
or SQLite fingerprints. Preserve the run and inspect the saved source hash. A
previous executable snapshot remains necessary if reverting code; results completed
after a patch must not be represented as an unmodified historical executable run.

## Measure the actual bottleneck

The independent Linux sampler reads the coordinator and its child-process tree,
process CPU deltas, proportional memory (PSS), I/O counters, system memory, and
optional physical-GPU samples. It imports no neural library, runs no model, and
does not open/change the experiment database. It may run alongside the experiment:

```bash
python -u scripts/resource_telemetry.py --pid PYTHON_PID --gpu 0 --duration-seconds 60
```

Use the Python coordinator PID shown by `nvidia-smi`/process tools, not a shell or
`tee` PID. `--gpu` is a physical NVIDIA index/UUID. The report under
`artifacts/analyses/performance/` distinguishes cores used from memory occupancy.
100% process CPU corresponds to one core, while `cpu_cores_used` sums the measured
process-tree deltas. PSS apportions shared pages; summed RSS can double count them.
Periodic GPU sampling can miss short kernels and includes other processes.

Busy correction workers with idle CPU cores support testing a larger pool. A busy
single coordinator with idle workers indicates a different serial bottleneck.
High GPU activity suggests measuring deterministic inference batches in a separate
new configuration. Training batch changes alter optimization and are outside this
continuation exception. Never infer a speedup from memory occupancy alone.

## Validation and remaining evidence

An actual archived predecessor execution was paused and continued using the new
source with a different CPU pool on controlled CPU fixtures. The audit checked
unchanged committed payloads/models, original fingerprint and sample seeds, exact
scientific outputs against the predecessor continuation, and exact weights for a
future native SAITS fit. Unit/integration tests also exercise scientific-edit,
unknown-release, target-hash, and active-coordinator rejection.

These checks do not measure V100 speedup or certify GPU determinism. Target-server
CPU utilization, useful worker count, queue effects, complete-run timing, and CUDA
recovery remain empirical work in the [development roadmap](development_status.md).
