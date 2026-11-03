# Report execution, progress, and audited continuation

The reporting performance update preserves the declared study, masks, fitted
models, committed predictions, contrast definitions, and bootstrap resampling.
It changes CPU scheduling, export batching, integrity-check reuse within a pass,
and operational progress. It does not establish a server speedup or close the
scientific work-in-progress items.

## Work and controls

Reports stream forecast, energy, and episode exports, aggregate losses, perform
the paired day-block bootstrap and its declared block-length sensitivities, and
produce figures and an integrity manifest. The primary query counter describes
HPO scoring and model inference; it does not count bootstrap draws or exported
rows as independent scientific queries.

A temporary compact report source reads the original prediction-heavy ledger
once, retaining forecast values, scalar metrics, diagnostics, and the unchanged
fit/energy-track payloads. It removes trajectory arrays only from this derived
inference copy; those arrays remain intact in the authoritative ledger and its
shards. Subsequent exports and aggregation scan the smaller temporary source.
The temporary databases live under the run's reports directory and are removed
on normal exit or handled interruption; sufficient scratch disk is still needed.

The bootstrap uses bounded spawned CPU workers with one BLAS thread each. The
parent alone writes SQLite and durable result shards in logical replica order.
Each replica retains its original ID, stable seed, day draws, arithmetic, and
invalid-result status. Existing partial bootstrap jobs resume at their confirmed
unit; committed draws are retained rather than regenerated. Worker processes
do not initialize a CUDA model. One pool is reused within report execution.

Supported run/resume options are:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--report-workers` | At most 16, bounded by requested threads and available CPU affinity | Parallel CPU bootstrap workers; independent of correction workers |
| `--report-chunk-size` | 10,000 | Bounded rows per export batch, up to 100,000; not a sampling cap |
| `--report-bootstrap-chunk-size` | 1,000 | Ordered replicas per durable publication, up to 10,000 |

The existing `--energy-cpu-workers` and `--energy-cpu-max-pending` still control
the numerical correction stage. They do not select the report worker count.
The effective report options are recorded in runtime execution history rather
than inserted into the frozen scientific configuration. Changed report execution
settings invalidate old report/bootstrap timing calibration, not scientific draws.

Use enough workers for measured CPU work, not memory occupancy. Very small
bootstrap batches can cost more to dispatch than they save; storage-bound exports
do not become proportionally faster with more bootstrap workers. CUDA, model
precision, training batch size, and inference batch size are unchanged.

## Stopping and continuing an existing run

1. Press Ctrl+C once in the coordinator terminal and wait for Python to exit.
   During reporting, signal handling requests a stop at a safe boundary. A
   currently executing indivisible operation may delay exit. Do not replace
   source files in the active process and do not start a second coordinator.
2. Keep the existing `runs/ID/`, raw data, fitted artifacts, environment, and
   downloaded checkpoints. Neither package extraction nor resume deletes them.
3. Extract `energygapbench-reporting-patch.zip` into the existing project root.
4. With the same `cuda_env` active, resume the same identity. Do not repeat setup.

```bash
unzip -o /path/to/energygapbench-reporting-patch.zip
bash scripts/server.sh resume conference-20261101T165101Z-1789282 \
  --threads 32 \
  --energy-cpu-workers 16 --energy-cpu-max-pending 32 \
  --report-workers 16 --report-chunk-size 10000 \
  --report-bootstrap-chunk-size 1000 --allow-runtime-patch
```

The continuation registry accepts only its explicitly reviewed source releases
and the exact target release. It still checks all scientific fields, data/model/
runtime identities, and the complete target source-file map. The original frozen
configuration, scientific fingerprint, RNG identity, models, and confirmed
records remain intact. A SQLite backup and patch lineage are recorded under
`runs/ID/runtime_history/` while holding the experiment lock. This is not a general
code-hash bypass. An unknown release or changed model/data/configuration is refused.

Completed site-level validation/test work avoids rebuilding model windows and
loading fold model caches solely to reach the report stage. Integrity verification
still checks the saved scientific evidence before continuation. It reads each
committed shard once and hashes shared referenced artifacts once per unchanged
file version in that pass. It never trusts timestamps instead of the first
content hash and does not skip verification across separate resumes. Long
reconciliation passes log their checked/total counts.

## Progress and estimates

The activity line identifies the current report stage and its own units. Export
rows, bootstrap replicas, figures, and checksum bytes are separate workloads.
Bootstrap context identifies the hypothesis, site, contrast, endpoint, and block
length. Displayed durable bootstrap counters advance only after parent publication.

An empirical estimate for the current stage needs observed progress. A complete
report ETA requires a known remaining plan and calibrated rates for every remaining
work family. Otherwise the display explicitly retains a partial estimate and
unknown stages. It does not divide work by the number of CPU threads or infer
report time from the completed inference percentage. Wall observations include
the actual scheduled batch and publication cost where measured. Observed ranges
are not confidence intervals or completion guarantees.

The ordinary status reader does not reconcile the run or start models:

```bash
PYTHONPATH=src python -m energygap status \
  --experiment-id conference-20261101T165101Z-1789282 --watch
```

`report_progress` and the report section of ETA metadata are operational status,
not additional scientific results. Low GPU use during CPU reporting is expected.

## Evidence, limitations, and completion

Check the validation record accompanying the update for the actual CPU test and
continuation evidence. Synthetic equality tests do not measure V100 performance
or guarantee the same wall-time improvement on shared storage. Remaining work
includes target-server timing, peak-memory measurements, and storage-specific
calibration. Unknown or unavailable comparisons retain their declared NA/support
rules; no bootstrap attempt count or scientific metric is reduced for speed.

The 2 November 2026 audit passed 489 tests with 14 explicit prerequisite skips
(CUDA hardware, compatible frozen-model artifacts, or an absent historical private
pilot). Actual archived-release continuation checks covered both accepted source
releases and extraction of the patch ZIP over its predecessor. They preserved
303 historical records, including 100 committed partial-bootstrap draws, and
produced 8,640 exactly matching final replica payloads, scientific tables, exports,
contrasts, and block sensitivities in each controlled case. These are small CPU
fixtures, not measurements of the running private server study.

A separate 180-record fixture containing 64-by-512 sample arrays reduced the
55.5 MB source ledger to a 135 KB temporary report source while retaining byte-exact
scientific CSV/JSONL exports. Local report wall time was 0.769 s versus 0.456 s.
Those illustrative measurements explain the optimization; they do not predict
completion time or a fixed percentage improvement on the V100 server.

Some report exports and figures may repeat if interrupted before their atomic
publication. Confirmed bootstrap results and model inferences remain reusable.
Wait for coordinator exit, the final status, `reports/report_complete.json`, and
the final coverage review before packaging results. Operational completion does
not itself establish scientific completeness or resolve meter semantics.
