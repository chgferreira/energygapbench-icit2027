# Execution optimizations and their scientific boundaries

The implementation improves execution in both [profiles](experiment_profiles.md).
Algorithm-preserving optimization and reduced experimental scope are separate
changes: focused explicitly changes methods, conditions, HPO, and epoch selection;
full retains its broader declared design. No CUDA speedup or completion deadline
is established by the local CPU checks.

## Native batching and input preparation

BRITS, SAITS, ImputeFormer, T1, and MOMENT can infer independent deterministic queries
in native batches, requested at 256 by the server YAMLs. Training remains batch 32.
Context 512, admissible/padding masks, observed restoration, evaluation mode, and
native float32 precision are preserved. CUDA OOM halves deterministic inference
batches and records requested/effective sizes and retries; it does not remove
queries. Failed retries still consume pipeline time.

BRITS prepares exact upstream elapsed-time deltas with vectorized CPU operations and
a bounded cache of sanitized inputs. `brits_execution: vectorized` groups decays and
gates that are independent of recurrent state, retaining native parameters, losses,
and optimizer state. Loss calculations unused by inference are omitted there.
The `native` path remains the reference. Local checks compare reconstructions,
training losses, gradients, and an Adam update, not just output shapes.

`vectorized_cuda_graph` is an experimental, opt-in BRITS inference path. It retains
one bounded captured shape, uses eager tails, and records capture/replay or fallback
reasons. Training stays eager float32. The default is `vectorized`; graph use needs
separate parity/performance evidence on the V100 and a new experiment identity.
No default path requires BF16, TF32, Triton, or newer GPU architecture features.

MOMENT batches independent queries with per-query masked normalization and observed
restoration. TS-ICL remains serial. Full-profile CSDI keeps one query at a time but
generates its 100 trajectories in native groups of 32/32/32/4, with all 50 diffusion
steps. Independent draws preserve the intended conditional sampling distribution;
RNG traversal changes, so old Monte Carlo samples are not promised bitwise identical.
The recorded policy/seed is part of the new identity. Focused does not select CSDI.

## Selection work and exact-score reuse

Full epoch validation uses all declared core episodes; focused uses its fixed
metadata-only panel. Both rank HPO candidates on full declared core validation.
The exact best checkpoint's complete episode scores can be reused only after
model/hash/epoch/coverage verification. Missing, stale, corrupted, or partial-panel
caches trigger normal complete scoring. No HPO checkpoint is promoted into a final
fit. Focused's smaller panel and HPO budget are declared design tradeoffs, not
claims of equivalence to full selection.

## Ridge and downstream batching

Vectorized preparation preserves the original chronological rows, features,
float64 arithmetic, valid zero observations, causal natural-gap filling, original
targets, physical horizons, and validation alpha grid. `ForecastQueue` copies the
last 288 reconstructed positions and groups `predict_many` by horizon, splitting
large groups into bounded batches. Eligibility remains evaluator-owned: targets
must be original, in the test partition, and within 90 seconds of physical time.
Missing targets stay NA; batching creates no truth or new fitted forecast model.

## Exact single-net interval-mean correction

For `interval_mean_start`/`interval_mean_end` with one net-energy constraint, the
net residual is affine in the interior correction d. With fixed boundaries,

\[
J(d)=d^TQd+\lambda\left[\max\left(0,\frac{|r+a^Td|}{\tau}-1\right)\right]^2,
\qquad Q=\frac{I/n+\gamma L/(n+1)}{s^2}.
\]

Here r is base net energy minus the counter increment, a is its interior gradient,
s is training scale, and L has diagonal 2 and adjacent entries −1. Q is positive
definite. If |r|≤tau the minimizer is zero; otherwise the unique minimizer is

\[
d=-\frac{\lambda\,[r-\operatorname{sign}(r)\tau]}
             {\tau^2+\lambda a^TQ^{-1}a}\,Q^{-1}a.
\]

The implementation solves/reuses the tridiagonal system and checks finite arithmetic
and objective reduction, retaining a general-solver fallback. This shortcut does
not apply to instantaneous integration, multiple constraints, or unidentified gross
import/export. An exact minimizer does not establish unknown instrument semantics.

## CPU correction workers and persistence

Both YAMLs request eight lazy spawned CPU workers with one BLAS thread each and a
bounded queue of 16 tasks. Workers receive episode correction inputs, not evaluator
truth, CUDA models, or SQLite connections. All configured candidates and trajectories
are retained for the selected profile. Results return in order; the coordinator
owns scientific scoring, commits, and safe-stop decisions. The queue prevents
materializing millions of corrections simultaneously.

Disabled-correction candidates (`lambda_energy=0`) retain their declared rows but
reuse the same identity score/failure counts within a hypothesis. Progress/status
uses grouped SQL and cached counts instead of querying every job at every validation
update. Default terminal/progress/status/heartbeat intervals are 1/10/60/15 seconds.
SQLite remains WAL with `synchronous=FULL`; safe scientific commits and epoch
checkpoints are not relaxed.

## Cost interpretation and remaining evidence

Batched query time is amortized, not isolated-request latency. Full CSDI's sample
batching remains within one query, allowing timing of that complete sampled query.
Parallel `correction_service_seconds` excludes queue wait and model inference;
`parallel_cpu_service_cost` does not populate isolated latency/throughput. Summing
parallel service times cannot estimate elapsed pipeline time.

Reports retain OOM metadata and separate cost modes. Successful-call amortized cost
omits failed retries; ledger elapsed work includes them. `oom_retry_seconds` is
cumulative per call and repeated on its query rows, so do not sum those rows as
independent retry durations. Neither inference throughput nor 32 threads predicts
whole-study runtime. Query fractions omit training, repeated validation, and other
work; use phase calibration and explicit unknown ETA.

Local parity tests cover scalar/batch composition and permutation, observed values,
padding, deterministic seeds, resumed state, all forecast horizons, and the original
energy objective/gradient. Controlled CSDI noise tests check trajectory paths without
assuming identical PRNG traversal. Rounding can affect near-tied model selection;
CUDA bitwise parity is not guaranteed. Historical CPU preparation timings are
component diagnostics, not measured V100 speedups.

Use the [separate GPU benchmark](performance_batched_gpu.md) without a competing
coordinator. [Development TODO](development_status.md) includes complete-server
runtime/storage, GPU parity/recovery, graph evaluation, and profile-selection
sensitivity. Scope reductions and compute ceilings must accompany performance claims.
