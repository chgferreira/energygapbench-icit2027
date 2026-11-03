# Measure batched inference on the V100

Independent benchmarks measure adapter inference, not whole-study duration or
scientific accuracy. Both profiles request deterministic inference batches up to
256, but the useful size depends on model, context, GPU, and preparation cost.
Low VRAM occupancy does not establish low/high useful throughput. Local CPU checks
do not provide measured V100 speedup.

## Run a selected benchmark

Stop the scientific coordinator safely and wait for exit before benchmarking.
The script shares the project lock and refuses another coordinator; it does not
terminate processes. Avoid competing GPU workloads. In the prepared `cuda_env`:

```bash
conda activate cuda_env
python -u scripts/benchmark_server_inference.py \
  --method SAITS --device cuda:0 --threads 32 \
  --batch-sizes 1 32 64 128 256 --queries 256 --repeats 3
```

Without a fitted local checkpoint, the trainable-model diagnostic fits one epoch
on 32 synthetic training windows and validates on eight later windows using the
first native grid architecture. Context is 512 and the same 256 queries are reused
for each measurement. This is a performance fixture, not convergence evidence;
its fitting time is excluded from inference comparisons.

To benchmark an existing matching final model:

```bash
python -u scripts/benchmark_server_inference.py \
  --method SAITS --checkpoint runs/ID/models/MODEL.pt \
  --device cuda:0 --threads 32 --batch-sizes 1 32 64 128 256
```

The final model/checksum is required; an in-progress `.epoch.pt` is not the final
artifact accepted here. The diagnostic does not change the fitted model, scientific
experiment, or training batch size. MOMENT uses its pinned frozen checkpoint and
needs no diagnostic local training:

```bash
python -u scripts/benchmark_server_inference.py \
  --method MOMENT-1-base --device cuda:0 --threads 32 \
  --batch-sizes 1 8 32 64 128 256 --queries 256 --repeats 3
```

## Compare numerical paths and effective batches

Warmup and repeated CUDA-synchronized measurements produce JSON under
`artifacts/analyses/performance/`: median seconds, queries/second, maximum allocated/
reserved memory, and scalar-reference comparisons. Observed entries must remain
identical; reconstructed values use atol/rtol 1e-5 checks. Requested batches and
actual native sizes are separate. OOM halving is explicit in fallback/retry metadata;
a requested 256 that executes 128 is not a measured effective batch of 256.
Persisting failures remain visible, not silently recommended.

Compare `median_queries_per_second` and `speedup_vs_batch_1` only for passing cases,
alongside `effective_native_batch_sizes`. Larger batches can be slower. Adapter input
preparation is included; scientific training, correction, forecasting, and reporting
are not. BRITS cache is cleared before each batch-size warmup; measured repeats
include warm reuse and record last-native-batch cache counters, not global totals.

For the full-profile BRITS path, compare identical weights/queries/requested batches:

```bash
python -u scripts/benchmark_server_inference.py \
  --method BRITS --brits-execution both --device cuda:0 --threads 32 \
  --batch-sizes 1 32 64 128 256 --queries 256 --repeats 3
```

`both` compares native/vectorized and records
`speedup_vs_native_same_requested_batch`. It does not enable the experimental
`vectorized_cuda_graph` path. Measure that opt-in separately and inspect graph/eager
fallback metadata. Inference checks do not certify gradients; selected full native
validation also checks BRITS loss, gradients, and an Adam step.

CSDI/TS-ICL are outside this deterministic batch comparison. Full CSDI keeps 100
trajectories and 50 diffusion steps with within-query native sample batching;
its sampling speed must be measured separately. TS-ICL stays serial. See
[execution optimization](performance_optimization.md) for timing modes and RNG limits.

A quick CPU-only diagnostic can use BRITS with `--device cpu --threads 1 --context 16
--queries 8 --batch-sizes 1 4 --repeats 1`; its report marks `cuda_tested: false`.
Actual V100 parity, throughput/memory, graph stability, and full-run resource
measurements remain TODO in [development status](development_status.md).
