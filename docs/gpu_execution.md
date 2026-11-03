# GPU execution and V100 compatibility

Both server profiles target a single V100 16 GB, exposed as logical `cuda:0`.
Use the active `cuda_env` and the [server workflow](../SERVER_GUIDE.md). Selected
native validation checks actual hardware operations and installed adapters:

```bash
conda activate cuda_env
bash scripts/server.sh validate native --preset focused --threads 32
# Use --preset full for the expanded model set.
```

`doctor --device cuda:0` probes the actual PyTorch build, GPU architecture, and a
float64 solve. `gpu-smoke` additionally exercises a small SAITS fit/inference and
ridge operation. Explicit CUDA requests fail rather than silently using CPU;
`auto` is a separate selection policy. Preserve a working V100-compatible PyTorch
build rather than replacing it with a historical local diagnostic version.

Native model tensors/weights use float32; ridge dispatches its solve/prediction to
float64. Device transfers, optimizer placement, CUDA RNG state, and synchronized
timing have explicit paths. The default introduces no AMP, BF16/TF32 requirement,
Triton dependency, or newer-architecture-only compile path. V100 requires effective
Volta `sm_70` support and actual kernel probes. The optional BRITS inference graph
mode remains experimental and disabled by default.

Data preparation, conventional baselines, correction, statistics, and plots run on
CPU. A request for 32 main-process threads does not launch 32 GPU experiments.
Correction uses eight bounded spawned CPU workers with one BLAS thread each. All
selected model integrations run natively in the active interpreter; historical
isolated model workers are not the current conference path.

Device, precision, package/runtime identities, and numerical execution settings
participate in compatibility. Resume normally retains the saved device. A CPU/GPU
switch or incompatible code/runtime requires a new identity; bitwise equality
across devices is not promised. Timing synchronizes CUDA at applicable boundaries;
CPU timings and VRAM allocation do not establish GPU throughput.

Local CPU regression/recovery tests and unavailable-CUDA checks are available.
Real V100 parity, performance, and interruption/recovery evidence must be retained
from the server; this source release does not certify a completed V100 study.
See [GPU measurement](performance_batched_gpu.md) and
[development TODO](development_status.md).
