# Server guide: focused and full experiments

**Work in progress.** This guide describes the supported single-server workflow.
The scientific designs and tradeoffs are defined in
[experiment profiles](docs/experiment_profiles.md); unfinished tests are listed in
[development status](docs/development_status.md).

## Requirements and package layout

Use Linux, the existing `cuda_env`, one logical device `cuda:0`, and local POSIX
storage with sufficient free space. The target is a V100 16 GB; the default
main-process thread request is 32. Worker counts are checked against the actual
CPU affinity rather than an assumed host-core inventory. Python 3.12/3.13 and an
installed CUDA PyTorch `>=2.5.1,<2.10` that actually supports this GPU are required.
The environment name and `nvidia-smi` alone do not validate CUDA kernels.

The server ZIP retains its historical filename `energygapbench-server-full.zip`
and includes **both** profiles; the entry-point default is **focused**. It extracts
flat into a repository root containing `pyproject.toml`. Real data, downloaded
weights/source, environments, and private run results are excluded. Preserve the
three original files:

```text
raw_data/H1.csv
raw_data/H2.csv
raw_data/I1.csv
```

Before replacing executable code/configuration, stop the old coordinator at a safe
boundary and preserve its complete `runs/ID/`. New scientific code or settings need
a new identity. A documentation-only edit does not itself change the Python source
hash, but do not replace an active installation with a different executable build.

## Prepare and start a new study

From the project root, with the archive transferred to the server:

```bash
conda activate cuda_env
unzip -o /path/to/energygapbench-server-full.zip
sha256sum -c SERVER_FULL_MANIFEST.sha256
bash scripts/server.sh setup --preset focused --threads 32
bash scripts/server.sh run --preset focused --threads 32 --budget-hours 24
```

Setup installs extra requirements in the active interpreter, preserves the exact
installed PyTorch/NVIDIA runtime packages, checks dependency resolution, obtains
pinned selected model sources/checkpoints, and runs relevant native/CUDA checks.
Network access is required for uncached installation/downloads. Focused needs the
MOMENT checkpoint and checks SAITS/MOMENT; full additionally prepares T1/TS-ICL
sources or weights and checks all seven model integrations. It does not create
separate model environments. See [single environment](docs/single_cuda_env.md).

For an installation already prepared with this executable version, use a selected
native check and start a fresh identity without reinstalling dependencies:

```bash
bash scripts/server.sh validate native --preset focused --threads 32
bash scripts/server.sh run --preset focused --threads 32 --budget-hours 24
```

Select the expanded design explicitly:

```bash
bash scripts/server.sh setup --preset full --threads 32
bash scripts/server.sh run --preset full --threads 32 --budget-hours 168
```

`run` assigns a new ID unless `--experiment-id NEW_ID` is provided. Existing run
directories are refused; earlier results are not deleted. Setup and run share a
project lock and refuse a competing coordinator. The workflow uses one GPU job at
a time and does not reserve remote resources.

## Custom configurations and budget semantics

An in-progress supported predecessor can use the exact-release
[reporting continuation](docs/reporting_performance.md). Stop it first, extract
the small reporting patch in the existing project, and resume its original ID
with `--allow-runtime-patch`. No setup or environment replacement is needed.
The additional `--report-workers`, `--report-chunk-size`, and
`--report-bootstrap-chunk-size` options are operational; they do not reduce
bootstrap attempts or change the study configuration. Report CPU scheduling
is separate from correction scheduling and GPU model execution.

Copy a profile YAML, edit the copy before evaluation, and pass it explicitly:

```bash
cp configs/experiments/server_focused.yaml configs/experiments/custom_study.yaml
bash scripts/server.sh setup --config configs/experiments/custom_study.yaml --threads 32
bash scripts/server.sh run --config configs/experiments/custom_study.yaml \
  --threads 32 --budget-hours 24
```

`--preset` and `--config` are mutually exclusive. Custom settings are validated and
recorded without focused overrides. The default cumulative ceiling is 24 hours for
focused/custom and 168 for full; `--budget-hours` replaces that default. A larger
budget does not switch methods/conditions. A reached ceiling pauses at a safe
boundary without reducing the declared scope. Setup/planning and indivisible work
mean this is not a strict elapsed-time deadline or completion estimate.

The compatibility wrapper `scripts/run_server_full.sh` explicitly starts full;
`scripts/restart_server.sh` starts a new run using the central entry point's options.
Direct Python conference calls should pass `--config`: their default remains the
full YAML, independently of the shell's focused default.

## Validation levels

Validation checks do not start the scientific experiment. Select the level relevant
to the change; there is no need to run all four in sequence.

| Command | Behavior |
| --- | --- |
| `bash scripts/server.sh validate fast --preset focused` | Python/shell syntax and selected configuration |
| `bash scripts/server.sh validate core --preset focused` | Fast checks and one non-heavy regression invocation |
| `bash scripts/server.sh validate native --preset focused --threads 32` | Fast checks, actual CUDA probes, and selected SAITS/MOMENT integration/parity checks |
| `bash scripts/server.sh validate full --preset focused` | Fast checks and one complete test-suite invocation, with explicit prerequisite skips |

The test level `validate full` and experiment preset `--preset full` are different.
Use the latter to select the expanded configuration for checks; `--config PATH`
selects a custom design. Native validation can use `--device cpu` locally, but that
does not validate V100 execution. Native/full checks refuse an active project
coordinator. Setup runs relevant native probes, not pytest; run performs environment
and CUDA preflight, not repeated setup or full regression suites. Detailed model
checks are written under `artifacts/analyses/server_environment/`.

## Monitor progress

The launch prints the experiment ID, log path, and monitoring command. In another
terminal, activate the same environment and enter the project root:

```bash
conda activate cuda_env
PYTHONPATH=src python -m energygap status --experiment-id ID --watch
```

Replace `ID` with the printed identifier. For a one-time machine-readable snapshot:

```bash
PYTHONPATH=src python -m energygap status --experiment-id ID --json
```

To find recent directories and follow the selected log:

```bash
ls -1dt runs/*/ | head -n 5
tail -f artifacts/analyses/logs/ID.log
```

The main display has confirmed jobs, committed queries, and current activity.
Activity identifies method, site, fold, seed, HPO ordinal, epoch, batch, and
validation progress. Focused HPO has two candidates and an epoch limit of 20;
the fixed epoch panel has at most 300 episodes. Final fits have their own
100-epoch limit. Final HPO scoring evaluates the full validation population.

The query counter increases on committed main inference/HPO scoring. Internal
training/epoch validation can advance while that counter stays unchanged.
A value such as 4,400/905,480 is about 0.49% of the audited focused query plan,
not a reliable elapsed-time fraction. Methods, correction, training, and reporting
have different costs. Partial ETA covers calibrated phases only; unknown phases
remain explicit. See [progress and ETA](docs/progress_and_eta.md).

Terminal refresh, compact progress persistence, status snapshots, and heartbeats
use default intervals of 1/10/60/15 seconds. These are separate from durable
scientific commits and completed-epoch checkpoints.

## Observe GPU activity separately

```bash
nvidia-smi --query-gpu=index,uuid,name --format=csv
python -u scripts/gpu_telemetry.py --gpu 0 --duration-seconds 120 --interval-seconds 1
```

`--gpu` is a physical NVIDIA index or UUID, not a remapped logical `cuda:0`.
Use the UUID when `CUDA_VISIBLE_DEVICES` remaps devices. The telemetry distinguishes
GPU activity, memory-controller activity, VRAM allocation, and power. Samples cover
the whole device, can include other processes, and can miss short kernels. VRAM
occupancy does not measure useful GPU throughput. Telemetry does not enter
scientific timing. Run independent inference benchmarks without another GPU job;
see [GPU benchmarking](docs/performance_batched_gpu.md).

## Stop and resume

Request interruption once and wait for the next safe boundary. Preserve all of
`runs/ID/`. With compatible code, data, environment, and settings:

```bash
bash scripts/server.sh resume ID --threads 32
```

Resume uses the locked profile and saved cumulative budget, rejecting `--preset`
and `--config`. To extend the ceiling explicitly:

```bash
bash scripts/server.sh resume ID --threads 32 --budget-hours 48
```

The 48 hours include work already accounted for. Offline time does not reset
scientific progress or contribute to measured throughput. Partial epochs/units
may repeat from the last valid checkpoint/commit. A completed compatible run does
not repeat committed fits/inference. Changed algorithms, selection policies, masks,
precision, or batching require a new identity; neither preset converts an old run.
See [recovery guarantees and limits](docs/resume_and_recovery.md).

## CPU scheduling-only continuation

The [audited worker patch](docs/runtime_worker_tuning.md) can preserve a supported
predecessor run while changing correction worker/queue counts. Safely stop before
updating, then resume the same ID with `--energy-cpu-workers 16
--energy-cpu-max-pending 32 --allow-runtime-patch`. The exact release pair and all
other scientific fields are checked; a database backup and execution history are
retained. This is a restricted exception to the general changed-code rule, not a
way to change batches, seeds, selection, or scientific computations.

## Review outputs before scientific use

`runs/ID/state.sqlite` is authoritative. The run contains protocol locks, immutable
mask/panel manifests, model/checkpoint artifacts, committed shards, and reports.
Diagnostics/logs remain under `artifacts/analyses/`. Reports expose scope, actual
budgets, panel evidence, committed coverage, unavailability, and failures. Inspect
those fields alongside reconstruction, forecast, and correction tables. A terminal
`completed` state is operational; it does not establish convergence, metrology,
all-catalogue reproduction, or manuscript completion.

The source ZIP and reporting commands do not publish private results or update the
manuscript. Follow the [TODO roadmap](docs/development_status.md) and
[release checklist](docs/release_checklist.md) before submission or public release.
