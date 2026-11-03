# Reference server and execution resources

The reference installation uses two Intel Xeon Gold 6130 processors, 386 GB of
system RAM and one NVIDIA Tesla V100-PCIE 16 GB GPU. CPU topology comes from the
authors' `lscpu` output; system RAM capacity was supplied by the authors on
3 November 2026. GPU and software details were recorded by the experiment.
The machine-readable record is
[`environments/reference_hardware.json`](../environments/reference_hardware.json).

## Host hardware

| Component | Reference configuration |
| --- | --- |
| CPU model | Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz |
| Architecture | x86_64, little-endian |
| Processor sockets | 2 |
| Physical cores per socket | 16 |
| Total physical cores | 32 |
| Hardware threads per physical core | 2 |
| Logical processors | 64, IDs 0–63 in the supplied host inventory |
| NUMA nodes | 2 |
| System memory | 386 GB RAM, author-reported capacity |
| GPU | NVIDIA Tesla V100-PCIE, 16 GB |
| GPU compute capability | 7.0, with tested `sm_70` kernels |

The reported 64 CPUs are logical processors, not 64 physical cores. In the
supplied topology, NUMA node 0 contains the even-numbered logical CPU IDs and
node 1 contains the odd-numbered IDs. The software does not assume that this
numbering or affinity is identical on another server or inside a container.
The CPU model's 2.10 GHz designation is not a measurement of sustained clock
frequency during the experiment.

## Executed runtime

| Setting | Recorded or requested value |
| --- | --- |
| Conda environment | `cuda_env` |
| Python | 3.12.16 |
| PyTorch | 2.9.1+cu126 |
| CUDA runtime | 12.6 |
| GPU device | `cuda:0` |
| Main-process thread request | 32 |
| Energy-correction workers | Initially 8; subsequently 16 |
| Pending correction tasks | Initially 16; subsequently 32 |
| BLAS threads in each correction worker | 1 |
| Training batch | 32 |
| Episode validation/inference batch request | 256, subject to final-batch size and recorded OOM reduction |
| Numerical precision | Native models in float32; ridge calculations in float64 |

GPU model execution is serial at the coordinator level; CPU correction workers
do not each train a separate GPU model. Main-process threads and worker counts
describe different parts of the pipeline. They should not be summed to infer
continuous CPU occupancy. Each CPU worker limits its numerical-library threads
to avoid nested oversubscription.

Host RAM and GPU memory are capacities, not recorded peak consumption. The
386 GB host capacity does not establish the memory quota or free memory available
to the JupyterHub container throughout the run. Similarly, available logical
processors do not establish the process affinity or CPU quota in every session.
Effective session settings and historical allocation remain separate from this
hardware inventory. The frozen protocol and existing results are unchanged.

## Check an execution environment

On the target Linux server, inspect the resources visible to the current session:

```bash
lscpu
free -h
nvidia-smi
python -c 'import os; print("Logical CPUs:", os.cpu_count()); print("Affinity:", sorted(os.sched_getaffinity(0)))'
bash scripts/server.sh validate native --preset focused --threads 32
```

The native validation command checks executable CUDA support; device names and
allocated VRAM alone do not establish kernel compatibility or useful throughput.
The reference V100 is a Volta device. The implementation retains its existing
float32 path and does not rely on newer-GPU precision modes.

Use the [server guide](../SERVER_GUIDE.md) for installation and execution,
[worker tuning](runtime_worker_tuning.md) for audited scheduling changes, and
[performance documentation](performance_optimization.md) for timing definitions.
Changing a model, batch-dependent training policy or scientific configuration
requires a separately identified experiment. Hardware capacity does not justify
silently changing the protocol of a completed run.
