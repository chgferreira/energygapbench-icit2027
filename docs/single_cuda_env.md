# Single server environment

All selected executable methods use the active `cuda_env` Python interpreter.
Setup supplements that environment while preserving its exact installed CUDA
PyTorch and NVIDIA runtime packages. The supported conference path creates no
per-model environments. Both [profiles](experiment_profiles.md) use this policy.

Python 3.12/3.13 and PyTorch `>=2.5.1,<2.10` with effective V100 support are required.
Preflight runs actual CUDA operations. An environment name or visible NVIDIA driver
alone is insufficient. The current source/dependency pins are in
`requirements-server.txt` and `constraints-server.txt`; historical diagnostic locks
are not replacements for the server requirements.

## Pinned native dependencies

| Component | Selected version | Role |
| --- | --- | --- |
| PyPOTS | 1.5 | Native BRITS, SAITS, CSDI, ImputeFormer |
| MOMENT | Official source 0.1.5 at the revision below | Compatible package metadata for the common environment |
| TS-ICL | 0.2.0 | Audited native imputation pipeline with compatible dependencies |
| Transformers | 4.57.6 | Common pinned dependency |
| Hugging Face Hub | 0.34.4 | Compatible checkpoint/client dependency |
| NumPy / pandas / SciPy | 2.2.6 / 2.2.3 / 1.15.3 | Jointly resolved scientific stack |

MOMENT source revision
[`38f7310ad594100747ca2a8357e9c7ca7d323e0e`](https://github.com/moment-timeseries-foundation-model/moment/commit/38f7310ad594100747ca2a8357e9c7ca7d323e0e)
declares version 0.1.5. Its source archive SHA-256 is
`987428c10b201f7af9598be8924c308d72c701bf6167175d0f8fb7a0f4271725`.
The local source audit found all 21 `momentfm/` Python files byte-identical to the
earlier audited official 0.1.4 package; metadata changes resolve installation
constraints. This is an official source revision, not rewritten package metadata.
Checkpoint revision/digest remain separately pinned in `model_registry.yaml`.

The audited TS-ICL 0.2.1 requirement `huggingface-hub>=1.19.0` conflicts with the
pinned Transformers requirement `huggingface-hub<1.0`. The selected official 0.2.0
revision is
[`6d4861e9eb29d59f05a0c253432afb2ae936e02a`](https://github.com/EDF-Lab/ts-icl/commit/6d4861e9eb29d59f05a0c253432afb2ae936e02a).
The [release comparison](https://github.com/EDF-Lab/ts-icl/compare/v0.2.0...v0.2.1)
and local audit found the imputation pipeline/model/data utilities unchanged; the
pipeline Git blob is `acd8c53f627e765429ba812d4ce14791c407c14d`. Versions are recorded
as actually installed, not relabeled as 0.2.1. Its original noncommercial terms
continue to apply. Required dependencies are resolved normally; unrelated optional
extras are not requested.

## Setup, validation, and limits

Setup checks a resolver plan before installing extras and validates dependencies
afterward. It records the installed environment and refuses a plan that replaces
the working PyTorch/NVIDIA runtime. Focused downloads/prepares only selected model
artifacts and checks SAITS/MOMENT; full also prepares T1/TS-ICL and checks all seven
integrations. Shared dependency requirements remain in the same interpreter.

Joint resolution was audited for Linux x86-64/Python 3.12. A clean temporary macOS
CPU environment verified installation/dependency consistency, small native fits
for five models, and real-checkpoint inference for MOMENT/TS-ICL. Mask poisoning,
padding, arrival cutoffs, same-process execution, and shard reuse have local tests.
These checks do not establish target-server CUDA behavior, convergence, or full-run
performance. Run selected native server validation and retain its version/kernel
probes. Outstanding evidence is listed in [development status](development_status.md).

Audit outputs are private local artifacts under `artifacts/analyses/`, not manuscript
results or source-package contents. Models without verified implementation/artifacts/
terms remain explicitly unavailable. Original licenses and hashes are preserved;
sharing one interpreter does not change component licenses.
