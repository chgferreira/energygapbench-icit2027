# Reproduction levels for ICIT 2027

## 1. Public aggregate reproduction

Follow [analysis instructions](../analysis/README.md). The default
`bash scripts/reproduce_icit.sh` regenerates three figures, comparison tables and
checks of 72 exploratory intervals from published aggregate draws. It requires
CPU Python and analysis dependencies, without private inputs or manuscript files.
It verifies fixed-cell means but does not resample episodes or audit calendar
geometry independently. See [data availability](../DATA_AVAILABILITY.md).

## 2. Open software checks

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m pytest -q analysis/tests analysis/forecast_v2/tests
PYTHONPATH=src .venv/bin/python -m energygap run --profile smoke --device cpu --max-episodes 1 --progress log --budget-total-hours .1
```

Native CUDA, upstream checkpoint and private historical-pilot tests may skip when
their prerequisites are absent. Record skip reasons. Synthetic fixtures establish
software behavior, not scientific effects on the real installations.

## 3. Restricted loss-level reanalysis

The scripts for original report verification, forecast common-target resampling,
scalar-control summaries and paired control resampling are included. Their inputs
are not public. Follow [the restricted-input commands](../analysis/README.md).
Original evidence is immutable. Revised estimates go to separate output folders.
The forecast revision changes the reporting population after the original results
were inspected; its inference is exploratory and conditional, without multiplicity
control. Historical validation manifests describe the private author workflow.

## 4. New end-to-end execution

Full reproduction requires authorized original H1/H2/I1 files, compatible model
artifacts and dependencies. Follow [server instructions](../SERVER_GUIDE.md) and
select the focused preset. Freeze a new run identity. The full preset is a separate
extension and cannot silently replace the focused experiment.

The observed reference server used Python 3.12.16, PyTorch 2.9.1+cu126, CUDA runtime
12.6 and Tesla V100-PCIE 16 GB with tested sm_70 support. The
[observed package subset](../environments/icit2027-runtime-observed.json) is not a
complete environment lock. Check kernels on the target hardware. Training uses
batch 32; episode inference requests 256. The frozen HPO ceiling is 20 epochs,
patience five, two fixed candidates. Details are in the
[protocol supplement](icit2027_supplement.md).

## Provenance and recovery

Six original aggregate exports retain their bytes. `protocol.public.json` is an
extract with distinct identity, not an executable replacement for the original
lock. The original `protocol.lock.sha256` represents a scientific fingerprint,
not the JSON byte hash; both values are recorded separately in the public extract.

The initial source hash was
`d62678487fdd69ee05fa444675099bb5f7ad823fa476994662f168da5d17554b`.
The final continuation source is retained under `src/`; its 81 Python-file hashes
are recorded in `environments/icit2027-source-files.json`. Exact-release patch
registries under `configs/runtime_patches/` document continuity. Runtime scheduling
changed from 8/16 to 16/32 workers/pending tasks while preserving scientific identity.

The public package cannot resume the old run. Recovery requires the complete
original server directory, ledger and checkpoints. Never copy this aggregate
results directory into `runs/` and treat it as a resumable experiment.

The article and LaTeX project are intentionally outside the repository. A GitHub
URL does not imply conference acceptance, publication or an archival DOI.
