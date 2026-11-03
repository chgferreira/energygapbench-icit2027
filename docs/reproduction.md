# Reproduction and validation paths

**Work in progress.** Reproducing software checks, a small diagnostic pilot, and a
complete scientific profile are different tasks. Start with the
[documentation index](README.md) and [profile comparison](experiment_profiles.md).
Real-site scientific reproduction requires private data access and external model
artifacts; the source archive alone does not supply them.

## ICIT article analysis from supplied results

This repository additionally includes compact evidence for the
completed focused run and scripts to regenerate manuscript tables and figures.
See [ICIT artifact reproduction](icit2027_reproduction.md). This CPU-only path is
separate from full raw-data reproduction and does not resume the original run.

## Open synthetic software checks

For a fresh CPU development environment with Python 3.12/3.13, use the minimal lock
separately from an existing CUDA server environment:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m energygap run --profile smoke \
  --device cpu --max-episodes 1 --progress log --budget-total-hours .1
```

Optional dependency/checkpoint/GPU tests can skip explicitly. The minimal lock is
not a target-server PyTorch installation procedure. A synthetic smoke needs no
real telemetry or foundation download. Convenience Makefile targets default to
`.venv/bin/python`; `make PYTHON=python test` uses another active interpreter.
`make plan` invokes the older broad planner, not an explicit current-profile plan.

## Current scientific profiles on the server

Use the existing compatible `cuda_env` and [server guide](../SERVER_GUIDE.md).
`requirements-server.txt`/`constraints-server.txt` govern additional packages while
preserving installed CUDA PyTorch/NVIDIA runtime. Select focused or full explicitly
when comparing studies. The corresponding YAMLs declare uncapped episodes/windows;
smoke and historical pilot caps are not profile defaults.

To generate an actual configured plan without training:

```bash
PYTHONPATH=src python -m energygap plan --profile conference \
  --config configs/experiments/server_focused.yaml --device cuda:0 \
  --threads 32 --budget-total-hours 24
```

Planning reads/hash-checks data/artifacts and writes manifests/state. Required
missing data/models can block preflight. For full, replace both the YAML and the
intended budget. Direct CLI defaults and the legacy planner are not the shell's
focused preset; keep `--config` explicit.

For ingest, preserve original source bytes/hashes and verify dataset evidence.
The statistical path accepts the documented unit/sign/timezone evidence while
retaining unknown physical semantics. Canonical physical `validate-config` can
remain unresolved; conditional energy hypotheses are separate, explicitly
exploratory branches. They do not silently fill missing instrument metadata.

## Retain evidence and resume

Keep `runs/ID/` with protocol locks, package/device identities, split/mask/panel
manifests, model/checkpoint checksums, state, result shards, timing, and reports.
Resume with `bash scripts/server.sh resume ID --threads 32`. Only a compatible
identity continues; changing selection settings or numerical science requires a
new run. Status reconstructs committed progress without loading models.

Run relevant tests, not every expensive diagnostic repeatedly. Selected native
checks validate the installed model/device path; recovery demonstrations validate
controlled interruption paths. Neither establishes real-site convergence or benefit.
The current gaps and required evidence are in [development TODO](development_status.md).

## Public package

`python scripts/package_release.py` builds allowlisted source ZIP/tar archives;
`python scripts/package_server.py` builds the flat server ZIP containing both
profiles. Manifests/checksums permit integrity checks. These source-only packagers exclude real telemetry, truth, results,
environments, weights, and downloaded upstream source. The separately prepared
ICIT publication package includes curated derived results under their own access
terms; consult `DATA_AVAILABILITY.md`. Model/data
licenses remain separate. Historical isolated-environment instructions are audit
records, not the supported current conference setup. No package command uploads,
publishes, or creates a DOI.
