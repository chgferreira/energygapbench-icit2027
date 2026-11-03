# EnergyGapBench for IEEE ICIT 2027

Research code, aggregate results and reproducible analysis for
**Energy-Regularized Imputation of Power Telemetry after Communication Outages**.

**Manuscript status: under review at IEEE ICIT 2027.**

This repository provides the experiment implementation, aggregate results and
versioned analyses for the completed focused study
`conference-20261101T165101Z-1789282`. The manuscript is maintained separately.
The code supports both the executed **focused** design and a configurable **full**
extension. Further validation and extensions are tracked in the
[development roadmap](docs/development_status.md).

**Data availability.** The data supporting this study are available from the
authors upon reasonable request. Aggregate results are included here. See
[access instructions and reproduction scope](DATA_AVAILABILITY.md).

The study asks whether admissible energy-register increments at the boundaries of
a communication gap can help recover its hidden power trajectory and support a
forecast issued at reconnection. A common correction preserves received power
values and trades departure from the base reconstruction against energy mismatch
and changes in the correction. Reconstruction error, counter consistency and
forecast utility are evaluated as distinct outcomes.

## Start here

| Task | Entry point |
| --- | --- |
| Rebuild aggregate tables and figures | [ICIT analysis instructions](analysis/README.md) |
| Inspect the versioned forecast follow-up | [Common-target analysis version 2](analysis/forecast_v2/README.md) |
| Inspect the actual run | [Evidence directory](results/conference-20261101T165101Z-1789282/) and [provenance guide](docs/icit2027_reproduction.md) |
| Understand what is and is not publicly reproducible | [Data availability](DATA_AVAILABILITY.md) |
| Run synthetic software checks | [Reproduction guide](docs/reproduction.md) |
| Inspect reference hardware and resource settings | [Server hardware](docs/reference_hardware.md) |
| Run a new experiment on the V100 | [Server guide](SERVER_GUIDE.md) and [experiment profiles](docs/experiment_profiles.md) |
| Read the compact protocol and runtime supplement | [ICIT supplement](docs/icit2027_supplement.md) |
| Review remaining work | [Development status](docs/development_status.md) and [release checklist](docs/release_checklist.md) |

## Executed focused study

The original frozen protocol is authoritative for this run. Its [public extract](results/conference-20261101T165101Z-1789282/protocol.public.json) retains experiment settings and records the original hashes without individual dataset metadata. Current YAML files describe
new runs and cannot retroactively alter the experiment.

| Design choice | Executed scope |
| --- | --- |
| Sites | H1 and H2, households with generation; I1, industrial facility without generation |
| Evaluation | Three chronological folds and three contiguous-gap conditions, 15, 60 and 180 minutes |
| Base methods | Median, LOCF, linear, seasonal, Kalman filter, Kalman smoother, SAITS and frozen MOMENT-1-base |
| Selection | SAITS only, two fixed candidates on the first fold of each site, seed 11, maximum 20 HPO epochs and patience 5 |
| Final training | SAITS seeds 11, 23 and 47 for each site/fold, maximum 100 epochs and patience 10 |
| Validation masks / test masks | Five / ten deterministic replicas |
| Epoch selection | Fixed metadata-selected panel, at most 300 episodes per site/fold |
| Final scoring | Complete eligible manifests within the declared focused scope |
| Batches | Training 32; inference/episode selection request 256, with recorded smaller final batches and OOM handling |
| Physical hypotheses | Instantaneous power, interval mean at interval start, interval mean at interval end; all exploratory |
| Forecast horizons | 30 minutes, one hour, three hours and six hours; one hour is primary |
| Planned bootstrap | 10,000 attempted day-block draws, seven-day main blocks and two-/14-day sensitivity |

The supplied report records operational completion, 33 fit records, no fit
failures and no unresolved jobs. It retains `scientific_complete=false` because
operational completion is distinct from scientific interpretation. The paper and
analysis document endpoint support and any deviations or additional analyses.
Do not treat an unavailable endpoint as zero, a completed job as a successful
model prediction, or an exploratory confidence interval as confirmatory evidence.

The full YAML declares 17 candidates, of which 13 have an execution path when
requirements pass and four remain unavailable. It expands methods, missingness
conditions and selection budgets. Its configuration is included for a future
separately identified experiment. Its results are not pooled with the focused run.

## Findings and interpretation

Under the instantaneous directional hypothesis, the correction reduces the
validation-selected conventional comparator's reconstruction nMAE by **1.83% for
H1, 14.69% for H2 and 44.13% for I1**. These are descriptive relative changes
after equal-duration/fold aggregation, not an overall pooled improvement or a
claim of benefit for every method and gap length. H1 does not satisfy the declared
temporal-support rule for the main seven-day bootstrap. All physical branches
remain exploratory. See the [selected-policy values](analysis/outputs/selected_policy_overall.csv)
and [original contrast support](analysis/outputs/frozen_contrasts.csv).

A secondary descriptive analysis shows lower corrected-linear nMAE in all
**27 site/fold/duration cells**, with reductions of **9.66–26.47%**. Site-level
MAE falls from 0.305 to 0.252 kW at H1, 0.327 to 0.276 kW at H2, and 2.454 to
1.944 kW at I1. These related cells are not independent tests. Linear remains
the declared reference and does not replace the validation-selected comparator.
The [engineering summary](analysis/outputs/engineering_summary.csv) and
[reproduction script](analysis/scripts/summarize_engineering.py) support the reported physical-unit interpretation and supplementary table. A future, separately identified experiment will compare the
current selection policy with joint base-and-correction selection on validation,
using common budgets and held-out test periods.

The frozen primary one-hour forecast endpoint is unavailable at site level because
some original future targets are absent under its complete-outcome rule. The
original NA values remain unchanged. The [version-2 forecast
analysis](analysis/forecast_v2/README.md) documents a follow-up on 68,217 episode instances with common
observed targets and verifies the day calendars against every original
reconstruction interval before computing paired forecast intervals.

Under the instantaneous hypothesis, relative forecast nMAE changes are **+1.62%,
+0.20% and -7.59%** for H1, H2 and I1. Positive values mean worse forecasting.
The seven-day pointwise interval for the I1 difference is **[-0.015900, -0.002985]**
in nMAE, with **unadjusted exploratory p = 0.0055**. H2's interval crosses zero.
H1 fails the temporal-support check, so its interval is diagnostic and no
exploratory p-value is reported. See the [complete version-2
contrasts](analysis/forecast_v2/outputs/forecast_contrasts.csv).

These results are conditional on observed targets, fitted models and realized
masks. They do not correct target-missingness bias or demonstrate general forecast
benefit. The revised reporting policy was developed after inspecting the completed
run and is explicitly exploratory. Its outputs and the earlier point-only
supplement remain separately versioned.

The [same-information energy controls](analysis/energy_controls/README.md) compare
constant gap offsets, with and without a tolerance zone, against the archived
flexible linear correction. All 68,580 test episode instances are retained and
89,175 validation/test power-only losses were reproduced before comparison.
The flexible correction lowers mean error against both controls in H1/H2. H2
has supported exploratory paired differences; H1 retains its support limitation.
At I1 the no-zone control has slightly lower mean error, with an interval crossing
zero. These comparisons hold information and integration fixed but do not isolate
the flexible regularizer or equalize the different search budgets.

A train-only [I1 diagnostic](analysis/outputs/energy_diagnostics/diagnostic_validation.json)
reproduces zero export-residual quantiles in all 27 calibration strata. The
original positive-tolerance rule turns those into unavailable export tolerances,
which blocks the interval-mean net constraint despite positive import tolerances.
This is a reproduced implementation mechanism, not recovered per-query solver
logs or confirmation of instrument semantics. Original results remain unchanged.

## Reproduce without training

Check the release inventory with `python scripts/check_public_release.py`.

The public workflow uses only aggregate tables and aggregate bootstrap contrast
estimates. It rebuilds the three reconstruction figures and numerical comparison
tables and checks all 54 forecast and 18 energy-control percentile intervals and
exploratory p-values. It needs CPU Python, without CUDA or a TeX installation.

```bash
python3.12 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r analysis/requirements.txt
ANALYSIS_PYTHON=.venv-analysis/bin/python bash scripts/reproduce_icit.sh
```

Outputs go to `analysis/public_outputs/`. The command does not reconstruct
individual episode pairing or rerun bootstrap resampling. The published bootstrap
arrays contain aggregate contrast estimates only. Full loss-level recomputation
requires restricted artifacts and is documented in the [analysis guide](analysis/README.md).
The article and individual records are maintained separately from this public
code-and-aggregate release.
See the [reproduction levels](docs/icit2027_reproduction.md) and [data terms](DATA_AVAILABILITY.md).

## Run a new focused experiment

A new full pipeline run requires authorized original H1/H2/I1 files, compatible
model artifacts and a CUDA environment. After preparing these prerequisites:

```bash
conda activate cuda_env
bash scripts/server.sh setup --preset focused --threads 32
bash scripts/server.sh run --preset focused --threads 32 --budget-hours 24
```

The 24-hour value is a configured compute ceiling, not a completion guarantee.
The reference machine has **386 GB RAM**, **two Intel Xeon Gold 6130 processors
at 2.10 GHz** (32 physical cores, 64 logical processors) and a **Tesla V100-PCIE
16 GB**. The recorded runtime uses Python 3.12.16 and PyTorch 2.9.1+cu126.

Host capacity, process affinity and resource usage are distinct. The experiment
requests 32 main-process threads. Correction workers initially used 8 workers
with 16 pending tasks; later sessions used 16 workers with 32 pending tasks.
Each correction worker uses one BLAS thread. The
[hardware guide](docs/reference_hardware.md) documents CPU topology, NUMA,
container visibility, batch sizes and provenance.

Check actual `sm_70` kernel support before training and preserve the CUDA build
when installing dependencies. The [observed environment snapshot](environments/icit2027-runtime-observed.json)
contains the recorded package subset; it is not a complete environment lock.

For full, explicitly use `--preset full` and a separately chosen budget. To resume
an existing compatible run, retain its complete original run directory and use
`bash scripts/server.sh resume ID --threads 32`. Exact-release continuation patches
are documented in [reporting performance](docs/reporting_performance.md); they
are not a general identity bypass.

## Repository structure

| Directory or file | Purpose |
| --- | --- |
| `src/energygap/` | Data contracts, availability rules, reconstruction, energy correction, forecasting, statistics and execution state |
| `configs/` | Focused/full configurations, dataset contracts and audited continuation registries |
| `scripts/` | Server setup, execution, monitoring, packaging and public aggregate reproduction |
| `analysis/scripts/` | Original report checks, common-target forecasts, energy-control replay and public summaries |
| `analysis/outputs/` | Reconstruction aggregates, control summaries, diagnostics and figures |
| `analysis/forecast_v2/` | Versioned common-observed-target forecast analysis |
| `analysis/energy_controls/` | Same-information scalar controls and paired exploratory comparisons |
| `results/` | Six original aggregate report exports and a clearly identified public protocol extract |
| `tests/` | Synthetic unit, integration, recovery and optional native-model checks |
| `docs/` | Methods, information availability, statistical plan, execution guides and research roadmap |
| `environments/` | Source hashes, dependency records, observed runtime and reference hardware |
| `ARTIFACT_MANIFEST.json` | SHA-256 inventory for the public release |
| `CHANGELOG.md` | Documentation and packaging changes, separate from scientific analysis versions |

## Citation, data access and licensing

The associated manuscript, **Energy-Regularized Imputation of Power Telemetry after Communication Outages**,
is **under review at IEEE ICIT 2027**. Author order and
citation metadata are provided in [`CITATION.cff`](CITATION.cff). Cite the software
version or commit used in a reproduction, and distinguish the original experiment
from its versioned follow-up analyses.

Study data are **available upon reasonable request** to the authors through the
[repository maintainer](https://github.com/chgferreira). Requests should identify
the required data or artifacts and the intended research use. Access to individual
telemetry and derived records is subject to the applicable privacy and data-holder
conditions; public aggregate results can be used immediately for the documented
reproduction workflow. See [data availability](DATA_AVAILABILITY.md).

Original code is covered by [MIT](LICENSE). Data and third-party assets retain
separate terms. Read [third-party notices](THIRD_PARTY_NOTICES.md) for model and
upstream-source licensing. The [release checklist](docs/release_checklist.md)
and [development roadmap](docs/development_status.md) distinguish completed
software checks from pending scientific validation.
