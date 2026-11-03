# Documentation index

**Status: work in progress.** This documentation describes the implemented designs
and remaining research work. A configured or successfully executed experiment is
not automatically an accepted scientific result.

## Start here

| Purpose | Document |
| --- | --- |
| Project overview and quick start | [Repository README](../README.md) |
| Reproduce the ICIT paper from supplied results | [ICIT artifact reproduction](icit2027_reproduction.md) and [analysis](../analysis/README.md) |
| Included derived data and sharing boundaries | [Data availability](../DATA_AVAILABILITY.md) |
| Original forecast NA values and follow-up analysis | [Version-2 contract](../analysis/forecast_v2/README.md) |
| Choose focused/full; understand rationale and differences | [Experiment profiles](experiment_profiles.md) |
| Run, validate, monitor, stop, and resume on the server | [Server guide](../SERVER_GUIDE.md) |
| Current evidence, pending tests, and acceptance criteria | [Development status and TODO](development_status.md) |
| Scope and unresolved assumptions | [Limitations](limitations.md) |

## Experimental design and interpretation

- [Protocol](protocol.md): task, chronology, masks, selection, and information cuts.
- [Focused profile](server_focused.md) and [full profile](server_full.md): concrete scope.
- [Statistical analysis plan](statistical_analysis_plan.md): endpoints, pairing, support, and multiplicity.
- [Information availability](information_availability.md): the model/evaluator boundary.
- [Methods and energy correction](methods.md): baselines, correction, CPI adaptation, and metrics.
- [Experiment decisions](experiment_decisions.md): dated rationale for scientific changes.
- [Related work](related_work.md) and [reference inventory](references.md): prior work and source pointers.

## Data and reproduction

- [Dataset card](dataset_card.md) and [data dictionary](data_dictionary.md).
- [Reproduction](reproduction.md), [release checklist](release_checklist.md), and [third-party notices](../THIRD_PARTY_NOTICES.md).
- [Optional external validation](external_validation.md): UCI ingestion contract, separate from current profiles.

## Execution and model integration

- [Single server environment](single_cuda_env.md) and [model integration status](model_integration_status.md).
- [GPU execution](gpu_execution.md), [execution optimization](performance_optimization.md), and [GPU benchmarking](performance_batched_gpu.md).
- [CPU worker tuning and audited continuation](runtime_worker_tuning.md): preserve a supported run while changing only correction scheduling.
- [Report performance and continuation](reporting_performance.md): bounded CPU bootstrap, stage progress, and preserved partial statistics.
- [Progress and ETA](progress_and_eta.md), [resume and recovery](resume_and_recovery.md), and [starting a new run](restart_server.md).
- [T1 integration](t1_integration.md): current conference path versus historical adapter.

## Historical records

[Earlier delivery](entrega.md), [model integration pilot](model_pilot.md),
[paired pilot](paired_pilot.md), [isolated MOMENT audit](moment_environment.md),
[isolated TS-ICL audit](tsicl_integration.md), and [environment compatibility audit](environment_compatibility.md)
retain earlier evidence and limitations. They are not current installation instructions.

## Sources of truth

The selected YAML defines a new design; `runs/ID/protocol.lock.json` freezes its
actual settings and environment. Immutable manifests define eligible episodes and
any epoch panel. `state.sqlite` records durable execution. Reports describe committed
coverage and failures. Historical prose and the manuscript draft cannot override
those artifacts. The current entry point is `scripts/server.sh`; direct CLI calls
should pass an explicit conference YAML because their default is the full YAML.
