# Focused experiment profile

**Work in progress.** The focused profile is the default server design for point
recovery and conditional energy correction after communication gaps. Its YAML is
[`server_focused.yaml`](../configs/experiments/server_focused.yaml). The design and
its omissions have task/cost rationales recorded in the project history. Their
declaration does not establish external preregistration or absence of prior data
exposure, and does not imply that omitted methods perform worse.

## Scope

Six conventional baselines, SAITS, and frozen MOMENT-1-base run on H1/H2/I1,
three chronological folds, and C3/C4/C5 (15/60/180-minute gaps). Local SAITS fits use
seeds 11/23/47. All methods share five validation and ten test mask replicas. Context
is 512; no training-window or evaluation-episode cap is configured.

HPO uses native grid indices 0/1: the width-64 SAITS architecture with learning
rates 1e-3 and 3e-4. Each candidate has at most 20 epochs with patience 5, on the
first fold of each site. Final fits use the selected configuration, three seeds,
and at most 100 epochs with patience 10. Training batch 32 and requested inference
batch 256 remain separate controls.

A fixed, metadata-selected epoch panel takes at most 20 episodes from each
(duration, validation-mask-seed) stratum, up to 300 per site/fold. Seed 1729 and
saved membership/hashes make this auditable. The best checkpoint of every HPO
candidate is ranked on all declared C3–C5 validation episodes. Correction selection,
final validation, and test retain all configured eligible episodes.

Four forecast horizons, three exploratory energy hypotheses, and the 10,000-attempt
bootstrap with 7-day blocks and 2/14-day sensitivities are retained. Physical
hypotheses do not become instrument facts. The generic `samples: 100` field remains
for custom probabilistic configurations; no joint-sample method is selected here.

This repository includes the operationally completed focused run
`conference-20261101T165101Z-1789282`. Its protocol lock, rather than this prose,
defines the executed settings. See [ICIT reproduction](icit2027_reproduction.md)
for results, interpretation and unavailable inputs.

## Execute and inspect

```bash
conda activate cuda_env
bash scripts/server.sh setup --preset focused --threads 32
bash scripts/server.sh run --preset focused --threads 32 --budget-hours 24
```

For a prepared installation, `validate native --preset focused --threads 32`
checks SAITS/MOMENT and the relevant CUDA probes. The 24-hour ceiling is cumulative
operational accounting; setup/planning and indivisible work prevent treating it as
a hard wall-clock promise. Monitor and resume with the [server guide](../SERVER_GUIDE.md).

The current private-data plan contains 6 HPO fits, 27 final fits, 891,750 main P
queries, and 13,730 HPO scoring queries. These counts describe planned work.
`summary.json` records the selected profile, omitted methods/conditions, actual
selection budgets, panel evidence, and verified committed coverage.

## Interpretation and remaining work

Focused retains conventional, locally trained, and frozen pretrained comparators,
but only one locally trained architecture. It supports a narrower correction
study than full. Shorter HPO and panel-based checkpoint selection can produce
suboptimal choices. Robustness to those choices, additional model families, and
mixed missingness remain pending tests. No probabilistic calibration or universal
ranking claim is supported by this scope.

Use the [profile comparison](experiment_profiles.md) for exact omissions and
rationale, and the [TODO roadmap](development_status.md) for required evidence.
Before manuscript submission, align the comparator table, HPO/epoch-selection
sections, limitations, and reported coverage with this design.
