# Limitations and scope of evidence

**Work in progress.** These limits apply to interpretation of the implemented
profiles. Completion and closure evidence are tracked in the
[development roadmap](development_status.md); a passing software test does not
remove a scientific limitation.

## Scope and selection

Focused evaluates six conventional baselines, SAITS, and MOMENT on contiguous
15/60/180-minute gaps. It has one locally trained architecture, no joint-sample
model, no pointwise C1/C2 analysis, and no C6 replay robustness. Its two-candidate,
20-epoch HPO and fixed epoch-validation panel can change hyperparameter/checkpoint
selection. Full expands those dimensions but still has four unavailable catalogue
candidates and does not by itself establish universal rankings or complete
reproduction. Full and focused are separate designs; do not pool them or select
between them using test performance without disclosing the adaptation.

Both use three installations from different periods on a known, previously studied
base. Three sites do not support broad population generalization. Seeds, masks,
overlapping contexts, and folds do not create independent datasets. Final-training
ceilings are not convergence evidence. Current-profile completed results are not
included in this source release, and historical pilots do not substitute for them.

## Physical and information assumptions

Units/sign/timezone rely on internal evidence, not independent calibration.
Instantaneous versus interval-mean power, timestamp placement, counter metrology,
synchronization, resets/rollovers, and arrival behavior remain unresolved. Energy
branches are conditional exploratory hypotheses. Lowering a constraint residual
alone does not establish a more accurate hidden trajectory or forecast.

Core execution covers R/P and exploratory R/E branches on C3–C5. C/O rankings,
auxiliary A/AE, native irregular-time comparisons, 15-minute-resolution sensitivity,
MAR/MNAR, and partial-truth targets remain separate extensions. C6 uses training-only
replay with a recorded sparse fallback. Gaps exceeding context capacity are excluded
with counts; uncapped configuration does not imply every possible gap is eligible.

CPI is a separately labeled interval-energy donor adaptation, not exact upstream
reproduction or a main instantaneous-power comparator. TS-ICL outputs marginal
quantiles, not joint paths; its native grid lacks 95% interval endpoints and supports
only a truncated quantile approximation to CRPS. Deterministic methods receive no
fabricated uncertainty. Corrected CSDI trajectories do not guarantee calibration.

## Statistical interpretation

Primary inference conditions on fitted models and realized masks. Main 7-day and
2/14-day bootstrap results depend on temporal support and weak-dependence/local-
stationarity assumptions. Training/selection uncertainty is not fully propagated.
Missing strata or invalid bootstrap replicas constrain inference and can leave NA.
Practical-effect thresholds and validation-only power/sensitivity analyses need
justification. Hypothetical energy branches do not receive confirmatory p values.
See the [statistical plan](statistical_analysis_plan.md).

## Execution and recovery

The supported backend is one local POSIX coordinator, SQLite WAL/FULL, and serial
GPU jobs. CPU correction tasks can execute in bounded spawned workers. Multi-machine,
NFS, distributed HPO, and arbitrary training-step continuation are unsupported.
The five native conference training loops have CPU completed-epoch recovery tests;
CUDA recovery/performance must be validated on the target V100. Historical diagnostic
adapters have different limits. CPU timing and VRAM occupancy do not establish
CUDA throughput or bitwise parity.

Setup and initial ingest/mask planning are outside the timed work ledger. Cumulative
ceilings are operational accounting, not strict wall-clock deadlines. Conservative
crash reservations are uncertain, not exact lost-time measurements. Hard unit timeouts,
automatic retry/backoff, full preparation accounting, and complete-run resource
calibration remain incomplete. Indivisible work may overrun a ceiling to a safe
boundary. Partial ETA excludes uncalibrated phases.

Report commands summarize committed evidence; they do not create missing fits or
inferences. Review available/declared coverage, failure/fallback rates, selection
metadata, and absent support before using tables. Operational completion does not
establish completion of the original aspirational specification or manuscript.

## Reproduction and release

Real telemetry, truth, identifying outputs, environments, and downloaded weights
are excluded. Public reproduction of H1/H2/I1 needs permission and an access policy.
The optional UCI adapter is a separate path with activation/DST rules and no independent
energy counters; an external benchmark protocol is still required. Pretraining
exposure to public datasets is unknown. No first-benchmark, first-energy-correction,
SOTA, or paper-acceptance claim is established by this release.
