# Development status and remaining work

**Study status, updated 3 November 2026.** The manuscript is **under review at
IEEE ICIT 2027**. The focused execution is operationally complete, with versioned
reconstruction, forecast and energy-control analyses. The public repository
contains code, documentation, aggregate results and analysis scripts. The
manuscript and individual records are maintained separately.

Further scientific validation is work in progress. The full design, physical
confirmation, selection sensitivities and external validation remain future work.
Data are available upon reasonable request to the corresponding author or
[repository maintainer](https://github.com/chgferreira), subject to applicable
confidentiality and data-holder requirements. See [data availability](../DATA_AVAILABILITY.md).

## Evidence and its limits

| Evidence | What it supports | What it does not establish |
| --- | --- | --- |
| Original frozen protocol and runtime history, retained separately; public protocol extract | Actual scope, environment, planned budgets and continuation provenance | External preregistration, blindness to previously studied data or complete raw-data reproducibility |
| Original summary/completion records and published aggregate audits | Operational completion, 33 recorded fits, no fit failures and no unresolved jobs | Correctness or adequate support of every scientific endpoint |
| Published aggregates and contrasts; individual losses retained separately | Public figure/table reproduction and authorized loss-level reanalysis | Complete prediction/solver audit, all forecast-horizon reanalysis or raw trajectory recovery |
| Analysis input checksums and regenerated tables/figures | Traceability from archived outputs to manuscript claims | Independent regeneration of those outputs from the original telemetry |
| Synthetic, leakage, recovery and CPU regression tests | Behavior on tested fixtures and environments | V100 numerical equivalence for every model or universal model performance |
| Recorded V100 runtime | Device/package settings used by this run | Hardware-independent determinism, saturation or a guaranteed 24-hour completion |

The original report retains `scientific_complete=false`. This field is preserved
as historical evidence. Scientific interpretation is documented in the manuscript
and analysis audit, not retroactively inserted into the frozen protocol or ledger.
An unavailable planned endpoint must not be replaced by an unlabelled post hoc one.

## Priority 1: close the focused-study audit

- [x] Inspect the supplied primary downstream endpoint and preserve its NA values.
  The original complete-outcome rule and its unavailable values remain archived.
- [x] Complete the separately versioned common-target forecast analysis with
  paired calendar-day intervals and explicit exploratory, conditional interpretation.
- [x] Add same-information scalar energy controls and paired exploratory contrasts,
  retaining unequal search budgets and the limits of component attribution.
- [x] Reconstruct the I1 training-calibration gate and distinguish that code-level
  mechanism from absent original per-query solver logs and meter documentation.
- [ ] Complete the target-missingness audit from full records and assess sensitivity
  before making stronger downstream claims or changing the planned estimand.
- [ ] Obtain the missing full forecast/episode records, selection diagnostics and
  solver reasons if needed to close detailed failure/fallback and multi-horizon
  audits. The compact transfer cannot answer every question.
- [ ] Check selected epochs, boundary selections, histories and seed variation.
  A 100-epoch cap and zero fit failures do not demonstrate convergence.
- [ ] Retain full mask/panel manifests and the original ledger/checkpoints on the
  server, then reconcile planned versus realized incidence and exclusions.
- [ ] Reconcile any changes arising during peer review with the versioned analysis,
  numerical interpretation, references and documented limitations.

## Priority 2: selection and robustness sensitivities

- [ ] Predeclare an epoch-panel sensitivity to panel size, panel seed or full epoch
  validation on a limited validation-only subset. Record checkpoint and candidate
  changes rather than assuming equivalence.
- [ ] Compare the two-candidate, 20-epoch HPO budget with a broader or longer search
  on a declared subset. The current short budget can weaken the learned comparator.
- [ ] Study directional versus net correction with the same measurement operator.
  Comparing instantaneous-directional and interval-mean-net branches changes both.
- [ ] Extend missingness mechanisms and include a separately identified additional
  architecture when resources permit. Do not select the extension solely because
  it improves the inspected test outcome.

These are future experiments after inspection of current results. Disclose that
chronology and use new held-out evidence when needed. They are not retrospectively
prespecified confirmatory checks.

## Priority 3: physical validity, external evidence and release

- [ ] Obtain instrument evidence for power semantics, timestamp placement, meter
  precision/resolution, counter cadence, resets, synchronization and arrival times.
  Internal power/counter agreement is not independent metrological calibration.
- [ ] Justify an application-specific practical-effect threshold. The conditional
  day-block bootstrap does not capture all training, selection or site uncertainty.
- [ ] Evaluate a separately frozen external installation or dataset with appropriate
  independent energy information. Three known case studies do not establish
  population-wide performance or real outage reconstruction accuracy.
- [x] Publish code, documentation and aggregate results with an explicit access
  policy. The manuscript and individual records remain outside the public repository.
- [ ] Evaluate individual data requests under the applicable confidentiality and
  data-holder requirements. The code license does not license telemetry,
  manuscript content or third-party weights.
- [ ] Prepare an immutable tagged release and an archival deposit when appropriate.
  Repository publication, archival deposit and DOI assignment are distinct actions.

## Priority 4: expanded full design

- [ ] Profile and execute the separately frozen full matrix under its declared
  budgets, methods and conditions. Do not mix historical pilots into focused results.
- [ ] Validate CSDI joint-sample correction and probabilistic calibration on the
  server. Do not manufacture predictive uncertainty for point-only outputs.
- [ ] Resolve or retain explicit integration/license blockers for FIM, TabPFN-TS,
  MoTM and CBiT. Validate any added implementation against its claimed upstream path.
- [ ] Review TS-ICL noncommercial terms before any intended use or redistribution.
- [ ] Validate upstream parity of the separately named CPI-inspired energy track.

Engineering TODO includes broader storage/hardware profiling, exact whole-pipeline
resource accounting, robust retry/timeouts and distributed-filesystem behavior.
These are distinct from changing the experiment's numerical/statistical design.
