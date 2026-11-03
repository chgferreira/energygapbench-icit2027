# Methods, energy correction, and metrics

Methods receive already masked values and physical-time timestamps. They must
preserve admissible observations and cannot receive evaluator truth or an original
unmasked window. `fit(..., split="train")` rejects other partitions; the caller
still owns correct training isolation. Selected methods differ by
[profile](experiment_profiles.md); the complete implementation catalogue is in
[model integration status](model_integration_status.md).

## Conventional references

| Identifier | Implementation and information boundary |
| --- | --- |
| `median` | Median of originally observed training values. |
| `locf` | Last admitted observation; training median before the first usable window observation. |
| `linear` | Physical-time interpolation between admitted boundaries; no extrapolation, and incompatible with causal task C. |
| `seasonal` | Training median by weekday/five-minute slot; fallback to slot, weekday, global median; at least 20 observations per group. |
| `kalman_filter` | Daily three-pair harmonic regression plus residual local-level state space, training-fitted variances, process variance scaled by physical elapsed time. |
| `kalman_smoother` | Separately identified RTS smoother within the already issuance-cut query; incompatible with C. |

Zero, negative, and extreme finite observations are retained. A common causal filler
for remaining natural gaps in the forecast history is separate from the imputer's
reconstruction and must be identified. Unknown method names are not silently
converted to a simple baseline.

## Native learned and frozen models

Full uses native BRITS/SAITS/CSDI/ImputeFormer via PyPOTS, pinned official T1, and
frozen MOMENT/TS-ICL. Focused selects SAITS and MOMENT. Models use the same admissible
L=512 core context, training-only scalers where applicable, balanced training masks,
and the profile's declared validation/HPO policy. Implementation orchestration is
not a claim to reproduce each source paper's complete training study.

MOMENT uses native masked reconstruction with observed-only normalization and no
local training. TS-ICL uses its pretrained imputer and native median/marginal
quantiles. CSDI supplies joint trajectories; deterministic point outputs are not
given fabricated distributions. See [integration status](model_integration_status.md)
and [T1 details](t1_integration.md).

## Energy operator and admissible constraints

`integrate_signed` takes kW and seconds and returns import, export, and net kWh.
For instantaneous piecewise-linear power it integrates positive/negative parts,
splits zero crossings, and includes both adjacent edge segments. This numerical
representation is not an independent energy measurement.

Interval-mean integration requires explicit start/end placement. Gross import and
export cannot be recovered from net interval means under unknown within-interval
flow reversals; additional confirmed assumptions are required. One signed trajectory
is used, not two independent freely adjustable directions. Negative counter changes
are not silently reduced modulo a guessed rollover. Known resets invalidate an
increment; rollover requires documented modulus. A constant counter can be valid.

Counter endpoints must have arrived by issuance and pass validity checks. Unknown
semantics keep the three profile hypotheses exploratory. No reliable constraint
means the base reconstruction is retained with an explicit status.

## Common correction layer

`correct_energy` changes only the target contiguous gap, holding boundaries and
other positions fixed. Its objective combines quadratic deviation from the base,
energy-tolerance excess, and smoothness of the correction. Training supplies scale
and calibrated tolerance; validation selects hyperparameters. `lambda_energy=0`
returns exact identity.

The general L-BFGS-B path uses deterministic base/seasonal/energy-scaled starts,
exact physical evaluation, and piecewise analytic gradients. Zero segments use a
symmetric subgradient. The available objective chooses the result, never truth
error. The general instantaneous problem is nonconvex and has no global-optimum
guarantee. A base identity candidate prevents numerical objective worsening;
failed solvers retain base and status. Diagnostics record starts, convergence,
iterations, objectives, residuals, and service cost. A single-net interval-mean
case has the exact convex solution described in
[execution optimization](performance_optimization.md).

`calibrate_tolerance` uses train-only residuals by duration/direction: an absolute
95th percentile with at least 30 intervals, then pooled training residuals, then
known counter resolution as an explicitly uncalibrated fallback. Unknown resolution
and insufficient support block the constraint; tolerance is not represented as
instrument accuracy. Calibration uses originally observed valid training intervals,
without claiming independent support across durations.

The common validation grid contains 16 combinations: energy weight {0, .1, 1, 10},
smoothness {0, .01}, and tolerance multiplier {1, 2}. Selection averages seed losses
before episode/duration aggregation, with equal C3–C5 duration weights. All selected
methods receive the same correction layer and search policy; fitted parameters may
differ by method. Candidate rows and calibration/selection diagnostics remain saved.

Point correction and probabilistic sample correction have distinct identities.
`correct_joint_samples` requires declared joint samples, corrects each trajectory,
and recomputes summaries/scores. Marginal quantiles are not promoted to paths.
Corrected medians and per-sample energy residuals differ; transformation does not
guarantee calibration or improved hidden-value accuracy.

## Separate CPI energy track

`cpi_energy_donor_adapted` is an explicitly named original adaptation inspired by
[Copy-Paste Imputation](https://github.com/KIT-IAI/CopyPasteImputation/tree/c1ec6facaf0a762d7e3c22ad988b26d0361085f9).
The inspected upstream uses cumulative daily energy, Prophet daily estimates, and
similar-day scaling. Its LGPL source is not copied into the MIT package.

The adaptation freezes complete originally valid training days, selects a donor
using weekday/season/known boundary energy, scales nonnegative increments to the
admitted gap total, and reintegrates while preserving received counters. Missing
donors, delayed/reset counters, irregular grid, or train/query overlap block it.
Material differences include segment matching and known boundary totals rather
than Prophet prediction. Output is cumulative energy and interval-mean power,
not recovered instantaneous power. It is not exact upstream reproduction and is
outside the main R/P ranking. Independent upstream parity remains TODO.

## Metrics and existing checks

Only originally observed, artificially hidden, nonpadding targets are scored.
Training q95−q05 normalizes error, falling back to positive standard deviation;
constant scale, absent support, and zero-denominator skills yield NA with reasons.
Metrics include MAE/RMSE/bias, gap-macro versus point-micro loss, peak/time/ramp
errors, sign/extreme classification, empirical-sample CRPS, pinball, coverage,
width, interval score, and event Brier where supported. Marginal CRPS does not
establish coherent joint trajectories.

Synthetic tests cover observed preservation, fitting isolation, physical-time
interpolation, filter/smoother differences, zero crossings/edges/irregular dt,
reset/rollover rules, tolerance, solver failures, identity correction, analytic
metrics, and sample transformation. They do not establish site-level scientific
performance. Pending server, sensitivity, and scientific checks are listed in
[development status](development_status.md).
