# Supplemental energy diagnostics and controls

This analysis is a **post-hoc exploratory extension** of the completed focused
run `conference-20261101T165101Z-1789282`. It does not replace its protocol, fitted
models, masks, selected parameters, or archived results. The original results
were already available when this extension was designed. Validation-only
selection within the extension does not make the extension preregistered.

## What is available

The restricted author compact bundle contains episode-level reconstruction losses for all
methods and hypotheses. It does not contain the original state SQLite database,
per-query corrected trajectories, correction-grid selections, or detailed solver
diagnostics. Equal losses alone cannot identify why an energy correction returned
its base trajectory.

For the present audit, the authors also had the original H1, H2, and I1 telemetry
files and a previously generated set of evaluation manifests. All raw-file
SHA-256 values matched the frozen run's embedded data contracts. Every replayed
manifest's gap IDs, durations, seeds, conditions, partitions, and elapsed-time
fold boundaries were checked against the archived linear P episode losses.
Raw telemetry, manifests and individual derived losses remain private. The public repository includes code, aggregate summaries and aggregate bootstrap estimates only.
The replay records all input and manifest hashes. Matching realized episode IDs
does not assert that the audit manifest files were copied from the server run.

## I1 interval-mean diagnosis

`scripts/diagnose_energy_constraints.py` first audits paired archived outcomes.
If a complete run includes `state.sqlite`, it queries the existing schema using
a read-only connection and exports recorded correction selections and status
counts. It never opens the mutable `StateStore` constructor on the source run.
Without that database, the corresponding JSON files are empty and explicitly
marked unavailable, rather than filled with inferred selections.

With hash-verified original telemetry, the script independently reconstructs
training residuals and verifies the resulting tolerances against the executed
calibration function. It additionally reports empirical q95, sample counts,
and zero-residual counts that the original compact report did not export.

For I1, all 27 export-direction strata across three folds, three durations, and
three measurement hypotheses have empirical q95 equal to zero. They contain
1,288 to 20,844 calibration intervals per stratum. The implemented positive-
tolerance rule therefore returns `tau_export=None` when counter resolution is
unknown, despite ample support. The original fallback reason string combines
this case with insufficient support; the supplemental diagnostic distinguishes
them. Import-direction tolerances remain positive.

The interval-mean wrapper requires both directional tolerances before forming
the net constraint. This reconstructed calibration explains its programmatic
availability gate. Instantaneous correction can retain the import-direction
constraint alone. This is a reconstruction of code behavior on frozen inputs,
not recovery of missing original per-query status logs, confirmation of the
meter's semantics, or evidence that energy cannot help industrial telemetry.
No zero-tolerance repair or retroactive result replacement is made. A revised
zero-residual policy requires its own documented experiment.

Outputs are under `outputs/energy_diagnostics/`. Every one of the 26,690 I1 test
episodes has equal P and interval-mean losses for each of the eight methods and
both interval placements. These equalities are reported separately from the
calibration-based explanation.

## Same-information scalar reconciliation controls

`scripts/replay_linear_energy_controls.py` reconstructs only deterministic linear
P. It first verifies each replayed validation and test episode's MAE and nMAE
against the original archived losses with relative tolerance `1e-10` and absolute
tolerance `1e-12`. No neural model is retrained. Original E comparisons use the
archived `linear+energy_consistent`, `E:instantaneous` losses on the same episode
keys. Original corrected trajectories are not claimed to have been reproduced.

Both controls preserve the two admissible boundary observations and add one
constant displacement `c` to all target-gap positions. They use the original
instantaneous piecewise-linear directional energy operator, including exact
zero-crossing integration and both boundary segments. Every direction admitted
by the original endpoint-availability and positive-training-tolerance rules is
included separately. For I1, this retains import and excludes export because the
export tolerance is unavailable, just as in the original instantaneous policy.

The scalar objective is

\[
J(c)=(c/s)^2+\lambda\sum_{d\in\mathcal D}
\left[\max\left(0,\frac{|I_d(p+c)-\Delta E_d|}{m\tau_d}-z\right)\right]^2.
\]

Here `s` is the training reconstruction scale. Displacement is zero outside the
target gap. `z=1` gives `uniform_shift_tolerant`; `z=0` gives
`uniform_shift_no_deadzone`. Both fix `gamma=0`. Each policy separately chooses
`lambda` from `{0, 0.1, 1, 10}` and tolerance multiplier `m` from `{1, 2}` using
all original validation episodes, equal duration weights, and fixed grid index
for ties. Lambda zero provides exact identity. The chosen parameters are written
before test queries are evaluated. The supplemental protocol is written before
any new test evaluation and identifies this extension as post hoc.

The fidelity term bounds any scalar solution improving identity by
`abs(c) <= s * sqrt(J(0))`. Deterministic bounded searches on both sides of zero
and an identity candidate are used. A global optimum is not claimed. If both
searches fail, identity is retained with an explicit failure status. Scoring
consumes hidden truth only after the control returns a trajectory. Controls see
no interior counters or hidden power. CPU workers limit numerical-library
threads to one; no GPU is needed.

The scalar controls test whether a much simpler energy-informed trajectory
family recovers the observed benefit. Their comparison with the flexible
correction changes trajectory parameterization and smoothness policy together,
so it does not isolate the original smoothness penalty. The two scalar policies
compare deadzone handling within that family, conditional on separate
validation selection. They are not a fixed-parameter causal ablation. A flexible
`gamma=0` ablation and a separately validated zero-export-tolerance policy remain
TODO. No forecast-control replay is included.

The complete replay outputs are under `outputs/energy_controls/`. Aggregation
first averages episode losses within site/fold/duration/policy cells, then gives
the three durations and three folds equal weight. Original P and E losses are
paired on identical test episode IDs. The controls and this script do not
perform inferential testing. Any paired bootstrap analysis is a separate
supplemental artifact with its own support and multiplicity limitations.

## Reproduce public summaries

With the analysis dependencies installed, `bash scripts/reproduce_icit.sh`
rebuilds the scalar-control comparison from published fold/duration means and
verifies its intervals from aggregate bootstrap estimates. It does not read
individual losses or rerun solvers.

Authorized holders of paired losses and their original completion/protocol
markers can perform the stronger loss-level summary check separately:

```bash
python analysis/scripts/summarize_energy_controls.py \
  --controls-dir /authorized/energy_controls \
  --output-dir restricted_analysis/control_summary
```

That command checks paired support, duplicate episode keys, original hashes and
all required strata. Its individual inputs and LaTeX output are excluded from the
public repository. Historical manifests describe the full author workflow.

## Reconstruct diagnosis on the original server

Run from the original experiment repository, using a publication copy that
contains these scripts. Use a new supplemental output directory outside the
frozen run. This command does not resume the experiment or modify its state.

```bash
PYTHONPATH=src python analysis/scripts/diagnose_energy_constraints.py \
  --run-dir runs/conference-20261101T165101Z-1789282 \
  --data-root . --sites I1 \
  --output-dir restricted_analysis/energy_diagnostics
```

Omit `--data-root` for a compact-loss/recorded-status audit without recalibration.
The source package used by the script must be the archived publication source,
whose digest is recorded in every validation record.

## Replay both controls on original private data and masks

```bash
PYTHONPATH=src python analysis/scripts/replay_linear_energy_controls.py \
  --run-dir runs/conference-20261101T165101Z-1789282 \
  --data-root . \
  --manifest-dir runs/conference-20261101T165101Z-1789282/manifests \
  --output-dir restricted_analysis/energy_controls --workers 8
```

Use the full run's existing manifests. The local audit used an independently
generated manifest directory and explicitly verified realized episode identity.
For a timing-only validation check, use `--profile-only` and a separate temporary
output directory. Profiling never evaluates test episodes and cannot substitute
for a completed replay. The complete comparison requires all three sites,
three folds, three durations, five validation mask seeds, and ten test mask seeds.
Interrupted supplemental runs remain marked incomplete. Re-run into a fresh
output directory; no original training or experiment results need to be repeated.

## Tests and remaining work

```bash
python -m pytest -q analysis/tests/test_energy_replay.py
```

Tests cover an independent analytic positive-flow solution, zero-crossing
sensitivity to both counter directions, observed-boundary preservation, identity
and failure fallbacks, zero export q95 with ample support, read-only access to a
real temporary ledger schema, and refusal to write inside an immutable run.

TODO items are the flexible correction's fixed-gamma ablation, validation of a
revised zero-residual calibration policy, original solver-log export, downstream
forecast comparisons for the new controls, independent metrological evidence,
and external validation. None is implied by passing software tests or completing
this descriptive replay.

## Paired exploratory control contrasts

The separately versioned `statistics/` supplement compares archived flexible
linear E minus each scalar control on exactly the same 68,580 test episode
instances. It verifies all saved validation selections and archived P/E values,
then applies the calendar-day bootstrap verified by the forecast v2 analysis.

```bash
python analysis/scripts/energy_control_contrasts.py \
  --run-dir /authorized/run \
  --controls-dir /authorized/energy_controls \
  --calendar-manifest /authorized/calendar_provenance.json \
  --output-dir restricted_analysis/control_statistics
```

Negative differences favor the flexible correction. At H2 the seven-day
intervals exclude zero against both scalar policies, with unadjusted exploratory
p-values of approximately 0.00010 (without a dead zone) and 0.00040 (with one).
At I1 the no-dead-zone scalar policy has slightly lower mean error, but the
flexible-minus-scalar interval includes zero and p is approximately 0.253.
This does not establish equivalence. H1 fails the seven-day temporal-support
heuristic and has no reported p-value at that block length.

The supplement retains 10,000 attempts per contrast, two- and fourteen-day
sensitivities, all draws, invalid-draw counts and support flags. Eighteen
site/control/block-length summaries are exploratory and unadjusted for
multiplicity. They condition on fitted and validation-selected policies and
realized masks, and are not independent confirmation or a factorial ablation.
No source run or replay input is changed.
