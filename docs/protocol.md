# Experimental protocol

**Work in progress.** This is a declaration of new analyses on a previously studied
dataset, not a blind experiment on an unknown dataset. The manuscript draft is
context; the selected YAML, frozen run identity, manifests, and committed coverage
determine the executed study. [Focused and full](experiment_profiles.md) are distinct
designs with different comparative and model-selection budgets.

## Questions and tasks

The focused question is whether admissible boundary energy information and a common
correction improve point reconstruction after communication gaps and subsequent
forecasting. Full adds broader missingness/model comparisons and trajectory-based
analysis. Architecture-wide rankings and uncertainty questions depend on that
expanded scope. Novelty and benefit remain hypotheses; simple-method wins, failures,
and negative findings must remain in outputs.

The executed core task R reconstructs a closed gap at reconnection. Its L=512
context ends at the first admitted received right boundary; later arrivals are
excluded. P supplies power/masks/times/calendar. E additionally supplies admissible
counter endpoints. C (causal, no right boundary), O (explicit offline look-ahead),
and auxiliary A/AE have contracts/tests but separate scientific rankings and are
not additional completed profile results.

## Chronology, masks, and fitting

Sites H1/H2/I1 are distinct installations and periods, not contemporaneous channels
of a microgrid. No battery or separate PV channel is invented. Rolling folds use
elapsed-time fractions:

| Fold | Train | Validation | Test |
| --- | --- | --- | --- |
| 1 | [0, .40) | [.40, .50) | [.50, .65) |
| 2 | [0, .55) | [.55, .65) | [.65, .80) |
| 3 | [0, .70) | [.70, .80) | [.80, 1] |

Only the last endpoint of the final fold is inclusive. Duplicate timestamps cannot
cross partitions. Test histories may precede test; scored future targets must lie
within the fold's test partition. Overlapping training periods/folds are not
independent replications.

C1/C2 are nominal 5%/20% pointwise MCAR masks. C3/C4/C5 are contiguous random
15/60/180-minute gaps under the declared 10% eligible-incidence policy. C6 uses
training-fitted natural-duration/calendar replay with a recorded sparse fallback.
Focused selects C3–C5; full selects C1–C6. Geometry and missingness mechanism are
different concepts. Generate each complete replica's hidden stream before extracting
windows. Original natural missingness and padding never become evaluation truth.
Report realized incidence, capacity exclusions, and support rather than counting
overlapping candidate starts as independent episodes.

Scalers, seasonal statistics, replay patterns, reconstruction scales, and energy
tolerances use train only. Hyperparameters and correction selection use validation.
Training seeds are 11/23/47 for local neural fits; frozen models and baselines do
not acquire fictitious training repetitions. Both profiles use validation masks
1001–1005 and test masks 2001–2010, with uncapped configured episodes/windows.

## Selection and evaluation

HPO uses the first fold of each site and the fixed native grid. Full searches six
candidates with 100-epoch/patience-10 ceilings. Focused searches indices 0/1 with
20-epoch/patience-5 ceilings. Final fits are separate jobs with three seeds and
100-epoch/patience-10 ceilings in both profiles.

Checkpoint selection minimizes equal-duration C3–C5 validation gap nMAE. Full uses
all declared core validation episodes each epoch; focused uses the fixed,
metadata-only stratified panel described in [profiles](experiment_profiles.md).
Candidate ranking always evaluates each best checkpoint on full core validation;
partial-panel scores cannot stand in for that pass. Final validation/test and
correction selection use complete eligible manifests within the selected scope.

The primary comparator B(site, fold) is chosen among the six conventional baselines
on validation P nMAE with equal duration weight, then latency and stable-name ties.
Learned models have descriptive comparisons; they are not silently included in B's
selection pool. Primary aggregation gives equal C3–C5 duration and fold weights;
missing required strata make the estimand unavailable.

## Conditional energy and forecasting

Energy operates on a single signed power trajectory. Instantaneous piecewise-linear
integration includes edge segments and splits zero crossings. Interval means need
declared start/end placement; gross directions require additional within-interval
identifiability assumptions. Unknown measurement semantics prohibit confirmatory
physical interpretation. The three profile branches are explicitly exploratory
hypotheses, not alternatives selected by test accuracy.

Ridge uses train-fitted, validation-selected weights shared across imputers, with
30-minute, one-hour, three-hour, and six-hour targets matched in physical time.
Only original eligible targets are scored; no interpolated future truth is created.
The primary forecast endpoint remains one hour. See the
[statistical plan](statistical_analysis_plan.md).

## Identity and completion

Protocol locks record scientific settings/source and supplied data/environment
identities before evaluation. SQLite records dependencies and durable results;
mask/panel artifacts identify evaluation and checkpoint-selection populations.
Presentation flags do not change numerical science; scientific adaptations require
a new identity and a [decision record](experiment_decisions.md). A local software
lock is not external scientific preregistration.

The shell entry point selects focused by default. Diagnostic smoke/pilot and the
legacy aspirational planner remain separate from either current profile. Calibrate
runtime by phase; cumulative ceilings pause without silently reducing scope.
Completed execution still requires support, failure, physical-evidence, and
manuscript review. Remaining tests are tracked in [development status](development_status.md).
