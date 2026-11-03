# ICIT 2027 experimental and operational supplement

This supplement records details moved out of the six-page manuscript. The frozen
run `conference-20261101T165101Z-1789282` remains authoritative. Analysis revisions
are separately identified and do not rewrite its protocol or original outcomes.

## Executed configuration

| Component | Frozen focused execution |
| --- | --- |
| Sites | H1, H2 and I1, evaluated separately |
| Fold 1 | Train [0, .40), validation [.40, .50), test [.50, .65) |
| Fold 2 | Train [0, .55), validation [.55, .65), test [.65, .80) |
| Fold 3 | Train [0, .70), validation [.70, .80), test [.80, 1], last endpoint inclusive |
| Time basis | Fractions of elapsed time, before windows or masks |
| Conditions | C3/C4/C5, contiguous 15/60/180-minute gaps, eligible incidence policy 10% |
| Mask seeds | Validation 1001–1005, test 2001–2010 |
| Context and stride | 512 positions and 288-position training stride |
| Training windows | No configured cap; eligibility still applies |
| SAITS | Two layers, width 64, four heads, d_k=d_v=16, feedforward width 128, dropout .1 |
| SAITS HPO | Learning rates .001/.0003, first fold per site, seed 11, 20 epochs, patience 5 |
| Final fits | Seeds 11/23/47, 100 epochs, patience 10, independently fitted by site/fold |
| Completed fits | Six HPO fits and 27 final fits, no recorded fit failure |
| Epoch panel | At most 20 episodes per duration/mask-seed stratum, maximum 300 per site/fold |
| Panel selection | Deterministic SHA-256 ranking, seed 1729, metadata only |
| Selection scoring | HPO candidate ranking and correction selection use complete eligible validation |
| Batch requests | Training 32; episode validation/inference 256, smaller final or recorded OOM-adjusted batches |
| Correction grid | Lambda 0/.1/1/10, gamma 0/.01, tolerance multiplier 1/2 |
| Forecast history | 288 five-minute positions, known calendar features, common causal fill |
| Ridge regularization | .1/1/10, chronological validation |
| Forecast horizons | One hour primary, 30 minutes/three hours/six hours secondary |
| Bootstrap | 10,000 attempts, base seed 20261031, calendar-day blocks 7 with 2/14 sensitivity |

Training corruption chooses equally between 20% point masking and blocks of
1/3/12/36 positions; half of blocks end before reconnection. Training-based model
normalization is distinct from the q95-minus-q05 reconstruction-score scale.

## Operational evidence

The run records Tesla V100-PCIE 16 GB, Python 3.12.16, PyTorch 2.9.1+cu126 and
CUDA runtime 12.6. Main thread requests were 32. Correction scheduling changed
from eight workers/16 pending items to 16 workers/32 pending items across recorded
sessions. Reporting later used 16 CPU workers. These are scheduling settings,
not independent experiment repetitions or claims of resource saturation.

The summary reports 2,949,150 inference records and 164 fallbacks, or approximately
0.00556% across validation and test. This is not a test-only solver-failure rate.
Accounted active time is 20.40 hours; calendar duration is 28.77 hours, including
interruptions and four resumptions. Neither is an uninterrupted 24-hour guarantee.
Per-query failure attribution still requires the original ledger.

Final SAITS fits used 100–292 training windows, completed 11–33 epochs and selected
checkpoints at epochs 1–23. These observations do not demonstrate convergence or
justify a broad ranking of architectures. Reduced HPO and panel sensitivities
remain future experiments.

## Analysis revisions and verification

The initial independent analysis reproduces 864 reconstruction cells, 96 site
aggregates and the original primary point estimates, preserving unavailable
forecast values. Version 2 uses 68,217 common-target episode instances and adds
conditional paired forecast intervals. Calendar metadata derived from
hash-verified source timestamps reproduces all 54 original reconstruction
intervals and support flags before forecast inference.

The additional energy-control replay uses all 68,580 test episode instances.
All 89,175 validation/test linear-P losses are matched before comparing scalar
controls with the archived flexible correction. Each scalar policy selects eight
candidates on validation; the original flexible policy used sixteen. Paired
control inference is a separately identified post-hoc supplement, not a factorial
ablation or independently confirmed test.

The I1 calibration audit reconstructs a zero-export-tolerance gate from original
training data and unchanged source. It does not recover original selected
parameters or solver statuses. A modified nonnegative directional-tolerance rule
requires a separately validated analysis before changing the original policy.

For executable commands, input hashes, complete results and statistical limits,
see [analysis](../analysis/README.md), [forecast v2](../analysis/forecast_v2/README.md),
[energy controls](../analysis/energy_controls/README.md), and the
[reproduction guide](icit2027_reproduction.md). Passing software and replay checks
establishes implementation consistency, not instrument semantics or universal
statistical coverage.
