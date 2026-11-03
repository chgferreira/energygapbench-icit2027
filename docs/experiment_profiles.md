# Focused and full experimental designs

**Work in progress.** Both designs are implemented; complete scientific results,
hardware measurements, and publication claims require the evidence listed in the
[development roadmap](development_status.md). Profile names describe scope and do
not certify completion, convergence, or a runtime guarantee.

## Purpose and rationale

The focused design addresses point recovery after contiguous communication gaps
and the incremental effect of admissible energy information and correction. It
retains conventional, locally trained, and frozen pretrained comparators while
reducing repeated model selection and the broadest method extensions. This makes a
complete first study more feasible under a constrained single-GPU budget.

The full design expands model families, missingness conditions, and probabilistic
analysis. It is available for a separately declared study when resources permit.
The documented rationale for focused omissions is task relevance and compute cost.
This rationale is not proof of blindness to previously studied data or external
preregistration. Neither design establishes that omitted methods are inferior.

## Design comparison

| Dimension | Focused | Full |
| --- | --- | --- |
| Configuration | `configs/experiments/server_focused.yaml` | `configs/experiments/server_full.yaml` |
| Study label | `focused_24h` | Expanded available-method study |
| Entry-point default | Yes | Explicit `--preset full` |
| Sites | H1, H2, I1 | Same |
| Rolling chronological folds | 1, 2, 3 | Same |
| Conditions | C3, C4, C5 | C1, C2, C3, C4, C5, C6 |
| Baselines | median, locf, linear, seasonal, kalman_filter, kalman_smoother | Same |
| Locally trained models | SAITS | BRITS, SAITS, CSDI, ImputeFormer, T1 |
| Frozen pretrained models | MOMENT-1-base | MOMENT-1-base, TS-ICL |
| Executable methods if required artifacts pass preflight | 8 | 13 |
| Declared unavailable candidates | Omitted by scope, catalogue status retained | FIM, TabPFN-TS, MoTM, CBiT explicitly declared unavailable |
| Training seeds | 11, 23, 47, for SAITS | Same, for each locally trained model |
| Validation mask seeds | 1001–1005 | Same |
| Test mask seeds | 2001–2010 | Same |
| Context / training stride | 512 / 288 positions | Same |
| Episode / training-window cap | None | None |
| Training / requested inference batch | 32 / 256 | Same |
| HPO candidates, first fold of each site | Indices 0/1 of the fixed native grid | All six indices 0–5 |
| HPO epochs / patience | At most 20 / 5 | At most 100 / 10 |
| Final-fit epochs / patience | At most 100 / 10 | Same |
| Repeated epoch validation | Up to 20 episodes per duration/mask seed; at most 300 per site/fold | All declared core validation episodes |
| Final HPO candidate ranking | Full C3–C5 validation manifests | Same |
| Correction selection and final evaluation | Full manifests within declared scope | Same |
| Forecast horizons | 30 min, 1 h, 3 h, 6 h | Same |
| Energy hypotheses on C3–C5 | Instantaneous, interval-mean-start, interval-mean-end; exploratory | Same |
| Joint probabilistic samples | No joint-sample model selected | CSDI: 100 trajectories, 50 native diffusion steps |
| Bootstrap | 10,000 attempts; 7-day blocks; 2/14-day sensitivities | Same |
| Default cumulative operational ceiling | 24 h | 168 h |

C1/C2 are pointwise MCAR masks at nominal 5%/20%; C3/C4/C5 are random contiguous
15/60/180-minute gaps with the declared eligible-incidence policy; C6 replays
training-derived gap/calendar patterns with a recorded sparse-support fallback.
Realized eligibility/incidence must be reported. Three sites × three folds × three
conditions give 27 focused site/fold/condition cells, not 27 independent datasets.
Energy hypotheses and forecast horizons are analysis branches, not independent
replications. Training seeds apply only to local fits, not to baselines or frozen models.

## Why each extension is deferred in focused

| Component | Rationale and resulting limitation |
| --- | --- |
| BRITS | Observed recurrent training cost motivated deferral; recurrent-model transfer of correction remains less tested. |
| CSDI | Diffusion and trajectory-wise correction expand computation substantially; focused supports point recovery, not calibration claims. |
| ImputeFormer and T1 | Deferred expanded learned-model comparison; focused has only one locally trained architecture. |
| TS-ICL | Deferred serial pretrained comparison; one frozen pretrained comparator remains. |
| C1/C2 | Pointwise absence is outside the focused contiguous communication-gap question. |
| C6 | Training-derived mixed-pattern robustness is deferred. |
| FIM, TabPFN-TS, MoTM, CBiT | Existing implementation/artifact/license verification gaps remain separately recorded in the registry. |

## Fixed epoch panel and selection safeguards

The focused panel is selected once per site/fold from validation metadata only:
duration, mask seed, and gap ID. Stable SHA-256 ranking uses seed 1729 and takes at
most 20 episodes in each duration/mask stratum. With three durations and five masks,
the maximum is 300. The same panel is shared across trials, models, and training
seeds at that site/fold. No value, prediction, validation error, or test outcome
selects its membership.

The panel selects checkpoints and triggers early stopping. Each HPO candidate's
best checkpoint is then scored on the full declared core validation population.
Correction hyperparameters and final validation/test also use full configured
manifests. A partial-panel score cache cannot replace that complete scoring pass.
Panel identities, membership, coverage, selection budgets, and hashes are saved.
Changing these choices requires a new experiment identity.

The smaller panel can select a different epoch, and a shorter/two-candidate HPO
search can choose weaker hyperparameters. These are scientific tradeoffs, not
numerical-equivalence optimizations. The final 100-epoch limit is a ceiling, not a
convergence demonstration. Sensitivity checks remain TODO.

## Planning cardinalities and runtime interpretation

An audit of the currently supplied private CSV files produced the following plan.
Counts depend on dataset hashes, eligibility, configuration, and model availability;
regenerate them for another dataset or implementation.

| Planned work | Full | Focused |
| --- | ---: | ---: |
| Mask manifests | 810 | 405 |
| Jobs | 19,001 | 4,168 |
| HPO fits | 90 | 6 |
| Final fits | 135 | 27 |
| Total fits | 225 | 33 |
| Main P inference queries | 13,973,535 | 891,750 |
| HPO scoring queries | 205,950 | 13,730 |
| Confirmable query-counter total | 14,179,485 | 905,480 |

The fit reduction is 85.3%; the confirmable-query reduction is 93.6%. These are
planning counts, not completed experiments, speedup measurements, or predictions
of finishing within 24 hours. Epoch-validation calls, correction branches, forecast
exports, and reporting are not additional independent units in that query total.
The nine focused panels each contained 300 episodes in the audited plan; full
candidate counts ranged from 1,625 to 2,685 per site/fold.

## Configuration and identity

Use `--preset focused` or `--preset full` with setup/run/validate. To customize,
copy a YAML and pass `--config PATH`; no focused overrides are silently applied.
Keep `hpo_trials` consistent with the number of distinct `hpo_trial_indices` (0–5).
`hpo_max_epochs` and `hpo_patience` control HPO separately from final `max_epochs`
and `patience`. `epoch_validation_max_per_stratum: null` selects full repeated
validation; a positive integer caps each stratum. `epoch_validation_seed` fixes
panel selection. Local neural training requires C3/C4/C5 for the implemented
selection objective; context remains fixed at 512.

A custom example starts with a real copy:

```bash
cp configs/experiments/server_focused.yaml configs/experiments/custom_study.yaml
# Edit the copied YAML and declare its scope before evaluation.
bash scripts/server.sh setup --config configs/experiments/custom_study.yaml --threads 32
bash scripts/server.sh run --config configs/experiments/custom_study.yaml --budget-hours 24
```

Resume uses the saved configuration, never a new preset. If a change follows
inspection of test results, declare that adaptation and use new held-out evidence
where appropriate; a new run ID alone does not restore test blindness. Do not
silently pool focused/full results or choose the reported profile by test performance.

CPU correction worker/queue counts can use the separately audited
[runtime continuation path](runtime_worker_tuning.md) without changing the model
or scientific selection design. Its effective settings and release lineage must
accompany computational comparisons. This exception does not apply to training or
inference batch changes.

## Manuscript reporting requirements

State the executed profile, methods, conditions, HPO grid and budgets, epoch-selection
population, full evaluation coverage, exclusions, failures, and data semantics.
Describe correction against the same base and a simple corrected reference using
paired reconstruction and forecasting endpoints. Report negative findings and
site/duration heterogeneity. Focused results do not establish universal model
rankings, broad architecture transfer, or probabilistic calibration. Completing
full also does not establish reproduction of every catalogue entry or resolve
physical metrology. Update the manuscript's comparator table and training/selection
sections to the executed design before submission.
