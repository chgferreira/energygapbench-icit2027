# Historical paired reconstruction and forecasting pilot

**Historical integration evidence.** This supplement is not a focused/full profile
result, convergence study, or confirmatory analysis. Its original private artifacts
and environments remain distinct from current run identities.

`scripts/paired_integrity_pilot.py` completed `paired-real-integrity-v3`: 13 methods,
18 shared queries, 234 reconstructions, and 117 one-hour forecasts. Forecasts came
from nine test issuances; validation queries remained reconstruction-only. Methods
were the six conventional baselines plus BRITS, SAITS, CSDI, ImputeFormer, T1,
MOMENT, and TS-ICL. Local models reused confirmed small-pilot predictions rather
than being retrained: PyPOTS had two epochs/16 training windows, T1 50 adapted steps.

Only isolated, officially pinned MOMENT outputs entered this paired table; earlier
dependency-override outputs were excluded. TS-ICL used its isolated noncommercial
runtime and native median/marginal quantiles. CSDI used the median of ten stored
joint samples. Variant/environment/point-estimator columns preserved those choices.

Source/prediction SHA-256 checks and logical input hashes verified shared support.
Only admitted observed values were reconstructed as inputs; unavailable positions
became NaN, never another method's imputed values. Evaluator truth was checked
separately. Baselines fit only the first 40% elapsed-time training partition.
Training scales matched the stored ridge records. One hash-verified ridge weight
set per site was shared across all 13 methods, using 288 reconstructed positions
and the common causal natural-gap filler.

Test targets were original observations at issuance + 3,600 seconds, within the
90-second allowance and fold test partition. All nine realized offsets were zero.
No interpolated targets were created. A pre-artificial-mask history diagnostic was
not treated as an available gap-recovery method.

Private reports include paired episode metrics, descriptive site/partition/method
summaries, fixed LOCF paired differences, and a support manifest. Negative outcomes
were retained, including large finite errors from short CSDI training. The three
selected industrial test gaps had constant future targets, limiting interpretation;
zero/constant values were not reclassified as failures.

Completed compatible resume preserved 345 transactional records, one session, and
the metric CSV SHA-256
`b02540f4b1121ade9899fd03299922fb0bf93a5a3a13f406d480f5a670050afa`.
The local record is `artifacts/reports/paired_completed_resume.json`. Its original
source/environment identity is needed for exact replay; current code is not an
instruction to resume that historical identity.

One fold, one mask per partition, nine test issuances, and reused short-fit outputs
do not establish general rankings or independent 13-method replications. There
were no confidence intervals, confirmatory tests, or energy correction. Use the
[profile comparison](experiment_profiles.md) and [TODO roadmap](development_status.md)
for the current study and remaining tests.
