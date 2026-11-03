# Version 2 forecast report

This versioned analysis revises one-hour forecast reporting for `conference-20261101T165101Z-1789282`. It includes **68,217 episode instances with common originally observed future targets from 68,580 eligible episode instances**, with all 32 evaluated policies on identical support. It preserves the immutable execution and the original report.

Version 1 required every episode outcome to be available before reporting a site aggregate or a primary paired forecast contrast. Missing future targets therefore propagated to unavailable site scores and contrasts. An earlier descriptive supplement reported conditional point estimates on available targets. Version 2 formalizes that population, retains all prescribed fold/duration weights, and adds explicitly exploratory paired calendar-day intervals.

This is a reporting revision made after the completed experiment was inspected. It is not a new preregistered or confirmatory analysis and does not retroactively change the original endpoint status. Missing targets are not imputed and their losses are not replaced by zero.

## Public reproduction and restricted reanalysis

The public package contains aggregate scores, support counts and bootstrap
contrast estimates. Run `bash scripts/reproduce_icit.sh` with the analysis
Python environment to verify intervals and regenerate aggregate figures/tables.
No episode records, membership, full calendars or manuscript files are included.

For authorized loss-level resampling, supply the restricted original run and
calendar metadata separately:

```bash
python analysis/scripts/forecast_common_target_v2.py \
  --run-dir /authorized/run \
  --calendar-manifest /authorized/calendar_provenance.json \
  --output-dir restricted_analysis/forecast_v2
```

This path needs restricted derived losses and calendar metadata, but no GPU,
training or raw telemetry. It verifies original reconstruction intervals before
calculating forecast intervals and preserves original input hashes. Independently
regenerating the calendar metadata additionally requires authorized raw files.

Run the independent weighting, pairing, missing-target and resampling tests with:

```bash
python -m unittest discover -s analysis/forecast_v2/tests -v
```

## Calendar evidence and verification

The compact run export did not include the original split manifests. Full day calendars have therefore been reconstructed from the local source timestamp files after verifying each file's SHA-256 against the frozen protocol. The restricted `calendar_provenance.json` records the source hashes, source row counts, time zone, aligned endpoints, exact fold boundaries and every calendar date. It contains no power or counter measurements.

The derivation follows the original rules. Timestamps are assigned to the nearest five-minute slot with the floor-based tie convention and a 90-second tolerance. The minimum and maximum accepted grid times define the elapsed-time span. The three chronological fold fractions are applied to this span. The report's day calendar includes the dates from `validation_end_exclusive` through `test_end`, including empty days and partial endpoint days. Calendar dates are never inferred from the minimum or maximum observed episode day. Blocks never bridge folds.

An authorized holder of the original raw files can independently regenerate the calendar evidence:

```bash
python analysis/scripts/derive_forecast_calendars_v2.py \
  --run-dir /authorized/run \
  --raw-root /path/to/original/project \
  --output-json restricted_analysis/calendar_provenance.json
```

The source files must be located at the frozen relative paths under that root. A changed input hash, ambiguous local timestamp, changed alignment rejection count or differing existing calendar manifest stops this derivation.

Before forecasting inference, the analysis reproduces **all 54 original reconstruction intervals** across the three sites, three physical assumptions, two contrasts and three block lengths. It also checks the original point estimates, invalid-draw counts, occupied-day counts and temporal-support flags. The observed maximum interval-endpoint difference is below `7e-18`. This validates consistency with the original calendar and resampling implementation; it is not proof of nominal frequentist coverage or meter semantics. The original server split manifests remain desirable provenance for the final archive.

## Population, pairing and weights

Each episode-policy input contains the mean of losses over the declared training seeds, rather than a loss computed from an averaged trajectory. SAITS has three seed evaluations and every other method has one. These multiplicities are checked.

An episode enters the revised forecast population only when the one-hour forecast losses are finite for **all 32 policies**. The observed missingness pattern is either available for every policy or absent for every policy. Any policy-specific pattern aborts the script for investigation rather than silently changing the comparison population. This filter depends on availability, not the magnitude or sign of errors.

All three durations and all three folds remain represented at each site. Episodes receive equal weights within a fold/duration cell, the three durations receive equal weights within each fold, and the three folds receive equal weights within each site. Sites remain separate. The same original validation-selected conventional baseline is used in each fold. No method, correction setting, physical hypothesis or threshold is selected using test performance.

Exclusions are 89 of 16,170 episode instances for H1, 226 of 25,720 for H2 and 48 of 26,690 for I1. These counts include mask realizations and are not independent installations or independent temporal samples. The full script exports per-cell counts and restricted per-episode membership. Only the aggregate counts are published.

## Exploratory interval calculation

For each site, paired loss differences are reduced to sums and counts by fold, reconnection day and duration. The sampling calendar includes dates with zero eligible forecast targets. Moving blocks of calendar days are resampled independently within each fold, preserving all associated episodes and the same resample across physical hypotheses and contrast columns.

The main block length is seven days. Two- and fourteen-day blocks are sensitivity analyses. Each calculation performs exactly 10,000 attempts with base seed `20261031`. Replicate `i` uses the first eight bytes of the SHA-256 of `20261031|i`, interpreted as an unsigned big-endian integer. Invalid draws retain their ID and are not redrawn. The aggregate bootstrap contrast estimates are supplied in a compressed NumPy archive.

Intervals are pointwise 95% percentile intervals, conditional on observed targets, fitted models, validation-selected policies and realized masks. They do not propagate full training or model-selection uncertainty, correct target-missingness bias, or establish cross-installation generalization. No multiple-comparison correction is applied. All `p_exploratory` values are unadjusted, `p_raw` and `p_holm` remain absent, and confirmatory flags are false. There are 54 exploratory combinations across sites, contrasts, physical assumptions and block lengths. No familywise-error or false-discovery-rate control is claimed.

The original conservative support heuristic is retained. It requires at least 20 block lengths over the folds, at least four per fold, at least 20 occupied days for every duration and zero invalid draws. H1 fails the seven-day support check, so its displayed interval is qualified and its exploratory p-value is absent. These rules are safeguards rather than a coverage theorem.

For instantaneous correction, the seven-day energy contrast `B(E) - B(P)` is approximately:

| Site | Difference in nMAE | 95% pointwise interval | Declared temporal support |
| --- | ---: | --- | --- |
| H1 | +0.003002 | [+0.002064, +0.003484] | Insufficient |
| H2 | +0.000436 | [−0.001240, +0.001405] | Sufficient by the stated heuristic |
| I1 | −0.009010 | [−0.015900, −0.002985] | Sufficient by the stated heuristic |

Positive differences indicate worse forecast loss after correction. The observed instantaneous relative changes are +1.62%, +0.20% and −7.59%, respectively. The I1 result is an exploratory conditional improvement. H2 does not establish an effect, and absence of an effect is not equivalence. Better reconstruction is therefore not sufficient to claim improved forecasting at every installation.

## Public aggregate outputs

| File under `outputs/` | Contents |
| --- | --- |
| `forecast_target_support.csv` | Eligible, observed-target and excluded episode counts in all 27 site/fold/duration cells |
| `forecast_all_methods.csv` | Equal-weight site scores for all 32 policies, in nMAE and kW |
| `forecast_all_method_cells.csv` | The corresponding fold/duration cell scores |
| `forecast_selected_policies.csv` | Original B and corrected-linear scores and descriptive changes |
| `forecast_contrasts.json` and `.csv` | Both contrasts under every physical assumption and block length, with explicit inference flags |
| `forecast_bootstrap_draws.npz` | All logical bootstrap attempts in stable column order |
| `bootstrap_draws_schema.json` | Draw-array identities, shape, seed derivation and column definitions |
| `reconstruction_interval_replay.csv` | Independent verification of every original reconstruction interval and support flag |
| `forecast_revision_summary.json` | Version, estimand, population and limitations |
| `forecast_revision_manifest.json` | Input, output and script hashes plus software versions |

The relative error `Change (%)` is `100 * (B(E)/B(P) - 1)`. Its interval applies to the absolute difference in nMAE, not the percentage change. The H1 asterisk flags insufficient temporal support. Alternative hypotheses remain available in the complete outputs and must not be selected by the most favorable test result.

## Remaining work

Retrieve the original server split manifests for archival provenance, examine mechanisms behind missing future targets, and assess whether a future experiment should preregister the available-target estimand. Audit inference fallbacks and selected correction parameters in the full ledger. Confirm meter semantics and test forecast transfer on additional installations before making a general operational claim. This revision updates reported evidence without resolving those scientific limitations.

Historical manifests retain hashes for excluded private inputs and private-workflow outputs. They are analysis provenance, not a public package inventory.
