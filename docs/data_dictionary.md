# Canonical data dictionary

Both [experiment profiles](experiment_profiles.md) share this representation. Field provenance and unresolved physical assumptions are described in the [dataset card](dataset_card.md).

`CanonicalRecord` retains one source observation or an explicitly absent grid slot. Original CSV/TXT/Parquet sources remain untouched. Excel readers are optional and are used only when such files are supplied. SHA-256 identifies each source; `source_record_id` combines that hash and original line number. `site_id` identifies an independent installation, never a spatial channel across H1/H2/I1.

| Field | Meaning |
|---|---|
| `source_file_id`, `source_record_id` | Source hash and stable original record identity |
| `measurement_time` | Original measurement timestamp with evidenced timezone; never replaced by its grid slot |
| `availability_time` | Earliest admitted arrival; absent arrival logs imply the explicit optimistic equality with measurement time |
| `grid_time` | Closest nominal slot if within the frozen tolerance |
| `value_original`, `unit_original` | Original numeric value and documented unit evidence; instrument confirmation remains separate |
| `value_canonical` | kW; positive import, negative export after explicit conversion |
| `observed_original` | Finite, nonconflicting actual target observation |
| `natural_missing` | Missing/invalid original target or absent slot; never evaluation truth |
| `artificial_hidden`, `mask_id`, `gap_id` | Perturbation metadata, applied across the temporal stream before extracting windows |
| `available_to_method` | Usability before the query-specific arrival and masking cutoff |
| `padding_mask` | Window extension without an original record; never an observation/loss target |
| `imputed` | Output provenance; imputations do not become observed ground truth |
| `quality_flag` | Retained anomaly/alignment/collision provenance |
| `fold_id` | Elapsed-time rolling origin membership |
| `elapsed_since_observed` | Seconds since the latest admitted unhidden observation, recomputed after masking |
| `import_counter`, `export_counter` | Distinct cumulative kWh channels after explicit conversion |
| `counter_availability_time` | Independent counter arrival when supplied; otherwise the declared synchronous assumption |
| `auxiliaries` | Same-installation channels with separately documented units |

`Query` contains only admitted values, epoch-second times, original IDs, missing/padding masks, target-gap/evaluation support masks, recalculated ages, allowed auxiliaries and admissible energy endpoint intervals. It contains no raw record collection or hidden target values. `EvaluationTruth` is returned separately and remains evaluator-owned; if serialized, it stays in excluded private evaluation artifacts and must never be passed to an imputer. Its values are NaN outside artificial, originally observed targets of the current episode. An episode is scored once per mask replica.

Alignment is not interpolation or aggregation. `regularize` adds absent nominal slots as NaN. Conflicting collisions remain unusable; an identical target/arrival duplicate uses deterministic first source identity and records all duplicate IDs. Zero, negative and extreme finite values remain valid observations. No prior manuscript value repairs a natural gap.
