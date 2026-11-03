# Forecast reporting revision history

## Original frozen execution

Run `conference-20261101T165101Z-1789282` completed the focused computation. The immutable report under `results/` requires complete episode outcomes for each aggregated metric. Originally unavailable future targets lead to unavailable site-level one-hour forecast scores and paired forecast contrasts. These outputs remain unchanged.

## Initial analysis supplement

The analysis under `analysis/outputs/` preserves the original report and adds explicitly post hoc point estimates on common available targets. It does not provide new forecast intervals. Its source data, scripts and outputs are retained for comparison.

## Version 2, 2026-11-03

Revision ID `forecast-common-target-v2` documents an available-target forecast population and adds exploratory paired moving-block intervals without rerunning training, inference or correction. It uses 68,217 common observed targets, preserves all 32 policies and all nine fold/duration strata per site, and retains the original validation-selected comparator. The relative changes agree with the initial point-only supplement.

Full day calendars were reconstructed from checksum-verified original source timestamps. The derivation exports metadata only. Before new inference, every original reconstruction interval and support field was reproduced for the main and sensitivity block lengths. The new forecast bootstrap uses the existing 10,000-attempt, stable-seed calendar-day procedure. Individual draws, support flags, version metadata and integrity hashes are exported.

This revision changes the forecast aggregation population from the original all-episode completeness requirement to explicitly common originally observed future targets. It was adopted after inspecting the completed execution. It must be described as a versioned exploratory reporting revision, not as an originally completed confirmatory endpoint or a previously preregistered analysis. The original unavailable results remain part of the provenance.

No architecture, weights, predictions, correction choice, training budget, test mask, physical hypothesis or raw measurement was modified. No new installation data were added. The update does not establish validity under non-random target absence, confirm meter semantics, or provide multiplicity-adjusted significance.
