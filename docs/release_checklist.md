# Release and manuscript checklist

**Work in progress.** A source package can be prepared while scientific experiments
remain unfinished, provided that status and scope are explicit. An unchecked item
is pending work, not an implied result. Use [development TODO](development_status.md)
for detailed experiment/test acceptance criteria.

## This repository

The compact focused run is operationally complete, with result-derived analysis
published as aggregates only. The manuscript is distributed separately. Original primary forecast values remain unavailable; forecast version 2 provides explicitly exploratory common-target estimates and intervals.
The expanded full experiment and stronger physical/generalization claims remain
future work. This checklist governs public release, not whether files can be
prepared locally.

## Software and documentation

- [ ] Validate the selected profile and relevant changed behavior; retain environment,
  versions, test logs, and skip reasons. Run native CUDA checks on the target hardware.
- [ ] Check README/server commands against the implementation and focused/full YAMLs;
  distinguish validation level `full` from experiment preset `full`.
- [ ] Retain source, dependency, checkpoint, mask, and epoch-panel identities;
  mark historical pilots and obsolete setup procedures clearly.
- [ ] Build the allowlisted source/server archive and verify its manifest, CRC, and
  exclusions after extraction. Include the WIP statement and TODO roadmap.

## Scientific results and manuscript

- [ ] Identify the executed design and actual methods, conditions, selection budgets,
  panel/full scoring populations, seeds, horizons, and measurement hypotheses.
- [ ] Verify declared versus available coverage and all failure/fallback denominators;
  retain negative results, absent-support NA values, and algorithm identities.
- [ ] Review selection/convergence, paired effects, temporal support, bootstrap
  invalidity/sensitivity, multiplicity, and physical assumptions.
- [ ] Document focused HPO/panel tradeoffs and any adaptive decisions after test
  exposure. Do not present pending full/extensions as completed evidence.
- [ ] Align manuscript tables/figures/claims and data availability with the reviewed
  run artifacts. Verify bibliographic metadata and qualify novelty claims.

## Permissions and publication

- [ ] Confirm authorship/ownership and complete citation/deposit metadata.
- [ ] Review code/checkpoint/data terms, including TS-ICL noncommercial restrictions;
  do not redistribute unresolved R24 source or unlicensed artifacts.
- [ ] Obtain permission/access terms and disclosure review before sharing residential/
  industrial records, processed series, truth, predictions, or identifying metadata.
- [ ] Publish a repository/deposit/tag/DOI only through an explicitly authorized action.
  Local package creation does not constitute public release or license private data.
