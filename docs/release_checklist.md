# Release maintenance and scientific validation checklist

The manuscript is **under review at IEEE ICIT 2027**. The focused execution and
its versioned analyses are complete within their documented scope. Further
scientific validation remains work in progress. An unchecked item is pending
work, not an implied result. See [development status](development_status.md) for
detailed acceptance criteria.

## This repository

The public repository contains code, documentation and aggregate results. The
manuscript and individual records are maintained separately. Original primary
forecast values remain unavailable under the frozen complete-outcome rule;
forecast version 2 provides explicitly exploratory common-target estimates and
intervals. The expanded full experiment and stronger physical/generalization
claims remain future work.

Data are available upon reasonable request to the corresponding author or
[repository maintainer](https://github.com/chgferreira), subject to applicable
confidentiality and data-holder requirements. This checklist applies to subsequent
software releases and scientific revisions.

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

- [x] Record the current author order and the manuscript's under-review status in
  citation metadata; maintain the manuscript outside the public repository.
- [ ] Review code/checkpoint/data terms, including TS-ICL noncommercial restrictions;
  do not redistribute unresolved R24 source or unlicensed artifacts.
- [x] Publish the authorized code-and-aggregate repository and document data access
  upon reasonable request.
- [ ] Apply the appropriate access terms and disclosure review to requests for
  residential/industrial records, processed series, truth, predictions or
  identifying metadata.
- [ ] Record future tags, archival deposits and DOI assignments separately from
  the experiment and analysis versions. An under-review manuscript is not an
  accepted or published proceedings paper.
