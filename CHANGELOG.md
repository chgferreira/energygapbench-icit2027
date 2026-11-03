# Repository updates

## 2026-11-03

- Documented the manuscript status as under review at IEEE ICIT 2027 and the
  availability of study data from the authors upon reasonable request.
- Aligned the overview, dataset card, access policy, research roadmap and citation
  metadata with the public code-and-aggregate release.
- Added reference hardware documentation and a machine-readable inventory for
  the dual Xeon Gold 6130 server, its 32 physical cores / 64 logical processors,
  386 GB RAM and V100 16 GB GPU. Hardware capacity is kept distinct from measured
  resource consumption and frozen execution settings.
- Declared `threadpoolctl` as a base runtime dependency for parallel report
  workers. Regenerated the minimal dependency export without changing previously
  pinned versions and added an early dependency check to continuous integration.
- Corrected the root dataset exclusion so that `configs/datasets/` contracts
  are included in clean checkouts while raw dataset directories remain excluded.
- Retained the original experiment implementation, protocol identity and all
  numerical results. Documentation and packaging updates do not revise the
  scientific estimands or the recorded run configuration.

The focused execution, the common-target forecast revision and the additional
energy controls retain their separate analysis identities. Their methodology and
interpretation are documented under `analysis/`.
