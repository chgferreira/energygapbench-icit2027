# Statistical analysis plan

**Status: implemented analysis rules with scientific validation still in progress.**
Both experiment profiles share the primary paired estimands and support rules.
Focused changes methods, conditions, HPO, and checkpoint selection as declared in
[experiment profiles](experiment_profiles.md); those changes must accompany results.
Unknown instrument semantics keep current physical branches exploratory.

## Endpoints and aggregation

Primary reconstruction is per-episode MAE divided by the training q95−q05 scale,
falling back to positive training standard deviation. Constant training series
produce NA normalization. Average losses across training seeds for the same episode
first, then across episodes in each fold/duration; weight durations 15/60/180 minutes
equally and folds by 1/3. Do not substitute loss of an averaged reconstruction for
average loss. Point-micro aggregation is separate and has different weighting.

Primary downstream performance is normalized absolute error of a fixed direct ridge
forecast at issuance + one hour. Ridge uses 288 history positions and known calendar,
fits original training targets, and selects alpha from (0.1, 1, 10) chronologically
on validation with the common causal natural-gap filler. Each fold/horizon shares
its fitted weights across imputers. Original future targets are matched within
90 seconds with deterministic ties and recorded offsets. Both current profiles
also evaluate 30-minute, three-hour, and six-hour horizons as additional endpoints;
the historical reduced pilot evaluated only one hour.

## Comparators and planned family

B(site, fold) is selected exclusively among the six conventional baselines by
validation P nMAE, equally across C3–C5; latency then name break ties. Neural/frozen
models receive descriptive comparisons outside that primary comparator pool.

The planned primary family contains energy gain (B+correction minus B) and complexity
gain (B+correction minus linear+the same correction), two endpoints and three sites:
12 tests. Lower differences favor correction. P versus E includes both extra
information and its correction policy; it is not a matched-information architecture
comparison. Hypothetical measurement branches do not acquire confirmatory p values.
No profile or physical hypothesis is selected by the smallest test p value.

## Dependence, uncertainty, and missing support

The paired moving-block bootstrap assigns a complete episode, all methods, and
computational repetitions to reconnection day. Preserve full calendars including
empty days and stratify by site/fold; blocks never cross folds. The main block is
7 days, with 2/14-day sensitivity and 10,000 attempted replicas. Persistent replica
IDs and seed 20261031 retain invalid attempts rather than redrawing them. Training
seeds and masks do not create independent temporal days or new datasets.

The conservative support heuristic requires at least 20 nonoverlapping block lengths
summed across folds, at least four per fold, and each duration on at least 20 distinct
days. It is not a theorem. A point estimate needs every required fold/duration.
Bootstrap interpretation conditions on fitted models and realized masks, with weak
dependence/local stationarity assumptions; it does not capture all training,
selection, or between-site uncertainty.

Percentile 95% intervals are pointwise, not simultaneous. Centered bilateral
bootstrap p values use (1 + count(|Db−Dhat| ≥ |Dhat|))/(B+1). Holm covers the fixed
12-test family; unsupported raw p values stay absent and count as 1 for multiplicity.
Any invalidity suspends confirmatory p values; conditional-valid percentile intervals
are exploratory. BY at q=.05 is available for predefined exploratory families with
valid marginal tests. No mass pointwise tests or pseudo-replication of seeds/windows.
Non-significance does not establish equivalence.

A provisional 5% practical reduction requires application justification before
confirmatory interpretation. Validation-only 2/5/10% threshold, power, and support
sensitivity remains TODO. See [development status](development_status.md).

## Failure and scope reporting

Retain failed episodes with the declared common seasonal/median fallback, requested
method identity, and reason. Solver failure retains the base. Report algorithm-
conditional success separately from operational-policy performance. Candidates
unavailable before execution are not reproduced by fallback baselines.

Present profile, selection budgets/panel, denominator and temporal support, excluded
strata, missing targets, attempted/invalid replicas, and fallback rates alongside
effects. Absent results remain NA; do not create zero-error rows. Descriptive
multi-horizon/model extensions must be distinguished from the fixed primary family.
The draft's statistical claims require review against actual support and physical
evidence even after all eligible jobs finish.
