# Data availability and public release scope

## Data access

**The data supporting this study are available from the authors upon reasonable
request.** Contact the corresponding author through the
[repository maintainer](https://github.com/chgferreira) to discuss access. Specify
the sites or artifacts required and the intended research purpose. Raw telemetry,
individual derived results and additional execution artifacts are shared subject
to applicable privacy and data-holder conditions.

The associated manuscript is **under review at IEEE ICIT 2027**. The repository
provides public code and aggregate results for examination and reproduction
while the manuscript is maintained separately.

## Public contents

This repository contains original software, configurations, synthetic tests,
analysis scripts, aggregate study results and aggregate-derived figures. The
public release scope is **code and aggregated results only**.

The original six report tables/contrast files under `results/` retain their
original bytes. `protocol.public.json` is a separately identified extract of
experiment settings, with the original lock's byte hash and scientific fingerprint.
It is not the original lock and must not be substituted during run resumption.
Analysis summaries are grouped by site, fold, duration, method or policy.
Bootstrap NumPy arrays contain resampled aggregate contrast estimates, with no
individual losses, membership or sampling-index matrices.

## Individual records and supporting artifacts

Raw telemetry, individual episode losses, target membership, detailed execution
records, predictions and model checkpoints are maintained separately from the
public aggregate release. Case labels H1, H2 and I1 identify the study
installations without making individual records publicly available. Requests
for these supporting artifacts follow the access process above.

## Reproduction levels

The public command rebuilds figures and numerical tables from aggregates,
checks fixed fold/duration aggregation and verifies published percentile
intervals and centered exploratory p-values from aggregate bootstrap draws.
It does not repeat episode-level pairing, temporal-support auditing, bootstrap
resampling, data ingestion, fitting or prediction. The historical audit summaries
record checks made with restricted inputs, rather than checks independently
possible from this public package.

Full recomputation requires authorized original data or restricted loss exports
and calendar metadata, their recorded hashes, and the corresponding software.
These inputs are available upon reasonable request, subject to the applicable
access conditions. Their availability does not imply an unrestricted public
download or transfer of data ownership. Synthetic tests verify
software behavior and do not reproduce real-site effects.

## Terms

The MIT license covers original code. Public availability of aggregate results
is not a license for private telemetry, the separately distributed manuscript,
or third-party assets. No additional redistribution license is granted for
derived data by this software license. Preserve provenance and attribution when
reporting aggregate findings. External model artifacts retain upstream terms.
