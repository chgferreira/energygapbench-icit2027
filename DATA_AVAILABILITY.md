# Data availability and public release scope

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

## Excluded contents

The manuscript PDF and LaTeX project, editorial prompts, private raw telemetry,
per-episode losses, target-membership records, dated episode manifests,
full protocol/dataset metadata, runtime history, SQLite state, predictions,
training checkpoints and model weights are not distributed here. Original
restricted evidence and the author manuscript package are maintained separately.
Case labels H1, H2 and I1 are pseudonymous study identifiers, not a claim of
anonymity for the underlying records.

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
No open download or general data-access agreement has been established for those
inputs. Access inquiries can be addressed through the repository maintainer;
availability is subject to the data holders' authorization. Synthetic tests verify
software behavior and do not reproduce real-site effects.

## Terms

The MIT license covers original code. Public availability of aggregate results
is not a license for private telemetry, the separately distributed manuscript,
or third-party assets. No additional redistribution license is granted for
derived data by this software license. Preserve provenance and attribution when
reporting aggregate findings. External model artifacts retain upstream terms.
