# Dataset card and access status

H1/H2/I1 are previously studied residential and industrial installation records
used in the focused study **under review at IEEE ICIT 2027**. The public repository
contains aggregate results and reproducible code. Original data and restricted
derived records are **available upon reasonable request** to the corresponding
author or [repository maintainer](https://github.com/chgferreira), subject to
applicable confidentiality and data-holder requirements.

Both experiment profiles use the same original files and provenance rules. The
focused execution is complete; the expanded full design is a future extension.
This card records the available evidence and remaining measurement assumptions.

## Sources and installation roles

A complete experiment expects `raw_data/H1.csv`, `raw_data/H2.csv`, and
`raw_data/I1.csv`. Those private raw files are not bundled in this repository.
Dataset YAMLs record explicit mapping, SHA-256, UTF-8 BOM, semicolon separators,
comma decimals, and dot thousands separators. Ingestion supports discovery/mapping
without changing original bytes. Aggregate counts and coverage are documented in
the published analysis outputs. Individual timestamps, observations, masks and
membership remain separate from the public synthetic example and aggregate exports.

H1/H2 are residential sites with generation and I1 is industrial, according to the
supplied study context. They cover separate periods. No simultaneous microgrid,
battery, or separate photovoltaic channel is assumed. The study declares new
analyses on a known base; it is not a blind experiment on unseen data.

## Evidence and unresolved semantics

The `potenciaAtivaTotal` header alone does not specify units or measurement semantics.
The study configuration uses W, negative import, and America/Sao_Paulo timezone on
the basis of supplied internal evidence and power/counter audit. These are not
manufacturer confirmation or independent instrument calibration. Canonical power
is kW with positive import/negative export after the explicit conversion.

Instantaneous versus interval-mean power and the meaning/placement of `atualizadoEm`
remain unresolved. Strict physical configuration validation retains those unknowns.
Current server profiles run separately declared instantaneous/interval-mean-start/
interval-mean-end hypotheses as exploratory branches; no hypothesis is selected
as instrument truth by its test error or counter agreement.

Consumed/injected cumulative active energy is labeled in kWh. Counter resolution,
accuracy, update cadence, independent arrivals, rollover/reset policy, and power
synchronization are not documented sufficiently. Negative increments are not
silently wrapped, and constant increments can represent valid zero energy. Printed
precision is not calibration. Any tolerance fitting uses train-only support within
the declared hypothesis, not hidden test power.

No receipt logs were supplied. The optimistic availability equality with measurement
time is explicit. Five-minute regularization and a 90-second nearest-slot tolerance
are declared, without interpolation of missing truth. Nominal timestamps remain
separate from original measurement times. Support/eligibility counts are not
independent episode counts or physical-identifiability evidence.

## Permissions, disclosure, and reproduction

The public release includes code, aggregate results, aggregate bootstrap contrast
estimates and figures. Original telemetry, processed individual series, evaluation
truth, per-episode losses, membership, dated masks, predictions and the complete
execution ledger are maintained separately. Case labels are pseudonymous study
identifiers and do not by themselves anonymize detailed records.

For a data request, describe the intended research use and the required artifacts
to the corresponding author or [maintainer](https://github.com/chgferreira).
Access is considered under the relevant confidentiality and data-holder
requirements. No unrestricted public download or independent data license is
implied. The MIT software license covers code and does not license telemetry.
See [data availability](../DATA_AVAILABILITY.md) for the release scope and access policy.

The public aggregate workflow rebuilds figures and tables and checks intervals
from aggregate bootstrap estimates. Full data ingestion, training, predictions
and episode-level resampling require the separately obtained authorized inputs.

Synthetic fixtures support open implementation checks. The [optional UCI adapter](external_validation.md)
has separate terms, activation/DST rules, and no independent energy counters; it
is not already an executed external benchmark. See [development TODO](development_status.md)
for further physical-metadata and external-validation work.
