# Dataset card and access status

**Work in progress.** H1/H2/I1 are private, previously studied installation records.
Both current profiles use the same original files and provenance rules. This card
states available evidence and unresolved assumptions; it is not a data-release license.

## Sources and installation roles

A complete experiment expects `raw_data/H1.csv`, `raw_data/H2.csv`, and
`raw_data/I1.csv`. Those private raw files are not bundled in this repository.
Dataset YAMLs record explicit mapping, SHA-256, UTF-8 BOM, semicolon separators,
comma decimals, and dot thousands separators. Ingestion supports discovery/mapping
without changing original bytes. Local audit counts, periods, coverage, and split
diagnostics are private artifacts, not copied into the public synthetic example.

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

Data are user-supplied; public redistribution permission and a data license have
not been granted. Collection/consent records, anonymization review, and an authorized
access mechanism remain pending. Raw/processed real series, truth, identifying
predictions/masks/dates, and private audits stay outside ordinary source packages.
This local ICIT publication candidate includes a separate compact derived-result
bundle with timestamps and execution metadata. Its intended public sharing scope
and permissions still require review under [Data availability](../DATA_AVAILABILITY.md).
MIT code terms do not license telemetry. Full public raw-data reproduction on these
three sites is therefore not established.

Synthetic fixtures support open implementation checks. The [optional UCI adapter](external_validation.md)
has separate terms, activation/DST rules, and no independent energy counters; it
is not already an executed external benchmark. See [development TODO](development_status.md)
for physical metadata, access, and external-validation work required before release.
