# Optional external UCI validation

**Work in progress — separate extension, selected by neither current profile.** The adapter has controlled tests, but an external experimental design, execution, and analysis remain TODO in [development status](development_status.md).

The [official UCI record](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) describes 370 independent consumers, 15-minute kW values, pre-activation zero placeholders and special Portuguese clock-change values. Its license is CC BY 4.0; cite Trindade (2015), DOI [10.24432/C58C86](https://doi.org/10.24432/C58C86). The archive is approximately 249 MB, with a roughly 678 MB uncompressed table. These are source facts, separate from the conservative benchmark policies below.

No external archive was downloaded or benchmark experiment executed during local adapter validation. Seven small local tests verify selection, ingestion, DST exclusions, download provenance and manifest integrity. Their tabular values are labeled test fixtures, not UCI observations. The downloader records a SHA-256 only after actual bytes are downloaded; there is no invented upstream reference checksum.

`energygap.data.external` exposes three functions:

- `download_uci(destination, expected_sha256=None, maximum_bytes=300_000_000)` streams the official archive only on explicit invocation, hashes its actual bytes and saves a sidecar download manifest. Existing verified files are reused. An existing archive without provenance is not silently adopted or overwritten.
- `freeze_uci_selection(source, train_end, count=30, min_train_observations=2880, min_train_coverage=.9, manifest_path=...)` reads consumer values only before the exclusive training cutoff, checks activation/support, sorts eligible exact IDs by SHA-256 and freezes the selected list. Fewer than 30 eligible consumers raises an error by default. A separate source hash covers all bytes for provenance; it does not enter the eligibility estimator. The train fingerprint covers only train rows.
- `load_uci_consumers(source, selection)` verifies source and selection hashes and returns one `SeriesData` per consumer. Each has 900-second nominal cadence, no invented auxiliaries and no energy counters. The ZIP can be read directly without extracting untrusted paths.

Freeze the list using the first planned training partition before inspecting test outcomes. Fit model/scaler/statistical components independently within each consumer's chronological training period. Do not treat consumers as contemporaneous channels of one device, or multiply their independent-unit count by masks or training seeds. Register the chosen time range and fold manifests before test evaluation; an example cutoff below illustrates the API and is not a completed scientific protocol.

```python
from energygap.data.external import (
    download_uci, freeze_uci_selection, load_uci_consumers,
)

archive = 'raw_data/external/electricityloaddiagrams20112014.zip'
download = download_uci(archive)  # Explicit network action: approximately 249 MB.
selection = freeze_uci_selection(
    archive,
    train_end='2012-01-01T00:00:00',  # Example predeclared cutoff, exclusive.
    count=30,
    manifest_path='data/splits/uci-consumers.json',
)
consumers = load_uci_consumers(archive, selection)
assert len(consumers) == 30
assert all(s.nominal_seconds == 900 for s in consumers)
```

The conservative activation proxy is the first positive observation available in train. Earlier zeros are excluded from truth with `quality_flag=preactivation_placeholder`; their original zero values remain preserved. Zeros after that point remain valid observations. Negative finite anomalies are preserved, not turned into missingness. A consumer first activated only after the training cutoff cannot become eligible from test information. This proxy can censor genuine initial zero load and biases support toward established consumers; it is a documented benchmark choice, not recovered creation metadata.

The source has 96 clock labels per day, including March zero placeholders and October aggregated readings. This adapter excludes each complete last-Sunday transition date in March/October, records those dates and creates no ground truth from them. Only other, unambiguous labels are mapped with an explicit mainland `Europe/Lisbon` benchmark assumption. Original local measurement labels retain their timezone; UTC grid times provide a strictly increasing physical coordinate. Excluded transition periods become natural missing grid slots. The adapter never resolves an ambiguous/nonexistent label by silently choosing a fold. The full-day exclusion is conservative and changes the support; report it rather than claiming an unchanged UCI evaluation. Actual receipt logs and interval timestamp start/end semantics are not established by this assumption.

This track can test imputation and subsequent forecasting. It cannot validate measured bidirectional energy consistency: it has no independent import/export counters. No function manufactures counters by integrating hidden power. Any later synthetic aggregate-information experiment requires a separate label and protocol. Foundation-model exposure to this public dataset during pretraining remains unknown.
