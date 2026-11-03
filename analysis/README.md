# Reproduce the published aggregates and figures

This analysis accompanies the focused execution
`conference-20261101T165101Z-1789282`. It preserves the original report and
separately identifies the common-target forecast revision and same-information
energy-control follow-up. All physical hypotheses and inferential comparisons
remain exploratory. The manuscript is distributed separately.

## Public workflow

From the repository root, using Python 3.12:

```bash
python3.12 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r analysis/requirements.txt
ANALYSIS_PYTHON=.venv-analysis/bin/python bash scripts/reproduce_icit.sh
```

No GPU, raw telemetry, episode-level records or TeX installation is required.
The command calls `analysis/scripts/reproduce_public.py` and writes to
`analysis/public_outputs/`, leaving archived aggregates unchanged.

| Generated output | Meaning |
| --- | --- |
| `figures/reconstruction_duration.{pdf,svg,png}` | B and linear, P/E, across durations for each site |
| `figures/reconstruction_contrasts.{pdf,svg,png}` | Original paired reconstruction intervals for all three physical hypotheses |
| `figures/reconstruction_methods.{pdf,svg,png}` | Descriptive P/E comparison of all eight methods |
| `methods_nmae.csv` | Eight-method site-level instantaneous P/E table |
| `energy_controls.csv` | Equal-duration/fold scalar-control comparison |
| `forecast_interval_checks.csv` | 54 checks of percentile intervals and exploratory p-values from aggregate draws |
| `controls_interval_checks.csv` | 18 corresponding energy-control checks |
| `validation.json` | Input hashes, aggregation checks and explicit reproduction scope |

The script verifies 96 site/policy summaries against all nine fixed fold/duration
cells. It does not rebuild episode-level losses, pairing, selection, sampling
calendars or temporal-support flags. The compressed bootstrap arrays contain
aggregate differences only. Recomputing their percentiles is distinct from
repeating the underlying day-block resampling.

The duration plot uses equal fold weights and site-specific axes starting at zero.
The contrast figure reads original seven-day percentile intervals, with hollow H1
markers indicating insufficient temporal support. Neither figure establishes
confirmatory superiority. The third plot includes all methods in a fixed order.
Fonts, sizes, timestamps and SVG salts are controlled for reproducible rendering
within the pinned analysis environment.

## Evidence and estimands

The selected comparator B is chosen on validation from six conventional methods.
SAITS and MOMENT are outside that pool. Energy contrast is B(E) minus B(P);
comparator contrast is B(E) minus linear(E). Negative values favor the first term.
Reconstruction first averages training-seed losses within episodes, then episodes
within a fold/duration cell, followed by equal duration and fold weights. Sites
remain separate. Missing required outcomes remain unavailable in original reports.

The initial `outputs/` summaries were computed from restricted episode losses and
checked against all 864 original cells, 96 site aggregates and 36 point estimates,
including unavailable forecasts. Their audit manifests are historical provenance,
not a claim that restricted inputs are included here. All retained CSV files are
aggregates; `episode_support.csv` contains counts per stratum, not episode records.

The [forecast version 2](forecast_v2/README.md) evaluates 68,217 common observed
one-hour targets, preserves all 27 strata and supplies exploratory conditional
intervals. H1 fails the main seven-day support heuristic. P-values are unadjusted,
not Holm-confirmatory. Target missingness and selection uncertainty remain limits.

The [energy controls](energy_controls/README.md) use the same admitted counters
and instantaneous directional operator with a scalar shift, with/without a
tolerance zone. All 68,580 test episode instances are retained. Different search
budgets and correction families prevent interpreting this as an isolated gamma
ablation. The [I1 tolerance audit](outputs/energy_diagnostics/train_tolerance_audit.csv)
reconstructs a zero-export tolerance gate from verified training data, without
claiming recovered per-query solver logs or confirmed meter semantics.

## Restricted-input workflows

The full algorithms remain available for authorized reproduction. Place original
artifacts outside the public working tree. Paths below are examples to replace
with authorized locations, not downloadable public data.

```bash
python analysis/scripts/analyze_icit.py --run-dir /authorized/run --output-dir restricted_analysis/original
python analysis/scripts/summarize_engineering.py --run-dir /authorized/run --analysis-dir restricted_analysis/original
python analysis/scripts/forecast_common_target_v2.py --run-dir /authorized/run --calendar-manifest /authorized/calendar_provenance.json --output-dir restricted_analysis/forecast_v2
python analysis/scripts/summarize_energy_controls.py --controls-dir /authorized/energy_controls --output-dir restricted_analysis/control_summary
python analysis/scripts/energy_control_contrasts.py --run-dir /authorized/run --controls-dir /authorized/energy_controls --calendar-manifest /authorized/calendar_provenance.json --output-dir restricted_analysis/control_statistics
```

The original analysis requires the original protocol lock/fingerprint, episode
loss Parquet and compact report summaries. Forecast resampling also requires full
calendar provenance. Control summaries require paired control losses, validation
completion marker and the control protocol. Replaying controls from power data
additionally requires the experiment engine and hash-matched private telemetry;
see the control guide. The public extract cannot replace the original lock.
Generated restricted records are ignored by Git under `restricted_analysis/`.

## Tests and remaining work

In the CPU development environment (`requirements.lock.txt`, including pytest and
the experiment engine dependencies),
`python -m pytest -q analysis/tests analysis/forecast_v2/tests` runs synthetic
checks of weighting, pairing, missing outcomes, scalar controls and aggregate
summary validation. Full engine checks are documented in
[reproduction levels](../docs/icit2027_reproduction.md).

Further work includes meter semantics/resolution, original solver/fallback logs,
SAITS HPO/panel sensitivity, target-missingness mechanisms, a fixed-operator
component ablation with common search budgets and external validation. The full
preset remains a separately identified future extension. No completed software
check establishes convergence, nominal interval coverage or population validity.
